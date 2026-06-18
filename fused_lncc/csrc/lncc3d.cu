// Fully-fused 3D Local (squared) NCC: reads p,t once, computes all 5 box-sums
// (Sp,St,Spp,Stt,Spt) separably in shared memory, then ncc + cached backward adjoints
// in one pass. Forward only here; backward reuses the box-sum (see __init__.py).
//
// Semantics (matches MONAI rectangular LNCC / our _ncc_loss):
//   cross=Spt-Sp*St/n ; var_p=Spp-Sp^2/n ; var_t=Stt-St^2/n ; n=k^3
//   ncc = cross^2 / ((var_p+dr)(var_t+dr)) , clamped to [0,1]
// Outputs: ncc map, and adjoints A,B,Cc (dncc/dSp, dncc/dSpp, dncc/dSpt); zeroed where clamped.
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#define TX 8
#define TY 8
#define TZ 8

__device__ __forceinline__ float to_float(float x) { return x; }
__device__ __forceinline__ float to_float(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ void store_g(float* p, float g) { *p = g; }
__device__ __forceinline__ void store_g(__nv_bfloat16* p, float g) { *p = __float2bfloat16(g); }

// S = input dtype (float or __nv_bfloat16). Everything internal is fp32 (shared mem + accumulation):
// inputs are read and cast to float on load; outputs (ncc map + adjoints) are always fp32.
template <int HALO, typename S>
__global__ void lncc_fwd(const S* __restrict__ P, const S* __restrict__ T,
                         float* __restrict__ NCC, float* __restrict__ AA,
                         float* __restrict__ BB, float* __restrict__ CC,
                         int B, int D, int H, int W, float n, float dr) {
    const int SX = TX + 2 * HALO, SY = TY + 2 * HALO, SZ = TZ + 2 * HALO;
    const int K = 2 * HALO + 1;
    extern __shared__ float sm[];
    float* sp = sm;                       // p tile  SZ*SY*SX
    float* st = sp + SZ * SY * SX;        // t tile  SZ*SY*SX
    float* X = st + SZ * SY * SX;         // 5 X-sums  5*SZ*SY*TX
    float* Y = X + 5 * SZ * SY * TX;      // 5 Y-sums  5*SZ*TY*TX
    const int xsz = SZ * SY * TX, ysz = SZ * TY * TX;

    const int nxt = (W + TX - 1) / TX;
    int b = blockIdx.x / nxt, xt = blockIdx.x % nxt;   // batch folded into gridDim.x (z/y cap at 65535)
    int x0 = xt * TX, y0 = blockIdx.y * TY, z0 = blockIdx.z * TZ;
    int tid = threadIdx.z * (TY * TX) + threadIdx.y * TX + threadIdx.x, nthreads = TZ * TY * TX;

    // load p,t tile+halo
    for (int idx = tid; idx < SZ * SY * SX; idx += nthreads) {
        int sx = idx % SX, sy = (idx / SX) % SY, sz = idx / (SX * SY);
        int gx = x0 - HALO + sx, gy = y0 - HALO + sy, gz = z0 - HALO + sz;
        float vp = 0.f, vt = 0.f;
        if (gx >= 0 && gx < W && gy >= 0 && gy < H && gz >= 0 && gz < D) {
            long o = (((long)b * D + gz) * H + gy) * W + gx;
            vp = to_float(P[o]); vt = to_float(T[o]);   // bf16/float -> fp32
        }
        sp[idx] = vp; st[idx] = vt;
    }
    __syncthreads();
    // X-pass: 5 sums over x window
    for (int idx = tid; idx < xsz; idx += nthreads) {
        int tx = idx % TX, sy = (idx / TX) % SY, sz = idx / (TX * SY);
        float ap = 0, at = 0, app = 0, att = 0, apt = 0;
#pragma unroll
        for (int dx = 0; dx < K; ++dx) {
            float p = sp[(sz * SY + sy) * SX + tx + dx], t = st[(sz * SY + sy) * SX + tx + dx];
            ap += p; at += t; app += p * p; att += t * t; apt += p * t;
        }
        X[0 * xsz + idx] = ap; X[1 * xsz + idx] = at;
        X[2 * xsz + idx] = app; X[3 * xsz + idx] = att; X[4 * xsz + idx] = apt;
    }
    __syncthreads();
    // Y-pass
    for (int idx = tid; idx < ysz; idx += nthreads) {
        int tx = idx % TX, ty = (idx / TX) % TY, sz = idx / (TX * TY);
        float s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0;
#pragma unroll
        for (int dy = 0; dy < K; ++dy) {
            int j = (sz * SY + ty + dy) * TX + tx;
            s0 += X[0 * xsz + j]; s1 += X[1 * xsz + j]; s2 += X[2 * xsz + j];
            s3 += X[3 * xsz + j]; s4 += X[4 * xsz + j];
        }
        Y[0 * ysz + idx] = s0; Y[1 * ysz + idx] = s1; Y[2 * ysz + idx] = s2;
        Y[3 * ysz + idx] = s3; Y[4 * ysz + idx] = s4;
    }
    __syncthreads();
    // Z-pass: finalize at each output voxel
    int ox = x0 + threadIdx.x, oy = y0 + threadIdx.y, oz = z0 + threadIdx.z;
    if (ox < W && oy < H && oz < D) {
        float Sp = 0, St = 0, Spp = 0, Stt = 0, Spt = 0;
#pragma unroll
        for (int dz = 0; dz < K; ++dz) {
            int j = ((threadIdx.z + dz) * TY + threadIdx.y) * TX + threadIdx.x;
            Sp += Y[0 * ysz + j]; St += Y[1 * ysz + j]; Spp += Y[2 * ysz + j];
            Stt += Y[3 * ysz + j]; Spt += Y[4 * ysz + j];
        }
        float cross = Spt - Sp * St / n;
        // clamp raw variance to >=0 before the smoothing floor: guards the sum-of-squares
        // cancellation on near-constant high-magnitude windows (keeps ncc finite & >=0).
        float var_p = fmaxf(Spp - Sp * Sp / n, 0.f) + dr;
        float var_t = fmaxf(Stt - St * St / n, 0.f) + dr;
        float ncc = (cross * cross) / (var_p * var_t);
        long o = (((long)b * D + oz) * H + oy) * W + ox;
        float A = 0, Bc = 0, Cc = 0;
        if (ncc < 1.0f) {  // interior -> real adjoints
            Cc = 2.f * cross / (var_p * var_t);       // dncc/dcross
            Bc = -(cross * cross) / (var_p * var_p * var_t);  // dncc/dvar_p
            A = -(St * Cc + 2.f * Sp * Bc) / n;       // dncc/dSp
        } else if (ncc >= 1.0f) {  // genuinely >=1 -> clamp. NaN/Inf falls through both branches
            ncc = 1.0f;            // and stays NaN -> visible NaN loss (not silently masked to 1)
        }
        NCC[o] = ncc; AA[o] = A; BB[o] = Bc; CC[o] = Cc;
    }
}

template <typename S>
static void launch_lncc_fwd(const S* pp, const S* tp, float* np, float* ap, float* bp, float* cp,
                            int HALO, dim3 grid, dim3 block, size_t shmem,
                            int B, int D, int H, int W, float n, float dr) {
#define LAUNCH(h)                                                                                       \
    cudaFuncSetAttribute(lncc_fwd<h, S>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem);      \
    lncc_fwd<h, S><<<grid, block, shmem>>>(pp, tp, np, ap, bp, cp, B, D, H, W, n, dr);
    if (HALO == 1)      { LAUNCH(1) }
    else if (HALO == 2) { LAUNCH(2) }
    else if (HALO == 3) { LAUNCH(3) }
    else if (HALO == 4) { LAUNCH(4) }
    else TORCH_CHECK(false, "kernel_size not supported (odd 3..9)");
#undef LAUNCH
}

std::vector<torch::Tensor> lncc_forward(torch::Tensor P, torch::Tensor T, int kernel_size, double dr) {
    TORCH_CHECK(P.is_cuda() && P.is_contiguous() && P.dim() == 4, "P must be a contiguous 4D CUDA tensor");
    TORCH_CHECK(T.is_cuda() && T.is_contiguous() && T.sizes() == P.sizes() && T.dtype() == P.dtype(), "T must match P");
    auto fopt = P.options().dtype(torch::kFloat32);   // ncc map + adjoints always fp32 (precise backward)
    auto ncc = torch::empty(P.sizes(), fopt), A = torch::empty(P.sizes(), fopt),
         Bc = torch::empty(P.sizes(), fopt), Cc = torch::empty(P.sizes(), fopt);
    int B = P.size(0), D = P.size(1), H = P.size(2), W = P.size(3);
    int HALO = kernel_size / 2;
    float n = (float)(kernel_size * kernel_size * kernel_size);
    dim3 block(TX, TY, TZ);
    dim3 grid(((W + TX - 1) / TX) * B, (H + TY - 1) / TY, (D + TZ - 1) / TZ);
    int SX = TX + 2 * HALO, SY = TY + 2 * HALO, SZ = TZ + 2 * HALO;
    size_t shmem = (size_t)(2 * SZ * SY * SX + 5 * SZ * SY * TX + 5 * SZ * TY * TX) * sizeof(float);
    // Up-front shared-memory cap check: TZ=8 needs ~70 KB (k=7) / ~92 KB (k=9); GPUs with a 64 KB
    // opt-in cap (e.g. Turing sm_75) can't run the larger kernels. Fail clearly instead of a cryptic
    // launch error. (A40/Ampere+ have 99-163 KB and fit all of k=3..9.)
    int devid; cudaGetDevice(&devid);
    int smem_cap = 0; cudaDeviceGetAttribute(&smem_cap, cudaDevAttrMaxSharedMemoryPerBlockOptin, devid);
    TORCH_CHECK((int)shmem <= smem_cap, "fused_lncc: kernel_size=", kernel_size, " needs ", shmem / 1024,
                " KB shared memory, but this GPU's opt-in cap is ", smem_cap / 1024,
                " KB. Use a smaller kernel_size (k<=5 fits 64 KB Turing GPUs), or lower TZ in "
                "fused_lncc/csrc/lncc3d.cu and rebuild.");
    float* np = ncc.data_ptr<float>(); float* ap = A.data_ptr<float>();
    float* bp = Bc.data_ptr<float>(); float* cp = Cc.data_ptr<float>();
    if (P.scalar_type() == torch::kFloat32)
        launch_lncc_fwd<float>(P.data_ptr<float>(), T.data_ptr<float>(), np, ap, bp, cp,
                               HALO, grid, block, shmem, B, D, H, W, n, (float)dr);
    else if (P.scalar_type() == torch::kBFloat16)
        launch_lncc_fwd<__nv_bfloat16>(reinterpret_cast<const __nv_bfloat16*>(P.data_ptr<at::BFloat16>()),
                                       reinterpret_cast<const __nv_bfloat16*>(T.data_ptr<at::BFloat16>()),
                                       np, ap, bp, cp, HALO, grid, block, shmem, B, D, H, W, n, (float)dr);
    else TORCH_CHECK(false, "fused_lncc: float32 or bfloat16 only, got ", P.dtype());
    { cudaError_t e_ = cudaGetLastError(); TORCH_CHECK(e_ == cudaSuccess, "lncc_fwd launch failed: ", cudaGetErrorString(e_), " — if 'no kernel image', rebuild for your GPU arch via TORCH_CUDA_ARCH_LIST"); }
    return {ncc, A, Bc, Cc};
}

// Fully-fused backward: grad_p = scale * (box(A) + 2*p*box(B) + t*box(C)), scale = -grad_out/M.
// Loads the 3 cached adjoints once, does all 3 separable box-sums in shared memory, and combines
// with p,t in one pass — replaces 3 box3d_sep launches + ~7 elementwise kernels. Identical math.
template <int HALO, typename S>
__global__ void lncc_bwd(const float* __restrict__ AA, const float* __restrict__ BB,
                         const float* __restrict__ CC, const S* __restrict__ P,
                         const S* __restrict__ T, S* __restrict__ GP,
                         int B, int D, int H, int W, float scale) {
    const int SX = TX + 2 * HALO, SY = TY + 2 * HALO, SZ = TZ + 2 * HALO;
    const int K = 2 * HALO + 1;
    extern __shared__ float sm[];
    float* sA = sm;                       // A tile  SZ*SY*SX
    float* sB = sA + SZ * SY * SX;        // B tile  SZ*SY*SX
    float* sC = sB + SZ * SY * SX;        // C tile  SZ*SY*SX
    float* X = sC + SZ * SY * SX;         // 3 X-sums  3*SZ*SY*TX
    float* Y = X + 3 * SZ * SY * TX;      // 3 Y-sums  3*SZ*TY*TX
    const int xsz = SZ * SY * TX, ysz = SZ * TY * TX;

    const int nxt = (W + TX - 1) / TX;
    int b = blockIdx.x / nxt, xt = blockIdx.x % nxt;   // batch folded into gridDim.x (same as fwd)
    int x0 = xt * TX, y0 = blockIdx.y * TY, z0 = blockIdx.z * TZ;
    int tid = threadIdx.z * (TY * TX) + threadIdx.y * TX + threadIdx.x, nthreads = TZ * TY * TX;

    // load A,B,C tile+halo (zero outside the volume)
    for (int idx = tid; idx < SZ * SY * SX; idx += nthreads) {
        int sx = idx % SX, sy = (idx / SX) % SY, sz = idx / (SX * SY);
        int gx = x0 - HALO + sx, gy = y0 - HALO + sy, gz = z0 - HALO + sz;
        float va = 0.f, vb = 0.f, vc = 0.f;
        if (gx >= 0 && gx < W && gy >= 0 && gy < H && gz >= 0 && gz < D) {
            long o = (((long)b * D + gz) * H + gy) * W + gx;
            va = AA[o]; vb = BB[o]; vc = CC[o];
        }
        sA[idx] = va; sB[idx] = vb; sC[idx] = vc;
    }
    __syncthreads();
    // X-pass: 3 sums over x window
    for (int idx = tid; idx < xsz; idx += nthreads) {
        int tx = idx % TX, sy = (idx / TX) % SY, sz = idx / (TX * SY);
        float a = 0, bb = 0, c = 0;
#pragma unroll
        for (int dx = 0; dx < K; ++dx) {
            int j = (sz * SY + sy) * SX + tx + dx;
            a += sA[j]; bb += sB[j]; c += sC[j];
        }
        X[0 * xsz + idx] = a; X[1 * xsz + idx] = bb; X[2 * xsz + idx] = c;
    }
    __syncthreads();
    // Y-pass
    for (int idx = tid; idx < ysz; idx += nthreads) {
        int tx = idx % TX, ty = (idx / TX) % TY, sz = idx / (TX * TY);
        float s0 = 0, s1 = 0, s2 = 0;
#pragma unroll
        for (int dy = 0; dy < K; ++dy) {
            int j = (sz * SY + ty + dy) * TX + tx;
            s0 += X[0 * xsz + j]; s1 += X[1 * xsz + j]; s2 += X[2 * xsz + j];
        }
        Y[0 * ysz + idx] = s0; Y[1 * ysz + idx] = s1; Y[2 * ysz + idx] = s2;
    }
    __syncthreads();
    // Z-pass + combine at each output voxel
    int ox = x0 + threadIdx.x, oy = y0 + threadIdx.y, oz = z0 + threadIdx.z;
    if (ox < W && oy < H && oz < D) {
        float bA = 0, bB = 0, bC = 0;
#pragma unroll
        for (int dz = 0; dz < K; ++dz) {
            int j = ((threadIdx.z + dz) * TY + threadIdx.y) * TX + threadIdx.x;
            bA += Y[0 * ysz + j]; bB += Y[1 * ysz + j]; bC += Y[2 * ysz + j];
        }
        long o = (((long)b * D + oz) * H + oy) * W + ox;
        float p = to_float(P[o]), t = to_float(T[o]);
        store_g(&GP[o], scale * (bA + 2.f * p * bB + t * bC));   // cast back to input dtype
    }
}

template <typename S>
static void launch_lncc_bwd(const float* ap, const float* bp, const float* cp,
                            const S* pp, const S* tp, S* gp,
                            int HALO, dim3 grid, dim3 block, size_t shmem,
                            int B, int D, int H, int W, float scale) {
#define LAUNCHB(h)                                                                                       \
    cudaFuncSetAttribute(lncc_bwd<h, S>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem);       \
    lncc_bwd<h, S><<<grid, block, shmem>>>(ap, bp, cp, pp, tp, gp, B, D, H, W, scale);
    if (HALO == 1)      { LAUNCHB(1) }
    else if (HALO == 2) { LAUNCHB(2) }
    else if (HALO == 3) { LAUNCHB(3) }
    else if (HALO == 4) { LAUNCHB(4) }
    else TORCH_CHECK(false, "kernel_size not supported (odd 3..9)");
#undef LAUNCHB
}

torch::Tensor lncc_backward(torch::Tensor A, torch::Tensor Bc, torch::Tensor Cc,
                            torch::Tensor P, torch::Tensor T, int kernel_size, double scale) {
    TORCH_CHECK(A.is_cuda() && A.is_contiguous() && A.dim() == 4 && A.dtype() == torch::kFloat32,
                "A must be a contiguous 4D fp32 CUDA tensor");
    TORCH_CHECK(Bc.sizes() == A.sizes() && Cc.sizes() == A.sizes() && Bc.is_contiguous() && Cc.is_contiguous(),
                "B,C must match A");
    TORCH_CHECK(P.is_cuda() && P.is_contiguous() && P.sizes() == A.sizes() && T.is_contiguous() &&
                T.sizes() == A.sizes() && P.dtype() == T.dtype(), "P,T must match A shape");
    auto gp = torch::empty(P.sizes(), P.options());   // grad in the input dtype
    int B = A.size(0), D = A.size(1), H = A.size(2), W = A.size(3), HALO = kernel_size / 2;
    dim3 block(TX, TY, TZ);
    dim3 grid(((W + TX - 1) / TX) * B, (H + TY - 1) / TY, (D + TZ - 1) / TZ);
    int SX = TX + 2 * HALO, SY = TY + 2 * HALO, SZ = TZ + 2 * HALO;
    size_t shmem = (size_t)(3 * SZ * SY * SX + 3 * SZ * SY * TX + 3 * SZ * TY * TX) * sizeof(float);
    int devid; cudaGetDevice(&devid);
    int smem_cap = 0; cudaDeviceGetAttribute(&smem_cap, cudaDevAttrMaxSharedMemoryPerBlockOptin, devid);
    TORCH_CHECK((int)shmem <= smem_cap, "fused_lncc backward: kernel_size=", kernel_size, " needs ",
                shmem / 1024, " KB shared memory, but this GPU's opt-in cap is ", smem_cap / 1024, " KB.");
    if (P.scalar_type() == torch::kFloat32)
        launch_lncc_bwd<float>(A.data_ptr<float>(), Bc.data_ptr<float>(), Cc.data_ptr<float>(),
                               P.data_ptr<float>(), T.data_ptr<float>(), gp.data_ptr<float>(),
                               HALO, grid, block, shmem, B, D, H, W, (float)scale);
    else if (P.scalar_type() == torch::kBFloat16)
        launch_lncc_bwd<__nv_bfloat16>(A.data_ptr<float>(), Bc.data_ptr<float>(), Cc.data_ptr<float>(),
                                       reinterpret_cast<const __nv_bfloat16*>(P.data_ptr<at::BFloat16>()),
                                       reinterpret_cast<const __nv_bfloat16*>(T.data_ptr<at::BFloat16>()),
                                       reinterpret_cast<__nv_bfloat16*>(gp.data_ptr<at::BFloat16>()),
                                       HALO, grid, block, shmem, B, D, H, W, (float)scale);
    else TORCH_CHECK(false, "fused_lncc: float32 or bfloat16 only, got ", P.dtype());
    { cudaError_t e_ = cudaGetLastError(); TORCH_CHECK(e_ == cudaSuccess, "lncc_bwd launch failed: ", cudaGetErrorString(e_)); }
    return gp;
}

// Separable tiled box-sum (reused for the backward adjoint scatter). Single global read+write.
template <int HALO>
__global__ void box3d_sep(const float* __restrict__ x, float* __restrict__ y, int B, int D, int H, int W) {
    const int SX = TX + 2 * HALO, SY = TY + 2 * HALO, SZ = TZ + 2 * HALO;
    extern __shared__ float sm[];
    float* A = sm; float* Bf = A + SZ * SY * SX; float* Cf = Bf + SZ * SY * TX;
    const int nxt = (W + TX - 1) / TX;
    int b = blockIdx.x / nxt, xt = blockIdx.x % nxt;   // batch folded into gridDim.x
    int x0 = xt * TX, y0 = blockIdx.y * TY, z0 = blockIdx.z * TZ;
    int tid = threadIdx.z * (TY * TX) + threadIdx.y * TX + threadIdx.x, nthreads = TZ * TY * TX;
    for (int idx = tid; idx < SZ * SY * SX; idx += nthreads) {
        int sx = idx % SX, sy = (idx / SX) % SY, sz = idx / (SX * SY);
        int gx = x0 - HALO + sx, gy = y0 - HALO + sy, gz = z0 - HALO + sz;
        float v = 0.f;
        if (gx >= 0 && gx < W && gy >= 0 && gy < H && gz >= 0 && gz < D) v = x[(((long)b * D + gz) * H + gy) * W + gx];
        A[idx] = v;
    }
    __syncthreads();
    for (int idx = tid; idx < SZ * SY * TX; idx += nthreads) {
        int tx = idx % TX, sy = (idx / TX) % SY, sz = idx / (TX * SY);
        float s = 0.f;
#pragma unroll
        for (int dx = 0; dx < 2 * HALO + 1; ++dx) s += A[(sz * SY + sy) * SX + tx + dx];
        Bf[idx] = s;
    }
    __syncthreads();
    for (int idx = tid; idx < SZ * TY * TX; idx += nthreads) {
        int tx = idx % TX, ty = (idx / TX) % TY, sz = idx / (TX * TY);
        float s = 0.f;
#pragma unroll
        for (int dy = 0; dy < 2 * HALO + 1; ++dy) s += Bf[(sz * SY + ty + dy) * TX + tx];
        Cf[idx] = s;
    }
    __syncthreads();
    int ox = x0 + threadIdx.x, oy = y0 + threadIdx.y, oz = z0 + threadIdx.z;
    if (ox < W && oy < H && oz < D) {
        float s = 0.f;
#pragma unroll
        for (int dz = 0; dz < 2 * HALO + 1; ++dz) s += Cf[((threadIdx.z + dz) * TY + threadIdx.y) * TX + threadIdx.x];
        y[(((long)b * D + oz) * H + oy) * W + ox] = s;
    }
}

torch::Tensor box3d_sep_forward(torch::Tensor x, int kernel_size) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.dim() == 4 && x.dtype() == torch::kFloat32);
    auto y = torch::empty_like(x);
    int B = x.size(0), D = x.size(1), H = x.size(2), W = x.size(3), HALO = kernel_size / 2;
    dim3 block(TX, TY, TZ);
    dim3 grid(((W + TX - 1) / TX) * B, (H + TY - 1) / TY, (D + TZ - 1) / TZ);
    int SX = TX + 2 * HALO, SY = TY + 2 * HALO, SZ = TZ + 2 * HALO;
    size_t shmem = (size_t)(SZ * SY * SX + SZ * SY * TX + SZ * TY * TX) * sizeof(float);
    if (HALO == 1)      box3d_sep<1><<<grid, block, shmem>>>(x.data_ptr<float>(), y.data_ptr<float>(), B, D, H, W);
    else if (HALO == 2) box3d_sep<2><<<grid, block, shmem>>>(x.data_ptr<float>(), y.data_ptr<float>(), B, D, H, W);
    else if (HALO == 3) box3d_sep<3><<<grid, block, shmem>>>(x.data_ptr<float>(), y.data_ptr<float>(), B, D, H, W);
    else if (HALO == 4) box3d_sep<4><<<grid, block, shmem>>>(x.data_ptr<float>(), y.data_ptr<float>(), B, D, H, W);
    else TORCH_CHECK(false, "kernel_size not supported: ", kernel_size);
    { cudaError_t e_ = cudaGetLastError(); TORCH_CHECK(e_ == cudaSuccess, "box3d_sep launch failed: ", cudaGetErrorString(e_)); }
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("lncc_forward", &lncc_forward, "fused LNCC forward (ncc map + adjoints A,B,Cc)");
    m.def("lncc_backward", &lncc_backward, "fused LNCC backward (grad_p in one pass)");
    m.def("box3d_sep_forward", &box3d_sep_forward, "separable tiled box-sum");
}
