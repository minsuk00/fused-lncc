"""Verify the fused forward kernel's ncc map + loss vs a torch reference and MONAI."""
import os, torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

cc = torch.cuda.get_device_capability()            # build for whatever GPU we're on (portable)
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{cc[0]}.{cc[1]}")
_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fused_lncc", "csrc", "lncc3d.cu")
ext = load(name="lncc3d_fwd", sources=[_src], extra_cuda_cflags=["-O3"], verbose=False)
dev = "cuda"

def ref_ncc_map(p, t, k=7, dr=1e-5):       # torch reference (box-sum via conv3d)
    C = p.shape[1]; n = k ** 3; pad = k // 2
    box = lambda x: F.conv3d(x, x.new_ones(C, 1, k, k, k), padding=pad, groups=C)
    Sp, St = box(p), box(t)
    cross = box(p * t) - Sp * St / n
    var_p = box(p * p) - Sp * Sp / n + dr
    var_t = box(t * t) - St * St / n + dr
    return ((cross * cross) / (var_p * var_t)).clamp(max=1.0)

def fused_ncc_map(p, t, k=7, dr=1e-5):
    N, C, *S = p.shape
    ncc, A, B, Cc = ext.lncc_forward(p.reshape(N * C, *S).contiguous(),
                                     t.reshape(N * C, *S).contiguous(), k, dr)
    return ncc.reshape(N, C, *S)

print("=== forward ncc map: fused vs torch reference ===")
torch.manual_seed(0)
for k in [3, 5, 7]:
    for shp in [(2, 4, 24, 24, 24), (1, 8, 32, 40, 48)]:
        p = torch.randn(*shp, device=dev); t = torch.randn(*shp, device=dev)
        a = fused_ncc_map(p, t, k); b = ref_ncc_map(p, t, k)
        d = (a - b).abs().max().item()
        print(f"  k={k} {str(shp):20s} max|fused-ref|={d:.2e}  in[0,1]={a.min().item():.3f}..{a.max().item():.3f}  {'OK' if d<1e-3 else 'FAIL'}")

print("=== loss 1-mean(ncc) vs MONAI (1 + monai) ===")
from monai.losses import LocalNormalizedCrossCorrelationLoss as MLNCC
for k in [3, 7]:
    p = torch.rand(2, 4, 40, 40, 40, device=dev); t = torch.rand(2, 4, 40, 40, 40, device=dev)
    fused_loss = 1.0 - fused_ncc_map(p, t, k, dr=1e-5).mean().item()
    monai = MLNCC(spatial_dims=3, kernel_size=k, kernel_type="rectangular", smooth_nr=0, smooth_dr=1e-5, reduction="mean").to(dev)
    monai_loss = 1.0 + monai(p, t).item()
    print(f"  k={k}: fused={fused_loss:.6f}  monai={monai_loss:.6f}  |diff|={abs(fused_loss-monai_loss):.2e}")
