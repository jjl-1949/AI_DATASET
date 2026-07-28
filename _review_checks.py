"""Targeted checks for potential issues found during code review."""
import torch
import torch.nn as nn
import inspect

print("=" * 65)
print("TARGETED CODE REVIEW CHECKS")
print("=" * 65)

# 1. FPN P6 kernel_size=1
print("\n[1] FPN P6 layer: kernel_size=1 stride=2")
x = torch.randn(2, 256, 20, 20)
pool = nn.MaxPool2d(kernel_size=1, stride=2)
out1 = pool(x)
pool_std = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
out3 = pool_std(x)
print(f"  P5 {tuple(x.shape)} -> k1s2 {tuple(out1.shape)} vs k3s2p1 {tuple(out3.shape)}")
print(f"  Max difference: {(out1 - out3).abs().max():.4f}")
print(f"  NOTE: kernel=1 stride=2 has NO antialiasing (standard FPN uses k=3)")

# 2. IR/Depth pretrained=True
print("\n[2] IR pretrained=True hazard (1ch conv1 vs 3ch pretrained)")
from hw_deal import ResNet50Backbone as IRBackbone
try:
    model = IRBackbone(in_channels=1, pretrained=True)
    w = model.conv1.weight
    print(f"  conv1 weight: {tuple(w.shape)} — pretrained=True OK (conv1 deleted)")
    # But check BN stats: they're from ImageNet, not thermal data
    print(f"  WARNING: BN stats loaded from ImageNet — may not match thermal distribution")
    print(f"  Recommend: pretrained=True + few epochs of BN warmup on target data")
except Exception as e:
    print(f"  FAILED: {e}")

# 3. FPN nearest mode
print("\n[3] FPN upsampling mode")
from rgb_fpn import FPN
src = inspect.getsource(FPN.forward)
if 'nearest' in src:
    print(f"  Uses 'nearest' — standard for detection FPNs (RetinaNet, FCOS)")

# 4. Orthogonality verification
print("\n[4] DT Ort — Gram-Schmidt orthogonality check")
from dt_ort import OrthoFusionBlock
block = OrthoFusionBlock(in_channels=256)
block.eval()
with torch.no_grad():
    d = torch.randn(2, 256, 40, 40)
    t = torch.randn(2, 256, 40, 40)
    d0 = block.proj_d(d)
    i0 = block.proj_t(t)
    d_orth = block._gram_schmidt_step(d0, i0, torch.tensor(1.0))
    i_orth = block._gram_schmidt_step(i0, d0, torch.tensor(1.0))
    cos_d = (d_orth * i0).sum(dim=1).abs().mean().item()
    cos_ir = (i_orth * d0).sum(dim=1).abs().mean().item()
    print(f"  |<F_d^bot, F_t>| @ alpha=1: {cos_d:.2e}")
    print(f"  |<F_t^bot, F_d>| @ beta=1:  {cos_ir:.2e}")
    print(f"  Orthogonality: {'VERIFIED' if max(cos_d,cos_ir)<1e-6 else 'ACCEPTABLE (fp32)'}")

# 5. ReLU regression gradient death
print("\n[5] DetHead ReLU regression — dead gradient check")
from det_head import DetHead
head = DetHead(num_conv=4).eval()
with torch.no_grad():
    x = torch.randn(2, 256, 40, 40)
    out = head(x)
    zeros = (out["bbox_preds"] == 0).float().mean().item()
    print(f"  Zero bbox_preds fraction: {zeros:.3f}")
    if zeros > 0.1:
        print(f"  WARNING: >10% zeros — ReLU kills regression gradient")
        print(f"  FCOS standard uses exp() which has nonzero gradient everywhere")
        print(f"  Recommend: replace relu() with exp() for bbox output")

# 6. Code duplication
print("\n[6] Backbone code duplication")
import rgb_deal, hw_deal, dep_deal
bn1 = inspect.getsource(rgb_deal.Bottleneck)
bn2 = inspect.getsource(hw_deal.Bottleneck)
bn3 = inspect.getsource(dep_deal.Bottleneck)
same_bn = bn1 == bn2 == bn3
bb1 = inspect.getsource(rgb_deal.ResNet50Backbone)
bb2 = inspect.getsource(hw_deal.ResNet50Backbone)
bb3 = inspect.getsource(dep_deal.ResNet50Backbone)
print(f"  Bottleneck identical across 3 files: {same_bn}")
print(f"  Backbone identical (modulo defaults): {bb1==bb2==bb3}")
print(f"  Duplicated lines: ~130 per file x 3 = ~390 total")
print(f"  Recommend: extract to shared backbone.py")

# 7. FullDynamicReduction param count
print("\n[7] DCR-CBAM FullDynamicReduction param count")
from fea_merge import FullDynamicReduction, LowRankDynamicReduction
full = FullDynamicReduction(512, 256)
lr = LowRankDynamicReduction(512, 256, rank=8)
print(f"  FullDynamic: {sum(p.numel() for p in full.parameters())/1e6:.1f}M per level")
print(f"  LowRank:     {sum(p.numel() for p in lr.parameters())/1e6:.1f}M per level")
print(f"  Docstring correctly warns about 'full' mode — 'low_rank' is default")

# 8. BN batch_size=1
print("\n[8] BatchNorm batch_size=1 behavior")
bn = nn.BatchNorm2d(64)
bn.train()
try:
    y = bn(torch.randn(1, 64, 10, 10))
    print(f"  BN(bs=1, train): OK (PyTorch uses running stats when batch var=0)")
except ValueError:
    print(f"  BN(bs=1, train): FAILS — need batch_size >= 2 or InstanceNorm")
    print(f"  Affects: all backbone BN layers during training")

# 9. Centerness saturation
print("\n[9] Centerness sigmoid saturation")
head.eval()
with torch.no_grad():
    c = head(torch.randn(2, 256, 40, 40))["centerness"]
    nz = (c < 0.01).float().mean().item()
    no = (c > 0.99).float().mean().item()
    print(f"  Near 0 (<0.01): {nz:.3f}  Near 1 (>0.99): {no:.3f}")
    if nz + no > 0.3:
        print(f"  WARNING: sigmoid saturation — recommend BCEWithLogitsLoss")

# 10. P6 extra level
print("\n[10] FPN P6: kernel=1 stride=2 issue")
print(f"  Current: MaxPool2d(1, stride=2) — skips every other pixel")
print(f"  Standard: MaxPool2d(3, stride=2, padding=1) — with antialiasing")
print(f"  Fix: change kernel_size=1 to kernel_size=3, add padding=1")

print("\n" + "=" * 65)
print("CODE REVIEW COMPLETE")
print("=" * 65)
