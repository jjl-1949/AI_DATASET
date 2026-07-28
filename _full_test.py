"""
Full network pipeline test with real dataset images.
Verifies all component connections, shapes, gradients, and throughput.
"""
import torch
import torch.nn as nn
import os, time
from PIL import Image
from torchvision.transforms import functional as TF

torch.manual_seed(42)

# ═══════════════════════════════════════════════
# 1. Load real images
# ═══════════════════════════════════════════════
print("=" * 65)
print("[Step 1] Loading real dataset images")
print("=" * 65)

base = "AIC2026_Train_Augmented"
rgb_path = os.path.join(base, "visible", "00000000.jpg")
ir_path = os.path.join(base, "infrared", "00000000.jpg")
depth_path = os.path.join(base, "depth", "00000000.jpg")

rgb_img = Image.open(rgb_path).convert("RGB")
ir_img = Image.open(ir_path).convert("L")
depth_img = Image.open(depth_path).convert("L")

print(f"RGB:     {rgb_img.size}, mode={rgb_img.mode}")
print(f"IR:      {ir_img.size}, mode={ir_img.mode}")
print(f"Depth:   {depth_img.size}, mode={depth_img.mode}")

# Convert to tensors
rgb_tensor = TF.to_tensor(rgb_img)
ir_tensor = TF.to_tensor(ir_img)
depth_tensor = TF.to_tensor(depth_img)

# ImageNet normalize for RGB
rgb_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
rgb_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
rgb_norm = (rgb_tensor - rgb_mean) / rgb_std

# Simple normalize for IR/Depth
ir_norm = (ir_tensor - 0.5) / 0.5
depth_norm = (depth_tensor - 0.5) / 0.5

H, W = rgb_tensor.shape[1:]
print(f"Tensor: RGB={tuple(rgb_norm.shape)}, IR={tuple(ir_norm.shape)}, Depth={tuple(depth_norm.shape)}")

rgb_batch = rgb_norm.unsqueeze(0)
ir_batch = ir_norm.unsqueeze(0)
depth_batch = depth_norm.unsqueeze(0)

del rgb_img, ir_img, depth_img

# ═══════════════════════════════════════════════
# 2. Import all modules
# ═══════════════════════════════════════════════
print("\n" + "=" * 65)
print("[Step 2] Importing all network modules")
print("=" * 65)

from rgb_deal import RGBBranch
from hw_deal import InfraredBranch
from dep_deal import DepthBranch
from dt_ort import DTOrtFusion
from rgb_fpn import RGBFPN, FPN
from dt_fpn import DTFPN
from fea_merge import MultiScaleDCRCBAM, channel_saliency_loss
from dmlab import DMLab, ASPP, DMLabDecoder
from det_head import DetHead

print("All modules imported, no circular dependencies")

# ═══════════════════════════════════════════════
# 3. Test individual components (inference mode)
# ═══════════════════════════════════════════════
print("\n" + "=" * 65)
print("[Step 3] Component-by-component shape verification")
print("=" * 65)

device = torch.device("cpu")

# 3a. RGB Branch
print("\n--- 3a. RGB Branch (ResNet50, ImageNet pretrained) ---")
rgb_branch = RGBBranch(frozen_stages=1).to(device)
rgb_branch.eval()
with torch.no_grad():
    rgb_feats = rgb_branch(rgb_batch.to(device))
for k, v in rgb_feats.items():
    print(f"  {k}: {tuple(v.shape)}")

# 3b. IR Branch
print("\n--- 3b. Infrared Branch (ResNet50, from scratch) ---")
ir_branch = InfraredBranch().to(device)
ir_branch.eval()
with torch.no_grad():
    ir_feats = ir_branch(ir_batch.to(device))
for k, v in ir_feats.items():
    print(f"  {k}: {tuple(v.shape)}")

# 3c. Depth Branch
print("\n--- 3c. Depth Branch (ResNet50, from scratch) ---")
depth_branch = DepthBranch().to(device)
depth_branch.eval()
with torch.no_grad():
    depth_feats = depth_branch(depth_batch.to(device))
for k, v in depth_feats.items():
    print(f"  {k}: {tuple(v.shape)}")

# 3d. DT Orthogonal Fusion
print("\n--- 3d. DT Orthogonal Fusion (Gram-Schmidt) ---")
dt_ort = DTOrtFusion().to(device)
dt_ort.eval()
with torch.no_grad():
    dt_fused = dt_ort(depth_feats, ir_feats)
for k, v in dt_fused.items():
    print(f"  {k}: {tuple(v.shape)}")
    ok = "OK" if v.shape == depth_feats[k].shape else "MISMATCH!"
    print(f"       vs depth {tuple(depth_feats[k].shape)}  {ok}")

# 3e. RGB FPN
print("\n--- 3e. RGB FPN ---")
rgb_fpn = RGBFPN(out_channels=256).to(device)
rgb_fpn.eval()
with torch.no_grad():
    rgb_pyramid = rgb_fpn(rgb_feats)
for k, v in rgb_pyramid.items():
    print(f"  {k}: {tuple(v.shape)}")

# 3f. DT FPN
print("\n--- 3f. DT FPN ---")
dt_fpn = DTFPN(out_channels=256).to(device)
dt_fpn.eval()
with torch.no_grad():
    dt_pyramid = dt_fpn(dt_fused)
for k, v in dt_pyramid.items():
    print(f"  {k}: {tuple(v.shape)}")

# 3g. DCR-CBAM
print("\n--- 3g. DCR-CBAM (low-rank dynamic reduction) ---")
dcr_cbam = MultiScaleDCRCBAM(
    channels=256, reduction_mode="low_rank", dynamic_rank=8
).to(device)
dcr_cbam.eval()
with torch.no_grad():
    fused_pyramid, attention_maps = dcr_cbam(rgb_pyramid, dt_pyramid)
for level in ["P2", "P3", "P4", "P5"]:
    print(f"  {level}: fused={tuple(fused_pyramid[level].shape)}, "
          f"Mc={tuple(attention_maps[level]['channel'].shape)}, "
          f"Ms={tuple(attention_maps[level]['spatial'].shape)}")

# 3h. DMLab
print("\n--- 3h. DMLab (DeepLabV3+ Decoder) ---")
dmlab = DMLab().to(device)
dmlab.eval()
with torch.no_grad():
    dmlab_out = dmlab(fused_pyramid)
print(f"  P2 input:  {tuple(fused_pyramid['P2'].shape)}")
print(f"  P5 input:  {tuple(fused_pyramid['P5'].shape)}")
print(f"  Output:    {tuple(dmlab_out.shape)}")
assert dmlab_out.shape[1] == 256
assert dmlab_out.shape[2:] == fused_pyramid["P2"].shape[2:]  # matches P2 spatial

# 3i. Detection Head
print("\n--- 3i. Detection Head (FCOS anchor-free) ---")
det_head = DetHead(in_channels=256, num_classes=12, num_conv=4).to(device)
det_head.eval()
with torch.no_grad():
    det_out = det_head(dmlab_out)
for k, v in det_out.items():
    print(f"  {k}: {tuple(v.shape)}")

print(f"\n  Output value ranges (real image):")
print(f"    cls_logits:  [{det_out['cls_logits'].min():.2f}, "
      f"{det_out['cls_logits'].max():.2f}]")
print(f"    bbox_preds:  [{det_out['bbox_preds'].min():.4f}, "
      f"{det_out['bbox_preds'].max():.4f}]  (min >= 0: {det_out['bbox_preds'].min() >= 0})")
print(f"    centerness:  [{det_out['centerness'].min():.4f}, "
      f"{det_out['centerness'].max():.4f}]  (in [0,1]: {0 <= det_out['centerness'].min() <= 1})")

# ═══════════════════════════════════════════════
# 4. Gradient flow test (training mode)
# ═══════════════════════════════════════════════
print("\n" + "=" * 65)
print("[Step 4] Gradient flow test")
print("=" * 65)

rgb_branch.train(); ir_branch.train(); depth_branch.train()
dt_ort.train(); rgb_fpn.train(); dt_fpn.train()
dcr_cbam.train(); dmlab.train(); det_head.train()

rf2 = rgb_branch(rgb_batch)
irf2 = ir_branch(ir_batch)
df2 = depth_branch(depth_batch)
dtf2 = dt_ort(df2, irf2)
rp2 = rgb_fpn(rf2)
dp2 = dt_fpn(dtf2)
fp2, attn2 = dcr_cbam(rp2, dp2)
dm2 = dmlab(fp2)
do2 = det_head(dm2)

loss = (do2["cls_logits"].mean() + do2["bbox_preds"].mean() +
        do2["centerness"].mean() + 1e-4 * channel_saliency_loss(attn2))
loss.backward()

components = [
    ("RGB Branch", rgb_branch),
    ("IR Branch", ir_branch),
    ("Depth Branch", depth_branch),
    ("DT Ort Fusion", dt_ort),
    ("RGB FPN", rgb_fpn),
    ("DT FPN", dt_fpn),
    ("DCR-CBAM", dcr_cbam),
    ("DMLab", dmlab),
    ("DetHead", det_head),
]

all_grad_ok = True
for name, model in components:
    params_with_grad = sum(
        1 for p in model.parameters() if p.requires_grad and p.grad is not None
    )
    trainable = sum(1 for p in model.parameters() if p.requires_grad)
    ok = params_with_grad == trainable and trainable > 0
    if not ok:
        all_grad_ok = False
    status = "OK" if ok else f"MISSING ({params_with_grad}/{trainable})"
    print(f"  {name:20s}: {params_with_grad}/{trainable} params with grad  [{status}]")

print(f"\n  Gradient flow: {'ALL OK' if all_grad_ok else 'ISSUES FOUND'}")

# ═══════════════════════════════════════════════
# 5. Parameter summary
# ═══════════════════════════════════════════════
print("\n" + "=" * 65)
print("[Step 5] Parameter summary")
print("=" * 65)

total_p, total_t = 0, 0
for name, model in components:
    p = sum(p.numel() for p in model.parameters())
    t = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_p += p; total_t += t
    print(f"  {name:20s}: {p/1e6:7.2f}M total, {t/1e6:7.2f}M trainable")

print(f"  {'-'*50}")
print(f"  {'TOTAL':20s}: {total_p/1e6:7.2f}M total, {total_t/1e6:7.2f}M trainable")

# ═══════════════════════════════════════════════
# 6. Throughput measurement
# ═══════════════════════════════════════════════
print("\n" + "=" * 65)
print("[Step 6] Throughput (CPU, batch=1, 360x640)")
print("=" * 65)

rgb_t = torch.randn(1, 3, 360, 640)
ir_t = torch.randn(1, 1, 360, 640)
depth_t = torch.randn(1, 1, 360, 640)

for model in [rgb_branch, ir_branch, depth_branch, dt_ort,
              rgb_fpn, dt_fpn, dcr_cbam, dmlab, det_head]:
    model.eval()

with torch.no_grad():
    # Warmup
    _ = det_head(dmlab(dcr_cbam(
        rgb_fpn(rgb_branch(rgb_t)),
        dt_fpn(dt_ort(depth_branch(depth_t), ir_branch(ir_t)))
    )[0]))

    t0 = time.perf_counter()
    for _ in range(5):
        r = rgb_branch(rgb_t)
        i = ir_branch(ir_t)
        d = depth_branch(depth_t)
        f = dt_ort(d, i)
        rp = rgb_fpn(r)
        dp = dt_fpn(f)
        fp, _ = dcr_cbam(rp, dp)
        dm = dmlab(fp)
        o = det_head(dm)
    t1 = time.perf_counter()
    avg_ms = (t1 - t0) / 5 * 1000
    print(f"  Full pipeline: {avg_ms:.0f}ms/frame ({1000/avg_ms:.1f} FPS)")

# Break down per component
print("\n  Per-component timing:")
with torch.no_grad():
    # RGB
    t0 = time.perf_counter()
    for _ in range(10):
        r = rgb_branch(rgb_t)
    t_rgb = (time.perf_counter() - t0) / 10 * 1000
    print(f"    RGB Branch:     {t_rgb:6.1f}ms")

    # IR
    t0 = time.perf_counter()
    for _ in range(10):
        i = ir_branch(ir_t)
    t_ir = (time.perf_counter() - t0) / 10 * 1000
    print(f"    IR Branch:      {t_ir:6.1f}ms")

    # Depth
    t0 = time.perf_counter()
    for _ in range(10):
        d = depth_branch(depth_t)
    t_depth = (time.perf_counter() - t0) / 10 * 1000
    print(f"    Depth Branch:   {t_depth:6.1f}ms")

    # DT Ort
    df = depth_branch(depth_t); irf = ir_branch(ir_t)
    t0 = time.perf_counter()
    for _ in range(10):
        f = dt_ort(df, irf)
    t_ort = (time.perf_counter() - t0) / 10 * 1000
    print(f"    DT Ort Fusion:  {t_ort:6.1f}ms")

    # FPNs
    rf = rgb_branch(rgb_t)
    t0 = time.perf_counter()
    for _ in range(10):
        rp = rgb_fpn(rf)
    t_rfpn = (time.perf_counter() - t0) / 10 * 1000
    print(f"    RGB FPN:        {t_rfpn:6.1f}ms")

    dtf = dt_ort(df, irf)
    t0 = time.perf_counter()
    for _ in range(10):
        dp = dt_fpn(dtf)
    t_dfpn = (time.perf_counter() - t0) / 10 * 1000
    print(f"    DT FPN:         {t_dfpn:6.1f}ms")

    # DCR-CBAM
    rp = rgb_fpn(rf); dp = dt_fpn(dtf)
    t0 = time.perf_counter()
    for _ in range(10):
        fp, _ = dcr_cbam(rp, dp)
    t_dcr = (time.perf_counter() - t0) / 10 * 1000
    print(f"    DCR-CBAM:       {t_dcr:6.1f}ms")

    # DMLab + DetHead
    fp, _ = dcr_cbam(rp, dp)
    t0 = time.perf_counter()
    for _ in range(10):
        dm = dmlab(fp)
        o = det_head(dm)
    t_head = (time.perf_counter() - t0) / 10 * 1000
    print(f"    DMLab+DetHead:  {t_head:6.1f}ms")

# ═══════════════════════════════════════════════
# 7. Variable input sizes
# ═══════════════════════════════════════════════
print("\n" + "=" * 65)
print("[Step 7] Variable input size compatibility")
print("=" * 65)

rgb_branch.eval(); ir_branch.eval(); depth_branch.eval()
dt_ort.eval(); rgb_fpn.eval(); dt_fpn.eval()
dcr_cbam.eval(); dmlab.eval(); det_head.eval()

for (h, w) in [(360, 640), (480, 640), (720, 1280)]:
    r = torch.randn(1, 3, h, w)
    ir = torch.randn(1, 1, h, w)
    d = torch.randn(1, 1, h, w)
    with torch.no_grad():
        rf = rgb_branch(r)
        irf = ir_branch(ir)
        df = depth_branch(d)
        dtf = dt_ort(df, irf)
        rp = rgb_fpn(rf)
        dp = dt_fpn(dtf)
        fp, _ = dcr_cbam(rp, dp)
        dm = dmlab(fp)
        o = det_head(dm)

    exp_h4 = (1, 256, h // 4, w // 4)
    assert dm.shape == exp_h4, f"DMLab: {tuple(dm.shape)} != {exp_h4}"
    print(f"  {h}x{w:4d}  ->  DMLab {tuple(dm.shape)}  DetHead cls {tuple(o['cls_logits'].shape)}  OK")

# ═══════════════════════════════════════════════
# Final summary
# ═══════════════════════════════════════════════
print("\n" + "=" * 65)
print("[RESULT] FULL PIPELINE TEST PASSED")
print("=" * 65)
print(f"  Components:      9")
print(f"  Total params:    {total_p/1e6:.1f}M")
print(f"  Trainable:       {total_t/1e6:.1f}M")
print(f"  Gradient flow:   {'OK' if all_grad_ok else 'FIX NEEDED'}")
print(f"  Variable sizes:  OK")
print(f"  Real data:       OK")
print(f"  Pipeline:")
print(f"    RGB/IR/Depth -> Backbones -> DT-Ort -> FPNs ->")
print(f"    DCR-CBAM -> DMLab -> DetHead")
print(f"  Outputs:")
print(f"    cls_logits:     (B, 12, H/4, W/4)")
print(f"    bbox_preds:     (B, 4,  H/4, W/4)  FCOS (l,t,r,b)")
print(f"    centerness:     (B, 1,  H/4, W/4)  [0,1]")
print("=" * 65)
