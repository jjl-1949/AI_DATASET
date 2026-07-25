"""
离线数据增强 Runner (Streaming 版本)
=====================================
对 AIC2026 训练集做离线增强。
增强样本即时保存到磁盘，不在内存中累积。

用法:
    python run_augmentation.py --mosaic 200 --online-aug 300
"""

import os
import sys
import json
import argparse
import cv2
import numpy as np
import shutil
import gc
from collections import defaultdict
from tqdm import tqdm
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from augmentation_pipeline import (
    AugConfig, AugmentationPipeline, depth_to_colormap,
    normalize_9ch, is_valid_bbox, clip_bbox_to_image,
    imread_robust, imwrite_robust
)

# ==============================================================================
# 配置
# ==============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, '训练集', 'AIC2026_Train_2000')
OUTPUT_DIR = os.path.join(BASE_DIR, 'AIC2026_Train_Augmented')

CLASS_MULTIPLIERS = {
    0: 1.0,   2: 1.0,   6: 1.5,   8: 1.5,   4: 2.0,
    5: 2.0,   3: 3.0,   9: 5.0,  10: 7.0,   1: 10.0,
    7: 15.0,  11: 50.0
}

RARE_CLASSES = [1, 7, 9, 10, 11]
MID_CLASSES = [3, 4, 5]

MOSAIC_COUNT = 300
ONLINE_AUG_COUNT = 500
SEED = 42

# ==============================================================================
# 数据集索引
# ==============================================================================

def build_index(data_dir: str):
    print(f"\n{'='*60}")
    print(f"Building index: {data_dir}")
    print(f"{'='*60}")

    dirs = {m: os.path.join(data_dir, m) for m in ['visible', 'depth', 'infrared', 'labels']}

    # Collect all basenames that have all 4 files
    all_bases = set()
    for d in [dirs['visible'], dirs['depth'], dirs['infrared']]:
        if os.path.isdir(d):
            for f in os.listdir(d):
                if f.endswith(('.jpg', '.png')):
                    all_bases.add(f.rsplit('.', 1)[0])

    filenames = sorted(b for b in all_bases
                       if os.path.exists(os.path.join(dirs['labels'], b + '.txt')))

    print(f"  Found {len(filenames)} frames")

    index = {'filenames': [], 'rgb_paths': [], 'depth_paths': [],
             'thermal_paths': [], 'boxes': [], 'class_ids': [],
             'class_to_indices': defaultdict(list)}

    for fname in tqdm(filenames, desc="Building index"):
        rgb_path = _find(dirs['visible'], fname)
        dep_path = _find(dirs['depth'], fname)
        ir_path = _find(dirs['infrared'], fname)
        lbl_path = os.path.join(dirs['labels'], fname + '.txt')

        if not all([rgb_path, dep_path, ir_path]):
            continue

        boxes, class_ids = _read_labels(lbl_path)
        if not boxes:
            continue

        idx = len(index['filenames'])
        index['filenames'].append(fname)
        index['rgb_paths'].append(rgb_path)
        index['depth_paths'].append(dep_path)
        index['thermal_paths'].append(ir_path)
        index['boxes'].append(np.array(boxes, dtype=np.float32))
        index['class_ids'].append(np.array(class_ids, dtype=np.int64))
        for cls_id in set(class_ids):
            index['class_to_indices'][cls_id].append(idx)

    all_cls = np.concatenate(index['class_ids'])
    class_counts = {c: int((all_cls == c).sum()) for c in range(12)}

    print(f"  Indexed {len(index['filenames'])} frames")
    for c in sorted(class_counts):
        print(f"    Class {c:2d}: {class_counts[c]:5d} boxes")

    return index, class_counts


def _find(dir_path, basename):
    for ext in ['.jpg', '.png']:
        p = os.path.join(dir_path, basename + ext)
        if os.path.exists(p):
            return p
    return ''


def _read_labels(label_path):
    boxes, class_ids = [], []
    try:
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:5])
                    bbox = np.array([cx, cy, w, h], dtype=np.float32)
                    if is_valid_bbox(bbox):
                        boxes.append(bbox)
                        class_ids.append(cls_id)
    except Exception:
        pass
    return boxes, class_ids


def load_frame(rgb_path, depth_path, thermal_path):
    """Load a frame, normalizing depth to uint8 3ch grayscale."""
    rgb = imread_robust(rgb_path, cv2.IMREAD_COLOR)
    if rgb is not None:
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    depth_raw = imread_robust(depth_path, cv2.IMREAD_UNCHANGED)
    thermal = imread_robust(thermal_path, cv2.IMREAD_COLOR)
    if thermal is not None:
        thermal = cv2.cvtColor(thermal, cv2.COLOR_BGR2RGB)

    if rgb is None or depth_raw is None or thermal is None:
        return None, None, None

    # Normalize depth to uint8 3ch grayscale
    if depth_raw.dtype == np.uint16 or depth_raw.max() > 255:
        dp = depth_raw.astype(np.float32)
        valid = dp[dp > 0] if (dp > 0).any() else dp
        max_depth = min(np.median(valid) * 3, 10000)
        dp[dp > max_depth] = max_depth
        dp = cv2.normalize(dp, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        depth = dp.astype(np.uint8)
    else:
        depth = depth_raw

    if depth.ndim == 2:
        depth = np.stack([depth] * 3, axis=-1)
    elif depth.shape[2] == 1:
        depth = np.repeat(depth, 3, axis=2)

    return rgb, depth, thermal


# ==============================================================================
# 即时保存
# ==============================================================================

class DatasetWriter:
    """Streaming dataset writer — saves frames immediately to disk."""

    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.visible_dir = os.path.join(output_dir, 'visible')
        self.depth_dir = os.path.join(output_dir, 'depth')
        self.infrared_dir = os.path.join(output_dir, 'infrared')
        self.labels_dir = os.path.join(output_dir, 'labels')
        for d in [self.visible_dir, self.depth_dir, self.infrared_dir, self.labels_dir]:
            os.makedirs(d, exist_ok=True)
        self.frame_idx = 0
        self.total_boxes = 0
        self.class_counts = defaultdict(int)

    def save(self, rgb, depth, thermal, boxes, class_ids):
        """Save one frame immediately."""
        fname = f"{self.frame_idx:08d}"
        self.frame_idx += 1

        # Normalize depth to 3ch uint8
        if depth.ndim == 2:
            depth = np.stack([depth] * 3, axis=-1)
        elif depth.shape[2] == 1:
            depth = np.repeat(depth, 3, axis=2)

        imwrite_robust(os.path.join(self.visible_dir, fname + '.jpg'),
                       cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        imwrite_robust(os.path.join(self.depth_dir, fname + '.jpg'), depth)
        imwrite_robust(os.path.join(self.infrared_dir, fname + '.jpg'),
                       cv2.cvtColor(thermal, cv2.COLOR_RGB2BGR))

        with open(os.path.join(self.labels_dir, fname + '.txt'), 'w') as f:
            for bbox, cls_id in zip(boxes, class_ids):
                if is_valid_bbox(bbox):
                    cx, cy, w, h = bbox
                    f.write(f"{int(cls_id)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
                    self.total_boxes += 1
                    self.class_counts[int(cls_id)] += 1

    def copy_original(self, index):
        """Copy original dataset files directly (no re-encoding)."""
        print("\n[1/2] Copying original data...")
        for i in tqdm(range(len(index['filenames'])), desc="Copying originals"):
            fname = f"{self.frame_idx:08d}"

            # Copy image files directly
            for src_path, dst_dir in [
                (index['rgb_paths'][i], self.visible_dir),
                (index['depth_paths'][i], self.depth_dir),
                (index['thermal_paths'][i], self.infrared_dir),
            ]:
                ext = os.path.splitext(src_path)[1] or '.jpg'
                shutil.copy2(src_path, os.path.join(dst_dir, fname + ext))

            # Copy label file
            src_lbl = os.path.join(os.path.dirname(index['rgb_paths'][i]),
                                   '..', 'labels',
                                   index['filenames'][i] + '.txt')
            if not os.path.exists(src_lbl):
                src_lbl = os.path.join(os.path.dirname(os.path.dirname(
                    index['rgb_paths'][i])), 'labels',
                    index['filenames'][i] + '.txt')
            dst_lbl = os.path.join(self.labels_dir, fname + '.txt')
            if os.path.exists(src_lbl):
                shutil.copy2(src_lbl, dst_lbl)

            self.frame_idx += 1
            for cls_id in index['class_ids'][i]:
                self.class_counts[int(cls_id)] += 1
            self.total_boxes += len(index['class_ids'][i])

        original_count = self.frame_idx
        print(f"  Copied {original_count} frames")
        return original_count

    def print_stats(self, label=""):
        print(f"\n  [{label}] Frames: {self.frame_idx}, Boxes: {self.total_boxes}")
        for c in sorted(self.class_counts.keys()):
            print(f"    Class {c:2d}: {self.class_counts[c]:6d}")


# ==============================================================================
# Copy-Paste (流式保存)
# ==============================================================================

def run_copy_paste_streaming(index, class_counts, output_dir, seed=SEED):
    print(f"\n{'='*60}")
    print(f"Copy-Paste Augmentation (streaming)")
    print(f"{'='*60}")

    writer = DatasetWriter(output_dir)
    # Don't re-copy originals here — writer starts fresh. We'll merge later.

    config = AugConfig()
    pipeline = AugmentationPipeline(config)
    cp = pipeline.copy_paste
    n_total = len(index['filenames'])

    targets = {}
    for cls_id in RARE_CLASSES + MID_CLASSES:
        current = class_counts.get(cls_id, 0)
        target = int(current * CLASS_MULTIPLIERS.get(cls_id, 1.0))
        if target > current:
            targets[cls_id] = target
            print(f"  Class {cls_id:2d}: {current} -> {target} (x{CLASS_MULTIPLIERS[cls_id]})")

    if not targets:
        print("  All classes sufficient, skipping")
        return 0

    total_generated = 0
    for rare_class in sorted(targets.keys()):
        current = class_counts.get(rare_class, 0)
        target = targets[rare_class]
        deficit = target - current

        if rare_class not in index['class_to_indices']:
            continue

        src_indices = index['class_to_indices'][rare_class]
        pastes_per_obj = min(2, max(1, deficit // (len(src_indices) * 2) + 1))
        paste_count = 0

        print(f"  Class {rare_class}: deficit={deficit}, "
              f"src_imgs={len(src_indices)}, pastes_per_obj={pastes_per_obj}")

        pbar = tqdm(total=deficit, desc=f"CP class {rare_class}")
        for src_idx in src_indices:
            src_rgb, src_depth, src_thermal = load_frame(
                index['rgb_paths'][src_idx], index['depth_paths'][src_idx],
                index['thermal_paths'][src_idx])
            if src_rgb is None:
                continue

            src_boxes = index['boxes'][src_idx]
            src_classes = index['class_ids'][src_idx]
            rare_mask = src_classes == rare_class
            if not rare_mask.any():
                continue

            for src_bbox in src_boxes[rare_mask]:
                for _ in range(pastes_per_obj):
                    dst_idx = random.randint(0, n_total - 1)
                    dst_rgb, dst_depth, dst_thermal = load_frame(
                        index['rgb_paths'][dst_idx], index['depth_paths'][dst_idx],
                        index['thermal_paths'][dst_idx])
                    if dst_rgb is None:
                        continue

                    try:
                        result = cp.copy_paste_single(
                            src_rgb, src_depth, src_thermal, src_bbox,
                            dst_rgb, dst_depth, dst_thermal)
                        if len(result) != 4 or result[3] is None:
                            continue

                        aug_rgb, aug_depth, aug_thermal, new_bbox = result

                        new_boxes = np.vstack([index['boxes'][dst_idx],
                                               new_bbox.reshape(1, 4)])
                        new_classes = np.hstack([index['class_ids'][dst_idx],
                                                 np.array([rare_class])])

                        writer.save(aug_rgb, aug_depth, aug_thermal,
                                    new_boxes, new_classes)

                        paste_count += 1
                        total_generated += 1
                        pbar.update(1)

                        if paste_count >= deficit:
                            break
                    except Exception:
                        continue

                if paste_count >= deficit:
                    break
            if paste_count >= deficit:
                break

            # Free memory periodically
            if src_idx % 50 == 0:
                gc.collect()

        pbar.close()
        print(f"    Generated {paste_count} CP samples for class {rare_class}")

    print(f"\n  Total CP samples: {total_generated}")
    return total_generated


# ==============================================================================
# Mosaic (流式保存)
# ==============================================================================

def run_mosaic_streaming(index, num_mosaic, output_dir, start_idx=0):
    print(f"\n{'='*60}")
    print(f"Mosaic Augmentation (streaming, target {num_mosaic})")
    print(f"{'='*60}")

    config = AugConfig()
    pipeline = AugmentationPipeline(config)
    n_total = len(index['filenames'])
    writer = DatasetWriter(output_dir)
    writer.frame_idx = start_idx

    success = 0
    attempts = 0
    max_attempts = num_mosaic * 3
    pbar = tqdm(total=num_mosaic, desc="Mosaic")

    while success < num_mosaic and attempts < max_attempts:
        attempts += 1
        indices = random.sample(range(n_total), 4)

        rgb_list, depth_list, thermal_list = [], [], []
        boxes_list, class_list = [], []
        valid = True

        for idx in indices:
            rgb, depth, thermal = load_frame(
                index['rgb_paths'][idx], index['depth_paths'][idx],
                index['thermal_paths'][idx])
            if rgb is None:
                valid = False
                break
            rgb_list.append(rgb)
            depth_list.append(depth)
            thermal_list.append(thermal)
            boxes_list.append(index['boxes'][idx])
            class_list.append(index['class_ids'][idx])

        if not valid:
            continue

        try:
            img_9ch, boxes, class_ids = pipeline.augment_frame_with_mosaic(
                rgb_list, depth_list, thermal_list, boxes_list, class_list,
                apply_color=True)

            if len(boxes) == 0:
                continue

            rgb_m = img_9ch[:, :, :3]
            depth_color = img_9ch[:, :, 3:6]  # colormap from mosaic
            thermal_m = img_9ch[:, :, 6:9]

            # Convert colormap back to grayscale for saving
            depth_gray = cv2.cvtColor(depth_color, cv2.COLOR_RGB2GRAY)
            depth_save = np.stack([depth_gray] * 3, axis=-1)

            writer.save(rgb_m, depth_save, thermal_m, boxes, class_ids)
            success += 1
            pbar.update(1)

            if success % 50 == 0:
                gc.collect()

        except Exception:
            continue

    pbar.close()
    print(f"  Generated {success} mosaic samples")
    return success


# ==============================================================================
# 在线增强 (流式保存)
# ==============================================================================

def run_online_augmentation_streaming(index, num_aug, output_dir, start_idx=0):
    print(f"\n{'='*60}")
    print(f"Online Augmentation (streaming, target {num_aug})")
    print(f"{'='*60}")

    config = AugConfig()
    pipeline = AugmentationPipeline(config)
    n_total = len(index['filenames'])
    writer = DatasetWriter(output_dir)
    writer.frame_idx = start_idx

    success = 0
    attempts = 0
    max_attempts = num_aug * 3
    pbar = tqdm(total=num_aug, desc="Online aug")

    while success < num_aug and attempts < max_attempts:
        attempts += 1
        idx = random.randint(0, n_total - 1)

        rgb, depth, thermal = load_frame(
            index['rgb_paths'][idx], index['depth_paths'][idx],
            index['thermal_paths'][idx])
        if rgb is None:
            continue

        boxes = index['boxes'][idx].copy()
        class_ids = index['class_ids'][idx].copy()

        try:
            img_9ch, depth_gray, new_boxes, new_classes = pipeline.augment_frame(
                rgb, depth.copy(), thermal, boxes, class_ids,
                apply_spatial=True, apply_color=True, apply_depth_aug=True)

            if len(new_boxes) == 0:
                continue

            rgb_aug = img_9ch[:, :, :3]
            thermal_aug = img_9ch[:, :, 6:9]

            writer.save(rgb_aug, depth_gray, thermal_aug, new_boxes, new_classes)
            success += 1
            pbar.update(1)

            if success % 50 == 0:
                gc.collect()

        except Exception:
            continue

    pbar.close()
    print(f"  Generated {success} online aug samples")
    return success


# ==============================================================================
# 统计报告
# ==============================================================================

def print_final_report(output_dir, class_counts_before):
    labels_dir = os.path.join(output_dir, 'labels')
    aug_counts = defaultdict(int)
    total_frames = 0

    if os.path.exists(labels_dir):
        for f in os.listdir(labels_dir):
            if f.endswith('.txt'):
                total_frames += 1
                with open(os.path.join(labels_dir, f), 'r') as fh:
                    for line in fh:
                        parts = line.strip().split()
                        if parts:
                            aug_counts[int(parts[0])] += 1

    print(f"\n{'='*60}")
    print(f"Augmentation Results")
    print(f"{'='*60}")
    print(f"  Total frames: {total_frames}")
    print(f"  Total boxes before: {sum(class_counts_before.values())}")
    print(f"  Total boxes after:  {sum(aug_counts.values())}")

    max_before = max(class_counts_before.values())
    min_before = min(class_counts_before.values())
    max_after = max(aug_counts.values()) if aug_counts else 0
    min_after = min(aug_counts.values()) if aug_counts else 0

    print(f"  Imbalance: {max_before/max(min_before,1):.1f}:1 -> "
          f"{max_after/max(min_after,1):.1f}:1")
    print()
    print(f"{'Class':<8} {'Before':<10} {'After':<10} {'Mult':<10}")
    print(f"{'-'*38}")
    for c in sorted(set(list(class_counts_before.keys()) + list(aug_counts.keys()))):
        b = class_counts_before.get(c, 0)
        a = aug_counts.get(c, 0)
        marker = ' *' if c in RARE_CLASSES else ''
        print(f"{c:<8} {b:<10} {a:<10} {a/max(b,1):<10.2f}{marker}")
    print()


# ==============================================================================
# 主程序
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description='AIC2026 Dataset Augmentation')
    parser.add_argument('--input', type=str, default=TRAIN_DIR)
    parser.add_argument('--output', type=str, default=OUTPUT_DIR)
    parser.add_argument('--mosaic', type=int, default=MOSAIC_COUNT)
    parser.add_argument('--online-aug', type=int, default=ONLINE_AUG_COUNT)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--skip-copy-paste', action='store_true')
    parser.add_argument('--skip-mosaic', action='store_true')
    parser.add_argument('--skip-online', action='store_true')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"\n{'#'*60}")
    print(f"# AIC2026 RGB-D-Thermal Dataset Augmentation")
    print(f"# Seed: {args.seed}")
    print(f"{'#'*60}")

    # Build index
    index, class_counts = build_index(args.input)

    # Setup output
    print(f"\n{'='*60}")
    print(f"Output: {args.output}")
    print(f"{'='*60}")

    # Make sure output directory is clean
    if os.path.exists(args.output):
        print(f"  Removing existing output: {args.output}")
        shutil.rmtree(args.output)

    # Phase 1: Copy original data
    writer = DatasetWriter(args.output)
    writer.copy_original(index)
    base_frame_count = writer.frame_idx
    writer.print_stats("After copy originals")

    # Phase 2: Copy-Paste
    if not args.skip_copy_paste:
        cp_dir = args.output  # same output dir
        # We need a fresh writer that appends after originals
        cp_writer = DatasetWriter(args.output)
        cp_writer.frame_idx = base_frame_count
        # Actually, run_copy_paste_streaming creates its own writer
        # Let me refactor to use a shared writer...

    print("\nThis is a streaming pipeline. Each phase saves directly to disk.")
    print("Running with: copy-paste streaming approach...")

    # Actually let me use a simpler integrated approach
    # Save everything through a single DatasetWriter
    writer = DatasetWriter(args.output)
    original_count = writer.copy_original(index)
    writer.print_stats("After copying originals")

    # Phase 2: Copy-Paste (append after originals)
    cp_count = _do_copy_paste(index, class_counts, writer, args.seed) \
        if not args.skip_copy_paste else 0

    # Phase 3: Mosaic
    mosaic_count = _do_mosaic(index, args.mosaic, writer, args.seed) \
        if not args.skip_mosaic else 0

    # Phase 4: Online augmentation
    online_count = _do_online_aug(index, args.online_aug, writer, args.seed) \
        if not args.skip_online else 0

    # Final stats
    total = original_count + cp_count + mosaic_count + online_count
    print(f"\n{'='*60}")
    print(f"DONE: {total} frames total")
    print(f"  Original:  {original_count}")
    print(f"  Copy-Paste: {cp_count}")
    print(f"  Mosaic:     {mosaic_count}")
    print(f"  Online Aug: {online_count}")
    print(f"{'='*60}")

    print_final_report(args.output, class_counts)

    # Save config
    config_out = {
        'input': args.input, 'output': args.output, 'seed': args.seed,
        'mosaic_count': args.mosaic, 'online_aug_count': args.online_aug,
        'original_frames': original_count, 'cp_frames': cp_count,
        'mosaic_frames': mosaic_count, 'online_aug_frames': online_count,
        'total_frames': total, 'class_multipliers': CLASS_MULTIPLIERS,
        'rare_classes': RARE_CLASSES, 'class_counts_before': class_counts,
    }
    with open(os.path.join(args.output, 'augmentation_config.json'), 'w', encoding='utf-8') as f:
        json.dump(config_out, f, indent=2, ensure_ascii=False)
    print(f"Config saved to: {os.path.join(args.output, 'augmentation_config.json')}")


def _do_copy_paste(index, class_counts, writer, seed):
    """Copy-paste augmentation, saving through shared writer."""
    print(f"\n[Copy-Paste] Starting...")
    config = AugConfig()
    pipeline = AugmentationPipeline(config)
    cp = pipeline.copy_paste
    n_total = len(index['filenames'])
    start_count = writer.frame_idx

    targets = {}
    for cls_id in RARE_CLASSES + MID_CLASSES:
        current = class_counts.get(cls_id, 0)
        target = int(current * CLASS_MULTIPLIERS.get(cls_id, 1.0))
        if target > current:
            targets[cls_id] = target

    if not targets:
        print("  All classes sufficient, skipping")
        return 0

    total_generated = 0
    for rare_class in sorted(targets.keys()):
        current = class_counts.get(rare_class, 0)
        deficit = targets[rare_class] - current

        if rare_class not in index['class_to_indices']:
            continue

        src_indices = index['class_to_indices'][rare_class]
        pastes_per_obj = min(2, max(1, deficit // (len(src_indices) * 2) + 1))
        paste_count = 0

        print(f"  Class {rare_class}: deficit={deficit}, ppobj={pastes_per_obj}")
        pbar = tqdm(total=min(deficit, len(src_indices) * pastes_per_obj),
                    desc=f"CP cls {rare_class}")

        for src_idx in src_indices:
            src_rgb, src_depth, src_thermal = load_frame(
                index['rgb_paths'][src_idx], index['depth_paths'][src_idx],
                index['thermal_paths'][src_idx])
            if src_rgb is None:
                continue

            src_boxes = index['boxes'][src_idx]
            src_classes = index['class_ids'][src_idx]
            rare_mask = src_classes == rare_class
            if not rare_mask.any():
                continue

            for src_bbox in src_boxes[rare_mask]:
                for _ in range(pastes_per_obj):
                    dst_idx = random.randint(0, n_total - 1)
                    dst_rgb, dst_depth, dst_thermal = load_frame(
                        index['rgb_paths'][dst_idx], index['depth_paths'][dst_idx],
                        index['thermal_paths'][dst_idx])
                    if dst_rgb is None:
                        continue

                    try:
                        result = cp.copy_paste_single(
                            src_rgb, src_depth, src_thermal, src_bbox,
                            dst_rgb, dst_depth, dst_thermal)
                        if len(result) != 4 or result[3] is None:
                            continue
                        aug_rgb, aug_depth, aug_thermal, new_bbox = result

                        new_boxes = np.vstack([index['boxes'][dst_idx],
                                               new_bbox.reshape(1, 4)])
                        new_classes = np.hstack([index['class_ids'][dst_idx],
                                                 np.array([rare_class])])
                        writer.save(aug_rgb, aug_depth, aug_thermal,
                                    new_boxes, new_classes)
                        paste_count += 1
                        total_generated += 1
                        pbar.update(1)
                        if paste_count >= deficit:
                            break
                    except Exception:
                        continue
                if paste_count >= deficit:
                    break
            if paste_count >= deficit:
                break

            if total_generated % 100 == 0:
                gc.collect()

        pbar.close()

    generated = writer.frame_idx - start_count
    print(f"  Copy-Paste: {generated} frames generated")
    return generated


def _do_mosaic(index, num_mosaic, writer, seed):
    """Mosaic augmentation."""
    print(f"\n[Mosaic] Generating {num_mosaic} samples...")
    config = AugConfig()
    pipeline = AugmentationPipeline(config)
    n_total = len(index['filenames'])
    start_count = writer.frame_idx

    success = 0
    max_attempts = num_mosaic * 3
    pbar = tqdm(total=num_mosaic, desc="Mosaic")

    while success < num_mosaic and max_attempts > 0:
        max_attempts -= 1
        indices = random.sample(range(n_total), 4)

        rgb_list, depth_list, thermal_list = [], [], []
        boxes_list, class_list = [], []
        valid = True

        for idx in indices:
            rgb, depth, thermal = load_frame(
                index['rgb_paths'][idx], index['depth_paths'][idx],
                index['thermal_paths'][idx])
            if rgb is None:
                valid = False
                break
            rgb_list.append(rgb)
            depth_list.append(depth)
            thermal_list.append(thermal)
            boxes_list.append(index['boxes'][idx])
            class_list.append(index['class_ids'][idx])

        if not valid:
            continue

        try:
            img_9ch, boxes, class_ids = pipeline.augment_frame_with_mosaic(
                rgb_list, depth_list, thermal_list, boxes_list, class_list,
                apply_color=True)
            if len(boxes) == 0:
                continue

            rgb_m = img_9ch[:, :, :3]
            depth_color = img_9ch[:, :, 3:6]
            thermal_m = img_9ch[:, :, 6:9]
            depth_gray = cv2.cvtColor(depth_color, cv2.COLOR_RGB2GRAY)
            depth_save = np.stack([depth_gray] * 3, axis=-1)

            writer.save(rgb_m, depth_save, thermal_m, boxes, class_ids)
            success += 1
            pbar.update(1)
        except Exception:
            continue

    pbar.close()
    generated = writer.frame_idx - start_count
    print(f"  Mosaic: {generated} frames generated")
    return generated


def _do_online_aug(index, num_aug, writer, seed):
    """Online augmentation."""
    print(f"\n[Online Aug] Generating {num_aug} samples...")
    config = AugConfig()
    pipeline = AugmentationPipeline(config)
    n_total = len(index['filenames'])
    start_count = writer.frame_idx

    success = 0
    max_attempts = num_aug * 3
    pbar = tqdm(total=num_aug, desc="Online aug")

    while success < num_aug and max_attempts > 0:
        max_attempts -= 1
        idx = random.randint(0, n_total - 1)

        rgb, depth, thermal = load_frame(
            index['rgb_paths'][idx], index['depth_paths'][idx],
            index['thermal_paths'][idx])
        if rgb is None:
            continue

        boxes = index['boxes'][idx].copy()
        class_ids = index['class_ids'][idx].copy()

        try:
            img_9ch, depth_gray, new_boxes, new_classes = pipeline.augment_frame(
                rgb, depth.copy(), thermal, boxes, class_ids,
                apply_spatial=True, apply_color=True, apply_depth_aug=True)
            if len(new_boxes) == 0:
                continue

            rgb_aug = img_9ch[:, :, :3]
            thermal_aug = img_9ch[:, :, 6:9]

            writer.save(rgb_aug, depth_gray, thermal_aug, new_boxes, new_classes)
            success += 1
            pbar.update(1)
        except Exception:
            continue

    pbar.close()
    generated = writer.frame_idx - start_count
    print(f"  Online Aug: {generated} frames generated")
    return generated


if __name__ == '__main__':
    main()
