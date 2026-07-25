"""
RGB-D-Thermal 三模态数据增强 Pipeline
=======================================

增强策略：
  ═══════════════════════════════════════════════════════════
  一、通用增强（每张图都做）
  ═══════════════════════════════════════════════════════════
  空间增强（三模态统一参数）
    1. 随机水平翻转           p=0.5
    2. 随机缩放旋转平移        p=0.5
    3. 随机裁剪               p=0.3
    4. Mosaic (4图拼接)       p=0.3

  颜色增强（各模态独立）
    5. RGB 颜色抖动           p=0.8
    6. Thermal 增强           p=0.6
    7. Depth: 不做颜色增强

  深度特有增强
    8. 深度噪声注入           p=0.3
    9. 深度随机丢失           p=0.1

  ═══════════════════════════════════════════════════════════
  二、类别平衡增强（Copy-Paste 离线处理）
  ═══════════════════════════════════════════════════════════
  稀有类: [1, 7, 9, 10, 11] → 目标倍数过采样

  ═══════════════════════════════════════════════════════════
  三、归一化（9通道 → 分三组 ImageNet 统计值）
  ═══════════════════════════════════════════════════════════
  mean = [0.485, 0.456, 0.406]
  std  = [0.229, 0.224, 0.225]

Usage:
    from augmentation_pipeline import AugmentationPipeline

    pipeline = AugmentationPipeline()
    img_9ch, boxes, class_ids = pipeline.augment_frame(
        rgb, depth_raw, thermal, boxes, class_ids
    )
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import random
import os


# ==============================================================================
# 路径/IO 工具（处理中文路径兼容性）
# ==============================================================================

def imread_robust(path: str, flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    """鲁棒的图片读取——处理中文路径问题。

    在 Windows 上，cv2.imread 无法处理含中文的路径。
    使用 np.fromfile + cv2.imdecode 绕过此问题。
    """
    try:
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, flags)
        return img
    except Exception:
        return None


def imwrite_robust(path: str, img: np.ndarray) -> bool:
    """鲁棒的图片写入——处理中文路径问题。"""
    try:
        ext = os.path.splitext(path)[1]
        success, encoded = cv2.imencode(ext, img)
        if success:
            encoded.tofile(path)
            return True
        return False
    except Exception:
        return False


# ==============================================================================
# 配置
# ==============================================================================

@dataclass
class AugConfig:
    """增强参数配置"""
    # --- 空间增强 ---
    hflip_p: float = 0.5
    scale_shift_rotate_p: float = 0.5
    shift_limit: float = 0.1          # 平移 ±10%
    scale_limit: float = 0.3          # 缩放 ±30%
    rotate_limit: float = 10.0        # 旋转 ±10°
    safe_crop_p: float = 0.3
    crop_height: int = 640
    crop_width: int = 640
    erosion_rate: float = 0.2         # 允许裁掉最多 20%
    mosaic_p: float = 0.3

    # --- RGB 颜色抖动 ---
    rgb_color_p: float = 0.8
    brightness: float = 0.2           # ±20%
    contrast: float = 0.2             # ±20%
    saturation: float = 0.2           # ±20%
    hue: float = 0.05                 # ±5%

    # --- Thermal 增强 ---
    thermal_p: float = 0.6
    thermal_brightness: float = 0.1   # ±10%
    thermal_contrast: float = 0.15    # ±15%

    # --- 深度增强 ---
    depth_noise_p: float = 0.3        # 深度噪声注入
    depth_noise_sigma_ratio: float = 0.03  # σ = median × 3%
    depth_dropout_p: float = 0.1      # 深度随机丢失
    depth_dropout_ratio: float = 0.1  # 丢失 10% 像素
    depth_clip_enabled: bool = True

    # --- Copy-Paste ---
    rare_classes: List[int] = field(default_factory=lambda: [1, 7, 9, 10, 11])
    # 增强倍数: class_id → multiplier
    class_multipliers: Dict[int, float] = field(default_factory=lambda: {
        0: 1.0,  2: 1.0,  6: 1.5,  8: 1.5,  4: 2.0,
        5: 2.0,  3: 3.0,  9: 5.0, 10: 7.0,  1: 10.0,
        7: 15.0, 11: 50.0
    })

    # --- 归一化 ---
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    # --- 输出 ---
    output_size: Optional[Tuple[int, int]] = None  # (H, W), None = 保持原图尺寸


# ==============================================================================
# 工具函数
# ==============================================================================

def clip_bbox_to_image(bbox: np.ndarray, img_h: int, img_w: int) -> np.ndarray:
    """将 YOLO 格式 bbox 裁剪到图像范围内。

    Args:
        bbox: [cx, cy, w, h] 归一化坐标
        img_h, img_w: 图像尺寸

    Returns:
        [cx, cy, w, h] 裁剪后的归一化坐标
    """
    cx, cy, w, h = bbox
    # 转换到像素坐标
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h

    # 裁剪
    x1 = max(0, min(x1, img_w))
    y1 = max(0, min(y1, img_h))
    x2 = max(0, min(x2, img_w))
    y2 = max(0, min(y2, img_h))

    # 转回归一化
    new_w = (x2 - x1) / img_w
    new_h = (y2 - y1) / img_h
    new_cx = (x1 + x2) / 2 / img_w
    new_cy = (y1 + y2) / 2 / img_h

    return np.array([new_cx, new_cy, new_w, new_h], dtype=np.float32)


def is_valid_bbox(bbox: np.ndarray, min_size: float = 0.002) -> bool:
    """检查 bbox 是否有效（不是太小/太大/NaN）。"""
    cx, cy, w, h = bbox
    if np.isnan([cx, cy, w, h]).any():
        return False
    if w < min_size or h < min_size:
        return False
    if w > 1.0 or h > 1.0:
        return False
    if cx < 0 or cy < 0 or cx > 1.0 or cy > 1.0:
        return False
    return True


def depth_to_colormap(depth_img: np.ndarray, depth_clip: bool = True) -> np.ndarray:
    """将深度图转换为 JET 伪彩色图。

    支持输入:
        - (H, W) uint16 原始深度值
        - (H, W) uint8 灰度深度图
        - (H, W, 3) uint8 灰度深度图 (R=G=B)

    Returns:
        (H, W, 3) uint8 伪彩色图
    """
    # 处理 3 通道灰度 → 1 通道
    if depth_img.ndim == 3 and depth_img.shape[2] >= 3:
        depth_img = depth_img[:, :, 0]  # 取第一个通道（R=G=B）

    dp = depth_img.astype(np.float32)

    if depth_clip and dp.max() > 255:
        # uint16 模式：裁剪异常值
        valid = dp[dp > 0] if (dp > 0).any() else dp
        median_val = np.median(valid)
        max_depth = min(median_val * 3, 10000)
        dp[dp > max_depth] = max_depth

    dp = cv2.normalize(dp, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    dp = np.asarray(dp, dtype=np.uint8)
    colormap = cv2.applyColorMap(dp, cv2.COLORMAP_JET)
    return colormap


def to_grayscale_1ch(img: np.ndarray) -> np.ndarray:
    """将图像转为单通道灰度。"""
    if img.ndim == 2:
        return img
    if img.shape[2] == 1:
        return img[:, :, 0]
    # 3 通道 → 取平均
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.shape[2] == 3 else img[:, :, 0]


def to_grayscale_3ch(img: np.ndarray) -> np.ndarray:
    """将单通道灰度转为 3 通道（匹配原始数据集格式）。"""
    if img.ndim == 3 and img.shape[2] == 3:
        return img
    return np.stack([img] * 3, axis=-1)


# ==============================================================================
# 空间增强（三模态统一参数）
# ==============================================================================

class SpatialAugmentations:
    """空间增强——对 RGB、Depth、Thermal 使用相同随机参数。"""

    def __init__(self, config: AugConfig):
        self.config = config

    def horizontal_flip(self, rgb: np.ndarray, depth_color: np.ndarray,
                        thermal: np.ndarray, boxes: np.ndarray,
                        force: Optional[bool] = None,
                        depth_gray: Optional[np.ndarray] = None) -> Tuple:
        """随机水平翻转。额外图像也会被同步翻转。"""
        do_flip = force if force is not None else (random.random() < self.config.hflip_p)

        if do_flip:
            rgb = cv2.flip(rgb, 1)
            depth_color = cv2.flip(depth_color, 1)
            thermal = cv2.flip(thermal, 1)
            if depth_gray is not None:
                depth_gray = cv2.flip(depth_gray, 1)
            if len(boxes) > 0:
                boxes[:, 0] = 1.0 - boxes[:, 0]

        if depth_gray is not None:
            return rgb, depth_color, thermal, depth_gray, boxes
        return rgb, depth_color, thermal, boxes

    def scale_shift_rotate(self, rgb: np.ndarray, depth_color: np.ndarray,
                           thermal: np.ndarray, boxes: np.ndarray,
                           class_ids: np.ndarray,
                           force: Optional[bool] = None,
                           depth_gray: Optional[np.ndarray] = None) -> Tuple:
        """随机缩放+平移+旋转。三模态统一参数，同步更新 bbox。"""
        do_ssr = force if force is not None else (random.random() < self.config.scale_shift_rotate_p)

        if not do_ssr or len(boxes) == 0:
            if depth_gray is not None:
                return rgb, depth_color, thermal, depth_gray, boxes, class_ids
            return rgb, depth_color, thermal, boxes, class_ids

        h, w = rgb.shape[:2]
        cfg = self.config

        scale = 1.0 + random.uniform(-cfg.scale_limit, cfg.scale_limit)
        angle = random.uniform(-cfg.rotate_limit, cfg.rotate_limit)
        dx = random.uniform(-cfg.shift_limit, cfg.shift_limit) * w
        dy = random.uniform(-cfg.shift_limit, cfg.shift_limit) * h

        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        M[0, 2] += dx
        M[1, 2] += dy

        rgb = cv2.warpAffine(rgb, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        depth_color = cv2.warpAffine(depth_color, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        thermal = cv2.warpAffine(thermal, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        if depth_gray is not None:
            # Handle both 2D and 3D
            if depth_gray.ndim == 2:
                depth_gray = cv2.warpAffine(depth_gray, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            else:
                depth_gray = cv2.warpAffine(depth_gray, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        # 更新 bbox
        new_boxes = []
        new_classes = []
        for bbox, cls_id in zip(boxes, class_ids):
            cx, cy, bw, bh = bbox
            corners = np.array([
                [cx - bw / 2, cy - bh / 2, 1],
                [cx + bw / 2, cy - bh / 2, 1],
                [cx + bw / 2, cy + bh / 2, 1],
                [cx - bw / 2, cy + bh / 2, 1],
            ]) * [w, h, 1]
            transformed = corners @ M.T
            transformed[:, 0] /= w
            transformed[:, 1] /= h

            min_x, min_y = transformed.min(axis=0)
            max_x, max_y = transformed.max(axis=0)
            new_bbox = np.array([(min_x + max_x) / 2, (min_y + max_y) / 2,
                                 max_x - min_x, max_y - min_y], dtype=np.float32)
            new_bbox = clip_bbox_to_image(new_bbox, h, w)
            if is_valid_bbox(new_bbox):
                new_boxes.append(new_bbox)
                new_classes.append(cls_id)

        if len(new_boxes) > 0:
            boxes = np.array(new_boxes, dtype=np.float32)
            class_ids = np.array(new_classes, dtype=np.int64)
        else:
            boxes = np.zeros((0, 4), dtype=np.float32)
            class_ids = np.zeros(0, dtype=np.int64)

        if depth_gray is not None:
            return rgb, depth_color, thermal, depth_gray, boxes, class_ids
        return rgb, depth_color, thermal, boxes, class_ids

    def safe_crop(self, rgb: np.ndarray, depth_color: np.ndarray,
                  thermal: np.ndarray, boxes: np.ndarray,
                  class_ids: np.ndarray,
                  force: Optional[bool] = None,
                  depth_gray: Optional[np.ndarray] = None) -> Tuple:
        """随机裁剪 80%~100% 区域，确保不裁掉目标。"""
        do_crop = force if force is not None else (random.random() < self.config.safe_crop_p)

        if not do_crop or len(boxes) == 0:
            if depth_gray is not None:
                return rgb, depth_color, thermal, depth_gray, boxes, class_ids
            return rgb, depth_color, thermal, boxes, class_ids

        h, w = rgb.shape[:2]
        cfg = self.config

        crop_ratio = random.uniform(1.0 - cfg.erosion_rate, 1.0)
        crop_h = int(h * crop_ratio)
        crop_w = int(w * crop_ratio)

        max_attempts = 20
        best_crop = None
        best_count = -1

        for _ in range(max_attempts):
            x1 = random.randint(0, w - crop_w) if w > crop_w else 0
            y1 = random.randint(0, h - crop_h) if h > crop_h else 0
            x2, y2 = x1 + crop_w, y1 + crop_h

            count = sum(1 for bbox in boxes
                       if x1 <= bbox[0] * w <= x2 and y1 <= bbox[1] * h <= y2)

            if count > best_count:
                best_count = count
                best_crop = (x1, y1, x2, y2)
                if count >= len(boxes):
                    break

        if best_crop is None or best_count == 0:
            if depth_gray is not None:
                return rgb, depth_color, thermal, depth_gray, boxes, class_ids
            return rgb, depth_color, thermal, boxes, class_ids

        x1, y1, x2, y2 = best_crop

        rgb = rgb[y1:y2, x1:x2]
        depth_color = depth_color[y1:y2, x1:x2]
        thermal = thermal[y1:y2, x1:x2]
        if depth_gray is not None:
            depth_gray = depth_gray[y1:y2, x1:x2]

        new_h, new_w = rgb.shape[:2]
        new_boxes = []
        new_classes = []
        for bbox, cls_id in zip(boxes, class_ids):
            new_cx = (bbox[0] * w - x1) / new_w
            new_cy = (bbox[1] * h - y1) / new_h
            new_w_n = bbox[2] * w / new_w
            new_h_n = bbox[3] * h / new_h
            new_bbox = np.array([new_cx, new_cy, new_w_n, new_h_n], dtype=np.float32)
            new_bbox = clip_bbox_to_image(new_bbox, new_h, new_w)
            if is_valid_bbox(new_bbox):
                new_boxes.append(new_bbox)
                new_classes.append(cls_id)

        if len(new_boxes) > 0:
            boxes = np.array(new_boxes, dtype=np.float32)
            class_ids = np.array(new_classes, dtype=np.int64)
        else:
            boxes = np.zeros((0, 4), dtype=np.float32)
            class_ids = np.zeros(0, dtype=np.int64)

        if depth_gray is not None:
            return rgb, depth_color, thermal, depth_gray, boxes, class_ids
        return rgb, depth_color, thermal, boxes, class_ids

    def mosaic(self, rgb_list: List[np.ndarray], depth_list: List[np.ndarray],
               thermal_list: List[np.ndarray], boxes_list: List[np.ndarray],
               class_list: List[np.ndarray],
               force: Optional[bool] = None) -> Tuple:
        """Mosaic 增强——4 张图拼成 1 张。"""
        do_mosaic = force if force is not None else (random.random() < self.config.mosaic_p)

        if not do_mosaic or len(rgb_list) < 4:
            # 返回第一张图
            return (rgb_list[0], depth_list[0], thermal_list[0],
                    boxes_list[0], class_list[0])

        # 选4张图（随机）
        indices = random.sample(range(len(rgb_list)), min(4, len(rgb_list)))
        if len(indices) < 4:
            return (rgb_list[0], depth_list[0], thermal_list[0],
                    boxes_list[0], class_list[0])

        imgs_rgb = [rgb_list[i] for i in indices]
        imgs_depth = [depth_list[i] for i in indices]
        imgs_thermal = [thermal_list[i] for i in indices]
        boxes_pool = [boxes_list[i] for i in indices]
        classes_pool = [class_list[i] for i in indices]

        # 统一 resize 到目标尺寸
        h, w = rgb_list[0].shape[:2]
        half_h, half_w = h // 2, w // 2
        mosaic_h, mosaic_w = h, w

        # 随机决定拼接中心点
        cx = random.randint(int(w * 0.3), int(w * 0.7))
        cy = random.randint(int(h * 0.3), int(h * 0.7))

        mosaic_rgb = np.zeros((mosaic_h, mosaic_w, 3), dtype=np.uint8)
        mosaic_depth = np.zeros((mosaic_h, mosaic_w, 3), dtype=np.uint8)
        mosaic_thermal = np.zeros((mosaic_h, mosaic_w, 3), dtype=np.uint8)

        all_boxes = []
        all_classes = []

        # 四个象限的放置位置
        placements = [
            (0, 0, cx, cy),           # 左上 → 图片0
            (cx, 0, mosaic_w, cy),    # 右上 → 图片1
            (0, cy, cx, mosaic_h),    # 左下 → 图片2
            (cx, cy, mosaic_w, mosaic_h),  # 右下 → 图片3
        ]

        for idx, (px1, py1, px2, py2) in enumerate(placements):
            region_w = px2 - px1
            region_h = py2 - py1

            if region_w <= 0 or region_h <= 0:
                continue

            # Resize 图片到对应区域大小
            img_rgb = cv2.resize(imgs_rgb[idx], (region_w, region_h))
            img_depth = cv2.resize(imgs_depth[idx], (region_w, region_h))
            img_thermal = cv2.resize(imgs_thermal[idx], (region_w, region_h))

            mosaic_rgb[py1:py2, px1:px2] = img_rgb
            mosaic_depth[py1:py2, px1:px2] = img_depth
            mosaic_thermal[py1:py2, px1:px2] = img_thermal

            # 更新 bbox（先缩放，再平移）
            scale_x = region_w / mosaic_w
            scale_y = region_h / mosaic_h
            offset_x = px1 / mosaic_w
            offset_y = py1 / mosaic_h

            for bbox, cls_id in zip(boxes_pool[idx], classes_pool[idx]):
                cx_b, cy_b, bw, bh = bbox
                new_cx = cx_b * scale_x + offset_x
                new_cy = cy_b * scale_y + offset_y
                new_w = bw * scale_x
                new_h = bh * scale_y
                new_bbox = np.array([new_cx, new_cy, new_w, new_h], dtype=np.float32)
                new_bbox = clip_bbox_to_image(new_bbox, mosaic_h, mosaic_w)
                if is_valid_bbox(new_bbox):
                    all_boxes.append(new_bbox)
                    all_classes.append(cls_id)

        if len(all_boxes) > 0:
            final_boxes = np.array(all_boxes, dtype=np.float32)
            final_classes = np.array(all_classes, dtype=np.int64)
        else:
            final_boxes = np.zeros((0, 4), dtype=np.float32)
            final_classes = np.zeros(0, dtype=np.int64)

        return mosaic_rgb, mosaic_depth, mosaic_thermal, final_boxes, final_classes


# ==============================================================================
# 颜色增强（各模态独立）
# ==============================================================================

class ColorAugmentations:
    """颜色增强——各模态独立处理。"""

    def __init__(self, config: AugConfig):
        self.config = config

    def rgb_color_jitter(self, rgb: np.ndarray,
                         force: Optional[bool] = None) -> np.ndarray:
        """RGB 颜色抖动: 亮度 ±20%, 对比度 ±20%, 饱和度 ±20%, 色相 ±5%"""
        do_jitter = force if force is not None else (random.random() < self.config.rgb_color_p)

        if not do_jitter:
            return rgb

        cfg = self.config
        img = rgb.astype(np.float32)

        # 亮度
        brightness = 1.0 + random.uniform(-cfg.brightness, cfg.brightness)
        img = img * brightness

        # 对比度（需要先转灰度算均值）
        if random.random() < 0.5:
            mean = img.mean(axis=(0, 1), keepdims=True)
            contrast = 1.0 + random.uniform(-cfg.contrast, cfg.contrast)
            img = (img - mean) * contrast + mean

        # 饱和度（HSV 空间）
        if random.random() < 0.5:
            hsv = cv2.cvtColor(img.clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
            saturation = 1.0 + random.uniform(-cfg.saturation, cfg.saturation)
            hsv = hsv.astype(np.float32)
            hsv[:, :, 1] = hsv[:, :, 1] * saturation
            hsv[:, :, 1] = hsv[:, :, 1].clip(0, 255)
            img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)

        # 色相
        if random.random() < 0.5:
            hsv = cv2.cvtColor(img.clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
            hue_shift = int(random.uniform(-cfg.hue * 180, cfg.hue * 180))
            hsv = hsv.astype(np.int32)
            hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
            hsv = hsv.clip(0, 255).astype(np.uint8)
            img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype(np.float32)

        # 高斯噪声
        if random.random() < 0.3:
            noise = np.random.randn(*img.shape) * random.uniform(5, 20)
            img = img + noise

        # ISO 噪声（泊松噪声）
        if random.random() < 0.2:
            img = img.clip(0, 255)
            noise = np.random.poisson(img / 255.0 * 20) / 20 * 255.0
            img = img + (noise - img) * 0.3

        return img.clip(0, 255).astype(np.uint8)

    def thermal_enhance(self, thermal: np.ndarray,
                        force: Optional[bool] = None) -> np.ndarray:
        """Thermal 增强: 对比度 ±15%, 亮度 ±10%"""
        do_enhance = force if force is not None else (random.random() < self.config.thermal_p)

        if not do_enhance:
            return thermal

        cfg = self.config
        img = thermal.astype(np.float32)

        # 亮度
        brightness = 1.0 + random.uniform(-cfg.thermal_brightness, cfg.thermal_brightness)
        img = img * brightness

        # 对比度
        mean = img.mean(axis=(0, 1), keepdims=True)
        contrast = 1.0 + random.uniform(-cfg.thermal_contrast, cfg.thermal_contrast)
        img = (img - mean) * contrast + mean

        return img.clip(0, 255).astype(np.uint8)


# ==============================================================================
# 深度特有增强
# ==============================================================================

class DepthAugmentations:
    """深度图增强——在深度可视化图像上操作。

    输入可以是:
        - (H, W) uint8 或 uint16
        - (H, W, 1) 或 (H, W, 3) uint8 灰度图
    """

    def __init__(self, config: AugConfig):
        self.config = config

    def noise_injection(self, depth_img: np.ndarray,
                        force: Optional[bool] = None) -> np.ndarray:
        """模拟传感器噪声: 添加高斯噪声。"""
        do_noise = force if force is not None else (random.random() < self.config.depth_noise_p)

        if not do_noise:
            return depth_img

        # 转为 1ch 处理
        is_3ch = depth_img.ndim == 3 and depth_img.shape[2] >= 3
        work = to_grayscale_1ch(depth_img).astype(np.float32)

        valid_mask = work > 0
        if not valid_mask.any():
            return depth_img

        # 噪声强度 = 像素中位数 × 3%
        median_val = np.median(work[valid_mask])
        sigma = max(median_val * self.config.depth_noise_sigma_ratio, 1.0)
        noise = np.random.randn(*work.shape) * sigma
        work = work + noise
        work = work.clip(0, 255 if depth_img.dtype == np.uint8 else 65535)

        result = work.astype(depth_img.dtype)
        if is_3ch:
            result = to_grayscale_3ch(result)
        return result

    def random_dropout(self, depth_img: np.ndarray,
                       force: Optional[bool] = None) -> np.ndarray:
        """随机 mask 掉 5%~15% 的深度像素为 0，模拟真实深度缺失。"""
        do_dropout = force if force is not None else (random.random() < self.config.depth_dropout_p)

        if not do_dropout:
            return depth_img

        is_3ch = depth_img.ndim == 3 and depth_img.shape[2] >= 3
        work = to_grayscale_1ch(depth_img)

        dropout_ratio = random.uniform(0.05, 0.15)
        dropout_mask = np.random.rand(*work.shape) > dropout_ratio
        result = work.copy()
        result[~dropout_mask] = 0

        if is_3ch:
            result = to_grayscale_3ch(result)
        return result


# ==============================================================================
# Copy-Paste 增强（离线处理稀有类）
# ==============================================================================

class CopyPasteAugmentation:
    """Copy-Paste 增强——将稀有类目标贴到其他图上。

    用于离线生成稀有类的增强数据，解决类别不平衡问题。
    """

    def __init__(self, config: AugConfig):
        self.config = config

    @staticmethod
    def _paste_with_mask(dst_img: np.ndarray, crop_img: np.ndarray,
                         y: int, x: int, h: int, w: int,
                         mask_2d: np.ndarray):
        """将 crop 按 mask 贴到 dst 上，自动适配通道数和维度。"""
        dst_patch = dst_img[y:y+h, x:x+w].astype(np.float32)
        crop_patch = crop_img[:h, :w].astype(np.float32)

        # 统一维度：如果两方维度不匹配，扩展低维的一方
        if dst_patch.ndim != crop_patch.ndim:
            if dst_patch.ndim == 2 and crop_patch.ndim == 3:
                dst_patch = dst_patch[:, :, np.newaxis]
            elif dst_patch.ndim == 3 and crop_patch.ndim == 2:
                crop_patch = crop_patch[:, :, np.newaxis]

        # 构建 mask
        if dst_patch.ndim == 2 or dst_patch.shape[2] == 1:
            _mask = mask_2d[:, :, np.newaxis] if dst_patch.ndim == 3 else mask_2d
        elif dst_patch.ndim == 3:
            _mask = np.stack([mask_2d] * dst_patch.shape[2], axis=-1)
        else:
            return

        # 确保 mask 和 patch 形状一致
        if _mask.shape != dst_patch.shape:
            # 如果还是不匹配，按通道数调整
            if _mask.ndim == 2 and dst_patch.ndim == 3:
                _mask = np.stack([_mask] * dst_patch.shape[2], axis=-1)
            elif _mask.ndim == 3 and dst_patch.ndim == 2:
                _mask = _mask[:, :, 0]

        result = dst_patch * (1 - _mask) + crop_patch * _mask
        # 还原到目标图像的原始形状
        if result.shape != dst_img[y:y+h, x:x+w].shape:
            result = result.reshape(dst_img[y:y+h, x:x+w].shape)
        dst_img[y:y+h, x:x+w] = result.astype(dst_img.dtype)

    def copy_paste_single(self, src_rgb: np.ndarray, src_depth: np.ndarray,
                          src_thermal: np.ndarray, src_bbox: np.ndarray,
                          dst_rgb: np.ndarray, dst_depth: np.ndarray,
                          dst_thermal: np.ndarray) -> Tuple[np.ndarray, ...]:
        """从源图裁剪目标区域，增强后贴到目标图。

        Args:
            src_rgb, src_depth, src_thermal: 源图（含稀有类目标）
            src_bbox: [cx, cy, w, h] 归一化，稀有类目标的 bbox
            dst_rgb, dst_depth, dst_thermal: 目标图

        Returns:
            dst_rgb, dst_depth, dst_thermal: 粘贴后的目标图
            new_bbox: 新目标在目标图中的归一化 bbox
        """
        h, w = src_rgb.shape[:2]
        cx, cy, bw, bh = src_bbox

        # 像素坐标，略微扩大裁剪区域
        padding = 0.1
        x1 = int((cx - bw / 2 - bw * padding) * w)
        y1 = int((cy - bh / 2 - bh * padding) * h)
        x2 = int((cx + bw / 2 + bw * padding) * w)
        y2 = int((cy + bh / 2 + bh * padding) * h)

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        crop_w, crop_h = x2 - x1, y2 - y1
        if crop_w <= 1 or crop_h <= 1:
            return dst_rgb, dst_depth, dst_thermal, None

        # 裁剪
        crop_rgb = src_rgb[y1:y2, x1:x2].copy()
        crop_depth = src_depth[y1:y2, x1:x2].copy()
        crop_thermal = src_thermal[y1:y2, x1:x2].copy()

        # 随机增强裁剪区域
        scale = random.uniform(0.8, 1.5)
        new_w = int(crop_w * scale)
        new_h = int(crop_h * scale)
        if new_w <= 2 or new_h <= 2:
            return dst_rgb, dst_depth, dst_thermal, None
        crop_rgb = cv2.resize(crop_rgb, (new_w, new_h))
        crop_depth = cv2.resize(crop_depth, (new_w, new_h))
        crop_thermal = cv2.resize(crop_thermal, (new_w, new_h))

        # 随机翻转
        if random.random() < 0.5:
            crop_rgb = cv2.flip(crop_rgb, 1)
            crop_depth = cv2.flip(crop_depth, 1)
            crop_thermal = cv2.flip(crop_thermal, 1)

        # 随机旋转
        if random.random() < 0.5:
            angle = random.uniform(-15, 15)
            center = (new_w // 2, new_h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            crop_rgb = cv2.warpAffine(crop_rgb, M, (new_w, new_h), borderMode=cv2.BORDER_CONSTANT)
            crop_depth = cv2.warpAffine(crop_depth, M, (new_w, new_h), borderMode=cv2.BORDER_CONSTANT)
            crop_thermal = cv2.warpAffine(crop_thermal, M, (new_w, new_h), borderMode=cv2.BORDER_CONSTANT)

        # 在目标图上随机位置粘贴
        dst_h, dst_w = dst_rgb.shape[:2]

        # 确保粘贴位置不会完全出界
        max_x = max(1, dst_w - new_w)
        max_y = max(1, dst_h - new_h)
        paste_x = random.randint(0, max_x)
        paste_y = random.randint(0, max_y)

        # 计算有效粘贴区域
        paste_w = min(new_w, dst_w - paste_x)
        paste_h = min(new_h, dst_h - paste_y)

        if paste_w <= 1 or paste_h <= 1:
            return dst_rgb, dst_depth, dst_thermal, None

        # 创建软边缘 mask（渐变边界，自然融合）
        mask = np.ones((paste_h, paste_w), dtype=np.float32)
        edge_width = min(3, paste_w // 4, paste_h // 4)
        if edge_width > 0:
            mask[:edge_width, :] = np.linspace(0, 1, edge_width)[:, np.newaxis]
            mask[-edge_width:, :] = np.linspace(1, 0, edge_width)[:, np.newaxis]
            mask[:, :edge_width] *= np.linspace(0, 1, edge_width)[np.newaxis, :]
            mask[:, -edge_width:] *= np.linspace(1, 0, edge_width)[np.newaxis, :]

        # 粘贴 RGB（3 通道）
        dst_rgb_patch = dst_rgb[paste_y:paste_y+paste_h, paste_x:paste_x+paste_w].astype(np.float32)
        crop_rgb_patch = crop_rgb[:paste_h, :paste_w].astype(np.float32)
        mask_3ch = np.stack([mask] * 3, axis=-1)
        dst_rgb[paste_y:paste_y+paste_h, paste_x:paste_x+paste_w] = (
            dst_rgb_patch * (1 - mask_3ch) + crop_rgb_patch * mask_3ch
        ).astype(np.uint8)

        # 粘贴 Depth（单通道或三通道）
        self._paste_with_mask(dst_depth, crop_depth, paste_y, paste_x, paste_h, paste_w, mask)

        # 粘贴 Thermal
        self._paste_with_mask(dst_thermal, crop_thermal, paste_y, paste_x, paste_h, paste_w, mask)

        # 计算新 bbox（归一化）
        new_cx = (paste_x + paste_w / 2) / dst_w
        new_cy = (paste_y + paste_h / 2) / dst_h
        new_bw = paste_w / dst_w
        new_bh = paste_h / dst_h
        new_bbox = np.array([new_cx, new_cy, new_bw, new_bh], dtype=np.float32)

        return dst_rgb, dst_depth, dst_thermal, new_bbox


# ==============================================================================
# 主 Pipeline
# ==============================================================================

class AugmentationPipeline:
    """RGB-D-Thermal 三模态增强 Pipeline。

    Usage:
        pipeline = AugmentationPipeline()

        # 在线增强（训练时）
        img_9ch, boxes, class_ids = pipeline.augment_frame(
            rgb, depth_raw, thermal, boxes, class_ids
        )

        # 离线 Copy-Paste
        aug_data = pipeline.copy_paste_dataset(rgb_list, depth_list,
                                                thermal_list, boxes_list,
                                                class_list)
    """

    def __init__(self, config: Optional[AugConfig] = None):
        self.config = config or AugConfig()
        self.spatial = SpatialAugmentations(self.config)
        self.color = ColorAugmentations(self.config)
        self.depth_aug = DepthAugmentations(self.config)
        self.copy_paste = CopyPasteAugmentation(self.config)

    def augment_frame(self, rgb: np.ndarray, depth_raw: np.ndarray,
                      thermal: np.ndarray, boxes: np.ndarray,
                      class_ids: np.ndarray,
                      apply_spatial: bool = True,
                      apply_color: bool = True,
                      apply_depth_aug: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """对一帧做完整增强 Pipeline。

        Args:
            rgb:        (H, W, 3) uint8 RGB 图像
            depth_raw:  (H, W) 或 (H, W, 3) uint8 灰度深度图
            thermal:    (H, W, 3) uint8 热红外图像
            boxes:      (N, 4) [cx, cy, w, h] 归一化 YOLO 格式
            class_ids:  (N,)  类别编号

        Returns:
            img_9ch:      (H, W, 9) uint8 9 通道融合图
            depth_aug:    (H, W) 或 (H, W, 3) 增强后的灰度深度图（保存用）
            boxes:        (N', 4)
            class_ids:    (N',)
        """
        boxes = np.array(boxes, dtype=np.float32).reshape(-1, 4)
        class_ids = np.array(class_ids, dtype=np.int64).reshape(-1)

        # ① 深度增强（在灰度深度上做，伪彩色转换之前）
        if apply_depth_aug:
            depth_raw = self.depth_aug.noise_injection(depth_raw)
            depth_raw = self.depth_aug.random_dropout(depth_raw)

        # ② 保存增强后的灰度深度（用于离线保存）
        depth_gray = depth_raw.copy()

        # ③ 深度 → 伪彩色
        depth_color = depth_to_colormap(depth_raw, depth_clip=self.config.depth_clip_enabled)

        # ④ 空间增强（三模态统一参数）
        if apply_spatial:
            # 需要把 depth_gray 也做同样的空间变换
            rgb, depth_color, thermal, depth_gray, boxes = self.spatial.horizontal_flip(
                rgb, depth_color, thermal, boxes, depth_gray=depth_gray
            )

            rgb, depth_color, thermal, depth_gray, boxes, class_ids = self.spatial.scale_shift_rotate(
                rgb, depth_color, thermal, boxes, class_ids, depth_gray=depth_gray
            )

            rgb, depth_color, thermal, depth_gray, boxes, class_ids = self.spatial.safe_crop(
                rgb, depth_color, thermal, boxes, class_ids, depth_gray=depth_gray
            )

        # 统一尺寸
        if self.config.output_size is not None:
            out_h, out_w = self.config.output_size
            rgb = cv2.resize(rgb, (out_w, out_h))
            depth_color = cv2.resize(depth_color, (out_w, out_h))
            thermal = cv2.resize(thermal, (out_w, out_h))
            if depth_gray.ndim == 2:
                depth_gray = cv2.resize(depth_gray, (out_w, out_h))
            else:
                depth_gray = cv2.resize(depth_gray, (out_w, out_h))

        # ⑤ 颜色增强（各模态独立）
        if apply_color:
            rgb = self.color.rgb_color_jitter(rgb)
            thermal = self.color.thermal_enhance(thermal)
            # depth 不做颜色增强

        # ⑥ Merge → 9 通道
        img_9ch = np.concatenate([rgb, depth_color, thermal], axis=-1)  # (H, W, 9)

        return img_9ch, depth_gray, boxes, class_ids

    def augment_frame_with_mosaic(self, rgb_pool: List[np.ndarray],
                                  depth_pool: List[np.ndarray],
                                  thermal_pool: List[np.ndarray],
                                  boxes_pool: List[np.ndarray],
                                  class_pool: List[np.ndarray],
                                  apply_color: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Mosaic 增强——4张图拼1张，然后做颜色增强。

        Args:
            rgb_pool:    候选 RGB 图列表
            depth_pool:  候选深度图列表（原始 uint16）
            thermal_pool: 候选热红外图列表
            boxes_pool:  每张图的 bbox 列表
            class_pool:  每张图的类别列表

        Returns:
            img_9ch, boxes, class_ids
        """
        # 先将所有深度转为伪彩色
        depth_color_pool = [depth_to_colormap(d, self.config.depth_clip_enabled) for d in depth_pool]

        # Mosaic
        rgb_m, depth_m, thermal_m, boxes, class_ids = self.spatial.mosaic(
            rgb_pool, depth_color_pool, thermal_pool, boxes_pool, class_pool
        )

        # 颜色增强
        if apply_color:
            rgb_m = self.color.rgb_color_jitter(rgb_m)
            thermal_m = self.color.thermal_enhance(thermal_m)

        # 统一尺寸
        if self.config.output_size is not None:
            out_h, out_w = self.config.output_size
            rgb_m = cv2.resize(rgb_m, (out_w, out_h))
            depth_m = cv2.resize(depth_m, (out_w, out_h))
            thermal_m = cv2.resize(thermal_m, (out_w, out_h))

        # Merge
        img_9ch = np.concatenate([rgb_m, depth_m, thermal_m], axis=-1)

        return img_9ch, boxes, class_ids

    def copy_paste_dataset(self, rgb_images: List[np.ndarray],
                           depth_raws: List[np.ndarray],
                           thermal_images: List[np.ndarray],
                           all_boxes: List[np.ndarray],
                           all_classes: List[np.ndarray],
                           target_counts: Optional[Dict[int, int]] = None) -> List[Dict]:
        """对稀有类做 Copy-Paste 增强，返回增强样本列表。

        Args:
            rgb_images:      所有 RGB 图
            depth_raws:      所有原始深度图
            thermal_images:  所有热红外图
            all_boxes:       每张图的 bbox 列表
            all_classes:     每张图的类别列表
            target_counts:   {class_id: target_count} 目标框数

        Returns:
            List[Dict]: 增强样本列表，每个样本包含:
                {rgb, depth, thermal, boxes, class_ids}
        """
        # 统计当前各类别框数
        class_counts = {}
        for classes in all_classes:
            for c in classes:
                class_counts[c] = class_counts.get(c, 0) + 1

        # 构建 类别→图片索引 的映射
        class_to_images = {}
        for img_idx, classes in enumerate(all_classes):
            for cls_id in set(classes):
                if cls_id not in class_to_images:
                    class_to_images[cls_id] = []
                class_to_images[cls_id].append(img_idx)

        # 默认目标：每个稀有类至少 1000 框
        if target_counts is None:
            target_counts = {c: 1000 for c in self.config.rare_classes}

        augmented_samples = []
        n_total = len(rgb_images)

        for rare_class in self.config.rare_classes:
            current_count = class_counts.get(rare_class, 0)
            target = target_counts.get(rare_class, 1000)
            deficit = target - current_count

            if deficit <= 0:
                print(f"  Class {rare_class}: {current_count} boxes (sufficient, skip)")
                continue

            print(f"  Class {rare_class}: {current_count} → target {target} "
                  f"(deficit: {deficit})")

            if rare_class not in class_to_images:
                print(f"    WARNING: No images found for class {rare_class}")
                continue

            src_indices = class_to_images[rare_class]
            paste_count = 0

            # 每个稀有类目标粘贴 1-3 次
            pastes_per_object = min(3, max(1, deficit // (len(src_indices) * 2) + 1))

            for src_idx in src_indices:
                src_rgb = rgb_images[src_idx]
                src_depth = depth_raws[src_idx]
                src_thermal = thermal_images[src_idx]
                src_boxes = all_boxes[src_idx]
                src_classes = all_classes[src_idx]

                # 找到稀有类的 bbox
                rare_mask = src_classes == rare_class
                if not rare_mask.any():
                    continue

                rare_boxes = src_boxes[rare_mask]

                for src_bbox in rare_boxes:
                    for _ in range(pastes_per_object):
                        # 随机选目标图
                        dst_idx = random.randint(0, n_total - 1)
                        dst_rgb = rgb_images[dst_idx].copy()
                        dst_depth = depth_raws[dst_idx].copy()
                        dst_thermal = thermal_images[dst_idx].copy()

                        # 做 copy-paste
                        dst_rgb, dst_depth, dst_thermal, new_bbox = \
                            self.copy_paste.copy_paste_single(
                                src_rgb, src_depth, src_thermal, src_bbox,
                                dst_rgb, dst_depth, dst_thermal
                            )

                        if new_bbox is None:
                            continue

                        # 合并原图 bbox + 新 bbox
                        new_boxes = np.vstack([
                            all_boxes[dst_idx],
                            new_bbox.reshape(1, 4)
                        ])
                        new_classes = np.hstack([
                            all_classes[dst_idx],
                            np.array([rare_class])
                        ])

                        augmented_samples.append({
                            'rgb': dst_rgb,
                            'depth': dst_depth,
                            'thermal': dst_thermal,
                            'boxes': new_boxes,
                            'class_ids': new_classes,
                            'source': f'copy_paste_class_{rare_class}',
                        })

                        paste_count += 1

                        if paste_count >= deficit:
                            break

                    if paste_count >= deficit:
                        break

                if paste_count >= deficit:
                    break

            print(f"    Generated {paste_count} augmented samples")

        return augmented_samples


# ==============================================================================
# 归一化工具
# ==============================================================================

def normalize_9ch(img_9ch: np.ndarray,
                  mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
                  std: Tuple[float, float, float] = (0.229, 0.224, 0.225)) -> np.ndarray:
    """9 通道归一化，分三组各自用 ImageNet 统计值。

    Args:
        img_9ch: (H, W, 9) uint8 [0, 255] 或 float32

    Returns:
        (9, H, W) float32 归一化后的 tensor 格式
    """
    img = img_9ch.astype(np.float32) / 255.0  # [0, 1]

    # 通道 0-2: RGB → ImageNet norm
    for c in range(3):
        img[:, :, c] = (img[:, :, c] - mean[c]) / std[c]

    # 通道 3-5: Depth colormap → ImageNet norm
    for c in range(3):
        img[:, :, c + 3] = (img[:, :, c + 3] - mean[c]) / std[c]

    # 通道 6-8: Thermal → ImageNet norm
    for c in range(3):
        img[:, :, c + 6] = (img[:, :, c + 6] - mean[c]) / std[c]

    # 转为 (C, H, W) 格式
    img = np.transpose(img, (2, 0, 1)).astype(np.float32)

    return img


def denormalize_9ch(img_tensor: np.ndarray,
                    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
                    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)) -> np.ndarray:
    """反归一化（用于可视化）。"""
    img = img_tensor.copy()
    if img.shape[0] == 9:
        img = np.transpose(img, (1, 2, 0))  # (C,H,W) → (H,W,C)

    for c in range(3):
        img[:, :, c] = img[:, :, c] * std[c] + mean[c]
        img[:, :, c + 3] = img[:, :, c + 3] * std[c] + mean[c]
        img[:, :, c + 6] = img[:, :, c + 6] * std[c] + mean[c]

    return (img.clip(0, 1) * 255).astype(np.uint8)


# ==============================================================================
# 统计工具（可选：统计自定义 mean/std）
# ==============================================================================

def compute_dataset_statistics(image_dir: str, modalities: List[str],
                               num_samples: int = 500) -> Dict[str, Tuple]:
    """统计数据集的 mean/std（在各模态的 [0,255] 值域上）。"""
    import os
    stats = {}
    for mod in modalities:
        mod_dir = os.path.join(image_dir, mod)
        files = [f for f in os.listdir(mod_dir) if f.endswith(('.jpg', '.png'))]
        files = files[:num_samples]

        pixels = []
        for f in files:
            img = cv2.imread(os.path.join(mod_dir, f))
            if img is None:
                continue
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 1:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            pixels.append(img.reshape(-1, 3).astype(np.float32))

        if pixels:
            all_pixels = np.concatenate(pixels, axis=0)
            mean = all_pixels.mean(axis=0) / 255.0  # 归一化到 [0,1]
            std = all_pixels.std(axis=0) / 255.0
            stats[mod] = (tuple(mean.tolist()), tuple(std.tolist()))

    return stats


# ==============================================================================
# 测试代码
# ==============================================================================

if __name__ == '__main__':
    print("增强 Pipeline 模块加载成功!")
    print(f"  - SpatialAugmentations: flip, scale/rotate, crop, mosaic")
    print(f"  - ColorAugmentations: RGB jitter, thermal enhance")
    print(f"  - DepthAugmentations: noise injection, random dropout")
    print(f"  - CopyPasteAugmentation: rare class oversampling")
    print(f"  - AugmentationPipeline: unified pipeline")
    print(f"  - normalize_9ch / denormalize_9ch")

    config = AugConfig()
    print(f"\n默认配置:")
    print(f"  hflip_p={config.hflip_p}, scale_shift_rotate_p={config.scale_shift_rotate_p}")
    print(f"  safe_crop_p={config.safe_crop_p}, mosaic_p={config.mosaic_p}")
    print(f"  rgb_color_p={config.rgb_color_p}, thermal_p={config.thermal_p}")
    print(f"  depth_noise_p={config.depth_noise_p}, depth_dropout_p={config.depth_dropout_p}")
    print(f"  mean={config.mean}, std={config.std}")
