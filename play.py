import os
import numpy as np
from PIL import Image

# ====== 颜色映射：仅保留 0~5（6 类）======
PALETTE_0_5 = np.array(
    [
        [255, 255, 255],  # 0 ImSurf
        [255, 0, 0],  # 1 Building
        [255, 255, 0],  # 2 LowVeg
        [0, 255, 0],  # 3 Tree
        [0, 255, 255],  # 4 Car
        [0, 0, 255],  # 5 Clutter
    ],
    dtype=np.uint8,
)

BOUNDARY_LABEL = 6  # 你的 Boundary 类别值
MERGE_TO_LABEL = 0  # 并入到背景/ImSurf


def read_label(path: str) -> np.ndarray:
    """
    读取 label 图，支持 .png/.tif/.jpg/.npy
    返回 (H, W) 的整型数组
    """
    ext = os.path.splitext(path)[1].lower()

    if ext == ".npy":
        label = np.load(path)
    else:
        # 强制单通道读取，避免读成 RGB
        label = Image.open(path).convert("L")
        label = np.array(label)

    if label.ndim != 2:
        raise ValueError(f"label 必须是 (H,W)，当前 shape={label.shape}")

    # 确保是整数类型
    if not np.issubdtype(label.dtype, np.integer):
        label = label.astype(np.int64)

    return label


def merge_boundary_to_background(label: np.ndarray) -> np.ndarray:
    """
    将 Boundary(label=6) 并入到 MERGE_TO_LABEL(默认0)
    """
    out = label.copy()
    out[out == BOUNDARY_LABEL] = MERGE_TO_LABEL
    return out


def label_to_rgb(label_0_5: np.ndarray) -> np.ndarray:
    """
    将 0~5 的 label 映射为 RGB
    """
    if label_0_5.min() < 0 or label_0_5.max() >= len(PALETTE_0_5):
        raise ValueError(
            f"label 超出 0~5 范围：min={label_0_5.min()}, max={label_0_5.max()}\n"
            f"提示：如果你的 ignore 不是 6（比如 255），需要先把 ignore 映射走。"
        )
    return PALETTE_0_5[label_0_5]


def save_rgb(rgb: np.ndarray, out_path: str):
    Image.fromarray(rgb, mode="RGB").save(out_path)


if __name__ == "__main__":
    in_path = "/home/wjx/data/dataset/RSS/Vaihingen/Vaihingen_1024/test_1024/masks/top_mosaic_09cm_area33_0_1.png"  # 改成你的输入路径（也可以是 label.npy）
    out_path = "/home/wjx/data/code/UNetMamba/top_mosaic_09cm_area33_0_1_label_rgb.png"

    label = read_label(in_path)
    label = merge_boundary_to_background(label)  # ⭐ 关键：6 -> 0
    rgb = label_to_rgb(label)  # 现在 label 必须只包含 0~5
    save_rgb(rgb, out_path)

    print("Saved:", out_path)
