# GeoPixelD Dataset (DOTA Patches) 数据集说明

## 1. 概述

本数据集基于 DOTA (Dataset for Object Detection in Aerial Images) 遥感图像，裁切成 800x800 的 patch 并标注了实例级别的多边形分割 mask，配合自然语言描述，构成用于**多模态对话式遥感图像分割**的训练数据（Grounded Caption Generation, GCG）。

## 2. 目录结构

```
/DOTA_patches/
├── GeoPixelD.json              # 主元数据文件（仅包含 train 集的 16,795 条记录）
├── train/                      # 训练集
│   ├── P0224_0_800_0_800.png   # 图像 patch (800x800 RGB)
│   ├── P0224_0_800_0_800.json  # 对应的多边形标注
│   ├── P0000_0_800_0_800.png   # 无标注的图像（只有 png，没有 json）
│   └── ...
├── test/                       # 测试集（结构同上）
│   ├── *.png
│   ├── *.json                  # 部分有标注
│   └── ...
└── vis_output/                 # 可视化输出（脚本生成）
```

## 3. 文件统计

| 项目 | Train | Test | 合计 |
|------|-------|------|------|
| PNG 图像 | 28,029 | 9,512 | 37,541 |
| JSON 标注 | 16,795 | 1,943 | 18,738 |
| 有标注的图像占比 | 59.9% | 20.4% | 49.9% |

> 注意：所有 JSON 标注文件都有对应的 PNG 图像，但不是所有 PNG 都有标注。无标注的 PNG 可用于无监督/半监督学习。

## 4. 文件命名规则

PNG 和 JSON 文件名格式：

```
P{source_id}_{y1}_{y2}_{x1}_{x2}.png
```

- `source_id`: 原始 DOTA 大图编号（如 `P0224`）
- `y1, y2`: 从原图裁切的纵向像素范围
- `x1, x2`: 从原图裁切的横向像素范围
- 裁切窗口大小固定为 800x800，步长 600（即相邻 patch 有 200 像素重叠）

示例：`P0000_1200_2000_600_1400.png` 表示从原图 P0000 裁切的 y=[1200,2000], x=[600,1400] 区域。

## 5. 数据格式详解

### 5.1 图像文件（*.png）

- **尺寸**: 全部为 800 x 800 像素
- **格式**: PNG, 8-bit RGB
- **内容**: 遥感/航拍图像 patch

### 5.2 多边形标注文件（*.json）

每个标注 JSON 包含以下字段：

```json
{
    "id": 3,
    "img": "P0224_600_1400_0_800.png",
    "gcg_description": "The image shows ... <p> multiple tennis courts </p> [SEG] visible ... <p> large vehicle </p> [SEG] ...",
    "polygons": [
        [                          // 第 1 个 [SEG] 对应的所有多边形
            [[x1,y1], [x2,y2], ...],  // 子多边形 1（一个实例或区域）
            [[x1,y1], [x2,y2], ...],  // 子多边形 2
            ...
        ],
        [                          // 第 2 个 [SEG] 对应的所有多边形
            [[x1,y1], [x2,y2], ...]
        ],
        ...
    ]
}
```

**字段说明：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | int | 与 GeoPixelD.json 中的 id 对应 |
| `img` | str | 对应图像文件名 |
| `gcg_description` | str | 带分割标记的自然语言描述 |
| `polygons` | list | 三层嵌套的多边形坐标列表 |

**`gcg_description` 格式说明：**

描述文本中使用特殊标记指示分割对象：
- `<p> object_name </p>` — 标记对象名称（phrase）
- `[SEG]` — 紧跟在 `</p>` 后面，表示该对象有对应的分割 mask

每个 `<p>...</p> [SEG]` 标记按顺序对应 `polygons` 数组中的一个元素。

**`polygons` 三层嵌套结构：**

```
polygons[i]       → 第 i 个 [SEG] 标记对应的多边形组
polygons[i][j]    → 该组中第 j 个子多边形（同一类别可能有多个实例）
polygons[i][j][k] → 第 k 个顶点坐标 [x, y]
```

**坐标系：**
- 像素坐标（整数），范围 [0, 800]
- 原点在图像左上角，x 向右，y 向下
- 多边形为近似闭合（首尾点相差约 1 像素，使用时建议显式闭合）

### 5.3 主元数据文件（GeoPixelD.json）

JSON 数组，共 16,795 条记录（仅 train 集），格式如下：

```json
{
    "id": 0,
    "conversations": [
        {
            "from": "human",
            "value": "Could you perform an in-depth analysis of this photo, ..."
        },
        {
            "from": "bot",
            "value": "The image is ... <p> tennis court-1 </p> [SEG] ..."
        }
    ],
    "image": "data/GeoPixelD/train/P0224_0_800_0_800.png",
    "polygons": "data/GeoPixelD/train/P0224_0_800_0_800.json"
}
```

| 字段 | 说明 |
|------|------|
| `id` | 记录编号（0 ~ 16794） |
| `conversations` | 对话对，`human` 为指令，`bot` 为带分割标注的回答 |
| `image` | 图像的相对路径（需映射到实际路径） |
| `polygons` | 多边形标注文件的相对路径 |

**`human` 指令模板（共 9 种，均匀分布）：**

所有指令均要求模型对图像进行详细描述并给出交错的分割 mask，只是措辞不同，例如：
- "Could you perform an in-depth analysis of this photo, including interleaved segmentation masks..."
- "Describe the objects in the image in detail. Use interleaved segmentation masks for clarity..."

## 6. 目标类别

数据集涵盖 DOTA 的典型遥感目标类别（按出现频次排序）：

| 类别 | 出现次数 | 类别 | 出现次数 |
|------|---------|------|---------|
| pier（码头） | 5,579 | soccer field（足球场） | 1,128 |
| small vehicle（小型车） | 4,917 | ground track field（田径场） | 1,011 |
| boat（船） | 4,802 | baseball diamond（棒球场） | 893 |
| large vehicle（大型车） | 4,686 | basketball court（篮球场） | 741 |
| plane（飞机） | 4,181 | roundabout（环形交叉口） | 730 |
| bridge（桥） | 3,107 | helicopter（直升机） | — |
| tennis court（网球场） | 1,404 | harbor（港口） | — |
| storage tank（储罐） | 1,381 | | |
| swimming pool（泳池） | 1,354 | | |

> 同一类别可能以单数/复数/修饰形式出现（如 "small vehicle" / "small vehicles" / "numerous small vehicles"），本质是同一类。

## 7. 数据读取示例

### 7.1 读取单个样本

```python
import json
import cv2
import numpy as np
import re

# 读取图像
img = cv2.imread("E:/DOTA_patches/train/P0224_600_1400_0_800.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # H=800, W=800, C=3

# 读取标注
with open("E:/DOTA_patches/train/P0224_600_1400_0_800.json") as f:
    anno = json.load(f)

# 提取对象名称列表（与 polygons 一一对应）
description = anno["gcg_description"]
object_names = re.findall(r'<p>\s*(.*?)\s*</p>', description)
# -> ['multiple tennis courts', 'large vehicle', 'small vehicles']

# 提取多边形 mask
for i, poly_group in enumerate(anno["polygons"]):
    name = object_names[i]
    mask = np.zeros((800, 800), dtype=np.uint8)
    for poly in poly_group:
        pts = np.array(poly, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 1)
    print(f"Object '{name}': {len(poly_group)} instance(s), mask sum = {mask.sum()}")
```

### 7.2 使用 GeoPixelD.json 遍历全部数据

```python
import json
import os

DATA_ROOT = "E:/DOTA_patches"

with open(os.path.join(DATA_ROOT, "GeoPixelD.json")) as f:
    metadata = json.load(f)

for entry in metadata:
    # 路径映射：将 "data/GeoPixelD/train/xxx" 映射到实际路径
    img_rel = entry["image"]            # "data/GeoPixelD/train/P0224_0_800_0_800.png"
    json_rel = entry["polygons"]        # "data/GeoPixelD/train/P0224_0_800_0_800.json"

    # 提取实际文件名
    img_filename = os.path.basename(img_rel)
    json_filename = os.path.basename(json_rel)
    split = "train" if "train" in img_rel else "test"

    img_path = os.path.join(DATA_ROOT, split, img_filename)
    json_path = os.path.join(DATA_ROOT, split, json_filename)

    # 对话内容
    human_query = entry["conversations"][0]["value"]
    bot_response = entry["conversations"][1]["value"]

    # 在此处理每个样本...
```

### 7.3 生成二值 mask 图

```python
import json
import numpy as np
import cv2

def load_masks(json_path, img_size=800):
    """
    从标注 JSON 加载所有分割 mask。

    Returns:
        masks: list of (name, binary_mask) tuples
        description: str, 原始描述文本
    """
    import re
    with open(json_path) as f:
        anno = json.load(f)

    desc = anno["gcg_description"]
    names = re.findall(r'<p>\s*(.*?)\s*</p>', desc)

    masks = []
    for i, poly_group in enumerate(anno["polygons"]):
        name = names[i] if i < len(names) else f"object-{i}"
        mask = np.zeros((img_size, img_size), dtype=np.uint8)
        for poly in poly_group:
            pts = np.array(poly, dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)
        masks.append((name, mask))

    return masks, desc
```

### 7.4 PyTorch Dataset 示例

```python
import json
import os
import re
import cv2
import numpy as np
from torch.utils.data import Dataset

class GeoPixelDDataset(Dataset):
    def __init__(self, data_root, split="train", transform=None):
        self.data_root = data_root
        self.split = split
        self.transform = transform

        with open(os.path.join(data_root, "GeoPixelD.json")) as f:
            self.metadata = json.load(f)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        entry = self.metadata[idx]

        # Load image
        img_name = os.path.basename(entry["image"])
        img_path = os.path.join(self.data_root, self.split, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load polygons
        json_name = os.path.basename(entry["polygons"])
        json_path = os.path.join(self.data_root, self.split, json_name)
        with open(json_path) as f:
            anno = json.load(f)

        # Parse object names and masks
        desc = anno["gcg_description"]
        object_names = re.findall(r'<p>\s*(.*?)\s*</p>', desc)

        masks = []
        for poly_group in anno["polygons"]:
            mask = np.zeros((800, 800), dtype=np.uint8)
            for poly in poly_group:
                pts = np.array(poly, dtype=np.int32)
                cv2.fillPoly(mask, [pts], 1)
            masks.append(mask)
        masks = np.stack(masks, axis=0)  # (N_objects, 800, 800)

        # Conversation
        query = entry["conversations"][0]["value"]
        response = entry["conversations"][1]["value"]

        sample = {
            "image": image,              # (800, 800, 3) uint8
            "masks": masks,              # (N, 800, 800) uint8, 0/1
            "object_names": object_names, # list of str
            "query": query,
            "response": response,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample
```

## 8. 注意事项

1. **路径映射**: `GeoPixelD.json` 中的 `image` 和 `polygons` 路径前缀为 `data/GeoPixelD/`，需要映射到实际存储路径。
2. **多边形闭合**: 多边形首尾点不严格相等（相差约 1px），`cv2.fillPoly` 会自动闭合，无需手动处理。
3. **一对多关系**: 一个 `<p>...</p> [SEG]` 标记可对应多个子多边形（同类别多个实例），例如 "small vehicles" 可能包含数十个车辆的多边形。
4. **无标注图像**: train 中约 40%、test 中约 80% 的 PNG 没有对应 JSON 标注。
5. **GeoPixelD.json 仅覆盖 train**: 该文件只包含 train 集的 16,795 条记录，test 集的 1,943 个标注需直接读取 `test/*.json`。
