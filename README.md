# SegEarth-R2-Qwen3.5

SegEarth-R2-Qwen3.5 是一个面向遥感图像语言引导分割任务的多模态分割项目。本项目基于 SegEarth-R2 进行工程化改造，将原始语言模型骨干迁移为 `Qwen3.5-4B`，并采用 `Swin Transformer + Mask2Former` 分割分支，实现“自然语言指令理解 + 像素级目标分割”的统一建模。

## 成果概览

截至 `2026-03-24`，当前方案在 AIRS-2026 赛道一「天眼智境」遥感图像语言引导分割挑战赛中的线上结果为：

- 总分：`0.786656`
- 分项指标：`0.729927`
- 分项指标：`0.843384`
- 当前排名：`第 3 名`

## 项目简介

遥感图像语言引导分割的目标是：

- 输入一张遥感图像
- 输入一段自然语言描述或目标指令
- 输出描述与该语言语义对应的像素级分割掩码

与传统语义分割不同，这类任务不仅要求模型具备高分辨率视觉感知能力，还要求模型能够理解复杂文本语义，并将文本语义准确映射到图像中的目标区域。
<div align="center">
  <p>
    <a href="https://www.ultralytics.com/events/yolovision?utm_source=github&utm_medium=org&utm_campaign=yv25_event" target="_blank">
      <img width="100%" src="https://github.com/user-attachments/assets/656d9b12-d829-4d7a-8d53-deb19334dc91" alt="Ultralytics YOLO banner"></a>
  </p>
</div>
本项目的核心思路是：

- 用 `Qwen3.5-4B` 负责多模态语义理解
- 用 `Swin Transformer + Mask2Former` 负责高质量像素级分割
- 用 `[SEG]` token 隐状态作为语言分支与分割分支之间的桥接表示

## 项目亮点

- 基于 `Qwen3.5-4B` 重构语言理解主干，完成从原始 SegEarth-R2 到新多模态 backbone 的迁移
- 保留 `Swin Transformer + Mask2Former` 高分辨率分割分支，兼顾语义理解与像素级定位能力
- 使用 `[SEG]` token 隐状态作为语言分支与分割分支的桥接表示
- 支持 LoRA 微调、DeepSpeed 分布式训练与 checkpoint 自动恢复
- 增加 hard mining 二阶段训练流程，提升难样本利用效率
- 打通训练、恢复训练、LoRA 合并、评估导出的完整工程链路
- 已在真实公开榜单上取得可验证结果

## 主要特性

- 支持遥感图像语言引导分割
- 基于 `Qwen3.5-4B` 的多模态语义建模
- 保留 `Swin Transformer + Mask2Former` 高分辨率分割能力
- 支持 LoRA 微调
- 支持 DeepSpeed 分布式训练
- 支持 checkpoint 自动恢复
- 支持 hard mining 难样本重训练
- 支持评估与预测结果导出

## 模型架构

整体架构由两条主路径构成：

### 1. 语义理解路径

- Qwen3.5 原生视觉编码器
- Qwen3.5 语言模型
- 负责图像内容理解与文本指令建模

### 2. 像素分割路径

- Swin Transformer 提取高分辨率视觉特征
- Pixel Decoder 聚合多尺度特征
- Mask2Former Predictor 输出最终分割掩码

### 3. 两条路径的连接方式

- 模型首先编码图像与语言指令
- 从语言模型隐藏状态中提取 `[SEG]` token 对应表示
- 使用 `SEG_token_projector` 将其映射到分割特征空间
- 将映射后的 query 输入到 Mask2Former 进行掩码预测

这种设计兼顾了：

- 大模型的语言理解能力
- 遥感图像分割对高分辨率视觉特征的需求
- 分割侧结构稳定、便于持续迭代

## 项目结构

```text
.
├── segearth_r2/
│   ├── eval/                     # 评估与预测导出
│   ├── model/                    # 模型实现
│   │   ├── language_model/       # 基于 Qwen3.5 的语言模型与桥接逻辑
│   │   ├── mask_decoder/         # Mask2Former 与 Pixel Decoder
│   │   └── mask_encoder/         # Swin Transformer
│   └── train/                    # 训练、LoRA 合并与 trainer 逻辑
├── scripts/                      # 训练 / 评估 / hard mining 脚本
├── qwen3_5/                      # 本地 Qwen3.5 相关资源
├── Qwen3.5_Migration_Summary.md  # Qwen3.5 迁移分析
└── README.md
```

## 成果

### 工程成果

除线上分数外，本项目还形成了以下可复用成果：

- 完成了 `SegEarth-R2 -> Qwen3.5` 的主干迁移
- 打通了训练、恢复训练、LoRA 合并、评估导出的完整流程
- 构建了 hard mining 二阶段训练方案
- 保留了较完整的架构分析与迁移说明文档

## 环境要求

推荐环境：

- Linux
- Python `>= 3.10`
- PyTorch `>= 2.0`
- CUDA GPU

主要依赖：

- `transformers >= 4.57.0.dev0`
- `deepspeed >= 0.12`
- `peft >= 0.6`
- `detectron2`

## 安装

### 1. 创建环境

```bash
conda create -n segearthr2 python=3.10
conda activate segearthr2

pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 2. 安装 detectron2

请参考官方文档：

```bash
https://detectron2.readthedocs.io/en/latest/tutorials/install.html
```

### 3. 编译 MSDeformAttn CUDA 扩展

```bash
cd segearth_r2/model/mask_decoder/Mask2Former_Simplify/modeling/pixel_decoder/ops
sh make.sh
```

## 数据准备

数据目录结构示例：

```text
data_path/
├── train/
│   ├── images/
│   └── annotations/
├── test/
│   ├── images/
│   └── annotations/
```

当前数据读取逻辑主要依赖如下字段：

- `image_name`
- `description`
- `answer`
- mask 标注信息

相关文件：

- `docs/Preparation.md`
- `segearth_r2/datasets/dataset.py`

## 预训练权重

在训练或评估前，需要准备以下资源：

- Qwen3.5 基础模型或本地模型权重
- Mask2Former 预训练权重
- 其他分割侧依赖权重

当前常用的分割权重路径示例：

```text
pretrained_model/mask2former/model_final_54b88a.pkl
```

具体路径可以在 `scripts/` 下的脚本中调整。

## 训练

主训练入口：

```bash
segearth_r2/train/train.py
```

推荐直接使用：

```bash
bash scripts/train.sh
```

示例命令：

```bash
python -m deepspeed.launcher.runner --master_port=29500 --include localhost:0 \
    segearth_r2/train/train.py \
    --model_name_or_path your_qwen_model \
    --vision_tower_mask pretrained_model/mask2former/model_final_54b88a.pkl \
    --base_data_path your_data_path \
    --output_dir your_output_dir \
    --lora_r 4 \
    --mask_config segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml
```

当前训练流程支持：

- LoRA 微调
- DeepSpeed 分布式训练
- 自动恢复 checkpoint
- 导出非 LoRA 可训练参数

## Hard Mining

仓库中提供了 hard mining 二阶段训练流程：

- `scripts/prepare_hard_mining.py`
- `scripts/train_hard_mining.sh`

执行方式：

```bash
bash scripts/train_hard_mining.sh
```

适用于对高损失样本进行针对性强化训练。

## LoRA 合并

LoRA 训练结束后，可将适配器权重合并为完整模型：

```bash
python segearth_r2/train/merge_lora_weights_and_save_hf_model.py \
    --model_path your_model_path \
    --vision_tower_mask pretrained_model/mask2former/model_final_54b88a.pkl \
    --mask_config segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml \
    --save_path your_save_path \
    --lora_r 4
```

## 评估与推理

主评估入口：

```bash
segearth_r2/eval/eval.py
```

示例命令：

```bash
python -m deepspeed.launcher.runner --master_port=29500 --include localhost:0 \
    segearth_r2/eval/eval.py \
    --base_data_path your_data_path \
    --model_path your_merged_model \
    --output_dir your_eval_dir \
    --mask_config segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml \
    --eval_batch_size 1
```

评估脚本会导出预测得到的 `.tif` 掩码文件，并完成输出目录打包。

## 关键文件说明

- `segearth_r2/model/language_model/llava_phi.py`  
  当前主模型实现，核心改造集中在这里。

- `segearth_r2/train/train.py`  
  训练主入口，包含 LoRA、恢复训练、模型保存逻辑。

- `segearth_r2/train/llava_trainer.py`  
  checkpoint 保存逻辑，补充了非 LoRA 参数导出。

- `segearth_r2/eval/eval.py`  
  推理与预测导出入口。

- `scripts/train.sh`  
  主训练脚本。

- `scripts/train_hard_mining.sh`  
  hard mining 训练脚本。

- `scripts/prepare_hard_mining.py`  
  hard sample 数据构建脚本。

- `Qwen3.5_Migration_Summary.md`  
  Qwen3.5 迁移分析说明。

## 后续计划

- 增加更清晰的实验配置管理
- 增加可视化推理与结果展示脚本
- 补充更完整的 ablation 与离线验证流程
- 提升自定义数据集接入体验

## 致谢

本项目建立在以下工作与工具基础之上：

- SegEarth-R2
- Qwen3.5
- Swin Transformer
- Mask2Former
- LoRA / PEFT
- DeepSpeed

## 许可说明

请遵循本仓库所依赖的上游代码、模型权重及第三方组件的许可证要求
