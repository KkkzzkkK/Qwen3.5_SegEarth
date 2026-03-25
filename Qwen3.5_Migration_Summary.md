# SegEarth-R2 Qwen3.5 Summary

## 1. Qwen3.5-4B Architecture Analysis

### 1.1 Overall Structure

```
Qwen3_5ForConditionalGeneration
  ├── model (Qwen3_5Model)
  │   ├── visual              ← Native ViT (24层, hidden=1024, out=2560)
  │   │   ├── patch_embed.proj
  │   │   ├── pos_embed
  │   │   ├── blocks[0-23]    ← {attn.qkv, attn.proj, mlp.*, norm1, norm2}
  │   │   └── merger           ← spatial_merge_size=2, linear_fc1/fc2
  │   └── language_model       ← Text Model (32层, hidden=2560)
  │       ├── embed_tokens     ← vocab_size=248320
  │       ├── layers[0-31]     ← Hybrid: 24 DeltaNet + 8 Full Attention
  │       └── norm
  ├── lm_head                  ← tied to embed_tokens (tie_word_embeddings=true)
  └── mtp                      ← Multi-Token Prediction (training only)
```

### 1.2 Hybrid Attention: DeltaNet + Full Attention

32层中按 `full_attention_interval=4` 交替分布:

| Layer Index | Type             | Attention Module   | Key Params                                         |
|-------------|------------------|--------------------|-----------------------------------------------------|
| 0,1,2       | linear_attention | `linear_attn`      | in_proj_qkv, in_proj_z/a/b, conv1d, A_log, dt_bias |
| **3**       | full_attention   | `self_attn`        | **q_proj, k_proj, v_proj, o_proj**, q_norm, k_norm  |
| 4,5,6       | linear_attention | `linear_attn`      | (同上)                                               |
| **7**       | full_attention   | `self_attn`        | (同上)                                               |
| ...         | ...              | ...                | ...                                                 |
| **31**      | full_attention   | `self_attn`        | (同上)                                               |

- **DeltaNet (Gated Linear Attention)**: 24层 (75%), 无标准 attention matrix, 线性复杂度
- **Full Attention (Gated)**: 8层 (25%), indices=[3,7,11,15,19,23,27,31], 标准 QKV attention

### 1.3 Native ViT

| Config Key           | Value  | Description                       |
|----------------------|--------|-----------------------------------|
| depth                | 24     | ViT block 数                     |
| hidden_size          | 1024   | ViT 内部维度                     |
| out_hidden_size      | 2560   | 通过 merger 映射到 LLM hidden_size |
| patch_size           | 16     | 图像 patch 大小                  |
| spatial_merge_size   | 2      | 空间合并因子 (2x2 patches → 1 token) |
| num_heads            | 16     | 注意力头数                        |

输出: `pixel_values` (扁平化 patches) + `image_grid_thw` (每张图的 temporal/height/width grid)

---

## 2. Model Structure

### 2.1 Architecture Overview

```
SegEarthR2(Qwen3_5ForConditionalGeneration)
  │
  ├── LLM Path (语义理解)
  │   ├── Qwen3.5 Native ViT     
  │   │   └── encode_images() → per-image features [N_patches, 2560]
  │   └── Qwen3.5 Language Model  
  │       └── forward → hidden_states → logits
  │
  ├── Segmentation Path (像素级分割)
  │   ├── Swin Transformer (vision_tower_mask)
  │   │   └── get_vision_tower_feature() → {res2, res3, res4, res5}
  │   ├── Pixel Decoder (MSDeformAttn)       
  │   └── Mask2Former Predictor              
  │
  └── Bridge
      └── SEG_token_projector: Linear(2560→256)
          └── [SEG] hidden_state → Mask2Former dynamic query
```

### 2.2 Data Flow

```
Image ──┬── Qwen3.5 ViT ──→ pixel_values/image_grid_thw ──→ encode_images()
        │                                                       │
        │                                          per-image features [N, 2560]
        │                                                       │
        │                                                       ▼
        │                           concat_image_seg_cls_embeds() ← input_ids + refer_embedding
        │                                                       │
        │                                              inputs_embeds [B, L, 2560]
        │                                                       │
        │                                                       ▼
        │                                        Qwen3.5 Language Model
        │                                                       │
        │                                              hidden_states [B, L, 2560]
        │                                                  │           │
        │                                          LM Head (logits)   extract [SEG] embedding
        │                                              │               │
        │                                          CE Loss         SEG_token_projector(2560→256)
        │                                              │               │
        └── Swin ──→ {res2-5} ──→ Pixel Decoder ──────┼───→ Mask2Former ← dynamic query
                                                       │               │
                                                       │          pred_masks
                                                       │               │
                                                       ▼               ▼
                                                   llm_loss + (mask_loss + dice_loss)
```

### 2.3 Loss Function

```
total_loss = llm_loss + mask_loss
```

| Loss Component  | Source             | Status       |
|-----------------|--------------------|-------------|
| llm_loss        | CrossEntropyLoss   | [不变]       |
| mask_loss       | BCE + Dice         | [不变]       |
| attention_loss  | Attention matrix   | [移除]    |

Attention Loss 移除原因:
- DeltaNet 层 (75%) 不产生 attention matrix
- 仅 Full Attention 层 (25%) 可获取, 且原始权重仅 0.01

## 3. LoRA Compatibility

LoRA `find_linear_layers` 搜索 `q_proj` / `v_proj`:
- **Qwen3.5 DeltaNet 层** (24层): 使用 `linear_attn.in_proj_qkv` — 不匹配, 自动跳过
- **Qwen3.5 Full Attention 层** (8层): 使用 `self_attn.q_proj` / `self_attn.v_proj` — 匹配

排除列表新增 `"visual"`, 确保 Qwen3.5 ViT 不被 LoRA 适配。

LoRA 作用于: 8 个 full_attention 层的 q_proj + v_proj = 16 个 Linear。

---

## 4. Dependencies

| Package        | Required Version  | Reason                              |
|----------------|-------------------|-------------------------------------|
| transformers   | >= 4.57.0.dev0    | Qwen3_5ForConditionalGeneration     |
| torch          | >= 2.0            | DeltaNet ops                        |
| deepspeed      | >= 0.12           | ZeRO training                       |
| peft           | >= 0.6            | LoRA                                |
| detectron2     | latest            | Mask2Former postprocessing          |

