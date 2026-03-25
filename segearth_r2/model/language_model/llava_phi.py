from typing import List, Optional, Tuple, Union
from addict import Dict
from dataclasses import dataclass
import torch.nn.functional as F
import fvcore.nn.weight_init as weight_init
import numpy as np
import pickle
import torch
import torch.nn as nn
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.utils.memory import retry_if_cuda_oom

from transformers import Qwen3_5ForConditionalGeneration

from segearth_r2.utils.constants import IGNORE_INDEX

from ..mask_decoder.Mask2Former_Simplify.modeling.transformer_decoder.mask2former_transformer_decoder import MultiScaleMaskedTransformerDecoderForOPTPreTrain
from ..mask_decoder.Mask2Former_Simplify.modeling.pixel_decoder.msdeformattn import MSDeformAttnPixelDecoder
from ..mask_encoder.swin_trans import build_swin_b, build_swin_l

from ..mask_decoder.Mask2Former_Simplify.modeling.transformer_decoder.position_encoding import PositionEmbeddingSine

from ..datasets_mapper.IVS_mapper import IVSDatasetMapper
from segearth_r2.model.mask_decoder.mask_criterion.Mask_Criterion import Criterion, hungarian_matcher_InstructSeg


@dataclass
class CausalOutputWithMask(CausalLMOutputWithPast):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    loss_mask: Optional[torch.FloatTensor] = None
    loss_dice: Optional[torch.FloatTensor] = None
    loss_llm: Optional[torch.FloatTensor] = None
    loss_seg_class: Optional[torch.FloatTensor] = None
    loss_attention: Optional[torch.FloatTensor] = None
    sample_diagnostics: Optional[list] = None


class AttentionLoss(nn.Module):
    """Attention supervision loss adapted for Qwen3.5.

    Encourages [SEG] token attention over image patches to align with
    ground-truth mask regions.  Only applied to Full Attention layers
    (8 out of 32 in Qwen3.5-4B).
    """

    def __init__(self, reduction='batchmean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, model_attention_logits: torch.Tensor, gt_mask: torch.Tensor) -> torch.Tensor:
        """Compute attention divergence loss.

        Args:
            model_attention_logits: [N_seg, N_image_patches] attention values
            gt_mask: [N_seg, N_image_patches] binary mask (0/1)
        """
        device = model_attention_logits.device
        loss = torch.tensor(0.0, device=device)
        epsilon = 1e-8
        for idx in range(model_attention_logits.shape[0]):
            attention_map_target = model_attention_logits[idx][gt_mask[idx] == 1]
            attention_map_else = model_attention_logits[idx][gt_mask[idx] == 0]
            if attention_map_target.numel() == 0:
                continue
            mean = (
                torch.mean(attention_map_else)
                if attention_map_else.numel() > 0
                else torch.tensor(0.0, device=device)
            )
            mse = torch.mean((attention_map_target - mean) ** 2)
            loss += -torch.log(mse + epsilon)
        if self.reduction == 'batchmean':
            loss = loss / max(model_attention_logits.shape[0], 1)
        return loss


class GatedSEGProjector(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.value_branch = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.gate_branch = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.Sigmoid(),
        )
        self.residual_branch = nn.Linear(input_dim, output_dim)
        self.output_norm = nn.LayerNorm(output_dim)

    def forward(self, hidden_states):
        value = self.value_branch(hidden_states)
        gate = self.gate_branch(hidden_states)
        residual = self.residual_branch(hidden_states)
        return self.output_norm(gate * value + (1.0 - gate) * residual)


def _finite_tensor_summary(name, tensor):
    flat = tensor.detach().float().reshape(-1)
    finite_mask = torch.isfinite(flat)
    finite_count = int(finite_mask.sum().item())
    total_count = flat.numel()
    if finite_count > 0:
        finite_vals = flat[finite_mask]
        min_val = float(finite_vals.min().item())
        max_val = float(finite_vals.max().item())
    else:
        min_val = float("nan")
        max_val = float("nan")
    return (
        f"{name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}, device={tensor.device}, "
        f"finite={finite_count}/{total_count}, min={min_val}, max={max_val}"
    )


def _raise_if_non_finite(name, tensor, extra_context=None):
    if tensor is None or torch.isfinite(tensor).all():
        return
    message = f"Non-finite values detected before segmentation criterion. {_finite_tensor_summary(name, tensor)}"
    if extra_context:
        message = f"{message}. {extra_context}"
    raise ValueError(message)


class SegEarthR2(Qwen3_5ForConditionalGeneration):
    """SegEarth-R2 with Qwen3.5-4B as LLM backbone.

    Architecture:
    - LLM path: Qwen3.5 native ViT (replaces SigLIP + mm_projector)
    - Segmentation path: Swin Transformer + Mask2Former (unchanged)
    - [SEG] token hidden states → SEG_token_projector → Mask2Former query
    """

    def __init__(self, config, mask_decoder_cfg=None, **kwargs):
        # Pop extra kwargs before super().__init__
        kwargs.pop('add_cross_attn', None)
        kwargs.pop('cross_attn_index', None)
        kwargs.pop('model_args', None)
        self.llm_loss_weight = kwargs.pop('llm_loss_weight', 1.0)
        self.mask_loss_weight = kwargs.pop('mask_loss_weight', 1.0)
        super(SegEarthR2, self).__init__(config)
        # After super().__init__:
        #   self.model.visual        — Qwen3.5 ViT (replaces SigLIP + mm_projector)
        #   self.model.language_model — Qwen3.5 text model (replaces PhiModel)
        #   self.lm_head             — tied to language_model.embed_tokens

        self.mask_decoder_cfg = mask_decoder_cfg
        self.init_config = config

        # Resolve hidden_size from Qwen3.5's nested config
        self._hidden_size = config.text_config.hidden_size  # 2560

        # Build Swin for segmentation path
        if mask_decoder_cfg is not None:
            swin_type = getattr(config, 'swin_type', 'base')
            if swin_type == 'base':
                self.vision_tower_mask = build_swin_b(None)
            else:
                self.vision_tower_mask = build_swin_l(None)
            self.vision_tower_mask.image_processor = IVSDatasetMapper(mask_decoder_cfg)

        self.is_train_mask_decode = getattr(config, 'mask_decode_train', False)
        if self.is_train_mask_decode:
            print('Mask Decoder has been trained, init directly')
            self.initial_mask_module()

        self.post_init()

    # ------------------------------------------------------------------ #
    #  Accessor helpers
    # ------------------------------------------------------------------ #

    def get_model(self):
        """Returns language_model sub-module (has embed_tokens)."""
        return self.model.language_model

    def get_vision_tower(self):
        """Returns Qwen3.5's native visual encoder."""
        return self.model.visual

    def get_vision_tower_mask(self):
        """Returns Swin Transformer for segmentation."""
        vt = getattr(self, 'vision_tower_mask', None)
        if isinstance(vt, list):
            vt = vt[0]
        return vt

    # ------------------------------------------------------------------ #
    #  Swin-only vision module init (called during training)
    # ------------------------------------------------------------------ #

    def initialize_vision_modules(self, model_args, fsdp=None):
        """Initialize Swin Transformer for segmentation path.
        Qwen3.5's visual encoder is already loaded via from_pretrained.
        """
        vision_tower_mask_path = model_args.vision_tower_mask
        swin_type = getattr(model_args, 'swin_type', 'base')
        self.config.swin_type = swin_type

        if swin_type == 'base':
            vision_tower_mask = build_swin_b(vision_tower_mask_path)
        else:
            print('current visual encoder is swin large')
            vision_tower_mask = build_swin_l(vision_tower_mask_path)

        if fsdp is not None and len(fsdp) > 0:
            self.vision_tower_mask = [vision_tower_mask]
        else:
            self.vision_tower_mask = vision_tower_mask

        vision_tower_mask.hidden_size = 256
        vision_tower_mask.image_processor = IVSDatasetMapper(self.mask_decoder_cfg)

    # ------------------------------------------------------------------ #
    #  Mask decoder initialization (unchanged logic)
    # ------------------------------------------------------------------ #

    def initial_mask_module(self, pretrained_path=None, model_args=None):
        if not self.is_train_mask_decode:
            print('Initialize mask modules...')
            self.config.mask_decode_train = True

        self.test_topk_per_image = self.mask_decoder_cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        input_shape = self.output_shape()
        self.pixel_decoder = self.pixel_decoder_init(cfg=self.mask_decoder_cfg, input_shape=input_shape)
        self.predictor = self.predictor_init(cfg=self.mask_decoder_cfg)

        hidden_dim = self.mask_decoder_cfg.MODEL.MASK_FORMER.HIDDEN_DIM  # 256
        self.seg_fusion_layer_indices = [15, 23, 27, 31]  # 0-indexed layer indices
        self.seg_scale_projectors = nn.ModuleList(
            [
                GatedSEGProjector(self._hidden_size, 1024, hidden_dim)
                for _ in self.seg_fusion_layer_indices
            ]
        )
        self.SEG_token_projector = GatedSEGProjector(hidden_dim, 512, hidden_dim)

        # Multi-layer SEG fusion: extract [SEG] from Full Attention layers
        # Qwen3.5 Full Attention layers: [3,7,11,15,19,23,27,31]
        # Use a broader four-layer pyramid to reduce reliance on the deepest states.
        num_fusion_layers = len(self.seg_fusion_layer_indices)
        # Learnable weights for weighted average (initialized equal)
        self.seg_layer_weights = nn.Parameter(
            torch.ones(num_fusion_layers) / num_fusion_layers
        )

        self.attention_loss = AttentionLoss()

        self.mask_decoder_training_init(self.mask_decoder_cfg)

        if pretrained_path is not None:
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
            def change_w(weights, old_name, new_name):
                weights[new_name] = weights[old_name]
                weights.pop(old_name)

            if pretrained_path.endswith('.pkl'):
                with open(pretrained_path, 'rb') as f:
                    ckpt = pickle.load(f)
            else:
                ckpt = torch.load(pretrained_path)
            pixel_decoder_weights = get_w(ckpt['model'], 'sem_seg_head.pixel_decoder')
            predictor_weights = get_w(ckpt['model'], 'sem_seg_head.predictor')
            pixel_decoder_weights = {k: torch.tensor(v) for k, v in pixel_decoder_weights.items()}
            predictor_weights = {k: torch.tensor(v) for k, v in predictor_weights.items()}

            change_w(pixel_decoder_weights, 'adapter_1.weight', 'adapter_1.0.weight')
            change_w(pixel_decoder_weights, 'adapter_1.norm.weight', 'adapter_1.1.weight')
            change_w(pixel_decoder_weights, 'adapter_1.norm.bias', 'adapter_1.1.bias')
            change_w(pixel_decoder_weights, 'layer_1.weight', 'layer_1.0.weight')
            change_w(pixel_decoder_weights, 'layer_1.norm.weight', 'layer_1.1.weight')
            change_w(pixel_decoder_weights, 'layer_1.norm.bias', 'layer_1.1.bias')
            if 'static_query.weight' in predictor_weights:
                change_w(predictor_weights, 'static_query.weight', 'query_feat.weight')
            if predictor_weights['query_embed.weight'].shape[0] == 200:
                predictor_weights['query_embed.weight'] = predictor_weights['query_embed.weight'][:100, :]
            diff_pixel_msg = self.pixel_decoder.load_state_dict(pixel_decoder_weights, strict=False)
            diff_predictor_msg = self.predictor.load_state_dict(predictor_weights, strict=False)
            print(diff_predictor_msg)
            print(diff_pixel_msg)

    # ------------------------------------------------------------------ #
    #  Swin feature extraction (unchanged)
    # ------------------------------------------------------------------ #

    def get_vision_tower_feature(self, images):
        features = self.get_vision_tower_mask()(images)
        features_dict = {
            'res2': features[0],
            'res3': features[1],
            'res4': features[2],
            'res5': features[3],
        }
        return features_dict

    # ------------------------------------------------------------------ #
    #  Mask decoder helpers (unchanged)
    # ------------------------------------------------------------------ #

    def mask_decoder_training_init(self, cfg):
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT

        matcher = hungarian_matcher_InstructSeg(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )

        weight_dict = {
            "loss_SEG_class": class_weight,
            "loss_mask": mask_weight,
            "loss_dice": dice_weight,
        }

        self.weight_dict = weight_dict
        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["SEG_labels", "masks"]
        adaptive_fg_point_sampling = bool(getattr(cfg.MODEL.MASK_FORMER, "ADAPTIVE_FG_POINT_SAMPLING", False))
        fg_ratio_min = float(getattr(cfg.MODEL.MASK_FORMER, "FG_RATIO_MIN", 0.05))
        fg_ratio_max = float(getattr(cfg.MODEL.MASK_FORMER, "FG_RATIO_MAX", 0.20))
        min_fg_points = int(getattr(cfg.MODEL.MASK_FORMER, "MIN_FG_POINTS", 64))
        fg_area_low = float(getattr(cfg.MODEL.MASK_FORMER, "FG_AREA_LOW", 0.002))
        fg_area_high = float(getattr(cfg.MODEL.MASK_FORMER, "FG_AREA_HIGH", 0.02))
        self.criterion = Criterion(
            matcher=matcher,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
            device=self.device,
            adaptive_fg_point_sampling=adaptive_fg_point_sampling,
            fg_ratio_min=fg_ratio_min,
            fg_ratio_max=fg_ratio_max,
            min_fg_points=min_fg_points,
            fg_area_low=fg_area_low,
            fg_area_high=fg_area_high,
        )
        self.size_divisibility = 32
        self.sem_seg_postprocess_before_inference = True

    def predictor_init(self, cfg):
        in_channels = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        hidden_dim = cfg.MODEL.MASK_FORMER.HIDDEN_DIM
        num_queries = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        nheads = cfg.MODEL.MASK_FORMER.NHEADS
        dim_feedforward = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD
        dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS - 1
        pre_norm = cfg.MODEL.MASK_FORMER.PRE_NORM
        mask_dim = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        enforce_input_project = False
        seg_norm = cfg.MODEL.MASK_FORMER.SEG_NORM
        seg_proj = cfg.MODEL.MASK_FORMER.SEG_PROJ
        seg_fuse_score = cfg.MODEL.MASK_FORMER.FUSE_SCORE
        return MultiScaleMaskedTransformerDecoderForOPTPreTrain(
            in_channels, hidden_dim, num_queries, nheads,
            dim_feedforward, dec_layers, pre_norm, mask_dim,
            enforce_input_project, seg_norm, seg_proj, seg_fuse_score,
        )

    def output_shape(self):
        out_features = self.mask_decoder_cfg.MODEL.SWIN.OUT_FEATURES
        out_feature_strides = {"res2": 4, "res3": 8, "res4": 16, "res5": 32}
        num_features = [
            int(self.mask_decoder_cfg.MODEL.SWIN.EMBED_DIM * 2 ** i)
            for i in range(len(self.mask_decoder_cfg.MODEL.SWIN.DEPTHS))
        ]
        out_feature_channels = {
            "res2": num_features[0], "res3": num_features[1],
            "res4": num_features[2], "res5": num_features[3],
        }
        backbone_feature_shape = dict()
        for name in out_features:
            backbone_feature_shape[name] = Dict(
                {'channel': out_feature_channels[name], 'stride': out_feature_strides[name]}
            )
        return backbone_feature_shape

    def pixel_decoder_init(self, cfg, input_shape):
        common_stride = cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE
        transformer_dropout = cfg.MODEL.MASK_FORMER.DROPOUT
        transformer_nheads = cfg.MODEL.MASK_FORMER.NHEADS
        transformer_dim_feedforward = 1024
        transformer_enc_layers = cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS
        conv_dim = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        mask_dim = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        transformer_in_features = cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES
        return MSDeformAttnPixelDecoder(
            input_shape, transformer_dropout, transformer_nheads,
            transformer_dim_feedforward, transformer_enc_layers,
            conv_dim, mask_dim, transformer_in_features, common_stride,
        )

    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.shape[-2:]
        new_targets = []
        has_gt_ids = hasattr(targets[0], 'gt_ids') if hasattr(targets[0], 'gt_ids') else False
        for targets_per_image in targets:
            if has_gt_ids:
                inst_ids = targets_per_image.gt_ids
                valid_id = inst_ids != -1
            else:
                inst_ids = None
                valid_id = None
            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, :gt_masks.shape[1], :gt_masks.shape[2]] = gt_masks
            new_targets.append({
                "labels": targets_per_image.gt_classes,
                "masks": padded_masks,
                "valid": valid_id,
                "inst_id": inst_ids,
            })
        return new_targets

    def _merge_sample_diagnostics(self, sample_records, diagnostics):
        merged = []
        sample_records = sample_records or []
        for batch_idx, metric in enumerate(diagnostics or []):
            record = {}
            if batch_idx < len(sample_records) and isinstance(sample_records[batch_idx], dict):
                record.update(sample_records[batch_idx])
            record['batch_idx'] = batch_idx
            record.update(metric)
            record['loss_dice_weighted'] = metric['loss_dice_raw'] * float(self.weight_dict.get('loss_dice', 1.0))
            record['loss_mask_weighted'] = metric['loss_mask_raw'] * float(self.weight_dict.get('loss_mask', 1.0))
            merged.append(record)
        return merged

    def get_special_token(self, SEG, EOS):
        self.SEG_id = SEG
        self.EOS_id = EOS

    # ------------------------------------------------------------------ #
    #  Native Qwen3.5 image encoding + MROPE helpers
    # ------------------------------------------------------------------ #

    def _encode_and_scatter_images(self, input_ids, pixel_values, image_grid_thw):
        """Encode images with Qwen3.5 ViT, then scatter into inputs_embeds.

        input_ids already contains native Qwen tokens
        (vision_start, image_pad × N, vision_end) thanks to DataCollator.

        Returns:
            inputs_embeds: [B, L, H] with image features in-place
        """
        inputs_embeds = self.get_model().embed_tokens(input_ids)
        _raise_if_non_finite('inputs_embeds.before_image_scatter', inputs_embeds)

        if pixel_values is not None:
            image_outputs = self.model.get_image_features(
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                return_dict=True,
            )
            image_embeds = image_outputs.pooler_output
            image_embeds = torch.cat(image_embeds, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            _raise_if_non_finite('image_embeds', image_embeds)
            try:
                image_mask, _ = self.model.get_placeholder_mask(
                    input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
                )
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
            except Exception as exc:
                # Rarely, multimodal token expansion can be truncated by model_max_length.
                # Keep text-only embeddings for this step to avoid crashing long training runs.
                if not getattr(self, '_warned_image_scatter_fallback', False):
                    print(
                        f"[WARN] image scatter fallback activated due to placeholder mismatch: {exc}"
                    )
                    self._warned_image_scatter_fallback = True
            _raise_if_non_finite('inputs_embeds.after_image_scatter', inputs_embeds)

        return inputs_embeds

    def _compute_mrope_positions(self, input_ids, image_grid_thw, attention_mask, mm_token_type_ids=None):
        """Compute 3D MROPE position_ids via Qwen3.5's native get_rope_index.

        mm_token_type_ids marks each token's modality: text=0, image=1, video=2.
        This is required by Qwen3.5 to assign correct (temporal, height, width)
        coordinates to image tokens.
        """
        if mm_token_type_ids is None:
            # Fallback: build mm_token_type_ids from input_ids
            image_token_id = self.config.image_token_id  # 248056
            mm_token_type_ids = torch.zeros_like(input_ids, dtype=torch.int)
            mm_token_type_ids[input_ids == image_token_id] = 1

        if attention_mask is None:
            attention_mask = input_ids.ne(self.config.pad_token_id)

        try:
            position_ids, _ = self.model.get_rope_index(
                input_ids=input_ids,
                mm_token_type_ids=mm_token_type_ids,
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask,
            )
            return position_ids
        except RuntimeError as exc:
            # Fallback for rare multimodal truncation mismatches.
            # Build standard 1D positions from attention_mask and broadcast to 3 axes.
            if not getattr(self, '_warned_mrope_fallback', False):
                print(f"[WARN] mRoPE fallback activated due to shape mismatch: {exc}")
                self._warned_mrope_fallback = True

            attn = attention_mask.to(dtype=torch.long)
            pos_1d = torch.cumsum(attn, dim=-1) - 1
            pos_1d = torch.clamp(pos_1d, min=0)
            pos_1d = pos_1d.masked_fill(attn == 0, 0)
            return pos_1d.unsqueeze(0).expand(3, -1, -1)

    # ------------------------------------------------------------------ #
    #  SEG embedding extraction (unchanged)
    # ------------------------------------------------------------------ #

    def get_SEG_embedding(self, hidden_states, SEG_embedding_indices):
        SEG_embedding_list = []
        for current_hidden_state, current_token_indice in zip(hidden_states, SEG_embedding_indices):
            current_refer_state = current_hidden_state[current_token_indice.bool()]
            SEG_embedding_list.append(current_refer_state)
        return torch.cat(SEG_embedding_list, dim=0).unsqueeze(1)

    def _fuse_seg_embeddings(self, all_hidden_states, SEG_embedding_indices):
        """Extract [SEG] embeddings from multiple layers and fuse via learned weights.

        Args:
            all_hidden_states: tuple of [B, L, H] from each layer (layer 0 = embedding output)
            SEG_embedding_indices: [B, L] binary mask for [SEG] positions

        Returns:
            Fused [SEG] embedding [N_seg, 1, H]
        """
        # all_hidden_states has (num_layers + 1) entries: embedding output + each decoder layer
        # Layer index i corresponds to all_hidden_states[i + 1]
        weights = torch.softmax(self.seg_layer_weights, dim=0)
        _raise_if_non_finite('seg_layer_weights_softmax', weights)
        fused = None
        for scale_idx, (w, layer_idx) in enumerate(zip(weights, self.seg_fusion_layer_indices)):
            hs = all_hidden_states[layer_idx + 1]  # +1 because index 0 is embedding output
            _raise_if_non_finite(f'all_hidden_states[{layer_idx + 1}]', hs)
            seg_emb = self.get_SEG_embedding(hs, SEG_embedding_indices)  # [N_seg, 1, H]
            _raise_if_non_finite(f'seg_emb_layer_{layer_idx}', seg_emb)
            scale_projector = self.seg_scale_projectors[scale_idx]
            scale_projector_dtype = next(scale_projector.parameters()).dtype
            seg_emb = seg_emb.to(scale_projector_dtype)
            seg_emb = scale_projector(seg_emb)
            _raise_if_non_finite(f'seg_scale_projected_layer_{layer_idx}', seg_emb)
            if fused is None:
                fused = w * seg_emb
            else:
                fused = fused + w * seg_emb
        return fused

    # ------------------------------------------------------------------ #
    #  Training forward
    # ------------------------------------------------------------------ #

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        mm_token_type_ids: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        seg_info=None,
        token_refer_id=None,
        SEG_token_embedding_indices=None,
        global_step=None,
        mask_num=None,
        dataset_type=None,
        sample_records=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if dataset_type is not None:
            assert all(item == dataset_type[0] for item in dataset_type), \
                f'this batch contain different dataset_type: {dataset_type}'

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        has_seg = SEG_token_embedding_indices is not None and (SEG_token_embedding_indices == 1).sum() != 0
        swin_features = None
        if has_seg and input_ids is not None and input_ids.shape[1] != 1:
            swin_features = self.get_vision_tower_feature(images)

        # --- Native Qwen3.5 multimodal processing ---
        # 1) MROPE 3D position_ids (temporal, height, width for image tokens)
        position_ids = self._compute_mrope_positions(
            input_ids, image_grid_thw, attention_mask, mm_token_type_ids
        )

        # 2) Embed tokens + scatter ViT features over <|image_pad|> positions
        inputs_embeds = self._encode_and_scatter_images(
            input_ids, pixel_values, image_grid_thw
        )

        # 3) Forward through Qwen3.5 language model with proper positions
        #    Only materialize attention maps when attention supervision is enabled.
        should_output_attentions = output_attentions
        if should_output_attentions is None:
            should_output_attentions = bool(getattr(self.config, 'use_attention_loss', False))

        outputs = self.model.language_model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=should_output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)

        # Extract [SEG] embeddings from multiple Full Attention layers → fuse → project
        fused_seg = self._fuse_seg_embeddings(
            outputs.hidden_states, SEG_token_embedding_indices
        )
        _raise_if_non_finite('fused_seg', fused_seg)
        projector_dtype = next(self.SEG_token_projector.parameters()).dtype
        fused_seg = fused_seg.to(projector_dtype)
        SEG_embedding = self.SEG_token_projector(fused_seg)
        _raise_if_non_finite('SEG_embedding', SEG_embedding)

        if mask_num is not None:
            expected_seg_num = int(sum(mask_num))
            if SEG_embedding.shape[0] != expected_seg_num:
                raise ValueError(
                    f"Mismatch between SEG token count ({SEG_embedding.shape[0]}) "
                    f"and mask_num sum ({expected_seg_num})."
                )

        # Pixel decoder + Mask2Former
        mask_features, transformer_encoder_features, multi_scale_features = \
            self.pixel_decoder.forward_features(swin_features)
        mask_num_tensor = torch.tensor(mask_num, device=mask_features.device)
        mask_features = torch.repeat_interleave(mask_features, repeats=mask_num_tensor, dim=0)
        multi_scale_features = [
            torch.repeat_interleave(feat, repeats=mask_num_tensor, dim=0)
            for feat in multi_scale_features
        ]
        mask_outputs = self.predictor(multi_scale_features, mask_features, None, None, SEG_embedding)
        pred_masks = mask_outputs.get('pred_masks')
        aux_outputs = mask_outputs.get('aux_outputs', [])
        debug_context = (
            f"mask_num={mask_num}; "
            f"{_finite_tensor_summary('SEG_embedding', SEG_embedding)}; "
            f"{_finite_tensor_summary('mask_features', mask_features)}"
        )
        _raise_if_non_finite('mask_outputs.pred_masks', pred_masks, extra_context=debug_context)
        _raise_if_non_finite('mask_outputs.pred_SEG_logits', mask_outputs.get('pred_SEG_logits'), extra_context=debug_context)
        for aux_idx, aux_output in enumerate(aux_outputs):
            _raise_if_non_finite(
                f'aux_outputs[{aux_idx}].pred_masks',
                aux_output.get('pred_masks'),
                extra_context=debug_context,
            )
            _raise_if_non_finite(
                f'aux_outputs[{aux_idx}].pred_SEG_logits',
                aux_output.get('pred_SEG_logits'),
                extra_context=debug_context,
            )

        # Compute losses
        loss = None
        llm_loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :]
            shift_labels = labels[..., 1:].to(shift_logits.device)
            vocab_size = shift_logits.shape[-1]

            total_loss = torch.zeros((), device=shift_logits.device, dtype=shift_logits.dtype)
            total_tokens = torch.zeros((), device=shift_logits.device, dtype=shift_logits.dtype)
            chunk_size = 256
            seq_len = shift_labels.shape[1]

            for start in range(0, seq_len, chunk_size):
                end = min(start + chunk_size, seq_len)
                logits_chunk = shift_logits[:, start:end, :].reshape(-1, vocab_size)
                labels_chunk = shift_labels[:, start:end].reshape(-1)
                valid_tokens = (labels_chunk != IGNORE_INDEX).sum()

                if valid_tokens.item() == 0:
                    continue

                chunk_loss = F.cross_entropy(
                    logits_chunk,
                    labels_chunk,
                    ignore_index=IGNORE_INDEX,
                    reduction='sum',
                )
                total_loss = total_loss + chunk_loss
                total_tokens = total_tokens + valid_tokens.to(total_loss.dtype)

            if total_tokens.item() == 0:
                llm_loss = total_loss
            else:
                llm_loss = total_loss / total_tokens

        mask_loss = None
        loss_mask = None
        loss_dice = None
        loss_seg_class = None
        sample_diagnostics = None
        if seg_info is not None:
            if 'padding_mask' in seg_info[0]:
                if isinstance(seg_info[0]["instances"], list):
                    gt_instances = [x["instances"][0].to(self.device) for x in seg_info]
                else:
                    gt_instances = [x["instances"].to(self.device) for x in seg_info]
                targets = self.prepare_targets(gt_instances, images)
            elif 'mask' in seg_info[0]:
                targets = []
                for gt_mask in seg_info:
                    targets.append({
                        'labels': torch.tensor([0]).to(mask_outputs['pred_masks'].device),
                        'masks': gt_mask['mask'].to(mask_outputs['pred_masks'].device),
                        'valid': None,
                        'inst_id': None,
                    })
            else:
                targets = None

            mask_losses = self.criterion(mask_outputs, targets)
            with torch.no_grad():
                sample_diagnostics = self._merge_sample_diagnostics(
                    sample_records,
                    self.criterion.get_sample_loss_diagnostics(mask_outputs, targets),
                )
            weight_dict = self.weight_dict

            loss_mask = 0.0
            loss_dice = 0.0
            loss_seg_class = 0.0
            for k in list(mask_losses.keys()):
                if k in weight_dict:
                    if mask_losses[k] is None:
                        continue
                    mask_losses[k] *= weight_dict[k]
                    if '_mask' in k:
                        loss_mask += mask_losses[k]
                    elif '_dice' in k:
                        loss_dice += mask_losses[k]
                    elif 'SEG_class' in k:
                        loss_seg_class += mask_losses[k]
                else:
                    mask_losses.pop(k)
            mask_loss = loss_mask + loss_dice + loss_seg_class

        # --- Attention Loss (from Full Attention layers only) ---
        loss_attention = None
        if has_seg and seg_info is not None and outputs.attentions is not None and mm_token_type_ids is not None:
            bs = inputs_embeds.shape[0]
            merge = self.model.visual.spatial_merge_size  # typically 2

            # Per-sample: downsample GT masks to match each image's patch grid
            # seg_info has one entry per SEG token; group masks by batch index
            # Build per-batch-item mask list using mask_num
            seg_idx = 0
            per_batch_masks_down = []
            for batch_idx in range(bs):
                n_masks = mask_num[batch_idx] if mask_num is not None else 1
                # Get this image's grid size
                t, h, w = image_grid_thw[batch_idx].tolist()
                down_h = int(h) // merge
                down_w = int(w) // merge
                sample_masks = []
                for _ in range(n_masks):
                    m = seg_info[seg_idx]['mask']
                    m_float = m.unsqueeze(0).float() if m.dim() == 2 else m.float()
                    if m_float.dim() == 3:
                        m_float = m_float.unsqueeze(0)
                    m_down = F.interpolate(m_float, size=(down_h, down_w), mode="bilinear", align_corners=False)
                    m_down = m_down.view(m_down.size(0), -1)  # [1, n_patches]
                    m_down[m_down > 0] = 1
                    sample_masks.append(m_down.squeeze(0))
                    seg_idx += 1
                if sample_masks:
                    per_batch_masks_down.append(torch.stack(sample_masks, dim=0).to(self.device))
                else:
                    per_batch_masks_down.append(None)

            # Compute attention loss across Full Attention layers, per-sample
            loss_attention = torch.tensor(0.0, device=self.device)
            num_valid_layers = 0
            for attn_weights in outputs.attentions:
                # attn_weights: (bs, num_heads, seq_len, seq_len)
                attn_summed = attn_weights.sum(dim=1)  # (bs, seq_len, seq_len)
                layer_loss = torch.tensor(0.0, device=self.device)
                num_valid_samples = 0
                for batch_idx in range(bs):
                    if per_batch_masks_down[batch_idx] is None:
                        continue
                    attn_map = attn_summed[batch_idx]  # (seq_len, seq_len)
                    seg_mask = SEG_token_embedding_indices[batch_idx].bool()
                    img_mask = (mm_token_type_ids[batch_idx] == 1)
                    seg_to_img = attn_map[seg_mask][:, img_mask]  # [n_seg, n_patches]
                    if seg_to_img.numel() > 0:
                        sample_loss = self.attention_loss(seg_to_img, per_batch_masks_down[batch_idx])
                        layer_loss = layer_loss + sample_loss
                        num_valid_samples += 1
                if num_valid_samples > 0:
                    loss_attention = loss_attention + layer_loss / num_valid_samples
                    num_valid_layers += 1

            if num_valid_layers > 0:
                loss_attention = loss_attention / num_valid_layers

        loss = 0.0
        if llm_loss is not None:
            loss = loss + self.llm_loss_weight * llm_loss
        if mask_loss is not None:
            loss = loss + self.mask_loss_weight * mask_loss
        if loss_attention is not None:
            loss = loss + 0.01 * loss_attention

        return CausalOutputWithMask(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=None,
            loss_mask=loss_mask.detach() if torch.is_tensor(loss_mask) else None,
            loss_dice=loss_dice.detach() if torch.is_tensor(loss_dice) else None,
            loss_llm=llm_loss.detach() if llm_loss is not None else None,
            loss_seg_class=loss_seg_class.detach() if torch.is_tensor(loss_seg_class) else None,
            loss_attention=(0.01 * loss_attention).detach() if loss_attention is not None else None,
            sample_diagnostics=sample_diagnostics,
        )

    # ------------------------------------------------------------------ #
    #  Evaluation forward
    # ------------------------------------------------------------------ #

    def eval_seg(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        mm_token_type_ids: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        seg_info=None,
        token_refer_id=None,
        SEG_token_embedding_indices=None,
        mask_num=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        swin_features = self.get_vision_tower_feature(images)

        # --- Native Qwen3.5 multimodal processing ---
        position_ids = self._compute_mrope_positions(
            input_ids, image_grid_thw, attention_mask, mm_token_type_ids
        )
        inputs_embeds = self._encode_and_scatter_images(
            input_ids, pixel_values, image_grid_thw
        )

        outputs = self.model.language_model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=return_dict,
        )

        hidden_states = outputs.last_hidden_state
        fused_seg = self._fuse_seg_embeddings(
            outputs.hidden_states, SEG_token_embedding_indices
        )
        projector_dtype = next(self.SEG_token_projector.parameters()).dtype
        fused_seg = fused_seg.to(projector_dtype)
        SEG_embedding = self.SEG_token_projector(fused_seg)

        mask_features, transformer_encoder_features, multi_scale_features = \
            self.pixel_decoder.forward_features(swin_features)

        images_list = [image.repeat((num, 1, 1, 1)) for image, num in zip(images, mask_num)]
        images_list = [s[0] for image_repeat in images_list for s in torch.split(image_repeat, 1, dim=0)]
        mask_num_tensor = torch.tensor(mask_num, device=mask_features.device)
        mask_features = torch.repeat_interleave(mask_features, repeats=mask_num_tensor, dim=0)
        multi_scale_features = [
            torch.repeat_interleave(feat, repeats=mask_num_tensor, dim=0)
            for feat in multi_scale_features
        ]

        mask_outputs = self.predictor(multi_scale_features, mask_features, None, None, SEG_embedding)

        mask_pred_results = mask_outputs["pred_masks"]
        images_imagelist = ImageList.from_tensors(images_list, self.size_divisibility)
        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=(images_imagelist.tensor.shape[-2], images_imagelist.tensor.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )

        processed_results = []
        for _seg_info, mask_pred_result in zip(seg_info, mask_pred_results):
            instance_r = {
                'pred': ((mask_pred_result.cpu().numpy() > 0) * 255).astype(np.uint8),
                'image_name': _seg_info['image_id'],
                'id': _seg_info['data_id'],
                'mask_id': _seg_info['mask_id'],
            }
            processed_results.append(instance_r)
        return processed_results
