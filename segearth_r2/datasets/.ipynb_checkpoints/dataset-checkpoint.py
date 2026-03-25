import os
import random
import re
import glob
from dataclasses import dataclass, field
import json
import pathlib

from typing import Dict, Sequence, Optional
import torch
from torch.utils.data import Dataset
import numpy as np
import transformers
import cv2
from PIL import Image
from pycocotools import mask as M

from fvcore.common.config import CfgNode

import warnings

from segearth_r2.utils.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, REFER_TOKEN_INDEX

from segearth_r2.model.mipha import conversation as conversation_lib
from segearth_r2.model import *
from segearth_r2.model.mask_decoder.mask_config.config import Config

warnings.filterwarnings('ignore')
local_rank = None

def preprocess_mask(mask, image_size, short_edge_length=1024, max_size=1024):
    if len(mask.shape) == 2:
        mask = np.expand_dims(mask, axis=0)
    bs, _, _ = mask.shape
    processed_masks = []
    for i in range(bs):
        single_mask = mask[i]
        hh, ww = single_mask.shape[:2]

        # Keep mask geometry exactly aligned with preprocess_image:
        # ResizeShortestEdge(short_edge=1024, max_size=1024) +
        # FixedSizeCrop(1024x1024) with right/bottom padding.
        size = short_edge_length * 1.0
        scale = size / min(hh, ww)
        if hh < ww:
            new_h, new_w = size, scale * ww
        else:
            new_h, new_w = scale * hh, size
        if max(new_h, new_w) > max_size:
            scale = max_size * 1.0 / max(new_h, new_w)
            new_h = new_h * scale
            new_w = new_w * scale
        new_w = int(new_w + 0.5)
        new_h = int(new_h + 0.5)

        resized_mask = cv2.resize(single_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        if new_w < image_size:
            padding = ((0, 0), (0, image_size - new_w))
        else:
            padding = ((0, image_size - new_h), (0, 0))

        padded_mask = np.pad(
            resized_mask,
            padding,
            mode="constant",
            constant_values=0,
        )
        processed_masks.append(padded_mask)
    processed_masks = np.stack(processed_masks, axis=0)
    
    return processed_masks

def preprocess_image(image_input, pad_value = 128.0, short_edge_length = 1024, max_size = 1024):
    if isinstance(image_input, Image.Image):
        img = image_input
    else:
        img = Image.open(image_input).convert('RGB')
    
    # ResizeShortestEdge
    w, h = img.size
    size = short_edge_length * 1.0
    scale = size / min(h, w)
    if h < w:
        newh, neww = size, scale * w
    else:
        newh, neww = scale * h, size
    if max(newh, neww) > max_size:
        scale = max_size * 1.0 / max(newh, neww)
        newh = newh * scale
        neww = neww * scale
    neww = int(neww + 0.5)
    newh = int(newh + 0.5)
    img_resize_shot_edge = img.resize((neww, newh), resample=Image.BILINEAR)
    
    # FixedSizeCrop
    img_np = np.array(img_resize_shot_edge)
    if neww < 1024:
        padding = ((0, 0), (0, 1024 - neww), (0, 0))
    else:
        padding = ((0, 1024 - newh), (0, 0), (0, 0))
    img_padded = np.pad(
        img_np,
        padding,
        mode="constant",
        constant_values=pad_value
    ) # (1024, 1024, 3)

    return img_padded.astype(np.uint8)

class RS_Base_Dataset(Dataset):

    def normalize_segmentation_answer(self, answer):
        answer = (answer or '').strip()
        if not answer:
            return answer

        seg_spans = re.findall(r'<p>\s*.*?\s*</p>\s*\[SEG\]', answer, flags=re.IGNORECASE | re.DOTALL)
        if not seg_spans:
            return answer

        normalized_spans = []
        for span in seg_spans:
            normalized = re.sub(r'\s+', ' ', span).strip()
            normalized = re.sub(r'\s*</p>\s*', '</p> ', normalized)
            normalized = re.sub(r'\s*\[SEG\]\s*', ' [SEG]', normalized)
            normalized_spans.append(normalized)

        return '\n'.join(normalized_spans)

    def build_segmentation_user_prompt(self, reference_answer):
        prompt_lines = [
            '<|im_start|>system',
            'You are a remote sensing segmentation assistant.',
            '<|im_end|>',
            '<|im_start|>user',
            '<image>',
            'Description:',
            '<refer>',
        ]
        if reference_answer:
            prompt_lines.extend([
                'Given answer:',
                reference_answer,
                'Output the given answer exactly.',
            ])
        return '\n'.join(prompt_lines)
    
    def tokenizer_special_tokens(self, prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, refer_token_index=REFER_TOKEN_INDEX, return_tensors=None):
        input_ids = []
        special_token_map = {'<image>': image_token_index, '<refer>':refer_token_index}
        prompt_chunks = re.split('(<image>|<refer>)', prompt)

        for chunk in prompt_chunks:
            if chunk in special_token_map:
                input_ids.append(special_token_map[chunk])
            elif chunk != '':
                input_ids.extend(tokenizer.encode(chunk, add_special_tokens=False))
        if return_tensors is not None:
            if return_tensors == 'pt':
                return torch.tensor(input_ids, dtype=torch.long).squeeze()
            raise ValueError(f'Unsupported tensor type: {return_tensors}')
        else:
            return input_ids
        
    def preprocess_llama2(self, sources, tokenizer):
        conv = conversation_lib.default_conversation.copy()
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

        # Apply prompt templates
        conversations = []
        for i, source in enumerate(sources):
            if roles[source[0]["from"]] != conv.roles[0]:
                # Skip the first one if it is not from human
                source = source[1:]

            if conv.version == 'qwen3_5':
                conversations.append("".join(sentence["value"] for sentence in source))
            else:
                conv.messages = []
                for j, sentence in enumerate(source):
                    role = roles[sentence["from"]]
                    assert role == conv.roles[j % 2], f"{i}"
                    conv.append_message(role, sentence["value"])
                conversations.append(conv.get_prompt())

        # Tokenize conversations

        input_ids = torch.stack(
            [self.tokenizer_special_tokens(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)

        targets = input_ids.clone()

        # Mask targets
        sep = conv.sep + conv.roles[1] + ": "
        qwen_assistant_sep = "<|im_start|>assistant\n"
        idx = 0
        for conversation, target in zip(conversations, targets):
            total_len = int(target.ne(tokenizer.pad_token_id).sum())

            if conv.version == 'qwen3_5':
                if qwen_assistant_sep not in conversation:
                    target[:] = IGNORE_INDEX
                    continue

                instruction = conversation.split(qwen_assistant_sep, 1)[0] + qwen_assistant_sep
                instruction_len = len(self.tokenizer_special_tokens(instruction, tokenizer))
                cur_len = len(self.tokenizer_special_tokens(conversation, tokenizer))

                target[:instruction_len] = IGNORE_INDEX
                target[cur_len:] = IGNORE_INDEX

                if cur_len < tokenizer.model_max_length and cur_len != total_len:
                    target[:] = IGNORE_INDEX
                    print(
                        f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                        f" (ignored)"
                    )
                continue

            rounds = conversation.split(conv.sep2)
            if conv.version == 'v0':
                cur_len = 0
                end_token_cnt = 0
                # target[:cur_len] = IGNORE_INDEX
                idx = 0
                for i, rou in enumerate(rounds):
                    if rou == "":
                        continue

                    parts = rou.split(sep)
                    if len(parts) != 2:
                        break
                    parts[0] += sep
                    if idx > 0:
                        round_len = len(self.tokenizer_special_tokens(rou, tokenizer)) + 1
                    else:
                        round_len = len(self.tokenizer_special_tokens(rou, tokenizer)) + 1
                    if idx > 0:
                        instruction_len = len(self.tokenizer_special_tokens(parts[0], tokenizer))
                    else:
                        instruction_len = len(self.tokenizer_special_tokens(parts[0], tokenizer)) - 2

                    target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

                    end_token_cnt += 1
                    cur_len += round_len
                    idx += 1
                target[cur_len:] = IGNORE_INDEX
                cur_len -= end_token_cnt
            else:
                cur_len = 1
                target[:cur_len] = IGNORE_INDEX
                for i, rou in enumerate(rounds):
                    if rou == "":
                        continue

                    parts = rou.split(sep)
                    if len(parts) != 2:
                        break
                    parts[0] += sep
                    round_len = len(self.tokenizer_special_tokens(rou, tokenizer))
                    instruction_len = len(self.tokenizer_special_tokens(parts[0], tokenizer)) - 2

                    target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

                    cur_len += round_len
                    idx += 1
                target[cur_len:] = IGNORE_INDEX

            if cur_len < tokenizer.model_max_length:
                if cur_len != total_len:
                    target[:] = IGNORE_INDEX
                    print(
                        f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                        f" (ignored)"
                    )

        return dict(
            input_ids=input_ids,
            labels=targets,
        )

class LaSeRSDataset(RS_Base_Dataset):
    
    def preprocess_referring_instruction(self, instruction, append_seg_token=False, refer_token='[SEG]'):
        tokenized = self.tokenizer.encode(instruction, add_special_tokens=False)
        if append_seg_token:
            refer_token_id = [self.tokenizer.encode(refer_token, add_special_tokens=False)[0]]
            tokenized = tokenized + refer_token_id

        token_refer_id = torch.tensor(tokenized)

        return token_refer_id
    
    def __init__(self, base_data_path, tokenizer, data_args, split='train_data.json'):
        self.pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)   
        self.pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
        
        self.base_data_path = base_data_path
        self.tokenizer = tokenizer
        self.require_seg_in_each_sample = getattr(data_args, 'require_seg_in_each_sample', True)
        self.drop_invalid_seg_sample = getattr(data_args, 'drop_invalid_seg_sample', True)
        self.compact_answer_supervision = getattr(data_args, 'compact_answer_supervision', False)
        self.answer_in_user_prompt = getattr(data_args, 'answer_in_user_prompt', True)
        self.max_seg_per_sample = int(getattr(data_args, 'max_seg_per_sample', 0) or 0)
        if "train" in split:
            self.LaSeRS_image_path = os.path.join(base_data_path, "train/images")
            self.LaSeRS_json_path = os.path.join(base_data_path, "train/annotations", split)
        elif "test" in split:
            self.LaSeRS_image_path = os.path.join(base_data_path, "test/images")
            self.LaSeRS_json_path = os.path.join(base_data_path, "test/annotations", split)

        self.SEG_token_id = self.tokenizer.convert_tokens_to_ids("[SEG]")
        
        with open(self.LaSeRS_json_path, "r") as f:
            data = json.load(f)
        self.reason_file = self._validate_and_filter_samples(data)

    def _validate_and_filter_samples(self, samples):
        if not self.require_seg_in_each_sample:
            return samples

        valid_samples = []
        invalid_count = 0
        over_seg_count = 0
        for sample in samples:
            answer = sample.get('answer', '')
            if self.compact_answer_supervision:
                answer = self.normalize_segmentation_answer(answer)
            mask_num = answer.count('[SEG]')

            if self.max_seg_per_sample > 0 and mask_num > self.max_seg_per_sample:
                over_seg_count += 1
                continue

            masks = sample.get('mask', None)

            has_valid_seg = mask_num > 0
            if has_valid_seg and masks is not None:
                has_valid_seg = len(masks) >= mask_num

            if has_valid_seg:
                valid_samples.append(sample)
            else:
                invalid_count += 1

        if invalid_count == 0 and over_seg_count == 0:
            return valid_samples

        if self.drop_invalid_seg_sample:
            print(
                f'drop invalid seg samples: {invalid_count}, '
                f'drop over max_seg_per_sample({self.max_seg_per_sample}): {over_seg_count}, '
                f'keep: {len(valid_samples)}'
            )
            if len(valid_samples) == 0:
                raise ValueError('No valid samples left after SEG filtering. Please check annotations.')
            return valid_samples

        raise ValueError(
            f'Found {invalid_count} invalid samples without valid [SEG]-mask alignment; '
            f'set drop_invalid_seg_sample=True to auto-drop them.'
        )
    
    def __len__(self):
        return len(self.reason_file)
    
    def __getitem__(self, idx):
        data_info = self.reason_file[idx]
        image_path = os.path.join(self.LaSeRS_image_path, data_info['image_name'])
        image_name = data_info['image_name']
        ref = data_info['description']
        raw_answer = data_info['answer']
        answer = raw_answer
        if self.compact_answer_supervision:
            answer = self.normalize_segmentation_answer(answer)
        data_id = data_info['id']

        if "mask" in data_info:        
            rle_list = data_info['mask']
            masks = []
            for rle in rle_list:
                mask = M.decode(rle)
                masks.append(mask)
            masks = np.stack(masks, axis=0)
        else:
            masks = None
        
        data_dict = {}
        data_dict['file_name'] = image_path
        pil_image = Image.open(image_path).convert('RGB')
        image_width, image_height = pil_image.size
        data_dict['height'] = image_height
        data_dict['width'] = image_width
        data_dict['image_id'] = idx

        # process image
        # ResizeShortestEdge + FixedSizeCrop
        image_RGB = preprocess_image(pil_image)
        # Use the same transformed image for both Swin and Qwen branches.
        data_dict['image_for_qwen'] = image_RGB
        image_tensor = torch.as_tensor(np.ascontiguousarray(image_RGB.transpose(2, 0, 1)))
        data_dict['image'] = (image_tensor - self.pixel_mean) / self.pixel_std
        
        data_dict['annotations'] = []
        
        mask_num = answer.count("[SEG]")
        processed_masks = None
        if masks is not None and mask_num > 0:
            processed_masks = preprocess_mask(masks[:mask_num], image_size=1024)

        for i in range(mask_num):
            data_dict['annotations'].append({
                'data_id': data_id,
                'mask_id': i,
                'mask': np.expand_dims(processed_masks[i], axis=0) if processed_masks is not None else None,
                'image_path': image_path,
                'height': image_height,
                'width': image_width,
                'image_id': os.path.basename(image_path).split(".")[0],
            })
            
        instruction = ref.strip()
        user_answer = raw_answer if self.answer_in_user_prompt else ''
        prefix_inst = self.build_segmentation_user_prompt(user_answer)

        token_refer_id = self.preprocess_referring_instruction(instruction, append_seg_token=False)

        sources = [[{'from': 'human', 'value': prefix_inst + '<|im_end|>'},
                    {'from': 'gpt', 'value': '\n<|im_start|>assistant\n' + answer + '<|im_end|>'}]]

        text_dict = self.preprocess_llama2(sources, self.tokenizer)
        input_ids = text_dict['input_ids'][0]
        labels = text_dict['labels'][0]
        
        SEG_token_embedding_indices = torch.zeros_like(input_ids)
        seg_positions = input_ids == self.SEG_token_id
        if self.answer_in_user_prompt:
            seg_positions = seg_positions & labels.eq(IGNORE_INDEX)
        SEG_token_embedding_indices[seg_positions] = 1
        
        refer_embedding_indices = torch.zeros_like(input_ids)
        refer_embedding_indices[input_ids == REFER_TOKEN_INDEX] = 1
        
        data_dict['input_ids'] = text_dict['input_ids'][0]
        data_dict['labels'] = labels
        data_dict['dataset_type'] = 'rs_reason_seg'
        
        data_dict['token_refer_id'] = token_refer_id    
        data_dict['refer_embedding_indices'] = refer_embedding_indices
        data_dict['SEG_token_embedding_indices'] = SEG_token_embedding_indices
        data_dict['sample_record'] = {
            'data_id': str(data_id),
            'image_name': image_name,
            'image_path': image_path,
            'image_id': os.path.basename(image_path).split(".")[0],
            'mask_num': mask_num,
        }
        
        data_dict['mask_num'] = mask_num
        
        return data_dict

@dataclass
class DataCollatorForCOCODatasetV2(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    clip_image_processor: object
    image_token_id: int = 248056
    vision_start_token_id: int = 248053
    vision_end_token_id: int = 248054
    spatial_merge_size: int = 2

    def _expand_placeholder_tokens(self, instances, image_grid_thw):
        """Expand IMAGE (-200) and REFER (-204) placeholders to real token IDs.

        IMAGE  → [vision_start, image_pad × n_patches, vision_end]
        REFER  → token_refer_id (instruction tokens)

        Labels and SEG_token_embedding_indices are adjusted in-place.
        """
        merge = self.spatial_merge_size
        for i, inst in enumerate(instances):
            ids = inst['input_ids']
            lab = inst['labels']
            seg = inst['SEG_token_embedding_indices']

            # --- expand IMAGE_TOKEN_INDEX (-200) ---
            img_pos = (ids == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0]
            if len(img_pos) > 0 and image_grid_thw is not None:
                idx = img_pos[0].item()
                t, h, w = image_grid_thw[i].tolist()
                n_patches = int(t) * (int(h) // merge) * (int(w) // merge)
                new_ids = torch.tensor(
                    [self.vision_start_token_id]
                    + [self.image_token_id] * n_patches
                    + [self.vision_end_token_id],
                    dtype=ids.dtype,
                )
                new_lab = torch.full((n_patches + 2,), IGNORE_INDEX, dtype=lab.dtype)
                new_seg = torch.zeros(n_patches + 2, dtype=seg.dtype)
                ids = torch.cat([ids[:idx], new_ids, ids[idx + 1:]])
                lab = torch.cat([lab[:idx], new_lab, lab[idx + 1:]])
                seg = torch.cat([seg[:idx], new_seg, seg[idx + 1:]])

            # --- expand REFER_TOKEN_INDEX (-204) ---
            ref_pos = (ids == REFER_TOKEN_INDEX).nonzero(as_tuple=True)[0]
            if len(ref_pos) > 0 and 'token_refer_id' in inst:
                idx = ref_pos[0].item()
                refer_ids = inst['token_refer_id'].to(dtype=ids.dtype)
                rlen = len(refer_ids)
                new_lab = torch.full((rlen,), IGNORE_INDEX, dtype=lab.dtype)
                new_seg = torch.zeros(rlen, dtype=seg.dtype)
                ids = torch.cat([ids[:idx], refer_ids, ids[idx + 1:]])
                lab = torch.cat([lab[:idx], new_lab, lab[idx + 1:]])
                seg = torch.cat([seg[:idx], new_seg, seg[idx + 1:]])

            inst['input_ids'] = ids
            inst['labels'] = lab
            inst['SEG_token_embedding_indices'] = seg

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # ---- 1. Process images FIRST to know n_patches ----
        image_grid_thw = None
        pixel_values = None
        if 'image' in instances[0]:
            if isinstance(instances[0]['file_name'], list):
                all_images = []
                for instance in instances:
                    frame_images = [Image.open(p).convert('RGB') for p in instance['file_name']]
                    all_images.extend(frame_images)
                processed = self.clip_image_processor(images=all_images, return_tensors="pt")
            else:
                if 'image_for_qwen' in instances[0]:
                    pil_images = [Image.fromarray(inst['image_for_qwen']) for inst in instances]
                else:
                    pil_images = [Image.open(inst['file_name']).convert('RGB') for inst in instances]
                processed = self.clip_image_processor(images=pil_images, return_tensors="pt")
            pixel_values = processed['pixel_values']
            image_grid_thw = processed['image_grid_thw']

        # ---- 2. Expand placeholder tokens (-200, -204) → real IDs ----
        self._expand_placeholder_tokens(instances, image_grid_thw)

        # ---- 3. Pad input_ids / labels ----
        input_ids = [inst['input_ids'] for inst in instances]
        labels = [inst['labels'] for inst in instances]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'sample_record' in instances[0]:
            batch['sample_records'] = [dict(inst['sample_record']) for inst in instances]

        # ---- mm_token_type_ids for Qwen3.5 MROPE ----
        # Mark image_pad tokens as 1 (image), everything else as 0 (text)
        mm_token_type_ids = torch.zeros_like(input_ids, dtype=torch.int)
        mm_token_type_ids[input_ids == self.image_token_id] = 1
        batch['mm_token_type_ids'] = mm_token_type_ids

        # ---- 4. Attach image tensors ----
        if pixel_values is not None:
            batch['pixel_values'] = pixel_values
            batch['image_grid_thw'] = image_grid_thw

            images = [inst['image'] for inst in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        # ---- 5. Clean up per-instance keys already consumed ----
        for inst in instances:
            for key in ['input_ids', 'labels', 'image', 'image_for_qwen', 'sample_record']:
                inst.pop(key, None)

        # ---- 6. Seg info ----
        if 'instances' in instances[0]:
            batch['seg_info'] = list(instances)
        else:
            batch['seg_info'] = []
            for inst in instances:
                for seg in inst['annotations']:
                    seg['mask'] = torch.as_tensor(seg['mask'], dtype=torch.uint8) if seg['mask'] is not None else None
                    batch['seg_info'].append(seg)

        if 'dataset_type' in instances[0]:
            batch['dataset_type'] = [inst['dataset_type'] for inst in instances]

        # ---- 7. SEG_token_embedding_indices ----
        if 'SEG_token_embedding_indices' in instances[0]:
            seg_indices = [inst['SEG_token_embedding_indices'] for inst in instances]
            seg_indices = torch.nn.utils.rnn.pad_sequence(
                seg_indices, batch_first=True, padding_value=0)
            seg_indices = seg_indices[:, :self.tokenizer.model_max_length]
            batch['SEG_token_embedding_indices'] = seg_indices

        if 'mask_num' in instances[0]:
            batch['mask_num'] = [inst['mask_num'] for inst in instances]

        return batch

class UnifyDatasetSingleDatasetForBatch(Dataset):
    """
    Dataset to concatenate multiple datasets.
    Purpose: useful to assemble different existing datasets, possibly
    large-scale datasets as the concatenation operation is done in an
    on-the-fly manner.
    Arguments:
        datasets (sequence): List of datasets to be concatenated
    """

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r


    def __init__(self,datasets,dataset_ratio,bs,fix_dataset_len=0):
        super(UnifyDatasetSingleDatasetForBatch, self).__init__()
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.fix_dataset_len = fix_dataset_len

        self.cnt = 0
        self.bs = bs

        self.datasets = list(datasets)
        self.datasets_index_list = list(range(len(datasets)))
        self.dataset_ratio = dataset_ratio
        self.cur_dataset_index=0
        self.dataset_length = [len(data) for data in self.datasets]
        self.cumulative_sizes = self.cumsum(self.datasets)
        
    def update_dataset_index(self):
        tempt = self.cur_dataset_index
        tempt += 1
        tempt = tempt % len(self.datasets)
        self.cur_dataset_index = tempt

    def __len__(self):
        if self.fix_dataset_len == 0:
            return self.cumulative_sizes[-1]
        else:
            return self.fix_dataset_len


    def __getitem__(self, idx):
        cur_dataset_len = self.dataset_length[self.cur_dataset_index]
        data_idx = idx % cur_dataset_len
        output_data = self.datasets[self.cur_dataset_index][data_idx]
        self.cnt += 1
        if self.cnt == self.bs:
            self.cnt = 0
            self.update_dataset_index()
        return output_data

def get_mask_config(config='./segearth_r2/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml'):
    cfg_coco = Config.fromfile(config)
    cfg_base = CfgNode.load_yaml_with_base(config, allow_unsafe=True)
    cfg_base.update(cfg_coco.__dict__.items())
    cfg = cfg_base
    cfg = Config(cfg)
    return cfg