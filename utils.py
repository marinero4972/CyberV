import matplotlib.pyplot as plt
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import re

import copy
def deep_copy_complex_list(original_list):
    copied_list = []
    for item in original_list:
        if type(item)==dict:
            copied_dict = {}
            for key, value in item.items():
                if isinstance(value, str):
                    copied_dict[key] = value
                elif isinstance(value, torch.Tensor):
                    copied_dict[key] = value.clone()
                elif isinstance(value, list):
                    copied_dict[key] = deep_copy_complex_list(value)
                else:
                    copied_dict[key] = copy.deepcopy(value)
            copied_list.append(copied_dict)
        elif type(item)==list:
            copied_list.append(deep_copy_complex_list(item))
        elif isinstance(item, str):
            copied_list.append(item)
        elif isinstance(item, torch.Tensor):
            copied_list.append(item.clone())
        elif isinstance(item, int):
            copied_list.append(item)
        
    return copied_list


def detect_repetition(text):

    text_length = len(text)
    substring_counts = {}
    
    if text_length <= 30 :
        return False
    
    for length in range(15,30):
        for i in range(text_length - length + 1):
            substring = text[i:i+length]
            if substring in substring_counts:
                substring_counts[substring] += 1
            else:
                substring_counts[substring] = 1
    for substring, count in substring_counts.items():
        if count >= 5:
            return True
    
    return False


import random
from collections import Counter

## majority voting
def majority_sampling(pred_list):
    if not pred_list:
        raise ValueError("pred_list cannot be empty")
    
    count = Counter(pred_list)
    
    max_count = max(count.values())
    most_common_elements = [element for element, cnt in count.items() if cnt == max_count]
    selected_element = random.choice(most_common_elements)
    
    selected_index = pred_list.index(selected_element)
    print(pred_list)
    
    return selected_element, selected_index

### core function for Controller
def select_output(output_list, pred_list, text=None, select_method="score_forest",\
                  max_frames=64, fps=1, n=8, topk=5, theta=0.5, all_origin=False):

    if select_method == "majority_sampling":
        print("Use majority sampling!")
        element, index = majority_sampling(pred_list)
        return output_list[index]
    
    if select_method == "score_forest":

        ### Score Forest in Controller ###
        print("Use score forest!")

        score1 = []
        for i in range(len(pred_list)):
            # signal "prob" from Sensor
            if output_list[i][0]['prob'] is not None:
                score1.append(output_list[i][0]['prob'])
            else:
                score1.append(1e-4)
        # print("score1",score1)

        score2 = []
        for i in range(len(pred_list)):
            ori = score1[0]
            # print((score1[i]- ori)/ori,end=",")
            if (score1[i]- ori)/ori < -0.5:
                score2.append(0.0)
            else:
                score2.append(1.0)
        # print("score2",score2)

        score3 = []
        for i in range(len(pred_list)):
            # signal "repetition" from Sensor
            if "repetition" not in output_list[i][0].keys():
                score3.append(0.0)
            elif output_list[i][0]["repetition"]==True:
                score3.append(0.0)
            else:
                score3.append(1.0)
        # print("score3",score3)

        score4 = [1.0]
        for i in range(1, len(pred_list)):
            # calculate signal "attn drift" (Sensor)
            origin = output_list[0][0]["attn"]
            s1 = output_list[i][0]["attn"]
            if origin is None or s1 is None:
                score4.append(0.0)
                continue

            if origin.shape[1]==2 and s1.shape[1]==2:
                # video
                matrice_ori = np.mean(np.array(origin),axis=0)[:,2:34]
                matrice_s1 = np.mean(np.array(s1),axis=0)[:,2:34]
                answer_token_diff = np.sum(matrice_s1[0] - matrice_ori[0])

                # asr
                if output_list[0][0]["asr_token_num"] > 0:
                    matrice_ori = np.mean(np.array(origin),axis=0)[:,36:36+output_list[0][0]["asr_token_num"]]
                    matrice_s1 = np.mean(np.array(s1),axis=0)[:,36:36+output_list[0][0]["asr_token_num"]]
                    answer_token_diff += np.sum(matrice_s1[0] - matrice_ori[0])

            else:
                score4.append(0.0)
                continue
            # print(i, answer_token_diff, end=" ")
            if answer_token_diff < -0.1:
                score4.append(0.0)
            else:
                score4.append(1.0)
        if all_origin:
            score4 = [1.0 for i in range(len(pred_list))]
        # print("score4",score4)

        sorted_indices = sorted(range(len(score1)), key=lambda i:score1[i])
        ranks = [0]*len(score1)
        for rank, idx in enumerate(sorted_indices):
            ranks[idx] = rank + 1
        score5 = [rank/n for rank in ranks]
        # print("score5",score5)

        # all beta_i equals 0.2
        score_list = [0.2*score1[i]+0.2*score2[i]+0.2*score3[i]+0.2*score4[i]+0.2*score5[i] for i in range(len(pred_list))]

        print(score_list)

        candidate_dict = {}
        for i in range(len(pred_list)):
            choice = pred_list[i]
            if choice not in candidate_dict.keys():
                candidate_dict[choice] = 0

            candidate_dict[choice] += score_list[i]
        
        print(candidate_dict)
        print(pred_list)

        max_score = 0
        best_choice = None

        # get the best choice and the max score
        for key,value in candidate_dict.items():
            if value > max_score:
                max_score = value
                best_choice = key

        candidate_indices = [i for i, choice in enumerate(pred_list) if choice == best_choice]
        index = max(candidate_indices, key=lambda i: score_list[i])

        print(max_score, n*theta, best_choice)

        # decision making
        if max_score > n*theta:
            # return the best choice
            return output_list[index]
        else:
            # need to self-correction

            # calculate attention-based key frames
            diff_list_video = []
            diff_list_asr = []
            for i in range(n-1):
                origin = output_list[0][0]["attn"]
                s1 = output_list[i+1][0]["attn"]
                if origin is None:
                    return output_list[index]
                if s1 is None:
                    continue
                # print("Attention Shape:",origin.shape, s1.shape)
                if origin.shape[1]==2 and s1.shape[1]==2:
                    matrice_ori = np.sum(np.array(origin),axis=0)[:,2:34]
                    matrice_s1 = np.sum(np.array(s1),axis=0)[:,2:34]
                    answer_token_diff = matrice_s1[0] - matrice_ori[0]
                    diff_list_video.append(answer_token_diff)

                    if output_list[0][0]["asr_token_num"] > 0:
                        matrice_ori = np.sum(np.array(origin),axis=0)[:,36:36+output_list[0][0]["asr_token_num"]]
                        matrice_s1 = np.sum(np.array(s1),axis=0)[:,36:36+output_list[0][0]["asr_token_num"]]
                        answer_token_diff = matrice_s1[0] - matrice_ori[0]
                        diff_list_asr.append(answer_token_diff)
            # video-part
            key_frame_1 = []
            if len(diff_list_video) > 0:    
                sum_diff = np.sum(np.array(diff_list_video),axis=0)
                bottom = np.argsort(sum_diff)[:topk].tolist()
                visual_based_key_frame = [item for item in bottom if sum_diff[item]<0]
                visual_based_key_frame = sorted(visual_based_key_frame)
                if max_frames == 64:
                    for frame in visual_based_key_frame:
                        key_frame_1.append(2*frame)
                        key_frame_1.append(2*frame+1)
                elif max_frames == 32:
                    for frame in visual_based_key_frame:
                        key_frame_1.append(frame)
                else:
                    raise ValueError("Max frames should be 32 or 64!")

            # print("visual_based:",key_frame_1)

            # asr-part
            key_frame_2 = []
            if len(diff_list_asr) > 0:
                sum_diff = np.sum(np.array(diff_list_asr),axis=0)
                num = min(topk, len(sum_diff))
                bottom = np.argsort(sum_diff)[:num].tolist()
                _, _, segments = get_segments(text)
                filtered_segments = [segments[k] for k in bottom]
                # print("asr diff:", sum_diff)
                # print("Filtered segments:", filtered_segments)
                # print("topk:", topk, "fps:", fps)
                if type(fps) == list:
                    # Here, fps is a list containing timestamp for each frame.
                    key_frame_2 = [
                        f_tmp for f_tmp in range(0, max_frames)
                        for seg in filtered_segments
                        if fps[f_tmp] > seg[0] and fps[f_tmp] <= seg[1]
                    ]
                else:
                    interval = 1/fps
                    key_frame_2 = [
                        f_tmp for f_tmp in range(0, max_frames)
                        for seg in filtered_segments
                        if interval * f_tmp > seg[0] and interval * f_tmp <= seg[1]
                    ]

                if len(key_frame_2) > 2*topk:
                    key_frame_2 = key_frame_2[:2*topk]
                
            # print("asr_based:",key_frame_2)

            # get final key frames
            key_frames = list(set(key_frame_1) | set(key_frame_2))
            key_frames = sorted(key_frames)
            
            if len(key_frames) > max_frames/2: 
                key_frames = sorted(key_frame_1)
            
            print("key frames:",key_frames, len(key_frames))
            if len(key_frames) > 0:
                return ("Add key frames!",[key_frames], output_list[index])
            else: 
                return output_list[index]
            
        

def find_focus_region(image_array, text_description, model, processor, device):

    image_array = np.transpose(image_array, (1, 2, 0))  # (H, W, C)
    pil_image = Image.fromarray(image_array)

    def generate_candidate_regions(image):
        regions = []
        width, height = image.size
        for scale in [1.0, 0.5, 0.25]:
            scaled_w = int(width * scale)
            scaled_h = int(height * scale)
            for x in range(0, width - scaled_w + 1, scaled_w // 4):
                for y in range(0, height - scaled_h + 1, scaled_h // 4):
                    regions.append((x, y, x + scaled_w, y + scaled_h))
        return regions

    candidate_regions = generate_candidate_regions(pil_image)

    text_inputs = processor(text=text_description, return_tensors="pt", padding=True, truncation=True, max_length=77)
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)  # shape: (1, D)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)  # 归一化

    crop_list = [pil_image.crop(region) for region in candidate_regions]

    image_inputs = processor(images=crop_list, return_tensors="pt", padding=True)
    image_inputs = {k: v.to(device) for k, v in image_inputs.items()}

    with torch.no_grad():
        image_features = model.get_image_features(**image_inputs)  # shape: (N, D)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # 归一化

    similarity = (text_features @ image_features.T).squeeze(0).cpu().numpy()  # shape: (N,)
    best_idx = np.argmax(similarity)
    best_region = candidate_regions[best_idx]

    return best_region

def zoom_in_image(image_array, region):

    image_array = np.transpose(image_array, (1, 2, 0))
    pil_image = Image.fromarray(image_array)

    x1, y1, x2, y2 = region
    cropped = pil_image.crop(region)
    original_width, original_height = pil_image.size
    zoom_factor = min(original_width/(x2-x1), original_height/(y2-y1))
    
    zoomed = cropped.resize(pil_image.size, resample=Image.LANCZOS)
    return np.array(zoomed).transpose(2, 0, 1).astype(np.uint8)

def visulize_frame(frame):
    frame = frame.permute(1, 2, 0).numpy()
    frame = np.clip(frame, 0, 255).astype(np.uint8)
    plt.imshow(frame)
    plt.show()


def convert_timestamp_format(text):
    if "does not have audio transcripts" in text:
        return text
    segments = []
    lines = text.strip().split('\n')
    
    for line in lines:
        timestamp_text, content = line.split('] ', 1)
        start_time, end_time = timestamp_text.strip('[]').split(' -> ')
        start_time = float(start_time.replace('s', ''))
        end_time = float(end_time.replace('s', ''))

        minutes = int(start_time//60)
        seconds = int(start_time - 60*minutes)
        if seconds < 10:
            start_time_new = str(minutes)+":0"+str(seconds)
        else:
            start_time_new = str(minutes)+":"+str(seconds)

        minutes = int(end_time//60)
        seconds = int(end_time - 60*minutes)
        if seconds < 10:
            end_time_new = str(minutes)+":0"+str(seconds)
        else:
            end_time_new = str(minutes)+":"+str(seconds)

        segments.append(f"[{start_time_new} -> {end_time_new}]: "+ content)
    
    return "\n".join(segments)

def time_to_seconds(minute, second):
    return int(minute) * 60 + int(second)


def get_segments(text):
    timestamps = re.findall(r'\[(\d+):(\d+)\s->\s(\d+):(\d+)\]', text)

    if len(timestamps) == 0:
        return 0, 0, []

    segments = []
    
    for start_min, start_sec, end_min, end_sec in timestamps:
        start_time_seconds = time_to_seconds(start_min, start_sec)
        end_time_seconds = time_to_seconds(end_min, end_sec)
        segments.append((start_time_seconds, end_time_seconds))

    min_time_seconds = time_to_seconds(*timestamps[0][:2])
    max_time_seconds = time_to_seconds(*timestamps[-1][2:])
    
    return min_time_seconds, max_time_seconds, segments

def contains_timestamp(text):
    pattern = r'\b(\d{1,2}):(\d{1,2})\b' 
    timestamps = re.findall(pattern, text)   
    return len(timestamps) > 0


def extract_segments(frames: list[int], fps: float, max_gap_frame: int = 3, isolate_time: float = 10.0):
    segments = []
    current = []

    frames = sorted(frames)

    max_gap_frame = max(fps*isolate_time, max_gap_frame)

    # print(max_gap_frame)

    for i, f in enumerate(frames):
        if not current:
            current.append(f)
        elif f - current[-1] <= max_gap_frame:
            current.append(f)
        else:
            segments.append(current)
            current = [f]
    if current:
        segments.append(current)

    # print(segments)
    filtered_segments = []
    for seg in segments:
        if len(seg) > 2:
            filtered_segments.append((seg[0],seg[-1]))
    
    # print(filtered_segments)

    return filtered_segments


import torch
import random
from decord import VideoReader


def resample_frames(
    video_path: str,
    inputs,
    segs: list[tuple[int, int]],
    fps: float,
    target_total_frames: int = 64,
    bias_ratio = 0.5
):
    vr = VideoReader(video_path)
    total_frames = len(vr)
    video_fps = vr.get_avg_fps()

    dense_time_intervals = []
    for start, end in segs:
        t_start = start / fps
        t_end = (end + 1) / fps
        dense_time_intervals.append((t_start, t_end))

    frame_times = np.arange(total_frames) / video_fps

    dense_flags = np.zeros(total_frames, dtype=bool)
    for t_start, t_end in dense_time_intervals:
        dense_flags |= (frame_times >= t_start) & (frame_times < t_end)

    dense_indices = np.where(dense_flags)[0]
    sparse_indices = np.where(~dense_flags)[0]

    
    dense_total_time = sum(t_end - t_start for t_start, t_end in dense_time_intervals)
    total_video_time = total_frames / video_fps
    raw_ratio = dense_total_time / total_video_time
    ratio = min(raw_ratio * (1 + bias_ratio), 1.0)

    # print(raw_ratio, ratio)

    n_dense = int(target_total_frames * ratio)
    n_sparse = target_total_frames - n_dense

    if len(dense_indices) > n_dense:
        step = len(dense_indices) / n_dense
        dense_sampled = sorted([dense_indices[int(i * step)] for i in range(n_dense)])
    else:
        dense_sampled = sorted(dense_indices.tolist())

    if n_sparse > 0:
        if len(sparse_indices) > n_sparse:
            step = len(sparse_indices) / n_sparse
            sparse_sampled = sorted([sparse_indices[int(i * step)] for i in range(n_sparse)])
        else:
            sparse_sampled = sorted(sparse_indices.tolist())
    else:
        sparse_sampled = []

    final_indices = sorted(dense_sampled + sparse_sampled)
    final_indices = final_indices[:target_total_frames]

    video = vr.get_batch(final_indices).asnumpy()  # (T, H, W, C)
    video_tensor = torch.tensor(video).permute(0, 3, 1, 2)  # (T, C, H, W)

    resized_height, resized_width = inputs[0]['video'].shape[2], inputs[0]['video'].shape[3]

    from torchvision import io, transforms
    from torchvision.transforms import InterpolationMode
    video_new = transforms.functional.resize(
        video_tensor,
        [resized_height, resized_width],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    ).float()

    return video_new


# Functions for InternVL
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img), T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC), T.ToTensor(), T.Normalize(mean=MEAN, std=STD)])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set((i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = ((i % (target_width // image_size)) * image_size, (i // (target_width // image_size)) * image_size, ((i % (target_width // image_size)) + 1) * image_size, ((i // (target_width // image_size)) + 1) * image_size)
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image, input_size=448, max_num=6):
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([int(start_idx + (seg_size / 2) + np.round(seg_size * idx)) for idx in range(num_segments)])
    return frame_indices


def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32, output_fps=False):

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    
    if isinstance(video_path, Image.Image):
        img = video_path.convert("RGB")
        img = transform(img)
        pixel_values = [img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list = [1]
        if output_fps:
            return pixel_values, num_patches_list, None
        return pixel_values, num_patches_list
    
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert("RGB")
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    if output_fps:
        return pixel_values, num_patches_list, float(fps/max_frame*num_segments)
    return pixel_values, num_patches_list


from typing import List, Optional, Tuple, Union
import re
import dataclasses
from enum import IntEnum, auto
from typing import Dict, List, Tuple, Union

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class SeparatorStyle(IntEnum):
    """Separator styles."""

    ADD_COLON_SINGLE = auto()
    ADD_COLON_TWO = auto()
    ADD_COLON_SPACE_SINGLE = auto()
    NO_COLON_SINGLE = auto()
    NO_COLON_TWO = auto()
    ADD_NEW_LINE_SINGLE = auto()
    LLAMA2 = auto()
    CHATGLM = auto()
    CHATML = auto()
    CHATINTERN = auto()
    DOLLY = auto()
    RWKV = auto()
    PHOENIX = auto()
    ROBIN = auto()
    FALCON_CHAT = auto()
    CHATGLM3 = auto()
    INTERNVL_ZH = auto()
    MPT = auto()


@dataclasses.dataclass
class Conversation:
    """A class that manages prompt templates and keeps all conversation history."""

    # The name of this template
    name: str
    # The template of the system prompt
    system_template: str = '{system_message}'
    # The system message
    system_message: str = ''
    # The names of two roles
    roles: Tuple[str] = ('USER', 'ASSISTANT')
    # All messages. Each item is (role, message).
    messages: List[List[str]] = ()
    # The number of few shot examples
    offset: int = 0
    # The separator style and configurations
    sep_style: SeparatorStyle = SeparatorStyle.ADD_COLON_SINGLE
    sep: str = '\n'
    sep2: str = None
    # Stop criteria (the default one is EOS token)
    stop_str: Union[str, List[str]] = None
    # Stops generation if meeting any token in this list
    stop_token_ids: List[int] = None

    def get_prompt(self) -> str:
        """Get the prompt for generation."""
        system_prompt = self.system_template.format(system_message=self.system_message)
        if self.sep_style == SeparatorStyle.ADD_COLON_SINGLE:
            ret = system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ': ' + message + self.sep
                else:
                    ret += role + ':'
            return ret
        elif self.sep_style == SeparatorStyle.ADD_COLON_TWO:
            seps = [self.sep, self.sep2]
            ret = system_prompt + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ': ' + message + seps[i % 2]
                else:
                    ret += role + ':'
            return ret
        elif self.sep_style == SeparatorStyle.ADD_COLON_SPACE_SINGLE:
            ret = system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ': ' + message + self.sep
                else:
                    ret += role + ': '  # must be end with a space
            return ret
        elif self.sep_style == SeparatorStyle.ADD_NEW_LINE_SINGLE:
            ret = '' if system_prompt == '' else system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + '\n' + message + self.sep
                else:
                    ret += role + '\n'
            return ret
        elif self.sep_style == SeparatorStyle.NO_COLON_SINGLE:
            ret = system_prompt
            for role, message in self.messages:
                if message:
                    ret += role + message + self.sep
                else:
                    ret += role
            return ret
        elif self.sep_style == SeparatorStyle.NO_COLON_TWO:
            seps = [self.sep, self.sep2]
            ret = system_prompt
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + message + seps[i % 2]
                else:
                    ret += role
            return ret
        elif self.sep_style == SeparatorStyle.RWKV:
            ret = system_prompt
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += (
                        role
                        + ': '
                        + message.replace('\r\n', '\n').replace('\n\n', '\n')
                    )
                    ret += '\n\n'
                else:
                    ret += role + ':'
            return ret
        elif self.sep_style == SeparatorStyle.LLAMA2:
            seps = [self.sep, self.sep2]
            if self.system_message:
                ret = system_prompt
            else:
                ret = '[INST] '
            for i, (role, message) in enumerate(self.messages):
                tag = self.roles[i % 2]
                if message:
                    if i == 0:
                        ret += message + ' '
                    else:
                        ret += tag + ' ' + message + seps[i % 2]
                else:
                    ret += tag
            return ret
        elif self.sep_style == SeparatorStyle.CHATGLM:
            # source: https://huggingface.co/THUDM/chatglm-6b/blob/1d240ba371910e9282298d4592532d7f0f3e9f3e/modeling_chatglm.py#L1302-L1308
            # source2: https://huggingface.co/THUDM/chatglm2-6b/blob/e186c891cf64310ac66ef10a87e6635fa6c2a579/modeling_chatglm.py#L926
            round_add_n = 1 if self.name == 'chatglm2' else 0
            if system_prompt:
                ret = system_prompt + self.sep
            else:
                ret = ''

            for i, (role, message) in enumerate(self.messages):
                if i % 2 == 0:
                    ret += f'[Round {i//2 + round_add_n}]{self.sep}'

                if message:
                    ret += f'{role}：{message}{self.sep}'
                else:
                    ret += f'{role}：'
            return ret
        elif self.sep_style == SeparatorStyle.CHATML:
            ret = '' if system_prompt == '' else system_prompt + self.sep + '\n'
            for role, message in self.messages:
                if message:
                    ret += role + '\n' + message + self.sep + '\n'
                else:
                    ret += role + '\n'
            return ret
        elif self.sep_style == SeparatorStyle.CHATGLM3:
            ret = ''
            if self.system_message:
                ret += system_prompt
            for role, message in self.messages:
                if message:
                    ret += role + '\n' + ' ' + message
                else:
                    ret += role
            return ret
        elif self.sep_style == SeparatorStyle.CHATINTERN:
            # source: https://huggingface.co/internlm/internlm-chat-7b-8k/blob/bd546fa984b4b0b86958f56bf37f94aa75ab8831/modeling_internlm.py#L771
            seps = [self.sep, self.sep2]
            ret = system_prompt
            for i, (role, message) in enumerate(self.messages):
                # if i % 2 == 0:
                #     ret += "<s>"
                if message:
                    ret += role + ':' + message + seps[i % 2] + '\n'
                else:
                    ret += role + ':'
            return ret
        elif self.sep_style == SeparatorStyle.DOLLY:
            seps = [self.sep, self.sep2]
            ret = system_prompt
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ':\n' + message + seps[i % 2]
                    if i % 2 == 1:
                        ret += '\n\n'
                else:
                    ret += role + ':\n'
            return ret
        elif self.sep_style == SeparatorStyle.PHOENIX:
            ret = system_prompt
            for role, message in self.messages:
                if message:
                    ret += role + ': ' + '<s>' + message + '</s>'
                else:
                    ret += role + ': ' + '<s>'
            return ret
        elif self.sep_style == SeparatorStyle.ROBIN:
            ret = system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ':\n' + message + self.sep
                else:
                    ret += role + ':\n'
            return ret
        elif self.sep_style == SeparatorStyle.FALCON_CHAT:
            ret = ''
            if self.system_message:
                ret += system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ': ' + message + self.sep
                else:
                    ret += role + ':'

            return ret
        elif self.sep_style == SeparatorStyle.INTERNVL_ZH:
            seps = [self.sep, self.sep2]
            ret = self.system_message + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ': ' + message + seps[i % 2]
                else:
                    ret += role + ':'
            return ret
        elif self.sep_style == SeparatorStyle.MPT:
            ret = system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + message + self.sep
                else:
                    ret += role
            return ret
        else:
            raise ValueError(f'Invalid style: {self.sep_style}')

    def set_system_message(self, system_message: str):
        """Set the system message."""
        self.system_message = system_message

    def append_message(self, role: str, message: str):
        """Append a new message."""
        self.messages.append([role, message])

    def update_last_message(self, message: str):
        """Update the last output.

        The last message is typically set to be None when constructing the prompt,
        so we need to update it in-place after getting the response from a model.
        """
        self.messages[-1][1] = message

    def to_gradio_chatbot(self):
        """Convert the conversation to gradio chatbot format."""
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def to_openai_api_messages(self):
        """Convert the conversation to OpenAI chat completion format."""
        ret = [{'role': 'system', 'content': self.system_message}]

        for i, (_, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                ret.append({'role': 'user', 'content': msg})
            else:
                if msg is not None:
                    ret.append({'role': 'assistant', 'content': msg})
        return ret

    def copy(self):
        return Conversation(
            name=self.name,
            system_template=self.system_template,
            system_message=self.system_message,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            stop_str=self.stop_str,
            stop_token_ids=self.stop_token_ids,
        )

    def dict(self):
        return {
            'template_name': self.name,
            'system_message': self.system_message,
            'roles': self.roles,
            'messages': self.messages,
            'offset': self.offset,
        }
    
conv_templates: Dict[str, Conversation] = {}
def get_conv_template(name: str) -> Conversation:
    """Get a conversation template."""
    return conv_templates[name].copy()

def register_conv_template(template: Conversation, override: bool = False):
    """Register a new conversation template."""
    if not override:
        assert (
            template.name not in conv_templates
        ), f'{template.name} has been registered.'

    conv_templates[template.name] = template

register_conv_template(
    Conversation(
        name='internvl2_5',
        system_template='<|im_start|>system\n{system_message}',
        system_message='你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。',
        roles=('<|im_start|>user\n', '<|im_start|>assistant\n'),
        sep_style=SeparatorStyle.MPT,
        sep='<|im_end|>\n',
    )
)

