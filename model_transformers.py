# For Qwen-2.5-VL models
import os
os.environ["DECORD_EOF_RETRY_MAX"] = "40960"
from flash_attn import flash_attn_func, flash_attn_varlen_func
import inspect
_flash_supports_window_size = "window_size" in list(inspect.signature(flash_attn_func).parameters)
import torch
from transformers import AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLFlashAttention2,apply_multimodal_rotary_pos_emb,repeat_kv
from qwen_vl_utils import process_vision_info
from typing import Tuple
from transformers.modeling_flash_attention_utils import fa_peft_integration_check, _upad_input, pad_input
from transformers.cache_utils import Cache
from transformers.utils import logging, is_flash_attn_greater_or_equal
from typing import Optional
flash_241 = is_flash_attn_greater_or_equal("2.4.1")
deterministic_g = os.environ.get("FLASH_ATTENTION_DETERMINISTIC", "0") == "1"
logger = logging.get_logger(__name__)
import numpy as np
from utils import convert_timestamp_format, detect_repetition, find_focus_region, zoom_in_image
import re


class QwenVL_Transformers:
    def __init__(self, llm_name, process_out_list, max_thinking_tokens=4096, max_frames=64,\
                num_reflection=0, spatial_zoom_in=False, temporal_zoom_in=False, **model_kwargs):
        """
        Initialize the QwenVL_Transformers model for video-language tasks.
        
        Args:
            llm_name (str): Name/path of the pretrained Qwen-2.5-VL model to load
            process_out_list (callable): Function to parse model outputs and get predictions
            max_thinking_tokens (int): Maximum number of tokens for thinking process
            max_frames (int): Maximum number of video frames to process (Here we use 64 frames)
            num_reflection (int): Number of reflection steps to perform after initial thinking (Add “Wait” before each reflection)
            spatial_zoom_in (bool): Whether to enable spatial zoom-in for key frames
            temporal_zoom_in (bool): Whether to enable temporal zoom-in (dense sampling)
            **model_kwargs: Additional arguments passed to model loading
        """
        
        model_name = llm_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2",
            **model_kwargs
        )
        self.model.config.output_attentions = True

        self.processor = AutoProcessor.from_pretrained(model_name, max_pixels=854 * 480)
        
        self.max_frames = max_frames
        assert max_frames == 64
        
        self.max_thinking_tokens = max_thinking_tokens

        self.num_reflection = num_reflection
        self.process_out_list = process_out_list

        # Initialize zoom-in capabilities
        self.spatial_zoom_in = spatial_zoom_in
        self.temporal_zoom_in = temporal_zoom_in

        # If spatial zoom-in is enabled, load CLIP model for region detection
        if self.spatial_zoom_in:
            from transformers import CLIPProcessor, CLIPModel
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_model.to(self.model.device)
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


    def get_batch_messages(self, video_paths, queries, query_image):

        messages = []
        for video_path, query, image in zip(video_paths, queries, query_image):
            content = [
                {
                    "type": "video",
                    "video": video_path,
                    "max_pixels": 360 * 420,
                    "max_frames": self.max_frames,
                },
                {"type": "text", "text": query},
            ]
            if image is not None:
                content.insert(1, {"type": "image", "image": image})
            messages.append([{"role": "user", "content": content}])

        texts = [self.processor.apply_chat_template(
            msg, tokenize=False, add_generation_prompt=True) for msg in messages]

        for i in range(len(texts)):
            text = texts[i]
            pattern = r"Audio transcripts of the video:\n(.*?)\nQuestion:"
            match = re.search(pattern, text, re.DOTALL)
            if match:
                extracted_text = match.group(1)
                extracted_text_new = convert_timestamp_format(extracted_text)
                texts[i] = text.replace(extracted_text,extracted_text_new)
        
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages, return_video_kwargs=True)

        self.fps_list = video_kwargs["fps"] 
        
        batch_inputs = []
        if image_inputs:
            for v_input, text, i_input in zip(video_inputs, texts, image_inputs):
                batch_inputs.append({
                    "text": text,
                    "video": v_input, 
                    "image": torch.tensor(np.array(i_input)), 
                })
            return batch_inputs
        
        for v_input, text in zip(video_inputs, texts):
            batch_inputs.append({
                "text": text,
                "video": v_input, 
            })
        return batch_inputs

    def __call__(self, video_path, query, query_image, doc= None,\
                 key_frames=None, inputs=None, think_prompt_num=0,\
                 tem=0.0, top_p=0.9, top_k=50, model_type=None,\
                 output_format_prompt="",**kwargs):
        if inputs == None:
            if isinstance(video_path, list) and isinstance(query, list):
                inputs = self.get_batch_messages(video_path, query, query_image)
            else:
                raise ValueError("video_path and query must be list or str")

        data_idx = 0
        input = inputs[0]

        if '<image 1>' in input['text']:
            input['text'] = input['text'].replace('<|vision_start|><|image_pad|><|vision_end|>','')
            input['text'] = input['text'].replace('<image 1>','<|vision_start|><|image_pad|><|vision_end|>')

        def cal_answer_prob(collected_logits, tokenizer, answer, token_id):
            target_token_id = tokenizer.encode(answer, add_special_tokens=False)[0]

            import torch.nn.functional as F
            idx_list = []
            prob_list = []
            for i, logits in enumerate(collected_logits):
                if target_token_id==token_id[i]:
                    probabilities = F.softmax(logits, dim=-1)
                    target_probability = probabilities[target_token_id].item()
                    prob_list.append(target_probability)

                    idx_list.append(i)
            return idx_list,prob_list

        def call_generate(input, max_thinking_tokens, check=False, vis_path= None,\
                          last_step_attn=None, tem=0.0, top_p=0.9, top_k=50):
        
            if type(input["video"]) != list:
                input["video"] = [input["video"]]
            
            if "image" in input.keys():
                if type(input["image"]) != list:
                    input["image"] = [input["image"]]
                processed_inputs = self.processor(
                    text=[input["text"]],
                    videos=input["video"],
                    images=input["image"],
                    padding = True,
                    return_tensors="pt",
                ).to(self.model.device)
            else:
                processed_inputs = self.processor(
                    text=[input["text"]],
                    videos=input["video"],
                    padding = True,
                    return_tensors="pt",
                ).to(self.model.device)

            logits_list = []
            attentions_list = []
            
            # Define a forward hook to capture logits and attentions
            def forward_hook(module, inputs, outputs):
                logits = outputs['logits'][:, -1, :][0]
                attention = outputs['attentions'][-1].squeeze(0)  # last layer attention
                logits_list.append(logits.detach().cpu().clone())
                attentions_list.append(attention.detach().cpu().clone())
            
            hook_handle = self.model.register_forward_hook(forward_hook)

            
            if tem==0.0:
                # Generate output with temperature 0.0 (deterministic)
                outputs = self.model.generate(**processed_inputs, do_sample=False, max_new_tokens=max_thinking_tokens)
            else: 
                outputs = self.model.generate(**processed_inputs, do_sample=True, temperature=tem, top_p=top_p, top_k=top_k, max_new_tokens=max_thinking_tokens)

            generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(processed_inputs.input_ids, outputs)]
            answers = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            
            all_token_ids = outputs[0]

            idx_list = list(range(len(attentions_list)))

            pred_answer = self.process_out_list(answers,doc[data_idx])[0]
            answer_list = ["A","B","C","D","E","F","G","H","I","J"]

            collected_attns = attentions_list
            num_heads = collected_attns[0].shape[0]
            prompt_len = collected_attns[0].shape[2]
            num_tokens = len(collected_attns)

            # get full attention map
            full_attention_map = torch.zeros((num_heads, num_tokens, prompt_len + num_tokens - 1))
            
            for i, attn in enumerate(collected_attns):
                full_attention_map[:, i, :prompt_len + i] = attn[:,0,:]
            
            full_attention_array = full_attention_map.to(dtype=torch.float32).numpy()

            # if last_step_attn is not None, refine the attention map
            if type(last_step_attn)==np.ndarray:
                # remove the last token "<|im_end|>"
                last_step_attn = last_step_attn[:,:-1,:-1]
                refined_attn_map = np.zeros((last_step_attn.shape[0],last_step_attn.shape[1]+full_attention_array.shape[1],full_attention_array.shape[2]))
                refined_attn_map[:,0:last_step_attn.shape[1], 0:last_step_attn.shape[2]] = last_step_attn
                refined_attn_map[:, last_step_attn.shape[1]:, :] = full_attention_array
                full_attention_array = refined_attn_map

            # if check is True, calculate the probability of the predicted answer and the attention map
            if check and pred_answer in answer_list:
                idx_list,prob_list = cal_answer_prob(logits_list, self.tokenizer, pred_answer, generated_ids_trimmed[0])
                if idx_list == []:
                    pred_answer_refine = " " + pred_answer
                    idx_list,prob_list = cal_answer_prob(logits_list, self.tokenizer, pred_answer_refine, generated_ids_trimmed[0])

                if type(last_step_attn)==np.ndarray:
                    attention_array, asr_token_num = process_attention_2(full_attention_array, all_token_ids, idx_list, None, self.tokenizer, last_step_shape=last_step_attn.shape)
                else:                        
                    attention_array, asr_token_num = process_attention(attentions_list, all_token_ids, idx_list, vis_path, self.tokenizer)

                if len(prob_list)>0:
                    prob = prob_list[-1]
                else:
                    prob = None

                logits_list.clear()
                attentions_list.clear()
                hook_handle.remove()     
                return answers[0], generated_ids_trimmed[0], asr_token_num, prob, attention_array
            
            logits_list.clear()
            attentions_list.clear()

            hook_handle.remove()

            return answers[0], generated_ids_trimmed[0], 0, None, full_attention_array
             
        # "origin" means do not use CoT 
        if model_type =='origin':
            # add key frames
            if key_frames != None and len(key_frames[data_idx])>0:
                key_frame = key_frames[data_idx]
                key_frame_string = "\nHere are some key frames that may be helpful:"
                cnt = 1
                for frame in key_frame:
                    added_image = input["video"][frame]
                    # if spatial_zoom_in is True, zoom in the image
                    if self.spatial_zoom_in:
                        question_pattern = r"Question:(.*?)\nQuestion Category:"
                        match = re.search(question_pattern, origin_input_prompt, re.DOTALL)
                        question_text = match.group(1)
                        added_image = np.array(added_image).astype(np.uint8)
                        device = self.model.device
                        region = find_focus_region(added_image, question_text, self.clip_model, self.clip_processor, device)
                        added_image = zoom_in_image(added_image, region)
                        added_image = torch.tensor(added_image)

                    if "image" in input.keys():
                        if type(input['image']) == list:
                            input['image'].append(added_image)
                        else:
                            input['image'] = [input['image'], added_image]
                    else:
                        input['image'] = added_image

                    interval = 1/self.fps_list[0]
                    time_now = round(frame* interval)

                    mins = time_now // 60
                    secs = time_now % 60
                    time_now = f"{mins}:{secs:02d}"

                    # key frame prompt
                    key_frame_string += f"Key Frame #{cnt}: Time {time_now}, <|vision_start|><|image_pad|><|vision_end|>"
                    cnt += 1

                # Answer after adding key frames
                key_frame_string += "You need to pay more attention to key frames and answer the question. "
                if output_format_prompt:
                    key_frame_string += output_format_prompt
                else:
                    key_frame_string += "Respond with only the letter of the correct option."
                    
                input["text"] = input["text"].replace("<|im_end|>\n<|im_start|>assistant", key_frame_string + "<|im_end|>\n<|im_start|>assistant")
                # get output text, token id, asr_token_num (K_2), probability, repetition
                o, o_token_id, _, prob, attention_array = call_generate(input, 512, check=True, tem=tem, top_p=top_p, top_k=top_k)
                repetition = detect_repetition(o)
                final_out = {"text":o,"attn": None, "prob":prob, "repetition": repetition}
                # print(input["text"] + o)
                return [final_out]
            
            # No key frame
            if "<OPEN-QUESTION>" in input['text']:
                check = False
            else:
                check = True

            origin_prompt = output_format_prompt
            input["text"] = input["text"].replace("<|im_end|>\n<|im_start|>assistant", origin_prompt+"<|im_end|>\n<|im_start|>assistant")

            o, o_token_id, asr_token_num, prob, attention_array= call_generate(input, 512, check=check, tem=0.0)
            repetition = detect_repetition(o)
            final_out = {"text":o, "attn": attention_array, "prob":prob, "repetition": repetition, "asr_token_num": asr_token_num}
            # print("-"*50)
            # print(input["text"] + o)
            # print("-"*50)

            torch.cuda.empty_cache()
            return [final_out]

        ### Think
        origin_input_prompt = input["text"]

        think_prompt_list = [
            "Thinking process:\n", 
            "Detailed reasoning process:",
            "First, let me think about it systematically. Logical Steps:",
            "Firstly, I need to write my thought process:",
        ]
        input["text"] += think_prompt_list[think_prompt_num%len(think_prompt_list)]
        
        o, o_token_id, _, _, full_attention_array = call_generate(input, self.max_thinking_tokens, tem=tem, top_p=top_p, top_k=top_k)
        max_tokens_thinking_tmp = self.max_thinking_tokens
        max_tokens_thinking_tmp -= len(o_token_id)

        input["text"] = input["text"] + o 

        ### reflection using "wait". The number of reflections is set to 0 in main experiments
        for reflection_idx in range(self.num_reflection):
            if max_tokens_thinking_tmp <= 0:
                break
            reflection_prompt = "\nWait,"
            input["text"] = input["text"] + reflection_prompt
            o, o_token_id, _, _, full_attention_array = call_generate(input, max(max_tokens_thinking_tmp,512), tem=tem, top_p=top_p, top_k=top_k, last_step_attn=full_attention_array)
            max_tokens_thinking_tmp -= len(o_token_id)
            input["text"] = input["text"] + o

        ### Add key frames after thinking process
        if key_frames != None:
            key_frame = key_frames[data_idx]
            key_frame_string = "\nHere are some key frames that may be helpful:"
            cnt = 1
            for frame in key_frame:
                added_image = input["video"][0][frame]
                if self.spatial_zoom_in:
                    question_pattern = r"Question:(.*?)\nQuestion Category:"
                    match = re.search(question_pattern, origin_input_prompt, re.DOTALL)
                    question_text = match.group(1)
                    added_image = np.array(added_image).astype(np.uint8)
                    device = self.model.device
                    region = find_focus_region(added_image, question_text, self.clip_model, self.clip_processor, device)
                    added_image = zoom_in_image(added_image, region)
                    added_image = torch.tensor(added_image)
                if "image" in input.keys():
                    if type(input['image']) == list:
                        input['image'].append(added_image)
                    else:
                        input['image'] = [input['image'], added_image]
                else:
                    input['image'] = added_image

                interval = 1/self.fps_list[0]
                time_now = round(frame* interval)

                mins = time_now // 60
                secs = time_now % 60
                time_now = f"{mins}:{secs:02d}"

                key_frame_string += f"Key Frame #{cnt}: Time {time_now}, <|vision_start|><|image_pad|><|vision_end|>"
                cnt += 1
            
            key_frame_string += "Next, I need to focus on the visual content of key frames, and recheck my thinking process." 
    
            input["text"] = input["text"] + key_frame_string

        ### Final answer ###
        if "<OPEN-QUESTION>" in input['text']:
            input['text'] += "Final Answer:"
        else:
            input['text'] += """\nNext, I will conclude by stating the final answer using the following format: "Therefore, the final answer is: LETTER" (without quotes), where $LETTER is the final option of the question. I need to ensure the consistency between the final output answer and the thinking process. """
            input['text'] += "\nTherefore, the final answer is:"

        if key_frames != None or "<OPEN-QUESTION>" in input['text']:
            check = False
        else:
            check = True
        o, o_token_id, asr_token_num, prob, attention_array = call_generate(input, 512, check=check, tem=0.0, last_step_attn=full_attention_array)
        
        output = input["text"] + o

        # print("-"*50)
        # print(output)
        # print("-"*50)

        repetition = detect_repetition(output[len(origin_input_prompt):])

        contains_letter = lambda s: any('A' <= c <= 'J' for c in s)
        if not contains_letter(o):
            o = output[len(origin_input_prompt):]

        final_out = {"text":o, "attn": attention_array,"prob":prob, "repetition":repetition, "asr_token_num": asr_token_num}

        torch.cuda.empty_cache()

        return [final_out]
    


def custom_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
    ):
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

    # Because the input can be padded, the absolute sequence length depends on the max position id.
    cos, sin = position_embeddings
    query_states, key_states = apply_multimodal_rotary_pos_emb(
        query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
    )

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    # repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    dropout_rate = 0.0 if not self.training else self.attention_dropout

    # In PEFT, usually we cast the layer norms in float32 for training stability reasons
    # therefore the input hidden states gets silently casted in float32. Hence, we need
    # cast them back in float16 just to be sure everything works as expected.
    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        # Handle the case where the model is quantized
        elif hasattr(self.config, "_pre_quantization_dtype"):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.q_proj.weight.dtype

        logger.warning_once(
            f"The input hidden states seems to be silently casted in float32, this might be related to"
            f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
            f" {target_dtype}."
        )

        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    # Reashape to the expected shape for Flash Attention
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    if (
        self.config.use_sliding_window
        and getattr(self.config, "sliding_window", None) is not None
        and self.layer_idx >= self.config.max_window_layers
    ):
        sliding_window = self.config.sliding_window
    else:
        sliding_window = None

    # our custom flash attention forward function
    attn_output, attn_weights = _flash_attention_forward_mine(
        query_states,
        key_states,
        value_states,
        attention_mask,
        q_len,
        dropout=dropout_rate,
        sliding_window=sliding_window,
        is_causal=self.is_causal,
        use_top_left_mask=self._flash_attn_uses_top_left_mask,
    )

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value

def _flash_attention_forward_mine(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor,
    query_length: int,
    is_causal: bool,
    dropout: float = 0.0,
    position_ids: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    sliding_window: Optional[int] = None,
    use_top_left_mask: bool = False,
    softcap: Optional[float] = None,
    deterministic: bool = None,
    cu_seq_lens_q: Optional[torch.LongTensor] = None,
    cu_seq_lens_k: Optional[torch.LongTensor] = None,
    max_length_q: Optional[int] = None,
    max_length_k: Optional[int] = None,
    target_dtype: Optional[torch.dtype] = None,
    **kwargs,
    ):
    """
    Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
    first unpad the input, then computes the attention scores and pad the final attention scores.

    Args:
        query_states (`torch.Tensor`):
            Input query states to be passed to Flash Attention API
        key_states (`torch.Tensor`):
            Input key states to be passed to Flash Attention API
        value_states (`torch.Tensor`):
            Input value states to be passed to Flash Attention API
        attention_mask (`torch.Tensor`):
            The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
            position of padding tokens and 1 for the position of non-padding tokens.
        dropout (`float`):
            Attention dropout
        softmax_scale (`float`, *optional*):
            The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        use_top_left_mask (`bool`, defaults to `False`):
            flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference.
        softcap (`float`, *optional*):
            Softcap for the attention logits, used e.g. in gemma2.
        deterministic (`bool`, *optional*):
            Determines if the deterministic option introduced in flash_attn>=2.4.1 is enabled.
    """
    if not use_top_left_mask:
        causal = is_causal
    else:
        # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1.
        causal = is_causal and query_length != 1

    # Assuming 4D tensors, key_states.shape[1] is the key/value sequence length (source length).
    use_sliding_windows = (
        _flash_supports_window_size and sliding_window is not None and key_states.shape[1] > sliding_window
    )
    flash_kwargs = {"window_size": (sliding_window, sliding_window)} if use_sliding_windows else {}

    if flash_241:
        if deterministic is None:
            deterministic = deterministic_g
        flash_kwargs["deterministic"] = deterministic

    if softcap is not None:
        flash_kwargs["softcap"] = softcap

    # PEFT possibly silently casts tensors to fp32, this potentially reconverts to correct dtype or is a no op
    query_states, key_states, value_states = fa_peft_integration_check(
        query_states, key_states, value_states, target_dtype
    )

    # print(query_states.shape, key_states.shape, value_states.shape)
    # print(softmax_scale, causal, flash_kwargs)



    #########calculate attention weights#########
    import torch.nn.functional as F
    if query_states.shape[1] != 1:
        query = query_states[:, -1].unsqueeze(1).clone()
    else:
        query = query_states.clone()
    key = key_states.clone()
    batch_size, seq_len, num_heads, d_k = query.size()

    query = query.transpose(1, 2)  # (batch_size, num_heads, seq_len, d_k)
    key = key.transpose(1, 2)      # (batch_size, num_heads, seq_len, d_k)

    attn_scores = torch.matmul(query, key.transpose(-2, -1))
    attn_scores = attn_scores / torch.sqrt(torch.tensor(d_k, dtype=torch.bfloat16))
    attn_weights = F.softmax(attn_scores, dim=-1)

    #########calculate attention weights#########

    # Contains at least one padding token in the sequence
    if attention_mask is not None:
        batch_size = query_states.shape[0]
        query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = _upad_input(
            query_states, key_states, value_states, attention_mask, query_length
        )
        cu_seqlens_q, cu_seqlens_k = cu_seq_lens
        max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

        attn_output_unpad = flash_attn_varlen_func(
            query_states,
            key_states,
            value_states,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_in_batch_q,
            max_seqlen_k=max_seqlen_in_batch_k,
            dropout_p=dropout,
            softmax_scale=softmax_scale,
            causal=causal,
            **flash_kwargs,
        )
        attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)

    # If position_ids is provided and check all examples do not contain only 1 sequence, If tensor in increasing
    # then we probably have one sequence, otherwise it is packed. Additionally check we are in pre-fill/training stage.
    # Use `flash_attn_varlen_func` to prevent cross-example attention and also allow padding free approach
    elif position_ids is not None and (
        max_length_q is not None or (query_length != 1 and not (torch.diff(position_ids, dim=-1) >= 0).all())
    ):
        batch_size = query_states.size(0)

        if cu_seq_lens_q is None or cu_seq_lens_k is None:
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = (
                prepare_fa2_from_position_ids(query_states, key_states, value_states, position_ids)
            )

            cu_seq_lens_q, cu_seq_lens_k = cu_seq_lens
            max_length_q, max_length_k = max_seq_lens

        else:
            query_states = query_states.reshape(-1, query_states.size(-2), query_states.size(-1))
            key_states = key_states.reshape(-1, key_states.size(-2), key_states.size(-1))
            value_states = value_states.reshape(-1, value_states.size(-2), value_states.size(-1))

        attn_output = flash_attn_varlen_func(
            query_states,
            key_states,
            value_states,
            cu_seqlens_q=cu_seq_lens_q,
            cu_seqlens_k=cu_seq_lens_k,
            max_seqlen_q=max_length_q,
            max_seqlen_k=max_length_k,
            dropout_p=dropout,
            softmax_scale=softmax_scale,
            causal=causal,
            **flash_kwargs,
        )

        attn_output = attn_output.view(batch_size, -1, attn_output.size(-2), attn_output.size(-1))

    else:
        attn_output = flash_attn_func(
            query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal, **flash_kwargs
        )

    return attn_output, attn_weights


Qwen2_5_VLFlashAttention2.forward = custom_forward

import matplotlib.pyplot as plt

def process_attention(collected_attns, all_token_ids, target_token_id_list, vis_path, tokenizer, save_fig=False):

    num_heads = collected_attns[0].shape[0]
    prompt_len = collected_attns[0].shape[2]
    num_tokens = len(collected_attns)

    video_token = 151656 # for qwen2.5-vl
    video_token_indices = [i for i, token in enumerate(all_token_ids) if token == video_token]

    video_start = video_token_indices[0]
    video_end = video_token_indices[-1]
    if target_token_id_list != [] and (video_end+1-video_start)%32==0:
        target_token_id = target_token_id_list[-1]
    else:
        return None, 0

    target_token_id += (prompt_len-1)
    all_tokens = {}

    for i in range(len(all_token_ids)):
        all_tokens[i] = [all_token_ids[i].item(),tokenizer.decode([all_token_ids[i]], skip_special_tokens=False, clean_up_tokenization_spaces=False)]

    group_list = [0, video_start-1] 
    interval = int((video_end+1-video_start)//32)

    # 32 video segments in group_list
    group_list.extend([i for i in range(video_start,video_end,interval)])
    group_list.append(video_end+1)  # vision end
    group_list.append(video_end+2)  # asr start

    # add asr segments in group_list
    txt = tokenizer.decode(all_token_ids)
    pattern = r"Audio transcripts of the video:\n(.*?)\nQuestion:"
    match = re.search(pattern, txt, re.DOTALL)
    asr_token_num = 0
    if match:
        extracted_text = match.group(1)
        pattern = r"\[\d+:\d+ -> \d+:\d+\]:"
        time_segments = re.findall(pattern, extracted_text)
        segment_token_ids = [tokenizer.encode(seg, add_special_tokens=True) for seg in time_segments]
        for seg_ids in segment_token_ids:
            for i in range(video_end, len(all_token_ids) - len(seg_ids) + 1):
                if list(all_token_ids[i:i + len(seg_ids)]) == seg_ids:
                    group_list.append(i)
                    asr_token_num +=1
                    break

    # add question segments in group_list
    for i in range(len(all_tokens)):
        if 'Question' in all_tokens[i][1] and ':' in all_tokens[i+1][1]:
            group_list.append(i)
            break
    
    for i in range(len(all_tokens)):
        if i+3 < len(all_tokens) and all_tokens[i][1]=='<|im_end|>' and all_tokens[i+2][1]=='<|im_start|>' and all_tokens[i+3][1]=='assistant' :
            group_list.append(i)
            break
    
    # add segment before answer token
    if target_token_id != prompt_len -1:
        group_list.append(prompt_len - 1)

    # add answer token
    group_list.append(target_token_id)
    
    group_list.append(target_token_id+1)
    group_list.append(len(all_tokens)-1)

    has_duplicates = len(group_list) != len(set(group_list))
    if has_duplicates:
        return None, 0

    full_attention_map = torch.zeros((num_heads, num_tokens, prompt_len + num_tokens - 1))

    for i, attn in enumerate(collected_attns):
        full_attention_map[:, i, :prompt_len + i] = attn[:,0,:]


    from matplotlib.colors import LogNorm

    grouped_attention_list = []

    for head in range(num_heads):

        # calculate attention map for each head
        
        head_att = full_attention_map[head]
        head_att_np = head_att.to(dtype=torch.float32).numpy()

        group_list_2 = [k-(prompt_len-1) for k in group_list[-3:]]
        small_map = np.zeros((len(group_list_2)-1, len(group_list)-1))

        for i in range(small_map.shape[0]):
            for j in range(small_map.shape[1]):
                row_start = group_list_2[i]
                row_end = group_list_2[i + 1]
                col_start = group_list[j]
                col_end = group_list[j + 1]
                small_map[i, j] = np.sum(head_att_np[row_start:row_end, col_start:col_end])/(row_end-row_start)
        
        grouped_attention_list.append(small_map)

        save_fig = False # for debug

        if save_fig:
            vis_path = "./logs/vis/origin/"
            os.makedirs(vis_path, exist_ok=True)

            head_att_np = small_map      
            
            head_att_np[head_att_np < 1e-8] = 1e-8    
            lognorm = LogNorm(vmin=np.min(head_att_np), vmax=1.0)


            plt.figure(figsize=(10, 5))  
            im = plt.imshow(head_att_np, cmap='jet', norm=lognorm)

            plt.axvline(x=1, color='black', linestyle='--', linewidth=1, label='video_start', clip_on=False)
            plt.axvline(x=34, color='black', linestyle='--', linewidth=1, label='video_end', clip_on=False)
            if asr_token_num > 0:
                plt.axvline(x=35, color='black', linestyle='--', linewidth=1, label='asr_start', clip_on=False)
                plt.axvline(x=36 + asr_token_num, color='black', linestyle='--', linewidth=1, label='asr_end', clip_on=False)

            plt.annotate('Answer',
                        xy=(0, 0), xytext=(-60, 10),
                        textcoords='offset points',
                        arrowprops=dict(facecolor='black', arrowstyle='->'),
                        fontsize=8, color='black')

            cbar = plt.colorbar(im, orientation='horizontal', pad=0.15, shrink=0.4)
            cbar.ax.tick_params(labelsize=8)

            plt.title(f'Attention Map - Head {head + 1}')
            plt.xlabel('Token Index')
            plt.ylabel('Token Index')
            save_path = os.path.join(vis_path, f"attention_map_head_{head + 1}.png")
            plt.savefig(save_path, dpi=600)
            plt.close()

    grouped_attention_list = np.stack(grouped_attention_list, axis=0)  
    return grouped_attention_list, asr_token_num

def process_attention_2(full_attention_map, all_token_ids, target_token_id_list, vis_path, tokenizer, last_step_shape, save_fig=False):
    
    num_heads = full_attention_map.shape[0]

    video_token = 151656
    video_token_indices = [i for i, token in enumerate(all_token_ids) if token == video_token]

    video_token_indices_filtered = [video_token_indices[0]]
    for i in range(1,len(video_token_indices)):
        if video_token_indices[i] == video_token_indices[i-1] + 1:
            video_token_indices_filtered.append(video_token_indices[i])
        else:
            break
    video_token_indices = video_token_indices_filtered

    video_start = video_token_indices[0]
    video_end = video_token_indices[-1]
    if target_token_id_list != [] and (video_end+1-video_start)%32==0:
        target_token_id = target_token_id_list[-1]
    else:
        return None, 0

    all_tokens = {}
    for i in range(len(all_token_ids)):
        all_tokens[i] = [all_token_ids[i].item(),tokenizer.decode([all_token_ids[i]], skip_special_tokens=False, clean_up_tokenization_spaces=False)]
   
    # calculate the answer token index
    target_token_id += (full_attention_map.shape[2]-(full_attention_map.shape[1]-last_step_shape[1]))

    group_list = [0, video_start-1] 
    interval = int((video_end+1-video_start)//32)

    group_list.extend([i for i in range(video_start,video_end,interval)])
    group_list.append(video_end+1)  # visual end
    group_list.append(video_end+2)  # ASR start

    txt = tokenizer.decode(all_token_ids)
    pattern = r"Audio transcripts of the video:\n(.*?)\nQuestion:"
    match = re.search(pattern, txt, re.DOTALL)
    asr_token_num = 0
    if match:
        extracted_text = match.group(1)
        pattern = r"\[\d+:\d+ -> \d+:\d+\]:"
        time_segments = re.findall(pattern, extracted_text)
        segment_token_ids = [tokenizer.encode(seg, add_special_tokens=True) for seg in time_segments]
        for seg_ids in segment_token_ids:
            for i in range(video_end, len(all_token_ids) - len(seg_ids) + 1):
                if list(all_token_ids[i:i + len(seg_ids)]) == seg_ids:
                    group_list.append(i)
                    asr_token_num +=1
                    break

    for i in range(len(all_tokens)):
        if all_tokens[i][1]=='Question' and ':' in all_tokens[i+1][1]:
            group_list.append(i)
            break
    
    for i in range(len(all_tokens)):
        if i+3 < len(all_tokens) and all_tokens[i][1]=='<|im_end|>' and all_tokens[i+2][1]=='<|im_start|>' and all_tokens[i+3][1]=='assistant' :
            group_list.append(i)
            break

    if target_token_id != (last_step_shape[2]-last_step_shape[1]):
        group_list.append(last_step_shape[2]-last_step_shape[1])
    group_list.append(target_token_id)
    
    group_list.append(target_token_id+1)
    group_list.append(len(all_tokens)-1)

    has_duplicates = len(group_list) != len(set(group_list))
    if has_duplicates:
        print(group_list)
        print(len(all_tokens))
        return None, 0

    from matplotlib.colors import LogNorm

    grouped_attention_list = []

    for head in range(num_heads):

        head_att = full_attention_map[head]
        head_att_np = head_att

        group_list_2 = []
        for idx in group_list[-3:]:
            group_list_2.append(idx - (full_attention_map.shape[2]-full_attention_map.shape[1]))

        small_map = np.zeros((len(group_list_2)-1, len(group_list)-1))

        for i in range(small_map.shape[0]):
            for j in range(small_map.shape[1]):
                row_start = group_list_2[i]
                row_end = group_list_2[i + 1]
                col_start = group_list[j]
                col_end = group_list[j + 1]
                small_map[i, j] = np.sum(head_att_np[row_start:row_end, col_start:col_end])/(row_end-row_start)
        
        grouped_attention_list.append(small_map)

        save_fig = False # for debug

        if save_fig:
            vis_path = "./logs/vis/cot/"
            os.makedirs(vis_path, exist_ok=True)

            head_att_np = small_map      
            head_att_np[head_att_np < 1e-8] = 1e-8    
            lognorm = LogNorm(vmin=np.min(head_att_np), vmax=1.0)

            plt.figure(figsize=(10, 5)) 
            im = plt.imshow(head_att_np, cmap='jet', norm=lognorm)

            plt.axvline(x=1, color='black', linestyle='--', linewidth=1, label='video_start', clip_on=False)
            plt.axvline(x=34, color='black', linestyle='--', linewidth=1, label='video_end', clip_on=False)
            if asr_token_num > 0:
                plt.axvline(x=35, color='black', linestyle='--', linewidth=1, label='asr_start', clip_on=False)
                plt.axvline(x=36 + asr_token_num, color='black', linestyle='--', linewidth=1, label='asr_end', clip_on=False)

            plt.annotate('Answer',
                        xy=(0, 0), xytext=(-60, 10),
                        textcoords='offset points',
                        arrowprops=dict(facecolor='black', arrowstyle='->'),
                        fontsize=8, color='black')

            cbar = plt.colorbar(im, orientation='horizontal', pad=0.15, shrink=0.4)
            cbar.ax.tick_params(labelsize=8)

            plt.title(f'Attention Map - Head {head + 1}')
            plt.xlabel('Token Index')
            plt.ylabel('Token Index')
            save_path = os.path.join(vis_path, f"attention_map_head_{head + 1}.png")
            plt.savefig(save_path, dpi=600)
            plt.close()

    grouped_attention_list = np.stack(grouped_attention_list, axis=0)  
    return grouped_attention_list, asr_token_num
