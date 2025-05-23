import logging
from typing import List, Tuple
import copy
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from utils import detect_repetition, load_video, get_conv_template, apply_rotary_pos_emb, repeat_kv
eval_logger = logging.getLogger("eval_logger")

DEFAULT_GEN_KWARGS = dict(
    num_beams=1,
    max_new_tokens=4096,
    do_sample=False,
)

import os
os.environ["DECORD_EOF_RETRY_MAX"] = "40960"
import torch
from transformers import AutoProcessor, AutoTokenizer
from typing import List, Optional, Tuple, Union
from utils import convert_timestamp_format
import re


class InternVL_Transformers:
    def __init__(self, llm_name, process_out_list, max_thinking_tokens=4096, max_frames=32,\
                num_reflection=0, **model_kwargs):
        """
        Initialize the QwenVL_Transformers model for video-language tasks.
        
        Args:
            llm_name (str): Name/path of the pretrained InternVL model to load
            process_out_list (callable): Function to parse model outputs and get predictions
            max_thinking_tokens (int): Maximum number of tokens for thinking process
            max_frames (int): Maximum number of video frames to process (Here we use 32 frames)
            num_reflection (int): Number of reflection steps to perform after initial thinking (Add “Wait” before each reflection)
            **model_kwargs: Additional arguments passed to model loading
        """
    
        model_name = llm_name
        device_map="auto"
        self.model = AutoModel.from_pretrained(model_name, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True, device_map=device_map).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, device_map=device_map)

        self.input_size = 448
        # self.model.num_image_token = self.input_size*self.input_size//(14*2*14*2)

        assert max_frames == 32
        max_pixels = self.input_size* self.input_size
        self.processor = AutoProcessor.from_pretrained(model_name, max_pixels=max_pixels, trust_remote_code=True)

        self.max_frames = max_frames
        self.max_thinking_tokens = max_thinking_tokens

        self.num_reflection = num_reflection
        self.process_out_list = process_out_list

        self.fps_list=[]
        self.DEFAULT_GEN_KWARGS = dict(
            num_beams=1,
            max_new_tokens=512,
            do_sample=False,
            output_attentions=True
        )
        self.model.language_model.model.layers[-1].self_attn.forward = self.custom_forward


    def get_batch_messages(self, video_paths, queries, query_image, duration=1.0):
        batch_inputs = []
        for video_path, query, image in zip(video_paths, queries, query_image):
            pattern = r"Audio transcripts of the video:\n(.*?)\nQuestion:"
            match = re.search(pattern, query, re.DOTALL)
            if match:
                extracted_text = match.group(1)
                extracted_text_new = convert_timestamp_format(extracted_text)
                query = query.replace(extracted_text,extracted_text_new)
            pixel_values, num_patches_list, fps = load_video(video_path, num_segments=self.max_frames, input_size=self.input_size, output_fps=True)
            self.fps_list.append(fps)
            pixel_values = pixel_values.to(torch.bfloat16).cuda()

            num_patches = len(num_patches_list)
            video_prefix = "".join([f"Frame{i+1}: <image>\n" for i in range(num_patches)])
            query = query.replace("<image 1>", "[input question image:<image>]")

            if image is not None:
                img_pixel_values, img_num_patches_list = load_video(image, num_segments=1, input_size=self.input_size)
                img_pixel_values = img_pixel_values.to(torch.bfloat16).cuda()

                pixel_values = torch.cat([pixel_values, img_pixel_values], dim=0)

                num_patches_list += img_num_patches_list
                if "[input question image:<image>]" not in query:
                    for i in range(len(img_pixel_values)):
                        video_prefix+= f"[input question image:<image>]\n"

            batch_inputs.append({"video_prefix":video_prefix, "text": query, "video": pixel_values, "num_patches_list": num_patches_list})
        return batch_inputs

    def __call__(self, video_path, query, query_image, doc= None,\
                 key_frames=None, inputs=None, think_prompt_num=0,\
                 tem=0.0, top_p=0.9, top_k=50, model_type=None,\
                 output_format_prompt="", **kwargs):
        
        if inputs == None:
            if isinstance(video_path, list) and isinstance(query, list):
                inputs = self.get_batch_messages(video_path, query, query_image)
            else:
                raise ValueError("video_path and query must be list or str")

        input = inputs[0]
        data_idx = 0

        question = input["video_prefix"] + input["text"]
        query = self.process_query(self.tokenizer, question)
        original_query = query

        # "origin" means do not use CoT 
        if model_type == "origin":

            # add key frames if exist
            if key_frames != None and len(key_frames[data_idx])>0:
                key_frame = key_frames[data_idx]
                key_frame_string = "\nHere are some key frames that may be helpful:"
                cnt = 1
                for frame in key_frame:
                    added_image = input["video"][frame].unsqueeze(0)
                    input["video"] = torch.cat([input["video"], added_image], dim=0)
                    interval = 1/self.fps_list[0]
                    time_now = round(frame* interval)
                    mins = time_now // 60
                    secs = time_now % 60
                    time_now = f"{mins}:{secs:02d}"
                    input["num_patches_list"].append(input["num_patches_list"][0])
                    key_frame_string += f"Key Frame #{cnt}: Time {time_now}, <image>"
                    cnt += 1

                key_frame_string += "You need to pay more attention to key frames and answer the question. "
                if output_format_prompt:
                    key_frame_string += output_format_prompt
                else:
                    key_frame_string += "Respond with only the letter of the correct option."
                    
                query = query.replace("<|im_end|>\n<|im_start|>assistant", key_frame_string + "<|im_end|>\n<|im_start|>assistant")

                query = query.replace("<|im_end|>\n<|im_start|>assistant", origin_prompt+"<|im_end|>\n<|im_start|>assistant")
                if tem == 0.0:
                    gen_configs = dict(
                        num_beams=1,
                        max_new_tokens=4096,
                        do_sample=False,
                        output_attentions=True
                    )
                else:
                    gen_configs = dict(
                        num_beams=1,
                        max_new_tokens=4096,
                        do_sample=True,
                        temperature=tem,
                        top_p=top_p,
                        top_k=top_k,
                        output_attentions=True
                    )
                
                pixel_values = input["video"]
                num_patches_list = input["num_patches_list"]

                o, o_token_id, asr_token_num, prob, attention_array = self.chat(self.tokenizer, pixel_values, query,\
                                                                            generation_config = gen_configs, doc=doc,\
                                                                            num_patches_list=num_patches_list, check=True)     
                repetition = detect_repetition(o)
                final_out = {"text":o, "attn": None, "prob":prob, "repetition": repetition, "asr_token_num": asr_token_num}
                # print("-"*50)
                # print(query + o)
                # print("-"*50)

                torch.cuda.empty_cache()
                return [final_out]
    
            if "<OPEN-QUESTION>" in input['text']:
                check = False
            else:
                check = True

            if len(output_format_prompt) == 0:
                output_format_prompt = "Respond with only the letter of the correct option."
            
            origin_prompt = output_format_prompt
            query = query.replace("<|im_end|>\n<|im_start|>assistant", origin_prompt+"<|im_end|>\n<|im_start|>assistant")
            gen_configs = dict(
                num_beams=1,
                max_new_tokens=512,
                do_sample=False,
                output_attentions=True
            )
            pixel_values = input["video"]
            num_patches_list = input["num_patches_list"]
            o, o_token_id, asr_token_num, prob, attention_array = self.chat(self.tokenizer, pixel_values, query,\
                                                                            generation_config = gen_configs, doc=doc,\
                                                                            num_patches_list=num_patches_list, check=True)     
            repetition = detect_repetition(o)
            final_out = {"text":o, "attn": attention_array, "prob":prob, "repetition": repetition, "asr_token_num": asr_token_num}

            # print(query + o)
            torch.cuda.empty_cache()
            return [final_out]


        ### Think ###
        query = original_query

        think_prompt_list = [
            "Thinking process:\n", 
            "Detailed reasoning process:",
            "First, let me think about it systematically. Logical Steps:",
            "Firstly, I need to write my thought process:",
        ]

        query += think_prompt_list[think_prompt_num%len(think_prompt_list)]
        if tem == 0.0:
            gen_configs = dict(
                num_beams=1,
                max_new_tokens=4096,
                do_sample=False,
                output_attentions=True
            )
        else:
            gen_configs = dict(
                num_beams=1,
                max_new_tokens=4096,
                do_sample=True,
                temperature=tem,
                top_p=top_p,
                top_k=top_k,
                output_attentions=True
            )
        pixel_values = input["video"]
        num_patches_list = input["num_patches_list"]
        o, o_token_id, _, _, full_attention_array= self.chat(self.tokenizer, pixel_values, query, generation_config = gen_configs, num_patches_list=num_patches_list, check=False, doc=doc)      

        max_tokens_thinking_tmp = self.max_thinking_tokens
        max_tokens_thinking_tmp -= len(o_token_id)

        query = query + o

        for reflection_idx in range(self.num_reflection):
            if max_tokens_thinking_tmp <= 0:
                break
            reflection_prompt = "\nWait,"
            query = query + reflection_prompt
            o, o_token_id, _, _, full_attention_array= self.chat(self.tokenizer, pixel_values, query, generation_config = gen_configs, num_patches_list=num_patches_list, check=False, doc=doc)      
            max_tokens_thinking_tmp -= len(o_token_id)
            query = query + o
        
        ### Think -> Add key frames ###
        if key_frames != None:
            key_frame = key_frames[data_idx]
            key_frame_string = "\nHere are some key frames that may be helpful:"
            cnt = 1
            for frame in key_frame:
                added_image = input["video"][frame].unsqueeze(0)
                input["video"] = torch.cat([input["video"], added_image], dim=0)
                interval = 1/self.fps_list[0]
                time_now = round(frame* interval)
                mins = time_now // 60
                secs = time_now % 60
                time_now = f"{mins}:{secs:02d}"
                input["num_patches_list"].append(input["num_patches_list"][0])
                key_frame_string += f"Key Frame #{cnt}: Time {time_now}, <image>"
                cnt += 1

            key_frame_string += "Next, I need to focus on the visual content of key frames, and recheck my thinking process." 

            query = query + key_frame_string

        
        ### Final answer ###
        if "<OPEN-QUESTION>" in input['text']:
            query += "Final Answer:"
        else:
            query += """\nNext, I will conclude by stating the final answer using the following format: "Therefore, the final answer is: LETTER" (without quotes), where $LETTER is the final option of the question. I need to ensure the consistency between the final output answer and the thinking process. """
            query += "\nTherefore, the final answer is:"

        if key_frames != None or "<OPEN-QUESTION>" in input['text']:
            check = False
        else:
            check = True

        pixel_values = input["video"]
        num_patches_list = input["num_patches_list"]
        gen_configs = dict(
            num_beams=1,
            max_new_tokens=512,
            do_sample=False,
            output_attentions=True
        )
        o, o_token_id, asr_token_num, prob, attention_array = self.chat(self.tokenizer, pixel_values, query,\
                                                                         generation_config = gen_configs,\
                                                                         num_patches_list=num_patches_list, check=check,\
                                                                         last_step_attn=full_attention_array,doc=doc)
        output = query + o

        # print(output)
        
        repetition = detect_repetition(output[len(original_query):])

        contains_letter = lambda s: any('A' <= c <= 'J' for c in s)
        if not contains_letter(o):
            o = output[len(original_query):]

        final_out = {"text":o, "attn": attention_array,"prob":prob, "repetition":repetition, "asr_token_num": asr_token_num}

        torch.cuda.empty_cache()

        return [final_out]
    
    def process_query(self, tokenizer, question, IMG_CONTEXT_TOKEN='<IMG_CONTEXT>'):

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.model.img_context_token_id = img_context_token_id

        template = get_conv_template(self.model.template)
        template.system_message = self.model.system_message

        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()
        return query

    def chat(self, tokenizer, pixel_values, query, generation_config,
             num_patches_list=None, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>', IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
             check=False, doc=None, last_step_attn=None):

        if num_patches_list is None:
            num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
        assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.model.img_context_token_id = img_context_token_id
        template = get_conv_template(self.model.template)
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())

        for num_patches in num_patches_list:
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.model.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)

        model_inputs = tokenizer(query, return_tensors='pt')
        input_ids = model_inputs['input_ids'].to(self.model.device)
        attention_mask = model_inputs['attention_mask'].to(self.model.device)
        generation_config['eos_token_id'] = eos_token_id

        logits_list = []
        attentions_list = []

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

        # define forward hook to get logits and attentions
        def forward_hook(module, inputs, outputs):
            logits = outputs['logits'][:, -1, :][0]
            attention = outputs['attentions'][-1].squeeze(0)  # 28 1 len
            logits_list.append(logits.detach().cpu().clone())
            attentions_list.append(attention.detach().cpu().clone())
        
        hook_handle = self.model.language_model.register_forward_hook(forward_hook)

        generation_output = self.model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
        response = response.split(template.sep.strip())[0].strip()

        all_token_ids =torch.cat([input_ids[0], generation_output[0]], dim=0)

        idx_list = list(range(len(attentions_list))) 

        my_answer = self.process_out_list([response],doc[0])[0]
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
            last_step_attn = last_step_attn[:,:-1,:-1]
            refined_attn_map = np.zeros((last_step_attn.shape[0],last_step_attn.shape[1]+full_attention_array.shape[1],full_attention_array.shape[2]))
            refined_attn_map[:,0:last_step_attn.shape[1], 0:last_step_attn.shape[2]] = last_step_attn
            refined_attn_map[:, last_step_attn.shape[1]:, :] = full_attention_array
            full_attention_array = refined_attn_map

        # if check is True, calculate the probability of the predicted answer and the attention map
        if check and my_answer in answer_list:
            idx_list,prob_list = cal_answer_prob(logits_list, self.tokenizer, my_answer, generation_output[0])
            if idx_list == []:
                my_answer_refine = " " + my_answer
                idx_list,prob_list = cal_answer_prob(logits_list, self.tokenizer, my_answer_refine, generation_output[0])

            if type(last_step_attn)==np.ndarray:
                attention_array, asr_token_num = process_attention_2(full_attention_array, all_token_ids, idx_list, None, self.tokenizer, last_step_shape=last_step_attn.shape)
            else:                        
                attention_array, asr_token_num = process_attention(attentions_list, all_token_ids, idx_list, None, self.tokenizer)

            if len(prob_list)>0:
                prob = prob_list[-1]
            else:
                prob = None

            logits_list.clear()
            attentions_list.clear()
            hook_handle.remove()     
            return response, generation_output[0], asr_token_num, prob, attention_array
        
        logits_list.clear()
        attentions_list.clear()
        hook_handle.remove()

        return response, generation_output[0], 0, None, full_attention_array

    # custom forward function for attention
    def custom_forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask,
        past_key_value,
        cache_position,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        attn = self.model.language_model.model.layers[-1].self_attn
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, attn.head_dim)

        query_states = attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, attn.layer_idx, cache_kwargs)

        sliding_window = None
        if (
            attn.config.use_sliding_window
            and getattr(attn.config, "sliding_window", None) is not None
            and attn.layer_idx >= attn.config.max_window_layers
        ):
            sliding_window = attn.config.sliding_window

        from transformers.integrations.flash_attention import flash_attention_forward
        attention_interface = flash_attention_forward

        ## query_states : 1 28 L 128
        ## key_states: 1 4 L 128 

        import torch.nn.functional as F
        btz, num_heads, seq_len, head_dim = query_states.shape
        q = query_states[:,:,-1,:].reshape(btz, num_heads, 1, head_dim)
        k = repeat_kv(key_states, num_heads//key_states.shape[1])
        # v = repeat_kv(value_states, num_heads//value_states.shape[1])
        attn_scores = torch.matmul(q, k.transpose(-2, -1))
        attn_scores = attn_scores / torch.sqrt(torch.tensor(head_dim, dtype=torch.bfloat16))
        attn_weights_manual = F.softmax(attn_scores, dim=-1)   # 1, 28, 1, 128
        # attn_output_manual = torch.matmul(attn_weights_manual, v) 

        attn_output, attn_weights = attention_interface(
            attn,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not attn.training else attn.attention_dropout,
            scaling=attn.scaling,
            sliding_window=sliding_window,  # main diff with Llama
            **kwargs,
        )
        attn_weights = attn_weights_manual
        
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn.o_proj(attn_output)
        return attn_output, attn_weights



import matplotlib.pyplot as plt

def process_attention(collected_attns, all_token_ids, target_token_id_list, vis_path, tokenizer, save_fig=False):

    num_heads = collected_attns[0].shape[0]
    prompt_len = collected_attns[0].shape[2]
    num_tokens = len(collected_attns)

    video_token = tokenizer.encode('<img>')[0]  # <img> # internvl2.5 92544  # qwen: 151665
    video_token_indices = [i for i, token in enumerate(all_token_ids) if token == video_token]

    video_end_token = tokenizer.encode('</img>')[0]  # </img> # internvl2.5 92545  # qwen: 151666 
    video_token_end_indices = [i for i, token in enumerate(all_token_ids) if token == video_end_token]

    if target_token_id_list != [] and len(video_token_indices) >= 32:
        video_token_indices = video_token_indices[:32]
        target_token_id = target_token_id_list[-1]
    else:
        return None, 0

    target_token_id += (prompt_len-1)
    all_tokens = {}

    for i in range(len(all_token_ids)):
        all_tokens[i] = [all_token_ids[i].item(),tokenizer.decode([all_token_ids[i]], skip_special_tokens=False, clean_up_tokenization_spaces=False)]

    group_list = [0]

    for i in range(len(all_tokens)):
        if i+2 < len(all_tokens) and all_tokens[i][1].strip()=='Frame' and all_tokens[i+1][1].strip()=='1' and all_tokens[i+2][1].strip()==':' :
            group_list.append(i) # video start
            break

    for i in range(len(video_token_indices)):
        group_list.append(video_token_indices[i])  # 32 video tokens
    
    group_list.append(video_token_end_indices[31])  # video end

    asr_start = 0
    for i in range(len(all_tokens)):
        if i+2 < len(all_tokens) and all_tokens[i][1].strip()=='Audio' and all_tokens[i+1][1].strip()=='transcripts' and all_tokens[i+2][1].strip()=='of':
            asr_start = i
            group_list.append(i) # asr start
            break
    
    txt = tokenizer.decode(all_token_ids)
    pattern = r"Audio transcripts of the video:\n(.*?)\nQuestion:"
    match = re.search(pattern, txt, re.DOTALL)
    asr_token_num = 0
    if match:
        extracted_text = match.group(1)
        pattern = r"\[\d+:\d+ -> \d+:\d+\]:"
        time_segments = re.findall(pattern, extracted_text)
        segment_token_ids = [tokenizer.encode(seg, add_special_tokens=False) for seg in time_segments]
        for seg_ids in segment_token_ids:
            for i in range(asr_start, len(all_token_ids) - len(seg_ids) + 1):
                if list(all_token_ids[i:i + len(seg_ids)]) == seg_ids:
                    group_list.append(i)
                    asr_token_num +=1
                    break
    # print("asr segment num:",asr_token_num)

    for i in range(len(all_tokens)):
        if 'Question' in all_tokens[i][1] and ':' in all_tokens[i+1][1]:
            group_list.append(i)
            break
    
    for i in range(len(all_tokens)):
        if i+3 < len(all_tokens) and all_tokens[i][1]=='<|im_end|>' and all_tokens[i+2][1]=='<|im_start|>' and all_tokens[i+3][1]=='assistant' :
            group_list.append(i)
            break
    
    if target_token_id != prompt_len -1:
        group_list.append(prompt_len - 1)
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

    video_token = tokenizer.encode('<img>')[0]  # <img>
    video_token_indices = [i for i, token in enumerate(all_token_ids) if token == video_token]

    video_end_token = tokenizer.encode('</img>')[0]  # </img>
    video_token_end_indices = [i for i, token in enumerate(all_token_ids) if token == video_end_token]

    if target_token_id_list != []  and len(video_token_indices) >= 32:
        video_token_indices = video_token_indices[:32]
        target_token_id = target_token_id_list[-1]
    else:
        return None, 0

    all_tokens = {}
    for i in range(len(all_token_ids)):
        all_tokens[i] = [all_token_ids[i].item(),tokenizer.decode([all_token_ids[i]], skip_special_tokens=False, clean_up_tokenization_spaces=False)]
   
    target_token_id += (full_attention_map.shape[2]-(full_attention_map.shape[1]-last_step_shape[1]))

    group_list = [0]

    for i in range(len(all_tokens)):
        if i+2 < len(all_tokens) and all_tokens[i][1].strip()=='Frame' and all_tokens[i+1][1].strip()=='1' and all_tokens[i+2][1].strip()==':' :
            group_list.append(i) # video start
            break

    for i in range(len(video_token_indices)):
        group_list.append(video_token_indices[i])  # 32 video tokens
    
    group_list.append(video_token_end_indices[31])  # video end

    asr_start = 0
    for i in range(len(all_tokens)):
        if i+2 < len(all_tokens) and all_tokens[i][1].strip()=='Audio' and all_tokens[i+1][1].strip()=='transcripts' and all_tokens[i+2][1].strip()=='of':
            asr_start = i
            group_list.append(i) # asr start
            break
    
    txt = tokenizer.decode(all_token_ids)
    pattern = r"Audio transcripts of the video:\n(.*?)\nQuestion:"
    match = re.search(pattern, txt, re.DOTALL)
    asr_token_num = 0
    if match:
        extracted_text = match.group(1)
        pattern = r"\[\d+:\d+ -> \d+:\d+\]:"
        time_segments = re.findall(pattern, extracted_text)
        segment_token_ids = [tokenizer.encode(seg, add_special_tokens=False) for seg in time_segments]
        for seg_ids in segment_token_ids:
            for i in range(asr_start, len(all_token_ids) - len(seg_ids) + 1):
                if list(all_token_ids[i:i + len(seg_ids)]) == seg_ids:
                    group_list.append(i)
                    asr_token_num +=1
                    break

    for i in range(len(all_tokens)):
        if 'Question' in all_tokens[i][1] and ':' in all_tokens[i+1][1]:
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

        if save_fig:
            vis_path = "./logs/vis/s1/"
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
