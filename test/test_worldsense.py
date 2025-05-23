import os
os.environ["DECORD_EOF_RETRY_MAX"] = "40960"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
import json
from datetime import datetime
from dataloader.worldsense import WorldSense_Bench, dump, worldsense_process_result, process_out_list
import argparse
import multiprocessing
import time
import pandas as pd
from utils import  deep_copy_complex_list, select_output


def parse_args():
    parser = argparse.ArgumentParser(description="Model Configuration Parameters")

    # Basic Paths
    parser.add_argument("--data_dir", type=str, default="/path/to/your/data",
                        help="Path to the data directory.")
    parser.add_argument("--model_path", type=str, default="/path/to/your/model",
                        help="Path to the model.")
    parser.add_argument("--asr_dir", type=str, default="/path/to/your/asr_data",
                        help="Path to the ASR data directory.")
    parser.add_argument("--exp_name", type=str, default="debug",
                        help="Experiment name.")

    # Model Configuration
    parser.add_argument("--score_policy", type=str, default="score_forest")
    parser.add_argument("--model_name", type=str, default="bon",
                        choices=['base', 'cot', 'bon'], help="Model name.")
    parser.add_argument("--models_per_gpu",type=int, default=2)
    parser.add_argument("--theta", type=float, default=0.7)
    parser.add_argument("--disable_asr", action="store_true", help="Whether to add ASR information.")
    parser.add_argument("--n", type=int, default=16,
                        help="Number of candidates for best-of-n.")  
    parser.add_argument("--n_loop2", type=int, default=8)
    parser.add_argument("--num_reflection", type=int, default=0)  
    parser.add_argument("--max_frames", type=int, default=64,
                        help="Number of frames to sample from the video.")      
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature of models.")  

    args = parser.parse_args()
    return args

def get_cuda_visible_devices():
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not cuda_visible_devices:
        return []  
    gpu_list = [int(gpu_id.strip()) for gpu_id in cuda_visible_devices.split(",") if gpu_id.strip()]
    return gpu_list

def build_model(model_path, max_frames, num_reflection):
    
    from model_transformers import QwenVL_Transformers
    model = QwenVL_Transformers(model_path, process_out_list, max_thinking_tokens=4096, max_frames=max_frames, num_reflection=num_reflection)
    return model


def evaluate_chunk(video_paths, image, text_input, docs, gpu_id, args, queue):
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        print(f"[GPU {str(gpu_id)}] Processing {len(video_paths)} examples.")

        results = []

        bs = 1  # Batch size

        model_bon = build_model(
            args.model_path,
            args.max_frames,
            args.num_reflection
        )

        def process_batch(batch_video_paths, batch_text_input, batch_image, batch_doc):

            if args.model_name=='base':
                output = model_bon(batch_video_paths, batch_text_input, query_image=batch_image, doc=batch_doc, model_type='origin')

            elif args.model_name=='cot':
                output = model_bon(batch_video_paths, batch_text_input, query_image=batch_image, doc=batch_doc, tem=args.temperature)
            
            ## BoN
            else:
                #### The first round ####
                #### MLLM Inference System ####
                inputs = model_bon.get_batch_messages(batch_video_paths, batch_text_input, batch_image)
                use_frames = len(inputs[0]['video'])==64
                pred_list = []
                output_list = []
                
                N = args.n

                # First, get the output of the base model.
                input1 = deep_copy_complex_list(inputs)
                output1 = model_bon(batch_video_paths, batch_text_input, query_image=batch_image, doc=batch_doc, inputs=input1, model_type='origin')
                del input1
                pred_list.append(process_out_list([output1[0]["text"]],batch_doc[0])[0])
                output_list.append(output1)

                # Then, get the output of the cot model.
                for i in range(N-1):
                    input2 = deep_copy_complex_list(inputs)
                    output2 = model_bon(batch_video_paths, batch_text_input, query_image=batch_image, doc=batch_doc, inputs=input2, think_prompt_num=i, tem=args.temperature, top_p=0.5, top_k=5)
                    del input2
                    pred_list.append(process_out_list([output2[0]["text"]],batch_doc[0])[0])
                    output_list.append(output2)

                # There are some signals in the output_list monitored by the Sensor.
                #### Sensor and Controller ####
                output = select_output(output_list, pred_list, text=inputs[0]["text"], select_method=args.score_policy,\
                                    max_frames=args.max_frames, fps=model_bon.fps_list[0], n=N, theta=args.theta) 

                #### Need to self-correction -> The second round ####
                if type(output)==tuple and use_frames:
                    print(output[0])
                    key_frames = output[1]
                    
                    reflect_kf_n = args.n_loop2

                    ## n = 1 in the second round

                    if reflect_kf_n == 1:
                        output_keyframe = model_bon(batch_video_paths, batch_text_input, query_image=batch_image, doc=batch_doc, key_frames=key_frames, inputs=inputs, model_type='origin',tem=0.01, top_p=0.5, top_k=5)
                        pred = process_out_list([output_keyframe[0]["text"]],batch_doc[0])[0]
                        print(pred, pred_list)

                        if len(pred_list) == 2:
                            if pred == "No Answer Found":
                                if pred_list[1] == "No Answer Found":
                                    output = output_list[0]
                                else:
                                    output = output_list[1]
                            else:
                                output = output_keyframe
                        else:
                            ### BoN
                            if pred not in pred_list or pred == "No Answer Found":
                                output = output[2] 
                            else:
                                output = output_keyframe

                    ## n > 1 in the second round
                    else:

                        output_keyframe_list = []
                        pred_keyframe_list = []

                        for i in range(reflect_kf_n):
                            tmp_inputs = deep_copy_complex_list(inputs)
                            import math, random
                            tmp_key_frames = sorted(random.sample(key_frames[0], math.ceil(len(key_frames[0]) / 2)))
                            output_keyframe = model_bon(batch_video_paths, batch_text_input, query_image=batch_image, doc=batch_doc, key_frames=[tmp_key_frames], inputs=tmp_inputs, model_type='origin', tem=args.temperature, top_p=0.5, top_k=5)
                            output_keyframe_list.append(output_keyframe)
                            pred_keyframe_list.append(process_out_list([output_keyframe[0]["text"]],batch_doc[0])[0])
                            del tmp_inputs

                        print(pred_list, pred_keyframe_list)

                        output = select_output(output_keyframe_list, pred_keyframe_list, text=inputs[0]["text"], select_method=args.score_policy,\
                                                max_frames=args.max_frames, fps=model_bon.fps_list[0], n=reflect_kf_n, theta=0.0, topk=5, all_origin=True)

                elif type(output)==tuple and not use_frames:
                    output = output_list[0]

            print(batch_doc[0]["video"], "GT:", batch_doc[0]["answer"], "Pred:", output[0]["text"])
            for out in output:
                results.append(out["text"])

        idx = 0
        while idx < len(video_paths):
            start = time.time()
            batch_size = min(bs, len(video_paths) - idx)
            batch_video_paths = video_paths[idx:idx+batch_size]
            batch_text_input = text_input[idx:idx+batch_size]
            batch_image = image[idx:idx+batch_size]
            batch_doc = docs[idx:idx+batch_size]

            process_batch(batch_video_paths, batch_text_input, batch_image, batch_doc)
            idx += batch_size
            print(f"GPU ID:{gpu_id},{idx}/{len(video_paths)}")

        df = pd.DataFrame(docs)
        df['prediction'] = df.index.map(lambda x: results[x])
        queue.put((df, None))
        print(f"[GPU {gpu_id}] Finished processing.")       

    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        queue.put((None, error_msg))


def evaluate(args, num_gpus, gpu_list):
    
    worldsense = WorldSense_Bench(args.data_dir, add_asr=not args.disable_asr, asr_dir=args.asr_dir)
    if len(gpu_list)==0:
        gpu_list = list(range(num_gpus))

    video_paths, image, text_input, docs = worldsense.get_data()
    total = len(video_paths)

    models_per_gpu = args.models_per_gpu
    gpu_list_new = [gpu_list[i] for i in range(len(gpu_list)) for j in range(models_per_gpu)]
    gpu_list = gpu_list_new

    print(f"Total examples: {total}")
    print(f"Number of GPUs: {num_gpus}")
    print(f"GPU list: {gpu_list}")

    num_gpus = num_gpus*models_per_gpu

    vis_dir="./logs/"
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    result_file_base = f'{args.model_name}_{args.exp_name}.xlsx'

    chunk_size = (total + num_gpus - 1) // num_gpus  # ceiling division to cover all examples
    chunks = []
    
    for i in range(num_gpus):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, total)
        if start >= end:
            break
        chunks.append((
            video_paths[start:end],
            image[start:end],
            text_input[start:end],
            docs[start:end]
        ))
    
    queue = multiprocessing.Queue()
    processes = []

    for i, (vp_chunk, img_chunk, txt_chunk, docs_chunk) in enumerate(chunks):
        p = multiprocessing.Process(
            target=evaluate_chunk,
            args=(vp_chunk, img_chunk, txt_chunk, docs_chunk, gpu_list[i], args, queue)
        )
        p.start()
        processes.append(p)
    
    # Collect the results from each process.
    all_df = []
    for _ in processes:
        df, error = queue.get()
        if error is not None:
            print(f"子进程出错: {error}")
            for p in processes:
                p.terminate()
            exit(1)
        all_df.append(df)

    for p in processes:
        p.join()

    result_file = os.path.join(vis_dir, result_file_base)

    df_final = pd.concat(all_df, ignore_index=True)

    dump(df_final, result_file)
    metrics = worldsense_process_result(result_file)

    queue.close()
    
    return metrics

def main():
    print("Start Time:",datetime.now())
    args = parse_args()
    print(args)

    num_gpus = int(os.getenv("NUM_GPUS"))
    gpu_list = get_cuda_visible_devices()

    metrics= evaluate(args,num_gpus=num_gpus, gpu_list=gpu_list)
    print(metrics)

    print("test finished")
    print("End Time:",datetime.now())


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    main()