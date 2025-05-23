import os
os.environ["DECORD_EOF_RETRY_MAX"] = "40960"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
import json
from datetime import datetime
from dataloader.videomme import  VideoMME_Bench, videomme_aggregate_results, videomme_process_results_new, process_out_list
import argparse
import multiprocessing
from utils import deep_copy_complex_list, select_output

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
    parser.add_argument("--theta", type=float, default=0.5)
    parser.add_argument("--disable_asr", action="store_true", help="Whether to add ASR information.")
    parser.add_argument("--n", type=int, default=2,
                        help="Number of candidates for best-of-n.")
    parser.add_argument("--num_reflection", type=int, default=0)  
    parser.add_argument("--max_frames", type=int, default=64,
                        help="Number of frames to sample from the video.")
    parser.add_argument("--temperature", type=float, default=0.0,
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
        metrics = []
       
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
            
            ### BoN
            else:
                #### The first round ####
                #### MLLM Inference System ####

                # First, get the output of the base model.
                inputs = model_bon.get_batch_messages(batch_video_paths, batch_text_input, batch_image)

                pred_list = []
                output_list = []
                
                N = args.n

                input1 = deep_copy_complex_list(inputs)
                output1 = model_bon(batch_video_paths, batch_text_input, query_image=batch_image, doc=batch_doc, inputs=input1, model_type='origin')
                del input1
                pred_list.append(process_out_list([output1[0]["text"]],batch_doc[0])[0])
                output_list.append(output1)

                # Then, get the output of the cot model.
                for i in range(N-1):
                    input2 = deep_copy_complex_list(inputs)
                    output2 = model_bon(batch_video_paths, batch_text_input, query_image=batch_image, doc=batch_doc, inputs=input2, tem=args.temperature, top_p=0.5, top_k=5)
                    del input2
                    pred_list.append(process_out_list([output2[0]["text"]],batch_doc[0])[0])
                    output_list.append(output2)

                #### Sensor and Controller ####
                output = select_output(output_list, pred_list, text=inputs[0]["text"], select_method=args.score_policy,\
                                max_frames=args.max_frames, fps=model_bon.fps_list[0], n=N, theta=args.theta)   # majority_sampling, adaptive_with_score_v2

                #### Need to self-correction -> The second round ####
                if type(output)==tuple:
                    # add key frames!
                    print(output[0])
                    key_frames = output[1]
                    output_keyframe = model_bon(batch_video_paths, batch_text_input, query_image=batch_image, doc=batch_doc, key_frames=key_frames, inputs=inputs, model_type='origin', tem=0.01, top_p=0.5, top_k=5)
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

            print(batch_doc[0]["videoID"], "GT:", batch_doc[0]["answer"], "Pred:", output[0]["text"])
            for out in output:
                results.append(out["text"])

        idx = 0
        while idx < len(video_paths):
            batch_size = min(bs, len(video_paths) - idx)
            batch_video_paths = video_paths[idx:idx+batch_size]
            batch_text_input = text_input[idx:idx+batch_size]
            batch_image = image[idx:idx+batch_size]
            batch_doc = docs[idx:idx+batch_size]

            process_batch(batch_video_paths, batch_text_input, batch_image, batch_doc)
            idx += batch_size
            print(f"GPU ID:{gpu_id},{idx}/{len(video_paths)}")

        metrics = [videomme_process_results_new(docs[i], results[i]) for i in range(len(docs))]
        queue.put((metrics, results, None))
        print(f"[GPU {gpu_id}] Finished processing.")
    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        queue.put((None, None, error_msg))
    
def evaluate(args, num_gpus, gpu_list):
    
    videomme = VideoMME_Bench(args.data_dir, add_asr=not args.disable_asr, asr_dir=args.asr_dir)

    if len(gpu_list)==0:
        gpu_list = list(range(num_gpus))

    video_paths, image, text_input, docs = videomme.get_data()
    total = len(video_paths)

    models_per_gpu = args.models_per_gpu
    gpu_list_new = [gpu_list[i] for i in range(len(gpu_list)) for j in range(models_per_gpu)]
    gpu_list = gpu_list_new

    print(f"Total examples: {total}")
    print(f"Number of GPUs: {num_gpus}")
    print(f"GPU list: {gpu_list}")

    num_gpus = num_gpus*models_per_gpu

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
    all_metrics = []
    all_results = []
    for _ in processes:
        metrics, results, error = queue.get()
        if error is not None:
            print(f"子进程出错: {error}")
            for p in processes:
                p.terminate()
            exit(1)
        all_metrics.extend(metrics)
        all_results.extend(results)

    for p in processes:
        p.join()
    
    acc = videomme_aggregate_results(all_metrics)
    print("Final accuracy:", acc)

    queue.close()
    
    return all_metrics, all_results



def main():
    print("Start Time:",datetime.now())
    args = parse_args()
    print(args)

    num_gpus = int(os.getenv("NUM_GPUS"))
    gpu_list = get_cuda_visible_devices()

    metrics, results = evaluate(args,num_gpus=num_gpus, gpu_list=gpu_list)
    
    metrics_path = f'./logs/metrics_{args.exp_name}.json'
    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=4)
        
    results_path = f'./logs/results_{args.exp_name}.json'
    with open(results_path, "w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=4)

    print("End Time:",datetime.now())



if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    main()