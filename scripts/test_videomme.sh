mkdir -p ./logs
export PYTHONPATH=/path/to/your/project/:$PYTHONPATH

NUM_GPUS=8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python ./test/test_videomme.py --exp_name 'videomme_bon' \
 --data_dir "/path/to/your/videomme_dataset/"\
 --model_path "/path/to/your/Qwen2.5-VL-7B-Instruct"\
 --asr_dir "/path/to/your/videomme_dataset/ASR_large/"\
 --models_per_gpu 2\
 --model_name bon\
 --theta 0.5\
 --temperature 0.0\
 --score_policy score_forest\
 --n 2 > ./logs/test_videomme_bon.log
