mkdir -p ./logs
export PYTHONPATH=/path/to/your/project/:$PYTHONPATH

NUM_GPUS=8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python ./test/test_worldsense.py --exp_name 'worldsense_bon' \
 --data_dir "/path/to/your/worldsense_dataset/"\
 --model_path "/path/to/your/Qwen2.5-VL-7B-Instruct"\
 --asr_dir "/path/to/your/worldsense_dataset/asr_large/"\
 --models_per_gpu 2\
 --model_name bon\
 --theta 0.7\
 --temperature 1.0\
 --score_policy score_forest\
 --n 16 > ./logs/test_worldsense_bon.log
