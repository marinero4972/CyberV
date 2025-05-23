mkdir -p ./logs
export PYTHONPATH=/path/to/your/project/:$PYTHONPATH

NUM_GPUS=8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python ./test/test_videommmu.py --exp_name 'videommmu_bon' \
 --data_dir '/path/to/your/video_mmmu_dataset/'\
 --model_path '/data/mjh/pretrained_models/Qwen2.5-VL-7B-Instruct'\
 --asr_dir '/path/to/your/video_mmmu_dataset/ASR_large/'\
 --models_per_gpu 2\
 --model_name bon\
 --theta 0.3\
 --temperature 1.0\
 --score_policy score_forest\
 --n 8 > ./logs/test_videommmu_bon.log
