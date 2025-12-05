mkdir -p logs/lmm_inference/clip-rn50
python 4-get_lmm_inference.py --dataset flower   --model clip-rn50 --use_ensemble 0 --topk 20 --lmm Qwen2.5-VL-32B 2>&1 | tee logs/lmm_inference/clip-rn50/flower.log
python 4-get_lmm_inference.py --dataset cub      --model clip-rn50 --use_ensemble 0 --topk 20 --lmm Qwen2.5-VL-32B 2>&1 | tee logs/lmm_inference/clip-rn50/cub.log
python 4-get_lmm_inference.py --dataset pet      --model clip-rn50 --use_ensemble 0 --topk 20 --lmm Qwen2.5-VL-32B 2>&1 | tee logs/lmm_inference/clip-rn50/pet.log
python 4-get_lmm_inference.py --dataset food     --model clip-rn50 --use_ensemble 0 --topk 20 --lmm Qwen2.5-VL-32B 2>&1 | tee logs/lmm_inference/clip-rn50/food.log
python 4-get_lmm_inference.py --dataset aircraft --model clip-rn50 --use_ensemble 0 --topk 20 --lmm Qwen2.5-VL-32B 2>&1 | tee logs/lmm_inference/clip-rn50/aircraft.log
python 4-get_lmm_inference.py --dataset car      --model clip-rn50 --use_ensemble 0 --topk 20 --lmm Qwen2.5-VL-32B 2>&1 | tee logs/lmm_inference/clip-rn50/car.log
python 4-get_lmm_inference.py --dataset dog      --model clip-rn50 --use_ensemble 0 --topk 20 --lmm Qwen2.5-VL-32B 2>&1 | tee logs/lmm_inference/clip-rn50/dog.log
python 4-get_lmm_inference.py --dataset sun      --model clip-rn50 --use_ensemble 0 --topk 20 --lmm Qwen2.5-VL-32B 2>&1 | tee logs/lmm_inference/clip-rn50/sun.log

mkdir -p logs/lmm_inference/clip-rn50-ensemble
python 4-get_lmm_inference.py --dataset flower   --model clip-rn50 --use_ensemble 1 --topk 20 --lmm Qwen2.5-VL-32B 2>&1 | tee logs/lmm_inference/clip-rn50-ensemble/flower.log
python 4-get_lmm_inference.py --dataset cub      --model clip-rn50 --use_ensemble 1 --topk 20 --lmm Qwen2.5-VL-32B 2>&1 | tee logs/lmm_inference/clip-rn50-ensemble/cub.log
python 4-get_lmm_inference.py --dataset pet      --model clip-rn50 --use_ensemble 1 --topk 20 --lmm Qwen2.5-VL-32B 2>&1 | tee logs/lmm_inference/clip-rn50-ensemble/pet.log
python 4-get_lmm_inference.py --dataset food     --model clip-rn50 --use_ensemble 1 --topk 20 --lmm Qwen2.5-VL-32B 2>&1 | tee logs/lmm_inference/clip-rn50-ensemble/food.log
python 4-get_lmm_inference.py --dataset aircraft --model clip-rn50 --use_ensemble 1 --topk 20 --lmm Qwen2.5-VL-32B 2>&1 | tee logs/lmm_inference/clip-rn50-ensemble/aircraft.log
python 4-get_lmm_inference.py --dataset car      --model clip-rn50 --use_ensemble 1 --topk 20 --lmm Qwen2.5-VL-32B 2>&1 | tee logs/lmm_inference/clip-rn50-ensemble/car.log
python 4-get_lmm_inference.py --dataset dog      --model clip-rn50 --use_ensemble 1 --topk 20 --lmm Qwen2.5-VL-32B 2>&1 | tee logs/lmm_inference/clip-rn50-ensemble/dog.log
python 4-get_lmm_inference.py --dataset sun      --model clip-rn50 --use_ensemble 1 --topk 20 --lmm Qwen2.5-VL-32B 2>&1 | tee logs/lmm_inference/clip-rn50-ensemble/sun.log

mkdir -p logs/lmm_inference/clip-b16
python 4-get_lmm_inference.py --dataset flower   --model clip-b16 --use_ensemble 0 --topk 20 --lmm Qwen2.5-VL-32B 2>&1 | tee logs/lmm_inference/clip-b16/flower.log
python 4-get_lmm_inference.py --dataset cub      --model clip-b16 --use_ensemble 0 --topk 20 --lmm Qwen2.5-VL-32B 2>&1 | tee logs/lmm_inference/clip-b16/cub.log
python 4-get_lmm_inference.py --dataset pet      --model clip-b16 --use_ensemble 0 --topk 20 --lmm Qwen2.5-VL-32B 2>&1 | tee logs/lmm_inference/clip-b16/pet.log
python 4-get_lmm_inference.py --dataset food     --model clip-b16 --use_ensemble 0 --topk 20 --lmm Qwen2.5-VL-32B 2>&1 | tee logs/lmm_inference/clip-b16/food.log
python 4-get_lmm_inference.py --dataset aircraft --model clip-b16 --use_ensemble 0 --topk 20 --lmm Qwen2.5-VL-32B 2>&1 | tee logs/lmm_inference/clip-b16/aircraft.log
python 4-get_lmm_inference.py --dataset car      --model clip-b16 --use_ensemble 0 --topk 20 --lmm Qwen2.5-VL-32B 2>&1 | tee logs/lmm_inference/clip-b16/car.log
python 4-get_lmm_inference.py --dataset dog      --model clip-b16 --use_ensemble 0 --topk 20 --lmm Qwen2.5-VL-32B 2>&1 | tee logs/lmm_inference/clip-b16/dog.log
python 4-get_lmm_inference.py --dataset sun      --model clip-b16 --use_ensemble 0 --topk 20 --lmm Qwen2.5-VL-32B 2>&1 | tee logs/lmm_inference/clip-b16/sun.log


mkdir -p logs/lmm_inference/clip-b16-ensemble
python 4-get_lmm_inference.py --dataset flower   --model clip-b16 --use_ensemble 1 --topk 20 --lmm Qwen2.5-VL-32B 2>&1 | tee logs/lmm_inference/clip-b16-ensemble/flower.log
python 4-get_lmm_inference.py --dataset cub      --model clip-b16 --use_ensemble 1 --topk 20 --lmm Qwen2.5-VL-32B 2>&1 | tee logs/lmm_inference/clip-b16-ensemble/cub.log
python 4-get_lmm_inference.py --dataset pet      --model clip-b16 --use_ensemble 1 --topk 20 --lmm Qwen2.5-VL-32B 2>&1 | tee logs/lmm_inference/clip-b16-ensemble/pet.log
python 4-get_lmm_inference.py --dataset food     --model clip-b16 --use_ensemble 1 --topk 20 --lmm Qwen2.5-VL-32B 2>&1 | tee logs/lmm_inference/clip-b16-ensemble/food.log
python 4-get_lmm_inference.py --dataset aircraft --model clip-b16 --use_ensemble 1 --topk 20 --lmm Qwen2.5-VL-32B 2>&1 | tee logs/lmm_inference/clip-b16-ensemble/aircraft.log
python 4-get_lmm_inference.py --dataset car      --model clip-b16 --use_ensemble 1 --topk 20 --lmm Qwen2.5-VL-32B 2>&1 | tee logs/lmm_inference/clip-b16-ensemble/car.log
python 4-get_lmm_inference.py --dataset dog      --model clip-b16 --use_ensemble 1 --topk 20 --lmm Qwen2.5-VL-32B 2>&1 | tee logs/lmm_inference/clip-b16-ensemble/dog.log
python 4-get_lmm_inference.py --dataset sun      --model clip-b16 --use_ensemble 1 --topk 20 --lmm Qwen2.5-VL-32B 2>&1 | tee logs/lmm_inference/clip-b16-ensemble/sun.log
