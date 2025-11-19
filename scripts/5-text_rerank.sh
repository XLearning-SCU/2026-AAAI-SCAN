mkdir -p logs/text_rerank/clip-rn50
python 5-text_rerank.py --dataset flower   --backbone clip-rn50 --use_ensemble 0  2>&1 | tee logs/text_rerank/clip-rn50/flower.log
python 5-text_rerank.py --dataset cub      --backbone clip-rn50 --use_ensemble 0  2>&1 | tee logs/text_rerank/clip-rn50/cub.log
python 5-text_rerank.py --dataset pet      --backbone clip-rn50 --use_ensemble 0  2>&1 | tee logs/text_rerank/clip-rn50/pet.log
python 5-text_rerank.py --dataset food     --backbone clip-rn50 --use_ensemble 0  2>&1 | tee logs/text_rerank/clip-rn50/food.log
python 5-text_rerank.py --dataset aircraft --backbone clip-rn50 --use_ensemble 0  2>&1 | tee logs/text_rerank/clip-rn50/aircraft.log
python 5-text_rerank.py --dataset car      --backbone clip-rn50 --use_ensemble 0  2>&1 | tee logs/text_rerank/clip-rn50/car.log
python 5-text_rerank.py --dataset dog      --backbone clip-rn50 --use_ensemble 0  2>&1 | tee logs/text_rerank/clip-rn50/dog.log
python 5-text_rerank.py --dataset sun      --backbone clip-rn50 --use_ensemble 0  2>&1 | tee logs/text_rerank/clip-rn50/sun.log

mkdir -p logs/text_rerank/clip-rn50-ensemble
python 5-text_rerank.py --dataset flower   --backbone clip-rn50 --use_ensemble 1  2>&1 | tee logs/text_rerank/clip-rn50-ensemble/flower.log
python 5-text_rerank.py --dataset cub      --backbone clip-rn50 --use_ensemble 1  2>&1 | tee logs/text_rerank/clip-rn50-ensemble/cub.log
python 5-text_rerank.py --dataset pet      --backbone clip-rn50 --use_ensemble 1  2>&1 | tee logs/text_rerank/clip-rn50-ensemble/pet.log
python 5-text_rerank.py --dataset food     --backbone clip-rn50 --use_ensemble 1  2>&1 | tee logs/text_rerank/clip-rn50-ensemble/food.log
python 5-text_rerank.py --dataset aircraft --backbone clip-rn50 --use_ensemble 1  2>&1 | tee logs/text_rerank/clip-rn50-ensemble/aircraft.log
python 5-text_rerank.py --dataset car      --backbone clip-rn50 --use_ensemble 1  2>&1 | tee logs/text_rerank/clip-rn50-ensemble/car.log
python 5-text_rerank.py --dataset dog      --backbone clip-rn50 --use_ensemble 1  2>&1 | tee logs/text_rerank/clip-rn50-ensemble/dog.log
python 5-text_rerank.py --dataset sun      --backbone clip-rn50 --use_ensemble 1  2>&1 | tee logs/text_rerank/clip-rn50-ensemble/sun.log

mkdir -p logs/text_rerank/clip-b16
python 5-text_rerank.py --dataset flower   --backbone clip-b16 --use_ensemble 0  2>&1 | tee logs/text_rerank/clip-b16/flower.log
python 5-text_rerank.py --dataset cub      --backbone clip-b16 --use_ensemble 0  2>&1 | tee logs/text_rerank/clip-b16/cub.log
python 5-text_rerank.py --dataset pet      --backbone clip-b16 --use_ensemble 0  2>&1 | tee logs/text_rerank/clip-b16/pet.log
python 5-text_rerank.py --dataset food     --backbone clip-b16 --use_ensemble 0  2>&1 | tee logs/text_rerank/clip-b16/food.log
python 5-text_rerank.py --dataset aircraft --backbone clip-b16 --use_ensemble 0  2>&1 | tee logs/text_rerank/clip-b16/aircraft.log
python 5-text_rerank.py --dataset car      --backbone clip-b16 --use_ensemble 0  2>&1 | tee logs/text_rerank/clip-b16/car.log
python 5-text_rerank.py --dataset dog      --backbone clip-b16 --use_ensemble 0  2>&1 | tee logs/text_rerank/clip-b16/dog.log
python 5-text_rerank.py --dataset sun      --backbone clip-b16 --use_ensemble 0  2>&1 | tee logs/text_rerank/clip-b16/sun.log


mkdir -p logs/text_rerank/clip-b16-ensemble
python 5-text_rerank.py --dataset flower   --backbone clip-b16 --use_ensemble 1  2>&1 | tee logs/text_rerank/clip-b16-ensemble/flower.log
python 5-text_rerank.py --dataset cub      --backbone clip-b16 --use_ensemble 1  2>&1 | tee logs/text_rerank/clip-b16-ensemble/cub.log
python 5-text_rerank.py --dataset pet      --backbone clip-b16 --use_ensemble 1  2>&1 | tee logs/text_rerank/clip-b16-ensemble/pet.log
python 5-text_rerank.py --dataset food     --backbone clip-b16 --use_ensemble 1  2>&1 | tee logs/text_rerank/clip-b16-ensemble/food.log
python 5-text_rerank.py --dataset aircraft --backbone clip-b16 --use_ensemble 1  2>&1 | tee logs/text_rerank/clip-b16-ensemble/aircraft.log
python 5-text_rerank.py --dataset car      --backbone clip-b16 --use_ensemble 1  2>&1 | tee logs/text_rerank/clip-b16-ensemble/car.log
python 5-text_rerank.py --dataset dog      --backbone clip-b16 --use_ensemble 1  2>&1 | tee logs/text_rerank/clip-b16-ensemble/dog.log
python 5-text_rerank.py --dataset sun      --backbone clip-b16 --use_ensemble 1  2>&1 | tee logs/text_rerank/clip-b16-ensemble/sun.log