echo "Start Experiment | Model: CLIP-RN50 | Ensemble Prompt: No"
mkdir -p logs/backbone/clip-rn50
python 1-backbone.py --dataset flower   --model clip-rn50 --use_ensemble 0 --topk 20 2>&1 | tee -a logs/backbone/clip-rn50/flower.log
python 1-backbone.py --dataset cub      --model clip-rn50 --use_ensemble 0 --topk 20 2>&1 | tee -a logs/backbone/clip-rn50/cub.log
python 1-backbone.py --dataset pet      --model clip-rn50 --use_ensemble 0 --topk 20 2>&1 | tee -a logs/backbone/clip-rn50/pet.log
python 1-backbone.py --dataset food     --model clip-rn50 --use_ensemble 0 --topk 20 2>&1 | tee -a logs/backbone/clip-rn50/food.log
python 1-backbone.py --dataset aircraft --model clip-rn50 --use_ensemble 0 --topk 20 2>&1 | tee -a logs/backbone/clip-rn50/aircraft.log
python 1-backbone.py --dataset car      --model clip-rn50 --use_ensemble 0 --topk 20 2>&1 | tee -a logs/backbone/clip-rn50/car.log
python 1-backbone.py --dataset dog      --model clip-rn50 --use_ensemble 0 --topk 20 2>&1 | tee -a logs/backbone/clip-rn50/dog.log
python 1-backbone.py --dataset sun      --model clip-rn50 --use_ensemble 0 --topk 20 2>&1 | tee -a logs/backbone/clip-rn50/sun.log

echo "Start Experiment | Model: CLIP-RN50 | Ensemble Prompt: Yes"
mkdir -p logs/backbone/clip-rn50-ensemble
python 1-backbone.py --dataset flower   --model clip-rn50 --use_ensemble 1 --topk 20 2>&1 | tee -a logs/backbone/clip-rn50-ensemble/flower.log
python 1-backbone.py --dataset cub      --model clip-rn50 --use_ensemble 1 --topk 20 2>&1 | tee -a logs/backbone/clip-rn50-ensemble/cub.log
python 1-backbone.py --dataset pet      --model clip-rn50 --use_ensemble 1 --topk 20 2>&1 | tee -a logs/backbone/clip-rn50-ensemble/pet.log
python 1-backbone.py --dataset food     --model clip-rn50 --use_ensemble 1 --topk 20 2>&1 | tee -a logs/backbone/clip-rn50-ensemble/food.log
python 1-backbone.py --dataset aircraft --model clip-rn50 --use_ensemble 1 --topk 20 2>&1 | tee -a logs/backbone/clip-rn50-ensemble/aircraft.log
python 1-backbone.py --dataset car      --model clip-rn50 --use_ensemble 1 --topk 20 2>&1 | tee -a logs/backbone/clip-rn50-ensemble/car.log
python 1-backbone.py --dataset dog      --model clip-rn50 --use_ensemble 1 --topk 20 2>&1 | tee -a logs/backbone/clip-rn50-ensemble/dog.log
python 1-backbone.py --dataset sun      --model clip-rn50 --use_ensemble 1 --topk 20 2>&1 | tee -a logs/backbone/clip-rn50-ensemble/sun.log

echo "Start Experiment | Model: CLIP-B16 | Ensemble Prompt: No"
mkdir -p logs/backbone/clip-b16
python 1-backbone.py --dataset flower   --model clip-b16 --use_ensemble 0 --topk 20 2>&1 | tee -a logs/backbone/clip-b16/flower.log
python 1-backbone.py --dataset cub      --model clip-b16 --use_ensemble 0 --topk 20 2>&1 | tee -a logs/backbone/clip-b16/cub.log
python 1-backbone.py --dataset pet      --model clip-b16 --use_ensemble 0 --topk 20 2>&1 | tee -a logs/backbone/clip-b16/pet.log
python 1-backbone.py --dataset food     --model clip-b16 --use_ensemble 0 --topk 20 2>&1 | tee -a logs/backbone/clip-b16/food.log
python 1-backbone.py --dataset aircraft --model clip-b16 --use_ensemble 0 --topk 20 2>&1 | tee -a logs/backbone/clip-b16/aircraft.log
python 1-backbone.py --dataset car      --model clip-b16 --use_ensemble 0 --topk 20 2>&1 | tee -a logs/backbone/clip-b16/car.log
python 1-backbone.py --dataset dog      --model clip-b16 --use_ensemble 0 --topk 20 2>&1 | tee -a logs/backbone/clip-b16/dog.log
python 1-backbone.py --dataset sun      --model clip-b16 --use_ensemble 0 --topk 20 2>&1 | tee -a logs/backbone/clip-b16/sun.log

echo "Start Experiment | Model: CLIP-B16 | Ensemble Prompt: Yes"
mkdir -p logs/backbone/clip-b16-ensemble
python 1-backbone.py --dataset flower   --model clip-b16 --use_ensemble 1 --topk 20 2>&1 | tee -a logs/backbone/clip-b16-ensemble/flower.log
python 1-backbone.py --dataset cub      --model clip-b16 --use_ensemble 1 --topk 20 2>&1 | tee -a logs/backbone/clip-b16-ensemble/cub.log
python 1-backbone.py --dataset pet      --model clip-b16 --use_ensemble 1 --topk 20 2>&1 | tee -a logs/backbone/clip-b16-ensemble/pet.log
python 1-backbone.py --dataset food     --model clip-b16 --use_ensemble 1 --topk 20 2>&1 | tee -a logs/backbone/clip-b16-ensemble/food.log
python 1-backbone.py --dataset aircraft --model clip-b16 --use_ensemble 1 --topk 20 2>&1 | tee -a logs/backbone/clip-b16-ensemble/aircraft.log
python 1-backbone.py --dataset car      --model clip-b16 --use_ensemble 1 --topk 20 2>&1 | tee -a logs/backbone/clip-b16-ensemble/car.log
python 1-backbone.py --dataset dog      --model clip-b16 --use_ensemble 1 --topk 20 2>&1 | tee -a logs/backbone/clip-b16-ensemble/dog.log
python 1-backbone.py --dataset sun      --model clip-b16 --use_ensemble 1 --topk 20 2>&1 | tee -a logs/backbone/clip-b16-ensemble/sun.log
