mkdir -p logs/image_features
python 3-get_image_features.py --dataset flower --lmm Qwen2.5-VL-32B 2>&1 | tee logs/image_features/flower-32B.log
python 3-get_image_features.py --dataset cub --lmm Qwen2.5-VL-32B 2>&1 | tee logs/image_features/cub-32B.log
python 3-get_image_features.py --dataset pet --lmm Qwen2.5-VL-32B 2>&1 | tee logs/image_features/pet-32B.log
python 3-get_image_features.py --dataset food --lmm Qwen2.5-VL-32B 2>&1 | tee logs/image_features/food-32B.log
python 3-get_image_features.py --dataset aircraft --lmm Qwen2.5-VL-32B 2>&1 | tee logs/image_features/aircraft-32B.log
python 3-get_image_features.py --dataset car --lmm Qwen2.5-VL-32B 2>&1 | tee logs/image_features/car-32B.log
python 3-get_image_features.py --dataset dog --lmm Qwen2.5-VL-32B 2>&1 | tee logs/image_features/dog-32B.log
python 3-get_image_features.py --dataset sun --lmm Qwen2.5-VL-32B 2>&1 | tee logs/image_features/sun-32B.log