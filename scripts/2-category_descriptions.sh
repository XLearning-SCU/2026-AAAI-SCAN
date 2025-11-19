echo "Get category description"
mkdir -p logs/category_description
python 2-get_category_descriptions.py --dataset flower   2>&1 | tee logs/category_description/flower.log
python 2-get_category_descriptions.py --dataset cub      2>&1 | tee logs/category_description/cub.log
python 2-get_category_descriptions.py --dataset pet      2>&1 | tee logs/category_description/pet.log
python 2-get_category_descriptions.py --dataset food     2>&1 | tee logs/category_description/food.log
python 2-get_category_descriptions.py --dataset aircraft 2>&1 | tee logs/category_description/aircraft.log
python 2-get_category_descriptions.py --dataset car      2>&1 | tee logs/category_description/car.log
python 2-get_category_descriptions.py --dataset dog      2>&1 | tee logs/category_description/dog.log
python 2-get_category_descriptions.py --dataset sun      2>&1 | tee logs/category_description/sun.log