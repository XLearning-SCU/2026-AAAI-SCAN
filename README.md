# Endowing Vision-Language Models with System 2 Thinking for Fine-Grained Visual Recognition

- ##### **Authors:** Yutong Yang, Lifu Huang, [Yijie Lin](https://lin-yijie.github.io/),  [Xi Peng](https://pengxi.me/),  [Mouxing Yang](https://mouxingyang.github.io/) <br>

- **Accepted by AAAI 2026**

------


## Setup Preparation

- ### 1. Environment Setup

```
git clone https://github.com/XLearning-SCU/2026-AAAI-SCAN.git

conda create -n SCAN python=3.10.18
conda activate SCAN

cd 2026-AAAI-SCAN
pip install -r requirements.txt
```

- ### 2. Dataset Preparation

- To reproduce all the results in this paper, please download the datasets used in the experiments, such as [CUB_200_2011](https://data.caltech.edu/records/65de6-vp158), [Flowers-102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html), [Food-101](https://data.vision.ee.ethz.ch/cvl/food-101.tar.gz), [Oxford-Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/), [Stanford Cars](http://ai.stanford.edu/~jkrause/cars/car_dataset.html), [Stanford Dogs](http://vision.stanford.edu/aditya86/ImageNetDogs/), [FGVC Aircraft](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/), [SUN397](https://aistudio.baidu.com/datasetdetail/125762/0). Place all downloaded datasets into a unified directory (e.g., `dataset/`), and then update the corresponding paths in `configs/defaults.yaml`.

- Since some datasets do not provide category name files, we created a unified set of [`class_names.txt`](https://drive.google.com/drive/folders/18rEpz4gH4iOsOmLvJhuiCdr6Jr15fzVU?usp=sharing) files for all datasets. Please download them and place each file into the corresponding dataset directory.

  The structure of the processed datasets is as follows:

  ```
  dataset/
  ├── CUB_200_2011/
  │   ├── ...
  │   └── cub_class_names.txt
  ├── flowers-102/
  │   ├── ...
  │   └── flower_class_names.txt
  ├──...
  ```

- Run the `dataset_split.py` , and a `./test_sample` folder will be generated. This folder contains the test split for each dataset, saved in JSON format.
```
 python dataset_split.py
```

- ### 3. Download Modal

  [Qwen2.5-VL-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct), [Qwen3-Reranker-8B](https://huggingface.co/Qwen/Qwen3-Reranker-8B).



# Pipeline of SCAN

### 1. System-1 for Candidate Identification

First, obtain the Top-20 candidates predicted by CLIP. Run the following command, and a `backbone` folder will be created in the project directory. The Top-20 results of CUB dataset will be saved in JSON format under `./backbone/`

```
python 1-backbone.py --dataset cub --model clip-rn50 --use_ensemble 0 --topk 20
```

### 2. Concretization

After obtaining the discriminative attributes using GPT, we construct prompts based on the discriminative attribute set and **use GPT-4.1-mini to expand each category name into an attribute-aligned category description.** The prompt template for each dataset is defined in `./prompts/category_descriptions_prompt.json`

Before running this step, you must fill in your **OpenAI API key** in `configs/defaults.yaml` . Then, send batched requests to OpenAI by running

```
python 2-get_category_descriptions.py --dataset cub
```

**If you cannot access GPT, category description files for all datasets are already provided under `./category_descriptioins/gpt_output/`. You can parse these raw GPT outputs and save them as Excel files by running `./category_descriptioins/data_process.py`. **

### 3. Abstraction

We leverage Qwen2.5-VL-32B to abstract the image into an attribute-level textual description.

First, start the Qwen2.5-VL-32B on the server by running the following command:

```
CUDA_VISIBLE_DEVICES=4,5,6,7 \
$$Path to Anaconda$$/envs/SCAN/bin/python -m sglang.launch_server \
--model-path $$Path to$$/Qwen2.5-VL-32B-Instruct  \
--tp-size=4    \
--mem-fraction-static 0.8  \
--host "0.0.0.0"  \
--disable-cuda-graph  \
--port=12345
```

Then run the following commands to obtain the textual description and the initial reasoning prediction for each image using Qwen2.5-VL-32B.

```
python 3-get_image_features.py --dataset cub --lmm Qwen2.5-VL-32B
python 4-get_lmm_inference.py --dataset cub --model clip-rn50 --use_ensemble 0 --topk 20 --lmm Qwen2.5-VL-32B
```

### 4. Text Rerank

We use Qwen3-Reranker-8B to calculate the matching degree between the image description and each candidate category description.

```
python 5-text_rerank.py --dataset cub --backbone clip-rn50 --use_ensemble 0
```

### 5. Uncertainty-aware Integration

The final recognition results could be obtained by the following command.

```
python 6-fusion.py --dataset cub --backbone clip-rn50 --use_ensemble 0
```

---
# Resource

Due to GitHub’s 25 MB file size limit, all intermediate files are provided via external download links:

After Step 1, you can obtain the CLIP Top-20 prediction results.
Download here: [Google Drive](https://drive.google.com/drive/folders/1xKvyzsz8CwZp1HLJO62vYKwWyszYwX5Y?usp=sharing)

After Step 2, you can obtain the category descriptions
Download here: [Google Drive](https://drive.google.com/drive/folders/1GOnaDBgCeC6hACcNCtue9cxqI7tZrXI_?usp=sharing)

After Step 3, you can obtain the textual descriptions generated for each image.
Download here: [Google Drive](https://drive.google.com/drive/folders/1oYxdY5Ec1UhzOsUyDio6JS_ecHdlpdIg?usp=sharing)
After Step 4, you can obtain the reranked similarity scores.
Download here: [Google Drive](https://drive.google.com/drive/folders/1lRMUfiqBgJFuGz02YCD_boX-Sr5mT_xe?usp=sharing)

After downloading all files from Steps 1 and 4, run:
‘’‘
./scripts/6-fusion.sh
’‘’
to reproduce the results reported in Table 1 of the paper.

alternative link: [modelscope](https://modelscope.cn/datasets/XLearning/SCAN/files), [huggingface](https://huggingface.co/datasets/XLearning-SCU/AAAI-SCAN)


# Bibtex
If you find this repository helpful, please consider giving it a ⭐️ and citing our work — your support is greatly appreciated!

```
@inproceedings{ding2025visual,
  title={Endowing Vision-Language Models with System 2 Thinking for Fine-grained Visual Recognition},
  author={Yang, Yutong and Huang, Lifu and Lin, Yijie and Peng, Xi and Yang, Mouxing},
  booktitle={AAAI},
  year={2026},
}
```




