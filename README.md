# Endowing Vision-Language Models with System 2 Thinking for Fine-Grained Visual Recognition

- ##### **Authors:** Yutong Yang, Lifu Huang,[Yijie Lin](https://lin-yijie.github.io/), [Xi Peng](https://pengxi.me/), [Mouxing Yang](https://mouxingyang.github.io/) <br>
##### **Accepted by AAAI 2026**


## Install

First, 

```
conda create -n SCAN python=3.10.18
conda activate SCAN
cd 2026-AAAI-SCAN
pip install -r requirements.txt
```

Then, download the `.whl` files for [**FlashAttention**](https://github.com/Dao-AILab/flash-attention/releases/), then install it using pip:

```
pip install flash_attn-2.7.1.post4+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```
