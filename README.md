# Bacteria git

---

# Introduction

---

In this paper, we developed novel approaches for bacterial species detection and identification methods with single-cell sensitivity using super-resolution microscopy and AI-based image analysis: a protein quantification-based method and an AI-based bacterial image analysis method. 

It has been confirmed that metric learning outperformed general classification in bacteria detection. It consists with both general classification models and metric learning models for bacteria experimentation . 

---

# Usage

### **Install**

- Clone this repo

```bash
git clone https://github.com/JINAILAB/bacteria-classifcation
cd bacteria-classifcation
```

- installation

```bash
pip install pytorch-metric-learning
pip install faiss-gpu
pip install torchmetrics
```

- Our code is developed on [metric learning pytorch](https://kevinmusgrave.github.io/pytorch-metric-learning/) and pytorch.

## data preparation

The directory structure is the standard layout for the torchvision [`datasets.ImageFolder`](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder), and the training and validation data is expected to be in the `train/` folder and `val/` folder respectively. and you can also use `test/` for inference. And your log will be saved at `/model_log`

```bash
data
├── train
│   ├── aur
│   │   ├── 1.png
│   │   ├── x.png
│   ├── epi
│   │   ├── 1.png
│   │   ├── x.png
├── val
│   ├── aur
│   │   ├── 1.png
│   │   ├── x.png
│   ├── epi
│   │   ├── 1.png
│   │   ├── x.png
├── test
│   ├── 1.png
│   ├── 2.png
│   ├── x.png
```

## Model Summary

All models have been trained on 2 x NVIDIA Gforce V100 GPUs with the following parameters:

| Parameter | value |
| --- | --- |
| --batch_size | 32 |
| --epochs | 50 |
| --lr | 0.001 |
| --momentum | 0.9 |
| --optimizer | sgd |

## **Test existing models**

We provide 2 models (Resnet18, efficientnetv2_s) for evaluating the batacteria classfication. 

- metric learning evaluate

```bash
python3 pytorch_metric/main.py --valid-only --data-path ${datapath} --resume ${model.pth} --output-dir ${output_path}
```

- classification evaluate

```bash
python3 pytorch_classfication/main.py --valid-only --data-path ${datapath} --resume ${model.pth} --output-dir ${output_path}
```

## model pretrain available

- Classification

| Model | Aur/Epi/RT6 |
| --- | --- |
| resnet18 | 0.993 |
| https://drive.google.com/file/d/10h7tegdtAq0FCHt3bamu0IPxiHvKpUpv/view?usp=share_link | 1.0 |
- Metric learning

|  | Aur/Epi/RT45/RT6 |
| --- | --- |
| resnet18 | 0.849 |
| https://drive.google.com/file/d/16ijEremmC_0c6UTpmeQX-6kdKgBGE6-s/view?usp=share_link | 0.914 |
