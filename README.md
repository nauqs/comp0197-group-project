# Applied Deep Learning (COMP0197) group project

## Semi-supervised segmentation using deep neural networks

Implement and evaluate a semi-supervised neural network for semantic segmentation of images on the [The Oxford-IIIT Pet Data-set](https://www.robots.ox.ac.uk/~vgg/data/pets/).

### Project structure

```
comp0197-group-project/
├── main.py
├── config.ini
├── data/
├── models/
│   ├── model1.py
│   ├── model2.py
│   └── ...
├── utils/
│   ├── dataset.py
│   ├── metrics.py
│   └── ...
├── experiments/
│   ├── experiment1/
│   │   ├── train.py
│   │   ├── test.py
│   │   ├── models/
│   │   └── results/
│   ├── experiment2/
│   ├── ...
│   └── experimentN/
```

### Instructions

1. Create environment

```
conda create --name comp0197-group-project -c pytorch python=3.10 pytorch=1.13 torchvision=0.14
conda activate comp0197-group-project
```

2. Install required packages

```
pip install -r requirements.txt
```

3. Download data from [Sharepoint](https://liveuclac-my.sharepoint.com/:f:/g/personal/ucabtc6_ucl_ac_uk/EsA5CnS2RLRJstq7emQ_bykBY9_P9JRd5l9ZBzJW2Mtncg?e=RbNbt9).

Alternatively, download them from https://www.robots.ox.ac.uk/~vgg/data/pets/ and untar inside the data folder.
Do not push these files to Github. 
Structure as:

```
comp0197-group-project/data/
├── README.md
├── images/
└── annotations/
```
