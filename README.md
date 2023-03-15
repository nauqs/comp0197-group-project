# Applied Deep Learning (COMP0197) group project

## Semi-supervised segmentation using deep neural networks

Implement and evaluate a semi-supervised neural network for semantic segmentation of images on the [The Oxford-IIIT Pet Data-set](https://link-url-here.org).

### Project structure

```
comp0197-group-project/
├── main.py
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
conda create --name comp0197-group-project pytorch torchvision
conda activate comp0197-group-project
```

2. Install required packages

```
pip install -r requirements.txt
```

3. Download data from [Sharepoint](https://liveuclac-my.sharepoint.com/personal/ucabtc6_ucl_ac_uk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fucabtc6%5Fucl%5Fac%5Fuk%2FDocuments%2FApplied%20Deep%20Learning%2FDataset&ga=1).

Alternatively, download them from https://www.robots.ox.ac.uk/~vgg/data/pets/ and untar inside the data folder.
Do not push these files to Github. 
Structure as:

```
comp0197-group-project/data/
├── README.md
├── images/
└── annotations/
```
