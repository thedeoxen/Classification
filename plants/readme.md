# Image Classification Repository

This repository contains code for image classification, specifically designed for detecting apple diseases. The goal of this competition is to improve upon the previous year's results by handling additional diseases and providing more detailed information about leaves with multiple infections.

## Dataset Description

The dataset used in this project can be downloaded from [this link](https://www.kaggle.com/datasets/ankursingh12/resized-plant2021). Please download the zip archive folder `img_sz_256` and place the contents in the following directory:

```
plants
└── dataset
    └── images
        ├── 8a0be55d81f4bf0c.jpg
        ├── 8a0d7cad7053f18d.jpg
        ├── ...
```

Additionally, the CSV file can be obtained from [this link](https://www.kaggle.com/competitions/plant-pathology-2021-fgvc8/data?select=train.csv). Please place the file in the following directory:

```
plants
└── dataset
    └── train.csv
```


## Implementation Details

This project is implemented using PyTorch, a popular deep learning framework. Please make sure you have PyTorch installed before running the code.

## Model Performance on ImageNet

The table below shows the performance of different models on the ImageNet dataset, with the provided accuracy values (Acc@1 and Acc@5), number of parameters, and GFLOPS (floating-point operations per second).

| Model Weight                            | Acc@1  | Acc@5   | Params  | GFLOPS |
|-----------------------------------------|--------|---------|---------|--------|
| ViT_B_16_Weights.IMAGENET1K_V1          | 81.072 | 95.318  | 86.6M   | 17.56  |
| ResNet50_Weights.IMAGENET1K_V1          | 76.13  | 92.862  | 25.6M   | 4.09   |
| MobileNet_V2_Weights.IMAGENET1K_V2      | 72.154 | 90.822  | 3.5M    | 0.3    |
| EfficientNet_V2_M_Weights.IMAGENET1K_V1 | 85.112 | 97.156  | 54.1M   | 24.58  |

Please note that these values were obtained from [PyTorch's model zoo](https://pytorch.org/vision/stable/models.html) and represent the performance of each model on the ImageNet dataset.
