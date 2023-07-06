# PCTNet

---
## Pure large kernel convolutional neural network transformer for medical image registration

---

## Requirements

The code for PCTNet is primarily implemented using PyTorch. Here are the specific requirements:

- Python 3.8
- torch 1.7.1
- Numpy
- Monai
- tqdm

## Training and Testing

---

We trained our model on two publicly available datasets, namely [OASIS](http://www.oasis-brains.org/) and [LPBA40](https://www.loni.usc.edu/research/atlas_downloads). Please download these datasets and place them in the "datasets" folder.

### Training script.

```java
python train.py
```

### Testing script.

```java
python test.py
```
## Ackowledgement

---
This work is supported by National Natural Science Foundation of China under grant 61771322 and 61871186, and the Fundamental Research Foundation of Shenzhen under Grant JCYJ20220531100814033.


