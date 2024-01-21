# torch imports
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder

# data viz imports
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# project imports
from dataset import CardDataset


# Loading the train dataset
dataset = CardDataset(data_dir = "/kaggle/input/cards-image-datasetclassification/train")

# Creating a dictionary associating target values with target names
data_dir = "/kaggle/input/cards-image-datasetclassification/train"
target_to_class = {v:k for k, v in ImageFolder(data_dir).class_to_idx.items()}
# print(target_to_class)


# Creating a transform for the input data
# here it will simply resize to 128x128 and convert to tensor
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Loading the train dataset with transform
dataset = CardDataset("/kaggle/input/cards-image-datasetclassification/train", transform)