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