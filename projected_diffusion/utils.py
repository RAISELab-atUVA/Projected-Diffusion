import os
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import re
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


def plot_images(images):
    plt.figure(figsize=(64, 64))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu(), cmap='gray')
    plt.show()
    
    
def normalize_sample(tensor):
    max_val = torch.max(torch.abs(tensor))
    return (tensor / max_val)


def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    
    # Rescale from [-1, 1] to [0, 1]
    ndarr = (ndarr + 1) / 2

    # Then scale from [0, 1] to [0, 255] and convert to uint8
    ndarr = (ndarr * 255).astype(np.uint8)
    
    im = Image.fromarray(ndarr)
    im.save(path)


def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)
    
