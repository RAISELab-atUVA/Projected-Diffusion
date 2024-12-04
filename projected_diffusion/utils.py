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

def extract_values(file_path):
    with open(file_path, 'r') as f:
        text = f.read()

    keywords = ["Mean", "Variance", "Skewness", "Kurtosis", "Porosity Value"]
    values = [float(val) for keyword in keywords for val in re.findall(rf"{keyword}:\s+([-+]?\d*\.\d+|\d+)", text)]
    #print(values, file_path)
    # Convert list of values to a torch tensor
    values_tensor = torch.tensor(values)

    # Normalize the values
    mean = torch.mean(values_tensor)
    std = torch.std(values_tensor)
    normalized_values = (values_tensor - mean) / std

    return normalized_values

    
    
class CustomImageDataset(Dataset):
    def __init__(self, root_dir, mode, transform=None):
        self.root_dir = root_dir
        self.mode = mode
        self.text_dir = os.path.join("/path/to/dir/Moments", self.mode)
        self.image_dir = os.path.join(self.root_dir, "N5")  
        self.image_paths = [f for f in os.listdir(self.image_dir) if f.endswith('.bmp')]  # Use .tiff if required
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_paths[idx])
        image = default_loader(img_name)  # Make sure default_loader is defined or use an appropriate function from torchvision

        base_name = os.path.splitext(self.image_paths[idx])[0]
        moments_file = os.path.join(self.text_dir, base_name + '_moments.txt')

        tensor_values = extract_values(moments_file)  # Make sure extract_values is defined
        if tensor_values.size() != torch.Size([33]):
            tensor_values = None

        if self.transform:
            image = self.transform(image)
        
        #print(tensor_values, moments_file)

        return image, tensor_values


def default_loader(path):
    with open(path, 'rb') as f:
        ext = os.path.splitext(path)[1]
        if ext.lower() in ['.bmp']:  # Use .tiff if required
            img = Image.open(f)  # Simply open the image without converting to grayscale
            img.load()  # Force the image file to be fully loaded into memory
            return img
        else:
            raise ValueError(f"Unsupported image extension {ext}")


def get_data_conditional(batch_size, image_size, train_dataset_path, val_dataset_path):
    image_size = 64
    train_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(image_size + image_size // 4),
        torchvision.transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    val_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(image_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    train_dataset = CustomImageDataset(
        root_dir=train_dataset_path,
        mode = 'Train',
        transform=train_transforms  # Define your transformations if any
    )
    
    val_dataset = CustomImageDataset(
        root_dir=val_dataset_path,
        mode = 'Val',
        transform=val_transforms  # Validation transforms can be different from training
    )


    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_dataloader, val_dataloader



def get_data_unconditional(batch_size, image_size, train_dataset_path, val_dataset_path):
    # Given the desired output is 64x64, set the image_size to 64
    image_size = 64
    train_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(image_size + image_size // 4),
        torchvision.transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    val_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(image_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_dataset = torchvision.datasets.ImageFolder(train_dataset_path, transform=train_transforms)
    val_dataset = torchvision.datasets.ImageFolder(val_dataset_path, transform=val_transforms)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_dataloader, val_dataloader


def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)
    
