import torch
import ddpm_conditional

def normalize_sample(tensor):
    max_val = torch.max(torch.abs(tensor))
    return (tensor / max_val)

def convert_to_grayscale(image_tensor):
    image_tensor = image_tensor
    weights = torch.tensor([0.2989, 0.5870, 0.1140], device=torch.device('cuda:0')).view(3, 1, 1)
    grayscale_image = (weights * image_tensor).sum(dim=0, keepdim=True)
    return grayscale_image

def get_dataset(train_path, val_path):
    train_loader, validation_loader = ddpm_conditional.get_data_conditional(8, 64, train_path, val_path)
    train_loader = normalize_sample(torch.stack([convert_to_grayscale(img) for i, img in enumerate(train_loader)]))
