import os
import copy
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from utils import *
from modules import UNet_conditional, EMA
import logging
# from torch.utils.tensorboard import SummaryWriter
import random
import time
import pandas as pd

from projection import Projection

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=64, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.img_size = img_size
        self.device = device

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))
    
    # n = number of images you want
    def sample(self, model, n, labels, cfg_scale=3, projection=True, ub=0.21, lb=0.19, threshold=0):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        
        with torch.no_grad():
            # 3 vs 1 is the number of channels for the images you are creating
            # x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            x = torch.randn((n, 1, self.img_size, self.img_size)).to(self.device)
            
            #JC
            # Upper bound constraint
            project_ub = Projection(k=ub, threshold=threshold, img_size=x[0].numel())
            # project_ub.schedule(self.noise_steps-1, self.noise_steps)
            proj_data = []
            ub_data = []
            x_axis = []
            x_axis_alt = []

            # Lower bound constraint
            project_lb = Projection(k=lb, threshold=threshold, img_size=x[0].numel(), lower_bound=True)
            lb_data = []
            
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t, labels)
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, None)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
                # x = project_ub.apply(x.clone())
                # x = project_lb.apply(x.clone())
                # JC
                
#                 if projection and ((i % 10) == 0 or i == 1):
#                     # x_axis.append(i)
                    
#                     # Upper bound correction
#                     # project_ub.schedule(i, self.noise_steps)
#                     # ub_data.append((project_ub.k/project_ub.img_size)*100)
#                     x = project_ub.apply(x.clone())
                    
#                     # Lower bound correction
#                     project_lb.schedule(i, self.noise_steps)
#                     # project_lb.k = 4096 - (project_ub.k - 992)
#                     lb_data.append((100 - (project_lb.k/project_lb.img_size)*100))
#                     x = project_lb.apply(x.clone())

# #                     proj_data.append((x < 0).float().mean().clone().cpu().item() * 100)  

#                 proj_data.append((x < project_ub.threshold).float().mean().clone().cpu().item() * 100) 
#                 x_axis_alt.append(i)
                    
        # JC
        # if projection:
        #     plt.figure(figsize=(10, 5))
        #     plt.plot(x_axis_alt, proj_data, label='Samples')
        #     plt.plot(x_axis, ub_data, label='Tightening Feasible Set', color='red')
        #     plt.plot(x_axis, lb_data, color='red')
        #     plt.title('Tightening Constraints')
        #     plt.xlabel('Time Step')
        #     plt.ylabel('Black Pixel %')
        #     plt.legend()
        #     plt.gca().invert_xaxis()
        #     plt.fill_between(x_axis, ub_data, lb_data, color='green', alpha=0.1)
        #     # plt.gca().set_ylim(0, 80)
        #     plt.show()
            
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x



def extract_values(file_path, por=False):
    with open(file_path, 'r') as f:
        text = f.read()

    keywords = ["Mean", "Variance", "Skewness", "Kurtosis", "Porosity Value"]
    
    if por: keywords = ["Mean", "Variance", "Skewness", "Kurtosis"]
        
    values = [float(val) for keyword in keywords for val in re.findall(rf"{keyword}:\s+([-+]?\d*\.\d+|\d+)", text)]

    # Convert list of values to a torch tensor
    values_tensor = torch.tensor(values)
    # print(len(values_tensor), values_tensor[-1])
    #JC
    # values_tensor[-1] = 0.3787109375
    
    #values_tensor[-1] = 0.1787109375 # 0.3
    

    # Normalize the values
    mean = torch.mean(values_tensor)
    std = torch.std(values_tensor)
    normalized_values = (values_tensor - mean) / std

    return normalized_values, mean, std


def get_ranges(directory, por=False):
    normalization_params = []  # To store mean and std for each attribute
    all_normalized_values = [[] for _ in range(33)]

    # Collect all normalized values and normalization parameters for each attribute
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            normalized_values, mean, std = extract_values(os.path.join(directory, filename), por=por)
            for i, val in enumerate(normalized_values):
                all_normalized_values[i].append(val.item())
            normalization_params.append((mean, std))

    # Calculate normalized ranges based on the collected normalized values
    normalized_ranges = []
    for values in all_normalized_values:
        normalized_ranges.append((min(values), max(values)))

    return normalized_ranges, normalization_params

def generate_random_tensor(directory):
    normalized_ranges, normalization_params = get_ranges(directory)
#     normalized_ranges1, _ = get_ranges(directory, por=True)
#     print(normalized_ranges, "\n", (normalized_ranges1))
    # TODO: Come back to this when you have a better understanding of porosity
    # FIXME: Not even reading the porosity value
#     print(normalized_ranges)
#     random_values = [random.uniform(min_val, max_val) for min_val, max_val in normalized_ranges]
    random_values = [(min_val + max_val)/2 for min_val, max_val in normalized_ranges]
    random_tensor = torch.tensor(random_values)

    # return random_tensor
    return random_tensor, normalization_params



def train(run_name, epochs, batch_size, image_size, train_dataset_path, val_dataset_path, device, lr):
    setup_logging(run_name)
    
    # Lists to store losses
    training_losses = []
    validation_losses = []
    
    device = device
    
    train_dataloader, val_dataloader = get_data_conditional(batch_size, image_size, train_dataset_path, val_dataset_path)
    # TODO: Initialize with c_in=1, c_out=1 and train bw model
    model = UNet_conditional(c_in=1, c_out=1).to(device)
    ckpt = torch.load("./models/Test_Labels_bw/ckpt.pt")
    model.load_state_dict(ckpt)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=image_size, device=device)
    # logger = SummaryWriter(os.path.join("runs", run_name))
    ema = EMA(0.995)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)
    
    
    best_val_loss = float('inf')  # Initialize the best validation loss
    for epoch in range(epochs):
        logging.info(f"Starting epoch {epoch}:")
        train_loss_accumulator = 0.0
        val_loss_accumulator = 0.0
        train_steps = 0
        val_steps = 0
        
        model.train()  # Set the model to training mode
        pbar = tqdm(train_dataloader)
        
        data_list = []
        
        for i, (images, labels) in enumerate(pbar):
                
            #print(labels)
            images = images.to(device)
            
            # JC
            images = torch.stack([convert_to_grayscale(img) for i, img in enumerate(images)])
            
            labels[:,-1] = torch.tensor([(((img < 0).float().mean().clone().cpu().item() * 2) - 1) for i, img in enumerate(images)])
            labels = labels.to(device)
            
            if epoch == 0:
                bp_perc = ([((img < 0).float().mean().clone().cpu().item() * 100) for i, img in enumerate(images)])
                porosity = list(labels[:, -1])
                porosity = [x.item() for i, x in enumerate(porosity)]
                data_list += (zip(bp_perc, porosity))
            
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            if np.random.random() < 0.1:
                labels = None
            predicted_noise = model(x_t, t, labels)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.step_ema(ema_model, model)

            train_loss_accumulator += loss.item()
            train_steps += 1

            pbar.set_postfix(MSE=loss.item())

        if epoch == 0:
#             df = pd.DataFrame(data_list, columns=['bp_perc', 'label'])
#             df.to_csv('~/Diffusion_Structures-main/porosity_analysis.csv', index=False)
            plt.figure(figsize=(10, 5))
            x, y = zip(*data_list)
            plt.scatter(x, y, label='Samples')
            plt.title('Porosity Guidance Analysis')
            plt.xlabel('Black Pixel (%)')
            plt.ylabel('Porosity Label')
            plt.legend()
            plt.show()
    
        # Compute average training loss for the epoch
        avg_training_loss = train_loss_accumulator / train_steps
        # logger.add_scalar("Average_Training_MSE", avg_training_loss, global_step=epoch)
        #print(f"Epoch {epoch} - Average Training Loss: {avg_training_loss}")

        # Validation phase
        ema_model.eval()  # Set the EMA model to evaluation mode
        with torch.no_grad():
            for images, labels in val_dataloader:
                images = images.to(device)
                
                # JC
                images = torch.stack([convert_to_grayscale(img) for i, img in enumerate(images)])
                
                labels = labels.to(device)
                t = diffusion.sample_timesteps(images.shape[0]).to(device)
                x_t, noise = diffusion.noise_images(images, t)
                predicted_noise = ema_model(x_t, t, labels)
                val_loss = mse(noise, predicted_noise)

                val_loss_accumulator += val_loss.item()
                val_steps += 1

         # Compute average validation loss for the epoch
        avg_validation_loss = val_loss_accumulator / val_steps
        # At the end of each epoch, log the losses
        training_losses.append(avg_training_loss)
        validation_losses.append(avg_validation_loss)
        # logger.add_scalar("Average_Validation_MSE", avg_validation_loss, global_step=epoch)
    
        # Checkpointing
        if avg_validation_loss < best_val_loss:
            best_val_loss = avg_validation_loss
            #logging.info(f"Validation loss improved to {avg_validation_loss:.4f}, saving model...")
    
            # Save original model checkpoint
            model_save_path = os.path.join("models", run_name, f"ckpt.pt")
            torch.save(model.state_dict(), model_save_path)
    
            # Save EMA model checkpoint
            ema_model_save_path = os.path.join("models", run_name, f"ema_ckpt.pt")
            torch.save(ema_model.state_dict(), ema_model_save_path)
    
            # Optionally save the optimizer state
            optimizer_save_path = os.path.join("models", run_name, f"optim.pt")
            torch.save(optimizer.state_dict(), optimizer_save_path)


        if epoch % 10 == 0:
            labels = generate_random_tensor("/path/to/dir/Moments/Train")
            labels = labels.to(device)
            sampled_images = diffusion.sample(model, n=4, labels=labels)
            ema_sampled_images = diffusion.sample(ema_model, n=4, labels=labels)
            
            # JC
#             logging.info(f"Sample type {type(sampled_images)}, of shape {sampled_images.shape}")
#             grayscale_samples = torch.stack([convert_to_grayscale(img) for i, img in enumerate(sampled_images)])
#             plot_images(grayscale_samples)
                
            plot_images(sampled_images)
            save_images(sampled_images, os.path.join("results", run_name, f"{epoch}.bmp"))
            save_images(ema_sampled_images, os.path.join("results", run_name, f"{epoch}_ema.bmp"))
    
    plt.figure(figsize=(10, 5))
    plt.plot(training_losses, label='Training loss')
    plt.plot(validation_losses, label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Save the figure
    plt.savefig('test.png', dpi=300, bbox_inches='tight')
    
                    


            
def launch(run_name="Test_Labels_bw_analysis", epochs=400, batch_size=20, image_size=64,
          train_dataset_path="/path/to/dir/Iowa_img/Train",
           val_dataset_path="/path/to/dir/Iowa_img/Val",
           device="cuda", lr=3e-4):
   # You can pass these arguments directly to the train function
   train(run_name, epochs, batch_size, image_size, train_dataset_path, val_dataset_path, device, lr)            
            
            

#def launch():
#    import argparse
#    parser = argparse.ArgumentParser()
#    args = parser.parse_args()
#    args.run_name = "DDPM_conditional"
#    args.epochs = 300
#    args.batch_size = 14
#    args.image_size = 64
#    args.num_classes = 10
#    args.dataset_path = r"C:\Users\dome\datasets\cifar10\cifar10-64\train"
#    args.device = "cuda"
#    args.lr = 3e-4
#    train(args)


if __name__ == '__main__':
    launch()
    # device = "cuda"
    # model = UNet_conditional(num_classes=10).to(device)
    # ckpt = torch.load("./models/DDPM_conditional/ckpt.pt")
    # model.load_state_dict(ckpt)
    # diffusion = Diffusion(img_size=64, device=device)
    # n = 8
    # y = torch.Tensor([6] * n).long().to(device)
    # x = diffusion.sample(model, n, y, cfg_scale=0)
    # plot_images(x)
