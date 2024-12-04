import os
import torch
import torchvision
import scoremodel_pandas

# import sys

# sys.path.append('/path/to/dir/mpd-public')

from experiment_launcher import single_experiment_yaml, run_experiment
from mpd import trainer
from mpd.models import UNET_DIM_MULTS, TemporalUnet
from mpd.trainer import get_dataset, get_model, get_loss, get_summary
from mpd.trainer.trainer import get_num_epochs
from torch_robotics.torch_utils.seed import fix_random_seed
from torch_robotics.torch_utils.torch_utils import get_torch_device, dict_to_device


# epsilon of step size
eps = 1.5e-5

# sigma min and max of Langevin dynamic
sigma_min = 0.005
sigma_max = 10

# Langevin step size and Annealed size
n_steps = 10
annealed_step = 100

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

model = scoremodel_pandas.Model(device, n_steps, sigma_min, sigma_max)
optim = torch.optim.Adam(model.parameters(), lr = 0.005)
dynamic = scoremodel_pandas.AnnealedLangevinDynamic(sigma_min, sigma_max, n_steps, annealed_step, model, device, eps=eps)

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((dynamic.img_size, dynamic.img_size)),
    torchvision.transforms.ToTensor()
])


# device = get_torch_device(device=device)
tensor_args = {'device': device, 'dtype': torch.float32}

dataset_subdir = 'EnvSpheres3D-RobotPanda'
results_dir = 'logs'

batch_size = 32

train_subset, train_dataloader, val_subset, val_dataloader = get_dataset(
        dataset_class='TrajectoryDataset',
        include_velocity=True,
        dataset_subdir=dataset_subdir,
        batch_size=batch_size,
        results_dir=results_dir,
        save_indices=True,
        tensor_args=tensor_args
    )

dataset = train_subset.dataset

total_iteration = 30000
current_iteration = 0
display_iteration = 150
sampling_number = 8
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
only_final = True

run_name = "panda-sb"
os.makedirs('/path/to/dir/mpd-public/mpd/models/trained/', exist_ok=True)
best_val_loss = float('inf')  # Initialize the best validation loss
    
losses = scoremodel_pandas.AverageMeter('Loss', ':.4f')
progress = scoremodel_pandas.ProgressMeter(total_iteration, [losses], prefix='Iteration ')

while current_iteration != total_iteration:
    
    ## Training Routine ##
    
    model.train()
    
    for data in train_dataloader:
        
        print(data.keys(), data['traj_normalized'].shape)

        # Convert to tensor
        data_new = torch.zeros(data['traj_normalized'].shape[0], 14, 64)
        data_new[:, :, :64] = data['traj_normalized'].permute(0, 2, 1)
        
        data = data_new.to(tensor_args['device']).reshape((data['traj_normalized'].shape[0], 14, 8, 8))
                
        loss = model.loss_fn(data)

        optim.zero_grad()
        loss.backward()
        optim.step()

        losses.update(loss.item())
        
    progress.display(current_iteration)
    current_iteration += 1
    
    
    ## Validation Routine ##
    
    model.eval()
    
    val_loss_accumulator = 0.0
    val_steps = 0
    
    with torch.no_grad():
        for data in val_dataloader:
            
            data_new = torch.zeros(data['traj_normalized'].shape[0], 14, 64)
            data_new[:, :, :64] = data['traj_normalized'].permute(0, 2, 1)
            data = data_new.to(tensor_args['device']).reshape((data['traj_normalized'].shape[0], 14, 8, 8))
            
            val_loss = model.loss_fn(data)
            val_loss_accumulator += val_loss.item()
            val_steps += 1

    # Compute average validation loss for the epoch
    avg_validation_loss = val_loss_accumulator / val_steps
    
    # Checkpointing
    if avg_validation_loss < best_val_loss:
        best_val_loss = avg_validation_loss

        # Save original model checkpoint
        model_save_path = os.path.join("/path/to/dir/mpd-public/mpd/models/", run_name, f"ckpt.pt")
        torch.save(model.state_dict(), model_save_path)
    
        # Optionally save the optimizer state
        optimizer_save_path = os.path.join("/path/to/dir/mpd-public/mpd/models/", run_name, f"optim.pt")
        torch.save(optim.state_dict(), optimizer_save_path)
        
        
    ## Logging ##
    
    # if current_iteration % display_iteration == 0:
        
    #     # Save original model checkpoint
    #     model_save_path = os.path.join("models", run_name, f"ckpt_{current_iteration}.pt")
    #     torch.save(model.state_dict(), model_save_path)
        
    #     dynamic = scoremodel.AnnealedLangevinDynamic(sigma_min, sigma_max, n_steps, annealed_step, model, device, eps=eps)
    #     sample = dynamic.sampling(sampling_number, only_final)
    #     losses.reset()
        