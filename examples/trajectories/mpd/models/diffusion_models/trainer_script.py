from scoremodel_temporal import *
import torch
import functools
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision
import tqdm
import random

import sys
sys.path.append('/path/to/dir/mpd-public')

from mpd import trainer
from mpd.trainer import get_dataset, get_model, get_loss, get_summary

torch.cuda.empty_cache()


score_model = torch.nn.DataParallel(ScoreNet(marginal_prob_std=marginal_prob_std_fn))
score_model = score_model.to(device)
# checkpoint_path = './ckpt_noise_cond_1.pth'
# score_model.load_state_dict(torch.load(checkpoint_path, map_location=device))


n_epochs   =  10000
batch_size =  32
lr=1e-4


# from datasets.bouncing_balls import load_balls
from torch.utils.data import DataLoader


transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((64, 64)),
    torchvision.transforms.ToTensor()
])


# device = get_torch_device(device=device)
tensor_args = {'device': device, 'dtype': torch.float32}

dataset_subdir = 'EnvSpheres3D-RobotPanda'
# dataset_subdir = 'EnvSimple2D-RobotPointMass'
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


optimizer = Adam(score_model.parameters(), lr=lr)
min_loss = 14.0

print("start")

for epoch in (range(n_epochs)):
  avg_loss = 0.
  num_items = 0
  for data in train_dataloader:
    
    split = random.randrange(1, 59, 1)
    
#     c = data[:, :split, :, :]
    
#     c_batch = []
#     for batch in range(c.shape[0]):
#         c_batch.append(torch.stack([c_i.flatten() for _, c_i in enumerate(c[batch])]))   
#     c = torch.stack(c_batch)
    
    
#     data = data[:, split, :, :].unsqueeze(1)
#     print('1', c.shape, data.shape)
    print(data['traj_normalized'].shape)

    data_new = torch.zeros(data['traj_normalized'].shape[0], 14, 64)
    data_new[:, :, :64] = data['traj_normalized'].permute(0, 2, 1)
    
    
    c = data_new[:, :, :split].permute(0, 2, 1)
    
    data = data_new[:, :, split:(split+14)].unsqueeze(1)
    
    print('1', c.shape, data.shape)
    
    x = data.to(device)   
    c = c.to(device)


    loss = loss_fn(score_model, x, c, marginal_prob_std_fn)
    optimizer.zero_grad()
    loss.backward()    
    optimizer.step()
    avg_loss += loss.item() * x.shape[0]
    num_items += x.shape[0]
    
  # Print the averaged training loss so far.
    
  if (avg_loss / num_items) < min_loss:
    min_loss = (avg_loss / num_items)
    torch.save(score_model.state_dict(), '../autor/ckpt.pth')
    print('Average Loss: {:5f}, Epoch: {:5f}'.format(avg_loss / num_items, epoch))
    