from mpd import models, losses, datasets, summaries
from mpd.utils import model_loader, pretrain_helper
from torch.utils.data import DataLoader, random_split
import os
import torch


def get_dataset(train_path, results_dir, device):
    
    tensor_args = {'device': device, 'dtype': torch.float32}

    return load_dataset(dataset_class='TrajectoryDataset',
                        include_velocity=True,
                        dataset_subdir=train_path,
                        batch_size=32,
                        results_dir=results_dir,
                        save_indices=True,
                        tensor_args=tensor_args
        )


def load_dataset(dataset_class=None,
                dataset_subdir=None,
                batch_size=2,
                val_set_size=0.05,
                results_dir=None,
                save_indices=False,
                **kwargs):
    DatasetClass = getattr(datasets, dataset_class)
    print('\n---------------Loading data')
    full_dataset = DatasetClass(dataset_subdir=dataset_subdir, **kwargs)
    print(full_dataset)

    # split into train and validation
    train_subset, val_subset = random_split(full_dataset, [1-val_set_size, val_set_size])
    train_dataloader = DataLoader(train_subset, batch_size=batch_size)
    val_dataloader = DataLoader(val_subset, batch_size=batch_size)

    if save_indices:
        # save the indices of training and validation sets (for later evaluation)
        torch.save(train_subset.indices, os.path.join(results_dir, f'train_subset_indices.pt'))
        torch.save(val_subset.indices, os.path.join(results_dir, f'val_subset_indices.pt'))

    return train_dataloader, val_dataloader
