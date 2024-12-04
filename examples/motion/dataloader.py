from glob import glob
import os
import numpy as np
import random
import torch.utils.data as data
import json
import cv2
import datasets.video_transforms as vtransforms
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader



def get_dataset(train_path, val_path):
    train_dataset      = load_balls('', 6, True)
    validation_dataset = load_balls('', 6, False)

    data_loader = DataLoader(train_dataset, batch_size=64, shuffle=False,
                            drop_last=True)

    validation_loader = DataLoader(validation_dataset, batch_size=64, shuffle=False,
                                drop_last=True)
    
    return data_loader, validation_loader

def load_balls(root, n_frames, is_train=True):
    transform = transforms.Compose([vtransforms.Scale((64, 64)),
                                    vtransforms.ToTensor()])
    dset = BouncingBalls(root, is_train, n_frames,
                         n_frames, 64, transform)
    return dset


def make_dataset(root, is_train):
  instances = 2048 if is_train else 205

  data = []
  for step in range(instances):
    bw_frames = generate_bw_animation_frames(num_frames=20)
    data.append(torch.stack([ToTensor()(frame) for frame in bw_frames], dim=0).squeeze(1))

  dataset = torch.stack(data, dim=0)#.numpy()
  print("TYPE: ", (dataset.shape))
  return dataset

class BouncingBalls(data.Dataset):
  '''
  Bouncing balls dataset.
  '''
  def __init__(self, root, is_train, n_frames_input, n_frames_output, image_size,
               transform=None, return_positions=False):
    super(BouncingBalls, self).__init__()
    self.n_frames = n_frames_input + n_frames_output
    self.dataset = make_dataset(root, is_train)

    self.size = image_size
    self.scale = self.size / 800
    self.radius = int(60 * self.scale)

    self.root = root
    self.is_train = is_train

    self.n_frames_input = n_frames_input*2
    self.n_frames_output = n_frames_output*2
    self.n_frames = n_frames_output*2

    self.transform = transform
    self.return_positions = return_positions
    
    # FIXME
    self.with_target = False
    self.digit_size_ = 28
    self.step_length_ = 0.1
    self.num_digits = 1
    self.image_size_ = image_size
    self.mnist = self.dataset

  def __len__(self):
    return self.dataset.size(0)
  
  def __getitem__(self, idx):

    
    if torch.is_tensor(idx):
            idx = idx.tolist()
    
    traj = self.dataset[idx, :10, :, :]
    label = self.dataset[idx, 10:, :, :]

    
    return traj, label




from torchvision.transforms import ToTensor
import random
from PIL import Image, ImageDraw



def create_background_with_gray_lines(frame_size=(64, 64), num_lines=50):
    """Create a background with random gray lines."""
    background = Image.new("L", frame_size, "white")
    draw = ImageDraw.Draw(background)

    for idx in range(int(num_lines/2)):

        const_point = int((frame_size[0] / (num_lines/2)) * idx) +1

        gray_shade = 127
        draw.line([(const_point, 0), (const_point, 63)], fill=gray_shade)
        draw.line([(0, const_point), (63, const_point)], fill=gray_shade)

    return background

def create_bw_frame(background, ball_y, ball_x, ball_radius=5):
    """Create a single black and white frame with the ball at the specified y position."""
    frame = background.copy()  # Copy the background with gray lines
    draw = ImageDraw.Draw(frame)
    x_position = ball_x
    top_left = (x_position - ball_radius, ball_y - ball_radius)
    bottom_right = (x_position + ball_radius, ball_y + ball_radius)
    draw.ellipse([top_left, bottom_right], fill="black")
    return frame

def generate_bw_animation_frames(num_frames=10, frame_size=(64, 64)):
    """Generate a series of black and white frames for the animation."""
    frames = []
    # ball_x = random.randrange(8, 56, 1)
    ball_x = 32
    # max_height = random.randrange(48, 56, 1)
    max_height = 56
    background = create_background_with_gray_lines(frame_size)

    for i in range(num_frames):
        t = i / (num_frames - 1)
        ball_y = (64 - max_height) + int(position_change(9.8, i) / 2)
        frame = create_bw_frame(background, ball_y, ball_x, ball_radius=5)
        frames.append(frame)
    return frames

def position_change(acceleration, time):
        # change_in_position = initial_velocity * time + 0.5 * acceleration * time^2
        change_in_position = 0.5 * acceleration * time ** 2

        return change_in_position

