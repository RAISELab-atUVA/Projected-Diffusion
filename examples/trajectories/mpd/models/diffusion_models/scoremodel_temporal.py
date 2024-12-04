import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GaussianFourierProjection(nn.Module):
  """Gaussian random features for encoding time steps."""  
  def __init__(self, embed_dim, scale=30.):
    super().__init__()
    # Randomly sample weights during initialization. These weights are fixed 
    # during optimization and are not trainable.
    self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
  def forward(self, x):
    x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Dense(nn.Module):
  """A fully connected layer that reshapes outputs to feature maps."""
  def __init__(self, input_dim, output_dim):
    super().__init__()
    self.dense = nn.Linear(input_dim, output_dim)
  def forward(self, x):
    return self.dense(x)[..., None, None]


class ScoreNet(nn.Module):
  """A time-dependent score-based model built upon U-Net architecture."""

  def __init__(self, marginal_prob_std, channels=[32, 64, 128, 256], embed_dim=260):
    """Initialize a time-dependent score-based network.

    Args:
      marginal_prob_std: A function that takes time t and gives the standard
        deviation of the perturbation kernel p_{0t}(x(t) | x(0)).
      channels: The number of channels for feature maps of each resolution.
      embed_dim: The dimensionality of Gaussian random feature embeddings.
    """
    super().__init__()
    # Gaussian random feature embedding layer for time
    self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=(256)),
         nn.Linear((256), (256)))
    # Encoding layers where the resolution decreases
    self.conv1 = nn.Conv2d(1, channels[0], 3, stride=1, bias=False)
    self.dense1 = Dense(embed_dim, channels[0])
    self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])
    self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride=2, bias=False)
    self.dense2 = Dense(embed_dim, channels[1])
    self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])
    self.conv3 = nn.Conv2d(channels[1], channels[2], 3, stride=2, bias=False)
    self.dense3 = Dense(embed_dim, channels[2])
    self.gnorm3 = nn.GroupNorm(32, num_channels=channels[2])
    self.conv4 = nn.Conv2d(channels[2], channels[3], 3, stride=2, bias=False)
    self.dense4 = Dense(embed_dim, channels[3])
    self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])    

    # Decoding layers where the resolution increases
    self.tconv4 = nn.ConvTranspose2d(channels[3], channels[2], 3, stride=2, bias=False)
    self.dense5 = Dense(embed_dim, channels[2])
    self.tgnorm4 = nn.GroupNorm(32, num_channels=channels[2])
    self.tconv3 = nn.ConvTranspose2d(channels[2] + channels[2], channels[1], 3, stride=2, bias=False, output_padding=1)    
    self.dense6 = Dense(embed_dim, channels[1])
    self.tgnorm3 = nn.GroupNorm(32, num_channels=channels[1])
    self.tconv2 = nn.ConvTranspose2d(channels[1] + channels[1], channels[0], 3, stride=2, bias=False, output_padding=1)    
    self.dense7 = Dense(embed_dim, channels[0])
    self.tgnorm2 = nn.GroupNorm(32, num_channels=channels[0])
    self.tconv1 = nn.ConvTranspose2d(channels[0] + channels[0], 1, 3, stride=1)
    
    # The swish activation function
    self.act = lambda x: x * torch.sigmoid(x)
    self.marginal_prob_std = marginal_prob_std
    
    
    # TODO: Change to vision transformer with patching?
    # Conditioning Layers
    d_model = 4
    self.encoder_layer = nn.TransformerEncoderLayer(d_model=4, nhead=4, dim_feedforward=8192, dropout=0.1)
    self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)
    
    self.decoder_layer = nn.TransformerDecoderLayer(d_model=4, nhead=4, dim_feedforward=8192, dropout=0.1)
    self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=6)
    
    
    
    
    
  
  def forward(self, x, c, t): 
    # print("here")
    # Obtain the Gaussian random feature embedding for t   
    
    
    # Create conditioning representation from frame sequence
    # Do we need an encoder?
    # TODO: Vision transformer structure with patching?
    
    # print('x.shape', x.shape)
    c = c.permute(1, 0, 2)
    
    
    x_flat = []
    for batch in range(x.shape[0]):
        x_flat.append(torch.stack([c_i.flatten() for _, c_i in enumerate(x[batch])]))   
    memory = torch.stack(x_flat).permute(1, 0, 2)[:,:,:4]
    
    # print(c.shape, memory.shape, x.shape)
    
    print(">-- ", memory.shape, c.shape)
    
    # print('x', c.shape)
    c = self.transformer_encoder(c)
    # print('y', c.shape, memory.shape)
    c = self.transformer_decoder(c, memory)[-1, :, :]
    # print('z', c.shape)
    c = c.reshape(c.shape[0], 4)
    
    
    # TODO: Find where to introduce conditioning for implicit classifier
    #       in the encoding/decoding pipeline. Are there additional conv
    #       layers needed?

    
    embed = torch.cat((self.act(self.embed(t)), c), dim=1)
    # print(embed.shape)
    
    x = x.reshape(x.shape[0], x.shape[1], 4, 4)
    
    print("-->", embed.shape, c.shape, x.shape)
    
    # Encoding path
    h1 = self.conv1(x)        
    ## Incorporate information from t
    h1 += self.dense1(embed)
    ## Group normalization
    h1 = self.gnorm1(h1)
    h1 = self.act(h1)
    print(h1.shape)
    h2 = self.conv2(h1)
    h2 += self.dense2(embed)
    h2 = self.gnorm2(h2)
    h2 = self.act(h2)
    h3 = self.conv3(h2)
    h3 += self.dense3(embed)
    h3 = self.gnorm3(h3)
    h3 = self.act(h3)
    h4 = self.conv4(h3)
    h4 += self.dense4(embed)
    h4 = self.gnorm4(h4)
    h4 = self.act(h4)
    
    # Encoded 
    

    # Decoding path
    h = self.tconv4(h4)
    ## Skip connection from the encoding path
    h += self.dense5(embed)    
    h = self.tgnorm4(h)
    h = self.act(h)
    h = self.tconv3(torch.cat([F.pad(h, (1, 0, 1, 0)), h3], dim=1))
    h += self.dense6(embed)
    h = self.tgnorm3(h)
    h = self.act(h)
    h = self.tconv2(torch.cat([h, h2], dim=1))
    h += self.dense7(embed)
    h = self.tgnorm2(h)
    h = self.act(h)
    h = self.tconv1(torch.cat([h, h1], dim=1))

    # Normalize output
    h = h / self.marginal_prob_std(t)[:, None, None, None]
        
    return h

def normalize_sample(tensor):
    max_val = torch.max(torch.abs(tensor))
    return (tensor / max_val)


import functools

device = 'cuda' #@param ['cuda', 'cpu'] {'type':'string'}

def marginal_prob_std(t, sigma):
  """Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$.

  Args:    
    t: A vector of time steps.
    sigma: The $\sigma$ in our SDE.  
  
  Returns:
    The standard deviation.
  """    
  t = torch.tensor(t, device=device)
  return torch.sqrt((sigma**(2 * t) - 1.) / 2. / np.log(sigma))

def diffusion_coeff(t, sigma):
  """Compute the diffusion coefficient of our SDE.

  Args:
    t: A vector of time steps.
    sigma: The $\sigma$ in our SDE.
  
  Returns:
    The vector of diffusion coefficients.
  """
  return torch.tensor(sigma**t, device=device)
  
sigma =  25.0#@param {'type':'number'}
marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)


def loss_fn(model, x, c, marginal_prob_std, eps=1e-5):
  """The loss function for training score-based generative models.

  Args:
    model: A PyTorch model instance that represents a 
      time-dependent score-based model.
    x: A mini-batch of training data.    
    marginal_prob_std: A function that gives the standard deviation of 
      the perturbation kernel.
    eps: A tolerance value for numerical stability.
  """
  random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps 
  z = torch.randn_like(x)
  std = marginal_prob_std(random_t)
    
  print(std.shape, z.shape, x.shape, c.shape)
  print('perturb shape: ', (z * std[:, None, None, None]).shape)
  perturbed_x = x + z * std[:, None, None, None]
  print('perturb shape: ', perturbed_x.shape)

    

  c_tilde = []
  for seq in range(c.shape[1]):
    # print(std[:, None, None, None].reshape(c.shape[0], 1, c.shape[2]).shape)
    c_seq = c[:, seq, :].unsqueeze(1) + ((z[:, :, 0, :] - (0.2*(c.shape[1]-seq))*torch.ones_like(z[:, :, 0, :])) * std[:, None, None])#.reshape(c.shape[0], 1, c.shape[2])
    c_tilde.append(c_seq)
  perturbed_c = torch.stack(c_tilde, dim=1).squeeze(2)

  print(perturbed_c.shape, perturbed_x.shape)
  score = model(perturbed_x, perturbed_c, random_t)
  loss = torch.mean(torch.sum((score * std[:, None, None, None] + z)**2, dim=(1,2,3)))
  return loss


