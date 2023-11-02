from cppn_torch.activation_functions import gauss

# from https://github.com/tancik/fourier-feature-networks
import torch
import numpy as np
def apply_mapping(x, B:torch.tensor, sin_and_cos=False):
    if B is None:
      return x
    else:
      device = x.device
      x = x.cpu().numpy()
      B = B.cpu().numpy()
      x_proj = (2.*np.pi*x) @ B.T
      if sin_and_cos:
        result = np.concatenate([np.sin(x_proj), np.cos(x_proj)], axis=-1)
      else:
        result = np.sin(x_proj)
      return torch.tensor(result, device=device).float()

    
    # x_proj = (2.*torch.pi*x) @ B.T
    # f0 = torch.sin(x_proj)
    # # return torch.cat([f0], dim=-1)
    # f1 = torch.cos(x_proj)
    # return torch.cat([f0,f1], dim=-1)

def input_mapping(x, b_scale:float, mapping_size:int, dims:int=2, sin_and_cos=False):
    mapping_size = mapping_size // 2 if sin_and_cos else mapping_size
    B_gauss = torch.randn((mapping_size, dims), device=x.device)
    B_gauss = B_gauss * b_scale
    return apply_mapping(x, B_gauss, sin_and_cos=sin_and_cos)
  

def add_fourier_features(x, n_features, B_scale=10.0, dims=2, include_original=False, mult_percent=.5, sin_and_cos=False):
    assert n_features % dims == 0, "mapping_size must be divisible by dims"
    
    # get first dims features
    feats = x[:,:, :dims]
    
    if mult_percent:
      orig_n_features = n_features
      n_features = orig_n_features - int(orig_n_features * mult_percent)
      
    f_feats = input_mapping(feats, B_scale, n_features, dims=dims, sin_and_cos=sin_and_cos)
    if mult_percent:
      while f_feats.shape[-1] < orig_n_features:
        two_rand = torch.randint(0, f_feats.shape[-1], (2,))
        m = f_feats[:,:, two_rand[0]] * f_feats[:, :, two_rand[1]]
        f_feats = torch.cat([f_feats, m.unsqueeze(-1)], dim=-1)
        

    if include_original:
      X = torch.cat([x, f_feats], dim=-1)
    else:
      X = f_feats

    return X

