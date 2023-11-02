import torch
from torch import nn
from typing import Optional, TypeVar, Union


available_normalizations = [
            "neat",
            "inv_neat",
            "sqr_neat",
            "clamp",
            "sigmoid",
            "sigmoid_like",
            "min_max_sigmoid_like",
            "min_max",
            "inv_min_max",
            "min_max_sqr",
            "inv_min_max_sqr",
            "min_max_channel",
            "min_max_channel_sqr",
            "inv_abs_min_max_sqr",
            "inv_abs_min_max_cube",
            "inv_abs_min_max",
            "abs_min_max",
            "abs_tanh",
            "inv_abs_tanh",
            "imagenet",
            "sigmoid_imagenet",
            "imagenet_min_max",
            "min_max_imagenet",
            "inv_abs_imagenet",
            "min_max_sqr_imagenet",
            "neat_sqr_imagenet",
            "softsign",
            "tanh_softsign"
            ]


def norm_min_max(X):
    max_value = torch.max(X)
    min_value = torch.min(X)
    image_range = max_value - min_value
    X = X - min_value
    X = X/(image_range+1e-8)
    X = torch.clamp(X, 0, 1)
    return X

def norm_neat(X):
    """http://eplex.cs.ucf.edu/papers/stanley_gpem07.pdf
    "function outputs range between [−1, 1]. However, ink level is
    darker the closer the output is to zero. Therefore, an output of
    either -1 or 1 produces white." """
    X = torch.abs(X)
    X = torch.clamp(X, 0, 1)
    return X

def norm_sigmoid_like(X):
    a = 2.4020563531719796
    X = -.5*torch.erf(X/a) + .5
    return X

def norm_min_max_sigmoid_like(X):
    X = norm_min_max(X)
    X = X*2 - 1 # center around 0
    return norm_sigmoid_like(X)

def norm_tanh(X,a,b,c):
    X = torch.tanh(a*X)
    X = b+c*torch.abs(X)
    X = torch.clamp(X, 0, 1)
    return X

def norm_min_max_channel(X):
    max_value = torch.max(X, dim=1, keepdim=True)[0]
    min_value = torch.min(X, dim=1, keepdim=True)[0]
    image_range = max_value - min_value
    X = X - min_value
    X = X/(image_range+1e-8)
    X = torch.clamp(X, 0, 1)
    return X


def norm_softsign(X):
    return 0.5 + X/(1+torch.abs(X))

def norm_tanh_softsign(X):
    return  norm_softsign(torch.tanh(X))
    

class Normalization(nn.Module):
    def __init__(self, device, mean=None, std=None):
        super().__init__()
        if mean is None:
            imagenet_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
            self.mean = imagenet_normalization_mean
        else:
            self.mean = mean
        if std is None:
            imagenet_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
            self.std = imagenet_normalization_std
        else:
            self.std = std
        self.mean = torch.as_tensor(self.mean).view(-1, 1, 1)
        self.std = torch.as_tensor(self.std).view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std

    def to(self: torch.nn.Module, device: Optional[Union[int, torch.device]] = ..., dtype: Optional[Union[torch.dtype, str]] = ...,
           non_blocking: bool = ...) -> torch.nn.Module:
        super().to(device, dtype, non_blocking)
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self


def handle_normalization(X, norm, imagenet_norm=None):
    assert callable(norm) or norm in available_normalizations, f"Unknown normalize_outputs value {norm}"
    if norm == "neat":
        X = norm_neat(X)
    elif norm == "inv_neat":
        X = 1.0-norm_neat(X)
    elif norm == "sqr_neat":
        X = norm_neat(X**2)
    elif norm == "clamp":
        X = torch.clamp(X, 0, 1)
    elif norm == "sigmoid":
        X = torch.sigmoid(X)
    elif norm == "sigmoid_like":
        X = norm_sigmoid_like(X)
    elif norm == "min_max_sigmoid_like":
        X = norm_min_max_sigmoid_like(X)
    elif norm == 'abs_tanh':
        # https://www.desmos.com/calculator/bluf2zbj8o
        X = norm_tanh(X, 3.0, 0, 1.0)
    elif norm == 'inv_abs_tanh':
        # https://www.desmos.com/calculator/no8l6sy1hh
        X = norm_tanh(X, 3.0, 1.15, -1.15)
    elif norm == 'min_max':
       X = norm_min_max(X)
    elif norm == 'inv_min_max':
        X = 1.0-norm_min_max(X)
    elif norm == 'min_max_sqr':
        X = norm_min_max(X) ** 2
    elif norm == 'inv_min_max_sqr':
        X = 1.0-norm_min_max(X)**2
    elif norm == 'inv_abs_min_max_cube':
        X = norm_min_max(abs(X))
        X = 1.13-X**3
    elif norm == 'inv_abs_min_max_sqr':
        X = norm_min_max(abs(X))
        X = 1.26-X**2
    elif norm == 'inv_abs_min_max':
        X = norm_min_max(abs(X))
        X = 1.5-X
    elif norm == 'abs_min_max':
        X = norm_min_max(abs(X))
    elif norm == 'min_max_sqr_imagenet':
        X = norm_min_max(X) **2
        X = imagenet_norm(X)
    elif norm == 'min_max_channel':
        X = norm_min_max_channel(X)
    elif norm == 'min_max_channel_sqr':
        X = norm_min_max_channel(X)**2
    elif norm == 'imagenet':
        X = imagenet_norm(X)
        X = torch.sigmoid(X)
    elif norm == 'sigmoid_imagenet':
        X = torch.sigmoid(X)
        X = imagenet_norm(X)
    elif norm == 'min_max_imagenet':
        X = norm_min_max(X)
        X = imagenet_norm(X)
    elif norm == 'imagenet_min_max':
        X = imagenet_norm(X)
        X = norm_min_max(X)
    elif norm == 'inv_abs_imagenet':
        X = 1.0 - torch.abs(X)
        X += 0.5
        X = imagenet_norm(X)
    elif norm == "neat_sqr_imagenet":
        X = 1.0 - torch.abs(X**2)
        X = imagenet_norm(X)
        X = torch.sigmoid(X)
    elif norm == 'softsign':
        X = norm_softsign(X)
    elif norm == 'tanh_softsign':
        X = norm_tanh_softsign(X)
    else:
        # callable
        X = norm(X)
    return X



if __name__== '__main__':
    import matplotlib.pyplot as plt
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # test the different normalization options:
    x_vals = torch.linspace(-1.0, 1.0, 256)
    y_vals = torch.linspace(-1.0, 1.0, 256)

    # initialize to 0s
    coords = torch.zeros((256, 256, 3))

    # assign values:
    coords[:, :, 0] = y_vals.unsqueeze(1).repeat(1, 256)
    coords[:, :, 1] = x_vals.unsqueeze(0).repeat(256, 1)
    # distance from center:
    coords[:, :, 2] = torch.sqrt(coords[:, :, 0] ** 2 + coords[:, :, 1] ** 2)
    
    plt.title("coords")
    plt.imshow(coords)
    plt.show()  
    # plt.imshow(coords[:,:,1])
    # plt.show()  

    img = coords.clone()
    img[:, :, 0] = .5+torch.sin(2.0*img[:, :, 0])
    img[:, :, 1] = torch.cos(0.5*img[:, :, 1])
    
    plt.title("img")
    plt.imshow(img)
    plt.show()  

    img = img.permute(2,0,1).to(device)
    coords = coords.permute(2,0,1).to(device)
    #%%
    from cppn.normalization import *
    imagenet = Normalization(device)
    for norm in available_normalizations:
        Y = handle_normalization(img, norm, imagenet)
        plt.title(f"{norm} {torch.min(Y):.3f} {torch.max(Y):.3f}")
        plt.imshow(Y.permute(1,2,0).cpu())
        plt.show()
