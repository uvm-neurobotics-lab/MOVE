import torch
from torch import nn

import math
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from torch import nn
import torch
from typing import List, Union
from cv2 import resize as cv2_resize
import itertools
import random

# from cppn.cppn import NodeType
from cppn.normalization import handle_normalization
   
from torchvision.transforms import GaussianBlur

def print_net(individual, show_weights=False, visualize_disabled=False):
    print(f"<CPPN {individual.id}")
    print(f"nodes:")
    for k, v in individual.node_genome.items():
        print("\t",k, "\t|\t",v.layer, "\t|\t",v.activation.__name__)
    print(f"connections:")
    for k, v in individual.connection_genome.items():
        print("\t",k, "\t|\t",v.enabled, "\t|\t",v.weight)
    print(">")
  


def name_to_fn(name):
    """
    Converts a string to a function.
    params:
        name: The name of the function.
    returns:
        The function.
    """
    raise NotImplementedError("")
    if isinstance(name, (Callable,)) or name is None:
        return name
    assert isinstance(name, str), f"name must be a string but is {type(name)}"
    if name == "":
        return None
    fns = inspect.getmembers(sys.modules[af.__name__])
    fns.extend(inspect.getmembers(sys.modules[ff.__name__]))
    
    fns.extend([("round", lambda x: torch.round(x))])
    
    if name == "Conv2d":
        return torch.nn.Conv2d
    
    try:
        return fns[[f[0] for f in fns].index(name)][1]
    except ValueError:
        raise ValueError(f"Function {name} not found.")


def get_max_number_of_hidden_nodes(population):
    max = 0
    for g in population:
        if len(list(g.hidden_nodes))> max:
            max = len(list(g.hidden_nodes))
    return max

def get_avg_number_of_hidden_nodes(population):
    count = 0
    if len(population) == 0:
        return 0
    for g in population:
        count+=len(g.node_genome) - g.n_in_nodes - g.n_outputs
    return count/len(population)

def get_max_number_of_connections(population):
    max_count = 0
    for g in population:
        count = len(list(g.enabled_connections))
        if(count > max_count):
            max_count = count
    return max_count

def get_min_number_of_connections(population):
    min_count = math.inf
    for g in population:
        count = len(list(g.enabled_connections)) 
        if(count < min_count):
            min_count = count
    return min_count

def get_avg_number_of_connections(population):
    count = 0
    if len(population) == 0:
        return 0
    for g in population:
        count+=len(list(g.enabled_connections))
    return count/len(population)



def upscale_conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=True, device="cpu"):
    # return ConvTranspose2d(in_channels,out_channels,kernel_size, stride=stride, padding=padding, output_padding=1,device=device)
    layer = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='bilinear'),
        nn.ReflectionPad2d(1),
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, device=device, bias=bias)
   )
    return layer

def show_inputs(inputs, cols=8, cmap='viridis'):
    if not isinstance(inputs, torch.Tensor):
        # assume it's an algorithm instance
        inputs = inputs.inputs
        try:
            inputs = handle_normalization(inputs, inputs.config.normalize_outputs)
        except:
            pass # no normalization
    inputs = inputs.permute(2,0,1)
    image_grid(inputs,
               cols=cols,
               show=True,
               cmap=cmap,
               suptitle="Inputs")


def image_grid(images,
                cols=4,
                titles=None,
                show=True,
                cmap='gray',
                suptitle=None,
                title_font_size=12,
                fig_size=(10,10)):
    
    if isinstance(images, torch.Tensor):
        images = images.detach().cpu().numpy()
        images = [i for i in images]
    fg = plt.figure(constrained_layout=True, figsize=fig_size)
    rows = 1 + len(images) // cols
    for i, img in enumerate(images):
        ax = fg.add_subplot(rows, cols, i + 1)
        ax.axis('off')
        ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
        if titles is not None:
            ax.set_title(titles[i])
    if suptitle is not None:
        fg.suptitle(suptitle, fontsize=title_font_size)
    if show:
        fg.show()
    else:
        return plt.gcf()


def custom_image_grid(images:Union[torch.Tensor, np.ndarray, List[torch.Tensor]],
               cols=8, titles=None, show=True, cmap="gray"):
    assert titles is None or len(titles) == len(images)
    if isinstance(images, List):
        images = torch.stack(images).detach().cpu()
    elif isinstance(images, np.ndarray):
        ...
    elif isinstance(images, torch.Tensor):
        images = images.detach().cpu()
    
    num = images.shape[0]
    
    rows = math.ceil(num / cols)
    fig, axs = plt.subplots(rows, cols, constrained_layout=True, figsize=(cols*2, rows*2))
    for i, ax in enumerate(axs.flatten()):
        ax.axis("off")
        if i >= num:
            ax.imshow(np.ones((num, images.shape[1], 3)), cmap="gray")
        else:    
            ax.imshow(images[:, :, i], cmap=cmap, vmin=0, vmax=1)
            if titles is not None:
                ax.set_title(f"Input {titles[i]}")
    if show:
        fig.tight_layout()
        fig.show()
    return fig
        


def gaussian_blur(img, sigma, kernel_size=(5,5)):
    return GaussianBlur(kernel_size=kernel_size, sigma=sigma)(img)
        
        
def resize(img, size):
    return cv2_resize(img, size)

def center_crop(img, r, c):
    h, w = img.shape[:2]
    r1 = int(round((h - r) / 2.))
    c1 = int(round((w - c) / 2.))
    return img[r1:r1 + r, c1:c1 + c]

def random_uniform(generator, low=0.0, high=1.0, grad=False):
    if generator:
        return ((low - high) * torch.rand(1, device=generator.device, requires_grad=grad, generator=generator) + high)[0]
    else:
        return ((low - high) * torch.rand(1, requires_grad=grad) + high)[0]
    
def random_normal (generator=None, mean=0.0, std=1.0, grad=False):
    if generator:
        return torch.randn(1, device=generator.device, requires_grad=grad, generator=generator)[0] * std + mean
    else:
        return torch.randn(1, requires_grad=grad)[0] * std + mean


def random_choice(options, count=1, replace=False):
    """Chooses a random option from a list of options"""
    if not replace:
        indxs = torch.randperm(len(options))[:count]
        output = []
        for i in indxs:
            output.append(options[i])
        if count == 1:
            return output[0]
        return output
    else:
        return options[torch.randint(len(options), (count,))]
    




def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


def initialize_inputs(res_h, res_w, use_radial_dist, use_bias, n_inputs, device, coord_range=(-.5,.5), dtype=torch.float32):
        """Initializes the pixel inputs."""
        if not isinstance(coord_range[0], tuple):
            # assume it's a single range for both x and y
            coord_range_x = coord_range
            coord_range_y = coord_range
        else:
            coord_range_x, coord_range_y = coord_range
            
        # Pixel coordinates are linear within coord_range
        x_vals = torch.linspace(coord_range_x[0], coord_range_x[1], res_w, device=device,dtype=dtype)
        y_vals = torch.linspace(coord_range_y[0], coord_range_y[1], res_h, device=device,dtype=dtype)

        # initialize to 0s
        inputs = torch.zeros((res_h, res_w, n_inputs), dtype=dtype, device=device, requires_grad=False)

        # assign values:
        inputs[:, :, 0] = y_vals.unsqueeze(1).repeat(1, res_w)
        inputs[:, :, 1] = x_vals.unsqueeze(0).repeat(res_h, 1)
            
        
        if use_radial_dist:
            # d = sqrt(x^2 + y^2)
            inputs[:, :, 2] = torch.sqrt(inputs[:, :, 0]**2 + inputs[:, :, 1]**2)
        if use_bias:
            inputs[:, :, -1] = torch.ones((res_h, res_w), dtype=dtype, device=device, requires_grad=False) # bias = 1.0
        
        repeat_dims = 2 # just y, x
        if use_radial_dist:
            repeat_dims += 1 # add radial dist
        n_repeats = 0   
        for i in range(n_repeats):
            inputs  = torch.cat((inputs, inputs[:, :, :repeat_dims]), dim=2)
        
        return inputs



# FROM: https://github.com/limacv/RGB_HSV_HSL
def hsl2rgb_torch(hsl: torch.Tensor) -> torch.Tensor:
    hsl = hsl.unsqueeze(0)
    # hsl = hsl.permute(2, 0, 1).unsqueeze(0)
    hsl_h, hsl_s, hsl_l = hsl[:, 0:1], hsl[:, 1:2], hsl[:, 2:3]
    _c = (-torch.abs(hsl_l * 2. - 1.) + 1) * hsl_s
    _x = _c * (-torch.abs(hsl_h * 6. % 2. - 1) + 1.)
    _m = hsl_l - _c / 2.
    idx = (hsl_h * 6.).type(torch.uint8)
    idx = (idx % 6).expand(-1, 3, -1, -1)
    rgb = torch.empty_like(hsl)
    _o = torch.zeros_like(_c)
    rgb[idx == 0] = torch.cat([_c, _x, _o], dim=1)[idx == 0]
    rgb[idx == 1] = torch.cat([_x, _c, _o], dim=1)[idx == 1]
    rgb[idx == 2] = torch.cat([_o, _c, _x], dim=1)[idx == 2]
    rgb[idx == 3] = torch.cat([_o, _x, _c], dim=1)[idx == 3]
    rgb[idx == 4] = torch.cat([_x, _o, _c], dim=1)[idx == 4]
    rgb[idx == 5] = torch.cat([_c, _o, _x], dim=1)[idx == 5]
    rgb += _m
    rgb = rgb.squeeze(0)#.permute(1, 2, 0)
    return rgb





def calculate_diversity_full(population):
    if len(population) == 0:
        return torch.zeros(1)[0], torch.zeros(1)[0], torch.zeros(1)[0]
    # very slow, compares every genome against every other
    diffs = []
    for i in population:
        for j in population:
            if i== j: continue
            diffs.append(i.genetic_difference(j))

    std_distance = np.std(diffs)
    avg_distance = np.mean(diffs)
    max_diff = np.max(diffs)if(len(diffs)>0) else 0
    return std_distance, avg_distance, max_diff

def calculate_diversity_stochastic(population):
    if len(population) == 0:
        return torch.zeros(1)[0], torch.zeros(1)[0], torch.zeros(1)[0]
    # compare 10% of population
    diffs = torch.zeros(len(population)//10, device=population[0].config.device)
    pop = population
    num = len(pop)//10
    pairs = itertools.combinations(pop, 2)
    pairs = random.sample(list(pairs), num)
    for i, (g1, g2) in enumerate(pairs):
        diffs[i] = g1.genetic_difference(g2)
        assert not torch.isnan(diffs[i]).any(), f"nan in diffs {i} {g1.id} {g2.id}" 
    max_diff = torch.max(diffs) if(len(diffs)>0) else torch.tensor(0).to(population[0].config.device)
    if max_diff == 0:
        return torch.zeros(1)[0], torch.zeros(1)[0], torch.zeros(1)[0]
    std_distance = torch.std(diffs)
    avg_distance = torch.mean(diffs)
    return std_distance, avg_distance, max_diff
