import matplotlib.pyplot as plt
from typing import List, Union
import math
import torch
import numpy as np


def show_move_map_images(move):
    pop = move.map.get_population()
    imgs = [g(move.inputs, channel_first=False).detach().cpu() for g in pop]
    image_grid(
        imgs,
        cols=10,
        titles=[f"{move.map.cell_names[i]}\n{g.id}:{g.fitness.item()}" for i, g in enumerate(pop)],
        show=True,
        fig_size=(30,30),
        suptitle=f"Generation {move.gen}",
        title_font_size=8,
    )
    

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
        


def get_dynamic_mut_rate(rate, run_progress, end_mod):
    return rate - (rate - end_mod * rate) * run_progress