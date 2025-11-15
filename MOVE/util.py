import matplotlib.pyplot as plt
from typing import List, Union
import math
import torch
import numpy as np
from contextlib import contextmanager, nullcontext


def show_move_map_images(move):
    pop = move.map.get_population()
    imgs = [g(move.inputs.cpu(), channel_first=False).detach().cpu() for g in pop]
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
        images = torch.stack(images)
    if isinstance(images, torch.Tensor):
        images = images.detach().cpu().numpy()
    elif isinstance(images, np.ndarray):
        images = np.array(images)

    num = images.shape[0]

    rows = math.ceil(num / cols)
    fig, axs = plt.subplots(rows, cols, constrained_layout=True, figsize=(cols*2, rows*2))
    axs = np.atleast_2d(axs)

    def _prepare_image(img: np.ndarray) -> np.ndarray:
        if img.ndim == 4:
            # Handle a batch within a batch – take the first element
            img = img[0]
        if img.ndim == 3:
            if img.shape[0] in (1, 3):
                img = np.moveaxis(img, 0, -1)
            elif img.shape[-1] in (1, 3):
                pass
            else:
                # Collapse unexpected middle dimension
                img = img[0]
        if img.ndim == 2:
            return img
        if img.ndim == 3 and img.shape[-1] == 1:
            return img[..., 0]
        return img

    empty_image = _prepare_image(images[0]).copy()

    for i, ax in enumerate(axs.flatten()):
        ax.axis("off")
        if i >= num:
            ax.imshow(empty_image, cmap=cmap if empty_image.ndim < 3 else None, vmin=0, vmax=1)
            continue

        img = _prepare_image(images[i])
        ax.imshow(img, cmap=cmap if img.ndim < 3 or img.shape[-1] == 1 else None, vmin=0, vmax=1)
        if titles is not None:
            ax.set_title(f"Input {titles[i]}")
    if show:
        fig.tight_layout()
        fig.show()
    return fig
        


def get_dynamic_mut_rate(rate, run_progress, end_mod):
    return rate - (rate - end_mod * rate) * run_progress


def is_canonical_image_batch(imgs: torch.Tensor, *, min_size: int = 33) -> bool:
    """Return ``True`` when ``imgs`` already matches MOVE's expected layout.

    Images that are 4-D, channel-first RGB, float32, finite, and clamped to
    $[0, 1]$ with minimum spatial size ``min_size`` can skip additional
    preprocessing work.
    """

    if not isinstance(imgs, torch.Tensor):
        return False
    if imgs.ndim != 4 or imgs.shape[1] != 3:
        return False
    if imgs.dtype != torch.float32:
        return False
    if imgs.shape[-2] < min_size or imgs.shape[-1] < min_size:
        return False
    if not torch.isfinite(imgs).all():
        return False
    if torch.any(imgs < 0) or torch.any(imgs > 1):
        return False
    return True


@contextmanager
def maybe_autocast(device: Union[str, torch.device], *, enabled: bool = True, dtype: torch.dtype = torch.float16):
    device = torch.device(device) if not isinstance(device, torch.device) else device
    if not enabled or device.type != "cuda":
        with nullcontext():
            yield
        return

    with torch.cuda.amp.autocast(dtype=dtype):
        yield
