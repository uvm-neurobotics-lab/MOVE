from cppn.cppn import CPPN
from move_config import MoveConfig, resize_target
import json
import os 
import torch
import tqdm
from cppn.cppn import Node
from move_config import MoveConfig, resize_target
import imageio.v2 as iio

import matplotlib.pyplot as plt
import numpy as np
import tqdm



        
def set_max_node_id(genomes_by_batch):
    max_id = max([max([int(node.id) for node in cppn.nodes.values()]) for cppn in genomes_by_batch.values()])
    CPPN.current_node_id = max_id + 1
    

def load_genome(path)->tuple[CPPN, MoveConfig]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(path, "r") as f:
        genome = json.load(f)
    if "config" in genome:
        # TODO REMOVE:
        genome["config"] = json.loads(genome["config"])
        if not "version" in genome["config"]:
            genome["config"]["version"] = "0.0.0"
        # END REMOVE
        config = MoveConfig.create_from_json(genome["config"], MoveConfig)
        if config.device != device:
            config.device = device # make sure we're using the right device regardless of what's in the file
        genome = genome["genome"]
    else:
        config_path = os.path.join(os.path.dirname(path), "../config.json")
        config = MoveConfig.create_from_json(config_path, MoveConfig)
        
    cppn = CPPN.create_from_json(genome, config=config)
    cppn.to(config.device)
    
    return (cppn, config)




def get_cppns_by_batch(path, modulo=1):
    batches = {}
    path = os.path.join(path, "genomes")
    sorted_list = sorted(os.listdir(path), key=lambda x: int(x.split("_")[1].split(".")[0]))
    pbar = tqdm.tqdm(total=len(sorted_list))
    for i, genome_path in enumerate(sorted_list):
        if i % modulo != 0:
            continue
        if genome_path.endswith(".json"):
            cppn, config = load_genome(os.path.join(path, genome_path))
            # print(f"{genome_path}: {len(cppn.nodes)} nodes, {len(cppn.connections)} connections")
            gen = genome_path.split("_")[1].split(".")[0]
            # this_cppn = cppn.clone(new_id=False, config=config)
            batches[int(gen)] = cppn
            assert isinstance(cppn, CPPN)
            assert isinstance(config, MoveConfig)
            for node in cppn.nodes.values():
                assert isinstance(node, Node)
            pbar.update(1)
    pbar.close()
    print("\nLoaded", len(batches), "genomes")
    return batches, config




def load_inputs(path, device='cpu'):
    path = os.path.join(path, "inputs.pt")
    inputs = torch.load(path, map_location=device)
    return inputs
    
    
def get_image(cppn, inputs):
    img = cppn(inputs, channel_first=False)
    return img.detach().cpu().numpy()


def show_image(cppn, inputs):
    plt.style.use("default")
    img = get_image(cppn, inputs)
    plt.imshow(img.detach().cpu().numpy(), cmap="gray")
    plt.show()


def show_image_grid(images, titles=[],save_name=None):
    num_images = len(images)
    
    # Calculate rows and cols based on the number of images
    cols = int(num_images**0.5)
    rows = cols if cols**2 == num_images else cols + 1

    # Adjust cols if necessary
    if rows * cols < num_images:
        cols += 1

    fig, axs = plt.subplots(rows, cols, figsize=(10,10), squeeze=False) # Ensure axs is always a 2D array
    
    # Hide any extra subplots
    for ax in axs.flat[num_images:]:
        ax.set_visible(False)

    for i in range(len(images)):
        ax = axs[i // cols, i % cols]
        ax.imshow(images[i], cmap="gray")
        if i < len(titles):
            ax.set_title(titles[i])
        ax.axis("off")
        
    plt.tight_layout()
    if save_name is not None:
        plt.savefig(save_name)
    plt.show()
    
    
def show_img_progression(genomes_by_batch, inputs, save_name):
    plt.style.use("default")
    plt.rcParams.update({'font.size': 12})
    sorted_keys = sorted(genomes_by_batch.keys())
    imgs = []
    titles = []
    for k in sorted_keys:
        imgs.append(get_image(genomes_by_batch[k], inputs))
        num_nodes = len(genomes_by_batch[k].nodes)
        num_cxs = len(genomes_by_batch[k].connections)
        titles.append(r"$\bf{"+str(k)+"}$: "+f"{num_nodes};{num_cxs}")
    
    
    show_image_grid(imgs, titles, save_name)
    


def load_target(path, config):
    with open(os.path.join(path, "target.txt"), "r") as f:
        target_path = f.read()
    config.target_name = target_path
    target_path = os.path.join("../",target_path) # move out of analysis directory
    pilmode = "RGB" if len(config.color_mode) == 3 else "L"
    target = torch.tensor(iio.imread(target_path, pilmode=pilmode), dtype=torch.float32, device=config.device)
    if target.max()>1.0:
        target /= 255.0
    config.target = target
    resize_target(config)
    config.target =  config.target.unsqueeze(0)
    if len(config.target.shape) == 3:
        # L -> RGB
        config.target = config.target.unsqueeze(1)
        config.target = config.target.repeat(1, 3, 1, 1)
    else:
        # RGB -> RGB
        config.target = config.target.permute(0,3,1,2)
    return config.target
