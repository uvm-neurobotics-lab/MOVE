import torch
from cppn_torch.graph_util import activate_population
from torchvision.transforms import Resize
from functorch.compile import aot_function, make_boxed_compiler

import __main__ as main
if not hasattr(main, '__file__'):
    try:
        from tqdm.notebook import trange
    except ImportError:
        from tqdm import trange
else:
    from tqdm import trange
from tqdm import tqdm
from norm import norm_tensor, norm_tensor_by_tensor

class EarlyStopping:
    def __init__(self, patience:int=1, min_delta:float=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_loss = torch.inf

    def check_stop(self, loss:float) -> bool:
        if loss < (self.min_loss + self.min_delta):
            self.min_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

resize = Resize((32, 32),antialias=True)
def min_resize(imgs):
    if imgs.shape[-1] < 32 or imgs.shape[-2] < 32:
        return resize(imgs)
    return imgs

def sgd_weights(genomes, mask, inputs, target, fns, norm, config, early_stop=3):
    if isinstance(genomes[0], tuple):
        genomes = [g for _, g in genomes]
    all_params = []
    
    for c in genomes:
        P = c.prepare_optimizer()  # create parameters
        # all_params.extend([cx.weight for cx in c.enabled_connections()]) # doesn't work
        all_params.extend(P)
        
    if len(all_params) == 0:
        return
    
    # All CPPN weights in one optimizer
    optimizer = torch.optim.Adam(all_params, lr=config.sgd_learning_rate)
    # optimizer = torch.optim.SGD(all_params, lr=config.sgd_learning_rate)


    # Compile function
    def f(X, *gs):
        if config.activation_mode == 'population':
            return activate_population(gs[0], config, X)
        else:
            
            return torch.stack([g(X, force_recalculate=True) for g in gs[0]])
    def fw(f,_): return f
    
    compiled_fn = f
    pbar = trange(config.sgd_steps, leave=False) if config.sgd_steps > 5 else range(config.sgd_steps)
    
    if hasattr(config, 'use_aot') and config.use_aot:
        pbar.set_description_str("Compiling population AOT function... ")
        if torch.__version__.startswith("1") or config.activation_mode != 'node':
            if hasattr(config, 'use_aot') and config.use_aot:
                # super slow unless there are a ton of SGD steps
                compiled_fn = aot_function(f, fw_compiler=make_boxed_compiler(fw))
        else:
            torch._dynamo.config.verbose=True
            compiled_fn = torch.compile(f)


    # loss function
    def loss_fn(imgs, target):
        # print("IMG IN LOSS_FN", imgs[0][0][0][0])
        # prepare images
        # remove nan
        imgs[torch.isnan(imgs)] = 0 # TODO
        assert not torch.isnan(imgs).any(), "NaNs in images"
        
        
        if len(config.color_mode) == 1:
            # Grayscale images need to be converted to RGB for loss functions
            imgs = imgs.unsqueeze(1).repeat(1, 3, 1, 1)
            if len(target.shape) == 3:
                target = target.unsqueeze(1).repeat(1, 3, 1, 1)
        
        # move color channel to first dimension
        
        imgs = min_resize(imgs)
        
        # return torch.nn.functional.mse_loss(imgs,
        #                             target,
        #                             reduction='mean')
        
        
        # calculate fitness
        pop_size = imgs.shape[0]
        normed = torch.zeros((pop_size, len(fns)), device=imgs.device)
        for i, fn in enumerate(fns):
            fitness = fn(imgs, target)
            # print(f"Fitness {fn.__name__}: {fitness.mean().item():.3f} +/- {fitness.std().item():.3f} (min: {fitness.min().item():.3f}, max: {fitness.max().item():.3f})")
            if norm is not None:
                normed_fit = norm_tensor(fitness, norm, fn.__name__, clamp=True, warn=False)
            else:
                normed_fit = fitness # no normalization
            normed[:, i] = normed_fit
                
        if mask is not None:
            # mask loss by the functions in each cell
            normed = normed * mask.T
            
        normed[torch.isnan(normed)] = 0 # TODO
        assert not torch.isnan(normed).any()

        # return the inverse of the mean fitness
        inv = torch.sub(1.0, normed.mean())
        # print("inv_max_normed_masked", inv.max().item(), "inv_min_normed_masked", inv.min().item(), "inv_mean_normed_masked", inv.mean().item(), "inv_std_normed_masked", inv.std().item())
        return inv
    
    # Optimize
    step = 0
    stopping = EarlyStopping(patience=early_stop if early_stop else config.sgd_steps, min_delta=config.sgd_early_stop_delta)
    for step in pbar:
        imgs = compiled_fn(inputs, genomes)
        loss = loss_fn(imgs, target)
        
        # print(loss.min().item())
        # print("\n\n\n\n")
        # print("*"*100)
        # print("PARAM:", all_params[0], "\nCPPN:", genomes[0].get_params()[0])
        # print("*"*100)
        # print("\n\n\n\n")

        # loss = loss.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # print(all_params[0], genomes[0].get_params()[0])
        if early_stop and stopping.check_stop(loss.item()):
            break
        
        if isinstance(pbar, tqdm):
            pbar.set_postfix_str(f"loss={loss.detach().clone().mean().item():.3f}")
            pbar.set_description_str(f"Optimizing {len(all_params)} params")
        
    return step
        
