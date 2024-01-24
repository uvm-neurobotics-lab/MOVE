import logging
import torch
from torchvision.transforms import Resize

import __main__ as main

# if not hasattr(main, '__file__'):
#     try:
#         from tqdm.notebook import trange
#     except ImportError:
#         from tqdm import trange
# else:
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


def sgd_weights(genomes, mask, inputs, target, fns, norm, config, early_stop=3, record_loss=None, skip_pbar=False):
    if isinstance(genomes[0], tuple):
        genomes = [g for c_,ci_,g in genomes]
    all_params = []

    if mask is not None:
        # filter fns to only the ones that are enabled in mask
        fns = [fn for i, fn in enumerate(fns) if mask[i].any()]
        mask = mask[mask.any(dim=1)]
    
    if len(fns) == 0:
        return 0
        
    for c in genomes:
        c.loss_delta = 0.0
        c.last_loss = 0.0
        all_params.extend([{'params': list(c.parameters()), 'lr': c.sgd_lr}])

    if len(all_params) == 0:
        return 0 # took no steps
    
    lr = config.sgd_learning_rate
    if config.batch_lr_mod:
        # Linear Scaling Rule: When the minibatch size is
        # multiplied by k, multiply the learning rate by k.
        # https://arxiv.org/pdf/1706.02677.pdf
        k = len(inputs)
        mod = config.batch_lr_mod * k
        lr = lr * mod
        for param_group in all_params:
            param_group['lr'] = param_group['lr'] * mod
    
    avg_lr = sum([p['lr'] for p in all_params])/len(all_params)
    
    # All CPPN weights in one optimizer # TODO allow for different learning rates per genome
    optimizer = torch.optim.AdamW(all_params, lr=lr, weight_decay=config.sgd_l2_reg)
    # optimizer = torch.optim.SGD(all_params, lr=lr)
    
    # Compile function
    def f(X, *gs):
        return torch.stack([g(X, force_recalculate=True, use_graph=True, channel_first=True) for g in gs[0]])
    def fw(f,_): return f
    
    compiled_fn = f
    if not skip_pbar:
        pbar = trange(config.sgd_steps, leave=False, disable=config.sgd_steps <= 5)
    
    if hasattr(config, 'use_aot') and config.use_aot:
        if not skip_pbar:
            pbar.set_description_str("Compiling population AOT function... ")
        if torch.__version__.startswith("1") or config.activation_mode != 'node':
            if hasattr(config, 'use_aot') and config.use_aot:
                # super slow unless there are a ton of SGD steps
                from functorch.compile import aot_function, make_boxed_compiler
                compiled_fn = aot_function(f, fw_compiler=make_boxed_compiler(fw))
        else:
            torch._dynamo.config.verbose=True
            compiled_fn = torch.compile(f)


    # loss function
    def loss_fn(imgs, target):
        # prepare images
        # remove nan
        assert torch.isfinite(imgs).all(), "NaNs in images"
        
        
        if len(config.color_mode) == 1:
            imgs = imgs.repeat(1, 3, 1, 1) # grayscale to RGB
        
        imgs = min_resize(imgs)
 
        # calculate fitness
        pop_size = imgs.shape[0]
        normed = torch.zeros((pop_size, len(fns)), device=imgs.device)
        for i, fn in enumerate(fns):
            fitness = fn(imgs, target)

            if norm is not None:
                normed_fit = norm_tensor(fitness, norm, fn.__name__, clamp=True, warn=False)
            else:
                normed_fit = fitness # no normalization
            normed[:, i] = normed_fit
        
        for i, c in enumerate(genomes):
            c.loss_delta += normed[i].mean().item() - c.last_loss
            c.last_loss = normed[i].mean().item()
                
        if mask is not None:
            # mask loss by the functions in each cell
            normed = normed * mask.T
            
        assert torch.isfinite(normed).all()

        # return the inverse of the mean fitness
        inv = torch.sub(1.0, normed.mean())
        
        return inv
    
    n_params = sum([len(list(c.parameters())) for c in genomes])
    
    # Optimize
    step = 0
    stopping = EarlyStopping(patience=early_stop if early_stop else config.sgd_steps, min_delta=config.sgd_early_stop_delta)
    if skip_pbar:
        pbar = range(config.sgd_steps)
    
    for step in pbar:
        imgs = compiled_fn(inputs, genomes)
        loss = loss_fn(imgs, target)
        
        if record_loss is not None:
            record_loss[step] = loss.item()
        
        if not loss.requires_grad:
            return step # no gradients, so we're done
        
        assert torch.isfinite(loss).all(), "Non-finite loss"
        
        optimizer.zero_grad()
        
        try:
            loss.backward() 
        except RuntimeError as e:
            logging.warning("RuntimeError in loss.backward()")
            import traceback
            traceback.print_exc()
            return step
        
        # make nan grads 0
        # TODO: prevent this upstream
        for param_group in all_params:
            for param in param_group['params']:
                if param.grad is not None:
                    # if not torch.isfinite(param.grad).all():
                        # logging.warning("Non-finite gradients")
                    param.grad[param.grad != param.grad] = 0       
                else:
                    param.grad = torch.zeros_like(param)
        
        if config.sgd_clamp_grad:
            torch.nn.utils.clip_grad_norm_(all_params, config.sgd_clamp_grad, error_if_nonfinite=True)
        
        if len(all_params) > 0:
            optimizer.step()
        
        for param_group in all_params:
            for param in param_group['params']:
                assert torch.isfinite(param).all(), "Non-finite parameters after step"

        if early_stop and stopping.check_stop(loss.item()):
            break
        
        if isinstance(pbar, tqdm):
            pbar.set_postfix_str(f"loss={loss.detach().clone().mean().item():.3f}")
            pbar.set_description_str(f"Optimizing {n_params} params on {len(genomes)} genomes and {len(fns)} fns lr: {lr:.2e} avg:{avg_lr:.2e}")
        
    return step+1
        
