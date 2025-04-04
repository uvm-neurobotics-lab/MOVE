import copy
import matplotlib.pyplot as plt
import logging
import os
import torch
from torchvision.transforms import Resize
import __main__ as main


from tqdm import trange
from tqdm import tqdm
from norm import norm_tensor, norm_tensor_by_tensor


class EarlyStopping:
    def __init__(self, patience:int=1, min_delta:float=0, n_genomes=0, device='cpu'):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_loss = torch.inf
        
        self.counter_tensor = torch.zeros(n_genomes, device=device)
        self.min_loss_tensor = torch.ones(n_genomes, device=device)*torch.inf

    def check_stop(self, loss:float) -> bool:
        if loss < (self.min_loss + self.min_delta):
            self.min_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    
    def mask_stop(self, loss):
        loss=loss.flatten()
        improve = loss < (self.min_loss_tensor + self.min_delta)
        
        self.counter_tensor = torch.where(improve, 0, self.counter_tensor+1)
        self.min_loss_tensor = torch.where(improve, loss, self.min_loss_tensor)
        
        result = self.counter_tensor >= self.patience

        # shorten tensors
        # self.min_loss_tensor = torch.where(result, torch.nan, self.min_loss_tensor) # replace the losses that are stopped with nan
        # self.min_loss_tensor = self.min_loss_tensor[~torch.isnan(self.min_loss_tensor)] # remove nan
        # self.counter_tensor = torch.where(result, torch.nan, self.counter_tensor) # replace the counters that are stopped with nan
        # self.counter_tensor = self.counter_tensor[~torch.isnan(self.counter_tensor)] # remove nan
        
        return result

resize = Resize((32, 32),antialias=True)
def min_resize(imgs):
    if imgs.shape[-1] < 32 or imgs.shape[-2] < 32:
        return resize(imgs)
    return imgs

def anneal_steps_sigmoid(gen, max_gens):
    # https://www.desmos.com/calculator/x8mntiv4bc
    a = 200 # max steps
    c = 0.01 # steepness
    x = torch.tensor(gen)
    d = int(max_gens*0.5) # halfway point of annealing
    steps = int(torch.floor(a / (1+torch.exp(-c*(x-d)))-1).item())
    steps = max(steps, 0)
    return steps

def anneal_steps_linear(gen, max_gens):
    # https://www.desmos.com/calculator/30cpcsfwzm
    a = 800 # max steps
    c = 4 # steepness
    d = 5 # delay
    x = torch.tensor(gen) if not isinstance(gen, torch.Tensor) else gen
    steps = int(torch.floor(c*(x-d)).item())
    steps = max(steps, 0)
    steps = min(steps, a)
    return steps
    

def save_annealing_plot(max_gens, fn, config):
    X = torch.linspace(0, max_gens, max_gens)
    Y = [fn(x, max_gens) for x in X]
    plt.plot(X, Y)
    plt.xlabel("Generation")
    plt.ylabel("SGD Steps")
    plt.title("SGD Steps by Generation")
    plt.savefig(os.path.join(config.run_output_dir, "sgd_steps_annealing.pdf"))
    plt.close()


def anneal(config, sgd_steps, genomes, current_gen):
    anneal_fn = None
    
    if isinstance(sgd_steps, str) and sgd_steps == 'annealing-sigmoid':
        anneal_fn = anneal_steps_sigmoid
    elif isinstance(sgd_steps, str) and sgd_steps == 'annealing-linear':
        anneal_fn = anneal_steps_linear
        
    # do annealing:
    sgd_steps = anneal_fn(current_gen, config.num_generations)
    
    print("Annealing steps:", sgd_steps)
    if current_gen == 0:
        save_annealing_plot(config.num_generations, anneal_fn, config)
    if sgd_steps == 0:
        # apply a small random mutation to the weights
        for c in genomes:
            for p in c.parameters():
                if isinstance(p, torch.nn.Parameter):
                    p.data += torch.randn_like(p.data) * config.sgd_learning_rate
                else:
                    p += torch.randn(1).item() * config.sgd_learning_rate

def batch_lr_mod(inputs, config, all_params, lr):
    # Modify the learning rate by the batch size
    # Linear Scaling Rule: When the minibatch size is
    # multiplied by k, multiply the learning rate by k.
    # https://arxiv.org/pdf/1706.02677.pdf
    k = len(inputs)
    mod = config.batch_lr_mod * k
    lr = lr * mod
    for param_group in all_params:
        param_group['lr'] = param_group['lr'] * mod


def prep_images(imgs, config):
    assert torch.isfinite(imgs).all(), "NaNs in images"
    
    if len(config.color_mode) == 1:
        imgs = imgs.repeat(1, 3, 1, 1) # grayscale to RGB
    
    imgs = min_resize(imgs)
    
    return imgs


def sgd_weights(genomes, mask, inputs, target, fns, norm, config, early_stop=3, record_loss=None, skip_pbar=False, current_gen=0, unequal_shape=False):
    lr = config.sgd_learning_rate
    sgd_steps = config.sgd_steps

    if isinstance(sgd_steps, str) and 'annealing' in sgd_steps:
       anneal(config, sgd_steps, genomes, current_gen)
    if sgd_steps == 0:
        return 0 # took no steps
                        
    
    # if isinstance(genomes[0], tuple):
        # genomes = [g for c_,ci_,g in genomes]

    if mask is not None:
        # filter fns to only the ones that are enabled in mask
        fns = [fn for i, fn in enumerate(fns) if mask[i].any()]
        mask = mask[mask.any(dim=1)]
    
        
    all_params = []
    for _,_,c in genomes:
        all_params.extend([{'params': list(c.parameters()), 'lr': c.sgd_lr}])
        for p in c.parameters():
            p.requires_grad = True

    if len(fns) == 0 or len(all_params) == 0:
        print("No fitness functions or no parameters to optimize. Skipping.")
        return 0 # took no steps
    
    if config.batch_lr_mod:
        batch_lr_mod(inputs, config, all_params, lr)
    
    avg_lr = sum([p['lr'] for p in all_params])/len(all_params)
    
    # All CPPN weights in one optimizer 
    optimizer = torch.optim.AdamW(all_params, lr=lr, weight_decay=config.sgd_l2_reg)
    
    # Compile function
    def f(X, *gs):
        if unequal_shape:
            outs= []
            if len(X.shape)>3:
                outs = [g(X[i], force_recalculate=True, use_graph=True, channel_first=True) for i,g in enumerate(gs[0])]
            else:
                outs = [g(X, force_recalculate=True, use_graph=True, channel_first=True) for g in gs[0]]
            for i,o in enumerate(outs):
                if o.shape[0]!=3:
                    outs[i] = outs[i].repeat(3,1,1)
            return torch.stack([o for o in outs])
        if len(X.shape)>3:
            return torch.stack([g(X[i], force_recalculate=True, use_graph=True, channel_first=True) for i,g in enumerate(gs[0])])
        return torch.stack([g(X, force_recalculate=True, use_graph=True, channel_first=True) for g in gs[0]])
    
    if not skip_pbar:
        pbar = trange(sgd_steps, leave=False, disable=sgd_steps <= 5)

    # loss function
    def loss_fn(imgs, target, mask, return_all=False):
        # prepare images
        imgs = prep_images(imgs, config)
 
        # calculate fitness
        normed = torch.zeros((imgs.shape[0], len(fns)), device=imgs.device)
        for i, fn in enumerate(fns):
            fitness = fn(imgs, target)

            if norm is not None:
                normed_fit = norm_tensor(fitness, norm, fn.__name__, clamp=True, warn=False)
            else:
                normed_fit = fitness # no normalization
            normed[:, i] = normed_fit
                
        if mask is not None:
            normed = normed * mask.T # mask loss by the functions in each cell
            
        assert torch.isfinite(normed).all()

        # loss is the inverse of the mean fitness
        inv = torch.sub(1.0, normed)
        if return_all:
            return inv.mean(), inv.mean(dim=1)
        return inv.mean()
        
    
    n_params_total = sum([len(list(c.parameters())) for _,_,c in genomes])
    
    # Optimize
    fwd_passes = 0
    step = 0
    stopping = EarlyStopping(patience=early_stop if early_stop else sgd_steps, min_delta=config.sgd_early_stop_delta, n_genomes=len(genomes), device=config.device)
    stop_mask = torch.zeros(len(genomes), dtype=torch.bool, device=config.device)
    if skip_pbar:
        pbar = range(sgd_steps)
    
    # this_genomes = genomes
    this_mask = mask
    
    for step in pbar:
        this_target = target[:torch.sum(~stop_mask)] # take only the non stopped targets
        this_genomes =  [n for i,(_,_,n) in enumerate(genomes) if not stop_mask[i]]
        if mask is not None:
            this_mask = mask[:, ~stop_mask]
            # stop_mask = stop_mask[~stop_mask]
            # mask = mask * ~stop_mask

        for i,(_,_,c) in enumerate(genomes):
            for p in c.parameters():
                # freeze the parameters of the stopped genomes
                if c not in this_genomes:
                    p.requires_grad = False # manually freeze the parameters (after optimizer is defined, this is okay)
                # else:
                    # p.requires_grad = True
        n_params = sum([len(list(c.parameters())) for i,c in enumerate(this_genomes)])
        
        imgs = f(inputs, this_genomes)
        loss, all_loss = loss_fn(imgs, this_target, this_mask, True)
        
        # fill in torch.inf for the stopped genomes
        for i,(_,_,c) in enumerate(genomes):
            if stop_mask[i]:
                all_loss = torch.cat((all_loss[:i], torch.tensor([torch.inf], device=config.device), all_loss[i:]))
        
        if record_loss is not None:
            record_loss[step] = loss.item()
        
        if not loss.requires_grad:
            print("No gradients")
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
                    if not torch.isfinite(param.grad).all():
                        logging.warning(f"Non-finite gradients: {param.grad}")
                    param.grad[param.grad != param.grad] = 0       
                else:
                    param.grad = torch.zeros_like(param)
        
        if config.sgd_clamp_grad:
            torch.nn.utils.clip_grad_norm_(all_params, config.sgd_clamp_grad, error_if_nonfinite=True)
        
        if len(all_params) > 0:
            optimizer.step()
            
        if config.max_weight:
            for p in all_params:
                for param in p['params']:
                    param.data.clamp_(-config.max_weight, config.max_weight)
        
        fwd_passes += len(this_genomes)
        
        if early_stop:
            stop_mask = stopping.mask_stop(all_loss)
            if torch.sum(~stop_mask) == 0:
                break
        
        for param_group in all_params:
            for param in param_group['params']:
                assert torch.isfinite(param).all(), "Non-finite parameters after step"

        if early_stop and stopping.check_stop(loss.item()):
            break
        
        
        if isinstance(pbar, tqdm):
            pbar.set_postfix_str(f"loss={loss.detach().clone().mean().item():.3f}")
            pbar.set_description_str(f"Optimizing {n_params}/{n_params_total} params on {len(this_genomes)}/{len(genomes)} genomes and {len(fns)} fns lr: {avg_lr:.2e}")
    
        
    return step+1


"""
Require: Distribution over tasks P (T ), outer step size η, regularization strength λ,
    while not converged do
        Sample mini-batch of tasks {Ti}B i=1 ∼ P (T )
        for Each task Ti do
            Compute task meta-gradient gi = Implicit-Meta-Gradient(Ti, θ, λ)
        end for
        Average above gradients to get ˆ∇F (θ) = (1/B) ∑B i=1 gi
        Update meta-parameters with gradient descent: θ ← θ − η ˆ∇F (θ) // (or Adam)
    end while
"""
def sgd_weights_imaml(genomes, mask, inputs, target, fns, norm, config, early_stop=3, record_loss=None, skip_pbar=False, current_gen=0):
    assert mask is not None, "IMAML requires a cell-function mask"
    
    param_deltas = []
    total_steps = 0
    losses = []
    
    for i,task in enumerate(fns):
        these_genomes = [(None,None,genome.clone(config, new_id=False)) for _,_,genome in genomes]
        params_before = [torch.tensor(list(c.parameters()), device=config.device) for _,_,c in these_genomes]

        print("Task", i, task.__name__)
    
        total_steps += sgd_weights(these_genomes,
                                   None,
                                   inputs,
                                   target,
                                   [task],
                                   norm,
                                   config,
                                   early_stop,
                                   record_loss,
                                   skip_pbar,
                                   current_gen
                                   )
        if record_loss is not None:
            losses.append(record_loss.clone())
        
        params_after = [torch.tensor(list(c.parameters()), device=config.device) for _,_,c in these_genomes]
        
        delta = [p1-p0 for p0,p1 in zip(params_before, params_after)]
        param_deltas.append(delta)
    
    if record_loss is not None:
        record_loss[:] = torch.stack(losses).mean(dim=0) # record the average loss by step
    
    delta_mag = 0 # for recording

    if True:
        
        total_index = 0
        # apply the average delta to the genomes
        for i in range(len(genomes)):               # for each child genome
            for j,cell_mask in enumerate(mask.T):   # for each cell
                avg_delta = torch.stack([d[i]*cell_mask[fn_index] for fn_index,d in enumerate(param_deltas)])
                avg_delta = avg_delta.mean(dim=0)   # average the deltas for the cell
                delta_mag += torch.norm(avg_delta)  # record      
                
                if total_index < len(genomes):
                    genomes[total_index] = (total_index,None,genomes[i][-1].clone(config, new_id=False))
                else:
                    genomes.append((total_index,None,genomes[i][-1].clone(config, new_id=False)))
                
                for p,d in zip(genomes[total_index][-1].parameters(), avg_delta):
                    p.data += d
                total_index += 1
                    
    else:
        # apply the average delta to the genomes
        for i in range(len(genomes)):               # for each child genome
            avg_delta = torch.stack([d[i] for d in param_deltas])
            avg_delta = avg_delta.mean(dim=0)
            delta_mag += torch.norm(avg_delta)
            for p,d in zip(genomes[i][-1].parameters(), avg_delta):
                p.data += d
                
                
    print("Ended up with", len(genomes), "genomes")
    print("\t Avg. parameter delta magnitude:", delta_mag.item()/len(genomes))
    
    return total_steps
        
    
    