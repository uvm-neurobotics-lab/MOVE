import pyiqa
import torch
from torch import nn

available_fns = pyiqa.list_models()
print(available_fns)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def wrap(fn):
   if hasattr(fn, "metric_name"):
      fn.__name__ = fn.metric_name
   else:
      fn.__name__ = fn.__class__.__name__
   return fn

def nr_wrap(fn):
   # wrap for no-reference IQA metrics
   fn = wrap(fn)
   fn.__call__ = lambda x,y,fn=fn: fn.forward(x)
   return fn

def copy_fwd(fn, fwd):
   fwd.metric_name = fn.metric_name
   fwd.lower_better = fn.lower_better
   return fwd

def init_clipiqa():
   loss = wrap(pyiqa.create_metric('clipiqa', device=device, loss_reduction='none', as_loss=True))
   def fwd(x, y):
      out = loss.forward(x).mean(dim=(1))
      return out
   fwd.metric_name = "clipiqa"
   fwd.lower_better = loss.lower_better
   return fwd

def init_ahiq():
   loss = wrap(pyiqa.create_metric('ahiq', device=device, loss_reduction='none', as_loss=True))
   def fwd(x, y):
      out = loss.forward(x, y).mean(dim=(1)) # TODO need to look at what the mean is
      return out
   fwd.metric_name = "ahiq"
   fwd.lower_better = loss.lower_better
   return fwd

def init_ckdn():
   loss = wrap(pyiqa.create_metric('ckdn', device=device, loss_reduction='none', as_loss=True))
   def fwd(x, y):
      out = loss.forward(x, y).mean(dim=(1)) # TODO need to look at what the mean is
      return out
   copy_fwd(loss, fwd)
   return fwd

def init_dists():
   return wrap(pyiqa.create_metric('dists', device=device, loss_reduction='none', as_loss=True))

def init_fsim():
   return wrap(pyiqa.create_metric('fsim', device=device, loss_reduction='none', as_loss=True))

def init_topiq_fr():
   loss = wrap(pyiqa.create_metric('topiq_fr', device=device, loss_reduction='none', as_loss=True))
   def fwd(x, y):
      out = loss.forward(x, y).mean(dim=(1)) # TODO need to look at what the mean is
      return out
   copy_fwd(loss, fwd)
   return fwd

def init_gmsd():
   loss = wrap(pyiqa.create_metric('gmsd', device=device, loss_reduction='none', as_loss=True))
   loss.metric_name = "pyiqa-gmsd" # different normalization range
   return loss

def init_ms_ssim():
   return wrap(pyiqa.create_metric('ms_ssim', device=device, loss_reduction='none', as_loss=True))

def init_cw_ssim():
   ms_ssim = wrap(pyiqa.create_metric('cw_ssim', device=device, loss_reduction='none', as_loss=True))
   return ms_ssim

def init_ssim():
   return wrap(pyiqa.create_metric('ssim', device=device, loss_reduction='none', as_loss=True))

def init_pieapp():
   return wrap(pyiqa.create_metric('pieapp', device=device, loss_reduction='none', as_loss=True))

def init_lpips():
   return wrap(pyiqa.create_metric('lpips', device=device, loss_reduction='none', as_loss=True))

def init_mad():
   return wrap(pyiqa.create_metric('mad', device=device, loss_reduction='none', as_loss=True))

def init_mse():
   loss = pyiqa.losses.MSELoss(reduction='none').to(device)
   def fwd(x,y):
      return loss(x,y).mean(dim=(1,2,3))
   loss.__call__ = fwd
   loss.metric_name = "mse"
   loss.lower_better = True
   return wrap(loss)

def init_nima():
   loss = wrap(pyiqa.create_metric('nima', device=device, loss_reduction='none', as_loss=True))
   def fwd(x, y):
      out = loss.forward(x).mean(dim=(1))
      return out
   fwd.metric_name = "nima"
   fwd.lower_better = loss.lower_better
   return fwd

def init_psnr():
   loss = wrap(pyiqa.create_metric('psnr', device=device, loss_reduction='none', as_loss=True))
   def fwd(x,y):
      return loss.forward(x, y) / 50.0
   fwd.metric_name = "psnr"
   fwd.lower_better = loss.lower_better
   return fwd

def init_vif():
   return wrap(pyiqa.create_metric('vif', device=device, loss_reduction='none', as_loss=True))

def init_vsi():
   return wrap(pyiqa.create_metric('vsi', device=device, loss_reduction='none', as_loss=True))

def init_nlpd():
   return wrap(pyiqa.create_metric('nlpd', device=device, loss_reduction='none', as_loss=True))


map_inits = {
   # "ahiq": init_ahiq,
   # "ckdn": init_ckdn,
   "clipiqa": init_clipiqa,
   "cw_ssim": init_cw_ssim,
   "fsim": init_fsim,
   "ssim": init_ssim,
   "lpips": init_lpips,
   "mse": init_mse,
   "topiq_fr": init_topiq_fr,
   "vif": init_vif,
   "vsi": init_vsi,
   "dists": init_dists,
   
   # "nlpd": init_nlpd,
   # "psnr": init_psnr,
   # "gmsd": init_gmsd,
   
   
   # "nima": init_nima,
   # "mad": init_mad,
   # "pieapp": init_pieapp,
   # "ms_ssim": init_ms_ssim, # needs larger images
   }
# haarpsi, dss, mdsi, style ?

def initialize_fns(names):
      fns = []
      for name in names:
         assert name in map_inits, f"Unknown fitness function {name}"
         fns.append(map_inits[name]())
         
      for i, f in enumerate(fns):
         if f.lower_better:
            # fns[i].__call__ = lambda x,y: 1.0-f.forward(x,y)
            fns[i] = lambda x,y,f=f: 1.0-f.__call__(x,y)
            fns[i].lower_better = False
            fns[i].__name__ = f.__name__
         
         if hasattr(f, "metric_name"):
            fns[i].__name__ = f.metric_name
      return fns


def correct_dims(candidates, target):
   f,r = candidates, target
   if len(f.shape) == 2:
      # unbatched L
      f = f.repeat(3, 1, 1) # to RGB
      f = f.unsqueeze(0) # create batch
   elif len(f.shape) == 3:
      # either batched L or unbatched RGB
      if min(f.shape) == 3:
         # color images
         if torch.argmin(torch.tensor(f.shape)) != 0:
            f = f.permute(2,0,1)
         f = f.unsqueeze(0) # batch
      else:
         # batched L
         f = torch.stack([x.repeat(3,1,1) for x in f])
   else:
      # color
      if f.shape[1] != 3:
         if f.shape[1] == 1:
            # L
            f = f.repeat(1,3,1,1)
         else:
            # RGB in wrong order
            f = f.permute(0,3,1,2)
   if len(r.shape) == 2:
      # unbatched L
      r = r.repeat(3,1,1) # to RGB
      r = r.unsqueeze(0) # create batch
   elif len(r.shape) == 3:
      # either batched L or unbatched RGB
      if min(r.shape) == 3:
         # color images
         if torch.argmin(torch.tensor(r.shape)) != 0:
            # move color to front
            r = r.permute(2,0,1)
         r = r.unsqueeze(0) # batch
      else:
         # batched L
         r = torch.stack([x.repeat(3,1,1) for x in r])         
   else:
      # color
      if r.shape[1] != 3:
         r = r.permute(0,3,1,2)

   f = f.to(torch.float32)
   r = r.to(torch.float32)
   
   # pad to 32x32 if necessary
   if f.shape[2] < 32 or f.shape[3] < 32:
      f = Resize((32,32),antialias=False)(f)
   if r.shape[2] < 32 or r.shape[3] < 32:
      r = Resize((32,32),antialias=False)(r)

   if f.shape[0] !=1 and r.shape[0] == 1:
      # only one target in batch, repeat for comparison
      r = torch.stack([r.squeeze() for _ in range(f.shape[0])])

   return f,r

all_available_fns = map_inits.keys()
# CW-SSIM

# GENOTYPE_FUNCTIONS = [min_nodes, max_nodes, min_connections, max_connections, max_activation_fns, modularity, partition_coverage, partition_performance, partition_quality, max_width, avg_width, depth, age, inv_age, std_of_weights, mean_of_weights, sgd_loss_delta, n_cells]
GENOTYPE_FUNCTIONS = []
NO_GRADIENT = GENOTYPE_FUNCTIONS + []
NO_MEAN = NO_GRADIENT
NO_NORM = GENOTYPE_FUNCTIONS + []
name_to_fn = {k:v for k,v in locals().items() if callable(v) and not k.startswith("__") and not k=="correct_dims"}
