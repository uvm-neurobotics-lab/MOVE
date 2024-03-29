"""Fitness functions."""
# from piq.fsim import FSIMLoss
# from piq.perceptual import DISTS as piq_dists
# from piq.perceptual import LPIPS as piq_lpips
import piq
from piq import StyleLoss
from skimage.transform import resize
import torch
from torchvision.transforms import Resize
# from torchvision.models import vgg16
import networkx as nx

from torchvision.models import vgg16, VGG16_Weights
from fitness.style_loss import StyleLoss
from fitness.dists import DISTS
from fitness.lpips import LPIPS
from fitness.dss import dss as piq_dss

FEATURE_EXTRACTOR = vgg16(weights=VGG16_Weights.DEFAULT).features

from piqa import HaarPSI

def control(candidates, target):
   return torch.rand(len(candidates), dtype=torch.float32, device=target.device)

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
   
   # replace NaNs with 0
   f = torch.nan_to_num(f) # TODO there should not be NaNs

   return f,r

def assert_images(*images):
   for img in images:
      assert img.dtype == torch.float32, "Fitness function expects float32 images"

def empty(candidates, target):
   raise NotImplementedError("Fitness function not implemented")

# Why not use MSE: https://ece.uwaterloo.ca/~z70wang/publications/SPM09.pdf
def mse(candidates, target):
   assert_images(candidates, target)
   return torch.sub(1.0, torch.mean((candidates-target).pow(2), dim=(1,2,3)))
   # return torch.sub(1.0, (candidates-target).pow(2).mean( dim=(1,2,3)))

def test(candidates, target):
   return (candidates/255).mean() # should get all white

def dists(candidates, target):
   if "DISTS_INSTANCE" in globals().keys() and candidates.device in globals()["DISTS_INSTANCE"].keys():
      dists_instance = globals()["DISTS_INSTANCE"][candidates.device]
   else:
      # dists_instance = piq_dists(reduction='none').eval().to(candidates.device)
      dists_instance = DISTS(FEATURE_EXTRACTOR).eval().to(candidates.device)
      if "DISTS_INSTANCE" not in globals().keys():
         globals()["DISTS_INSTANCE"] = {}
      globals()["DISTS_INSTANCE"][candidates.device] = dists_instance
   assert_images(candidates, target)
   # loss = dists_instance(candidates, target)
   # value = torch.tensor([1.0]*len(candidates)).to(loss) - loss
   # return torch.sub(1.0, dists_instance(candidates, target))
   
   val = dists_instance(candidates, target, require_grad=True, batch_average=False)
   if len(candidates) == 1:
      val = val.unsqueeze(0) # batch
   return torch.sub(1.0, val)
   
def lpips(candidates, target):
   # return 1.0 - lpips_instance(candidates, target)   
   if "LPIPS_INSTANCE" in globals().keys() and candidates.device in globals()["LPIPS_INSTANCE"].keys():
      lpips_instance = globals()["LPIPS_INSTANCE"][candidates.device]
   else:
      # lpips_instance = piq_lpips(reduction='none').eval()
      lpips_instance = LPIPS(FEATURE_EXTRACTOR, reduction='none').eval().to(candidates.device)
      if "LPIPS_INSTANCE" not in globals().keys():
         globals()["LPIPS_INSTANCE"] = {}
      globals()["LPIPS_INSTANCE"][candidates.device] = lpips_instance
   assert_images(candidates, target)
   value = torch.sub(1.0, lpips_instance(candidates, target))
   return value


def haarpsi(candidates, target):
   assert_images(candidates, target)
   if "HAARPSI_INSTANCE" in globals().keys() and candidates.device in globals()["HAARPSI_INSTANCE"].keys():
      haarpsi_instance = globals()["HAARPSI_INSTANCE"][candidates.device]
   else:
      haarpsi_instance = HaarPSI(reduction='none', value_range=1.0).to(candidates.device)
      if "HAARPSI_INSTANCE" not in globals().keys():
         globals()["HAARPSI_INSTANCE"] = {}
      globals()["HAARPSI_INSTANCE"][candidates.device] = haarpsi_instance
   
   value = haarpsi_instance(candidates, target)
   return value

def dss(candidates, target):
   assert_images(candidates, target)
   value = piq_dss(candidates, target, data_range=1., reduction='none')
   return value
   
def gmsd(candidates, target):
   assert_images(candidates, target)
   loss = piq.gmsd(candidates, target, data_range=1., reduction='none')
   return torch.sub(0.35, loss) # 0.35 is max value

def mdsi(candidates, target):
   assert_images(candidates, target)
   return 1.0 - piq.mdsi(candidates, target, data_range=1., reduction='none')

def msssim(candidates, target):
   assert_images(candidates, target)
   value = piq.multi_scale_ssim(candidates, target, data_range=1., kernel_size=3,k2=0.2,
                                reduction='none')
   return value

def style(candidates, target):
   beta = 1
   if "STYLE_INSTANCE" in globals().keys() and candidates.device in globals()["STYLE_INSTANCE"].keys():
      style_instance = globals()["STYLE_INSTANCE"][candidates.device]
   else:
      
      style_instance = StyleLoss(FEATURE_EXTRACTOR, candidates.device, target[0].unsqueeze(0))
      
      if "STYLE_INSTANCE" not in globals().keys():
         globals()["STYLE_INSTANCE"] = {}
      globals()["STYLE_INSTANCE"][candidates.device] = style_instance

   # Computes distance between Gram matrices of feature maps
   assert_images(candidates, target)
   loss = style_instance(candidates, target)
   value = -loss
   value = value  * beta
   return value

def content(candidates, target):
   if "CONTENT_INSTANCE" in globals().keys() and candidates.device in globals()["CONTENT_INSTANCE"].keys():
      content_instance = globals()["CONTENT_INSTANCE"][candidates.device]
   else:
      content_instance = piq.ContentLoss(
        feature_extractor=FEATURE_EXTRACTOR, reduction='none').eval().to(candidates.device)
      if "CONTENT_INSTANCE" not in globals().keys():
         globals()["CONTENT_INSTANCE"] = {}
      globals()["CONTENT_INSTANCE"][candidates.device] = content_instance
   assert_images(candidates, target)
   loss = content_instance(feature_extractor = feature_extractor, layers=("relu3_3",), reduction='none')(candidates,target)
   value = torch.tensor([1.0]*len(candidates)).to(loss) - loss
   return value

def pieAPP(candidates, target):
   assert_images(candidates, target)
   candidates,target = Resize((128,128))(candidates), Resize((128,128))(target)
   loss = piq.PieAPP(reduction='none', stride=32)(candidates, target)
   value = torch.tensor([1.0]*len(candidates)).to(loss) - loss
   return value


"""The principle philosophy underlying the original SSIM
approach is that the human visual system is highly adapted to
extract structural information from visual scenes. (https://ece.uwaterloo.ca/~z70wang/publications/SPM09.pdf pg. 105)"""
def ssim(candidates, target):
   assert_images(candidates, target)
   value = piq.ssim(candidates, target, data_range=1.0, reduction='none')
   return value

def psnr(candidates, target):
   assert_images(candidates, target)
   return torch.div(piq.psnr(candidates, target, data_range=1.0, reduction='none'), 50.0) # max is normally 50 DB

def vif(candidates, target):
   assert_images(candidates, target)
   candidates, target = Resize((41,41),antialias=False)(candidates), Resize((41,41),antialias=False)(target)
   value = piq.vif_p(candidates, target, data_range=1.0, reduction='none')
   return value

def vsi(candidates, target):
   assert_images(candidates, target)
   value = piq.vsi(candidates, target, data_range=1.0, reduction='none')
   return value

def srsim(candidates, target):
   assert_images(candidates, target)
   candidates, target = Resize((161,161),antialias=False)(candidates), Resize((161,161),antialias=False)(target)
   value = piq.srsim(candidates, target, data_range=1.0, reduction='none')
   return value

def fsim(candidates, target):
   assert_images(candidates, target)
   value = piq.fsim(candidates, target, data_range=1.0, reduction='none')
   return value

import zlib
def compression_ratio(candidates, _=None):
   sizes = torch.zeros(len(candidates), device=candidates.device)
   for i, candidate in enumerate(candidates):
      sizes[i] = len(zlib.compress(candidate.detach().cpu().numpy().tobytes()))
      orig = candidate.numel() * candidate.element_size()
      sizes[i] = sizes[i] / orig
   return sizes

######################
# GENOTYPE FUNCTIONS #
######################

def modularity(candidate_genomes,_=None):
   """Returns the modularity of the genotype"""
   fits = torch.zeros(len(candidate_genomes))
   for i, candidate in enumerate(candidate_genomes):
      G = candidate.to_networkx()
      fits[i] = nx.algorithms.community.modularity(G, nx.algorithms.community.greedy_modularity_communities(G, weight='weight'), weight='weight')
   return fits

def partition_coverage(candidate_genomes,_=None):
   """Returns the partition coverage [the ratio of the number of intra-community edges to the total number of edges in the graph] of the genotype"""
   fits = torch.zeros(len(candidate_genomes))
   for i, candidate in enumerate(candidate_genomes):
      G = candidate.to_networkx()
      coverage, _ = nx.algorithms.community.partition_quality(G, nx.algorithms.community.greedy_modularity_communities(G, weight='weight'))
      fits[i] = coverage
   return fits

def partition_performance(candidate_genomes,_=None):
   """Returns the partition performance [the number of intra-community edges plus inter-community non-edges divided by the total number of potential edges.] of the genotype"""
   fits = torch.zeros(len(candidate_genomes))
   for i, candidate in enumerate(candidate_genomes):
      G = candidate.to_networkx()
      _, performance = nx.algorithms.community.partition_quality(G, nx.algorithms.community.greedy_modularity_communities(G, weight='weight'))
      fits[i] = performance
   return fits

def partition_quality(candidate_genomes,_=None):
   """Returns the partition quality [the coverage and performance of a partition] of the genotype"""
   fits = torch.zeros(len(candidate_genomes))
   for i, candidate in enumerate(candidate_genomes):
      G = candidate.to_networkx()
      coverage, performance = nx.algorithms.community.partition_quality(G, nx.algorithms.community.greedy_modularity_communities(G, weight='weight'))
      fits[i] = (performance + coverage) / 2.0
   return fits

def min_nodes(candidate_genomes,_=None):
   """Returns the inverse of the count of nodes in the genotype"""
   fits = torch.zeros(len(candidate_genomes))
   for i, candidate in enumerate(candidate_genomes):
      fits[i] = 1.0 / candidate.count_nodes()
   return fits

def max_nodes(candidate_genomes,_=None):
   """Returns the count of nodes in the genotype"""
   fits = torch.zeros(len(candidate_genomes))
   for i, candidate in enumerate(candidate_genomes):
      fits[i] = candidate.count_nodes() / 100.0 # make it closer to 0-1
   return fits

def min_connections(candidate_genomes,_=None):
   """Returns the inverse of the count of enabled connections in the genotype"""
   fits = torch.zeros(len(candidate_genomes))
   for i, candidate in enumerate(candidate_genomes):
      fits[i] = 1.0 / candidate.count_enabled_connections()
   return fits

def max_connections(candidate_genomes,_=None):
   """Returns the count of enabled connections in the genotype"""
   fits = torch.zeros(len(candidate_genomes))
   for i, candidate in enumerate(candidate_genomes):
      fits[i] = candidate.count_enabled_connections() / 100.0 # /100 to make it closer to 0-1
   return fits

def max_activation_fns(candidate_genomes,_=None):
   """Returns the count of unique activation functions in the genotype"""
   fits = torch.zeros(len(candidate_genomes))
   for i, candidate in enumerate(candidate_genomes):
      fits[i] = candidate.count_activation_functions() / 10.0 # /100 to make it closer to 0-1
   return fits
   
def depth(genomes):
   """Returns the depth of the genotype"""
   metric = torch.zeros(len(genomes))
   for i, genome in enumerate(genomes):
      metric[i] = genome.depth()
   return metric

def age(genomes):
   metric = torch.zeros(len(genomes))
   for i, genome in enumerate(genomes):
      metric[i] = genome.age
   return metric

def inv_age(genomes):
   metric = torch.zeros(len(genomes))
   for i, genome in enumerate(genomes):
      metric[i] = -genome.age
   return metric
   
GENOTYPE_FUNCTIONS = [min_nodes, max_nodes, min_connections, max_connections, max_activation_fns, modularity, partition_coverage, partition_performance, partition_quality, depth, age, inv_age]
NO_GRADIENT = GENOTYPE_FUNCTIONS + [compression_ratio]
NO_MEAN = NO_GRADIENT
NO_NORM = GENOTYPE_FUNCTIONS + [compression_ratio]
name_to_fn = {k:v for k,v in locals().items() if callable(v) and k not in ["name_to_function", "GENOTYPE_FUNCTIONS"]}
