"""Fitness functions."""
# from piq.fsim import FSIMLoss
# from piq.perceptual import DISTS as piq_dists
# from piq.perceptual import LPIPS as piq_lpips
import logging
import piq
from piq import ContentLoss
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
# from piqa import fsim as p_fsim
# from piqa import vsi as p_vsi
import piqa


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
   # TODO NAN IN GRAD
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
      content_instance = piq.ContentLoss(FEATURE_EXTRACTOR, reduction='none').eval().to(candidates.device)
      if "CONTENT_INSTANCE" not in globals().keys():
         globals()["CONTENT_INSTANCE"] = {}
      globals()["CONTENT_INSTANCE"][candidates.device] = content_instance
   assert_images(candidates, target)
   loss = content_instance(candidates, target)
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
   # TODO NAN IN GRAD
   if "VSI_INSTANCE" in globals().keys() and candidates.device in globals()["VSI_INSTANCE"].keys():
      vsi_instance = globals()["VSI_INSTANCE"][candidates.device]
   else:
      
      vsi_instance = piqa.VSI(reduction=None, value_range=1.0).to(candidates.device)
      
      if "VSI_INSTANCE" not in globals().keys():
         globals()["VSI_INSTANCE"] = {}
      globals()["VSI_INSTANCE"][candidates.device] = vsi_instance

   
   assert_images(candidates, target)
   value = piq.vsi(candidates, target, data_range=1.0, reduction='none') # dep warning TODO
   # value = vsi_instance(candidates, target)
   return value

def srsim(candidates, target):
   assert_images(candidates, target)
   candidates, target = Resize((161,161),antialias=False)(candidates), Resize((161,161),antialias=False)(target)
   value = piq.srsim(candidates, target, data_range=1.0, reduction='none')
   return value

def fsim(candidates, target):
   # TODO NAN IN GRAD
   if "FSIM_INSTANCE" in globals().keys() and candidates.device in globals()["FSIM_INSTANCE"].keys():
      fsim_instance = globals()["FSIM_INSTANCE"][candidates.device]
   else:
      
      fsim_instance = piqa.FSIM(reduction=None, value_range=1.0).to(candidates.device)
      # fsim_instance = FSIMLoss(reduction=None, value_range=1.0).to(candidates.device)
      
      if "FSIM_INSTANCE" not in globals().keys():
         globals()["FSIM_INSTANCE"] = {}
      globals()["FSIM_INSTANCE"][candidates.device] = fsim_instance

   
   assert_images(candidates, target)
   value = piq.fsim(candidates, target, data_range=1.0, reduction='none') # dep warning TODO
   # value = fsim_instance(candidates, target)
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
import networkx as nx
import numpy as np


def _communities(G):
   return nx.algorithms.community.greedy_modularity_communities(
      G,
      weight='weight',
      )

def n_cells(genomes):
   """Returns the modularity of the genotype"""
   metric = torch.zeros(len(genomes))
   for i, genome in enumerate(genomes):
      metric[i] = 1.0 - (genome.n_cells/ genome.config.num_cells)
   return metric

def modularity(genomes):
   """Returns the modularity of the genotype"""
   metric = torch.zeros(len(genomes))
   for i, genome in enumerate(genomes):
      G = genome.to_networkx()
      metric[i] = nx.algorithms.community.modularity(G, _communities(G), weight=None)
   return metric

def partition_coverage(genomes):
   """Returns the partition coverage [the ratio of the number of intra-community edges to the total number of edges in the graph] of the genotype"""
   metric = torch.zeros(len(genomes))
   for i, genome in enumerate(genomes):
      G = genome.to_networkx()
      coverage, _ = nx.algorithms.community.partition_quality(G, _communities(G))
      metric[i] = coverage
   return metric

def partition_performance(genomes):
   """Returns the partition performance [the number of intra-community edges plus inter-community non-edges divided by the total number of potential edges.] of the genotype"""
   metric = torch.zeros(len(genomes))
   for i, genome in enumerate(genomes):
      G = genome.to_networkx()
      _, performance = nx.algorithms.community.partition_quality(G, _communities(G))
      metric[i] = performance
   return metric

def partition_quality(genomes):
   """Returns the partition quality [the coverage and performance of a partition] of the genotype"""
   metric = torch.zeros(len(genomes))
   for i, genome in enumerate(genomes):
      G = genome.to_networkx()
      coverage, performance = nx.algorithms.community.partition_quality(G, _communities(G))
      metric[i] = (performance + coverage) / 2.0
   return metric

def min_nodes(genomes):
   """Returns the inverse of the count of nodes in the genotype"""
   metric = torch.zeros(len(genomes))
   for i, genome in enumerate(genomes):
      metric[i] = 1.0 / genome.count_nodes()
   return metric

def max_nodes(genomes):
   """Returns the count of nodes in the genotype"""
   metric = torch.zeros(len(genomes))
   for i, genome in enumerate(genomes):
      metric[i] = genome.count_nodes() / 100.0 # make it closer to 0-1
   return metric

def min_connections(genomes):
   """Returns the inverse of the count of enabled connections in the genotype"""
   metric = torch.zeros(len(genomes))
   for i, genome in enumerate(genomes):
      metric[i] = 1.0 / genome.count_enabled_connections()
   return metric

def max_connections(genomes):
   """Returns the count of enabled connections in the genotype"""
   metric = torch.zeros(len(genomes))
   for i, genome in enumerate(genomes):
      metric[i] = genome.count_enabled_connections() / 100.0 # /100 to make it closer to 0-1
   return metric

def max_activation_fns(genomes):
   """Returns the count of unique activation functions in the genotype"""
   metric = torch.zeros(len(genomes))
   for i, genome in enumerate(genomes):
      metric[i] = genome.count_activation_functions() / 10.0 # /100 to make it closer to 0-1
   return metric
   
   
def avg_in_degree(genomes):
   """Returns the average in degree of the genotype"""
   metric = torch.zeros(len(genomes))
   for i, genome in enumerate(genomes):
      G = genome.to_networkx()
      metric[i] = torch.mean([G.in_degree(n) for n in G.nodes()])
   return metric

def avg_out_degree(genomes):
   """Returns the average out degree of the genotype"""
   metric = torch.zeros(len(genomes))
   for i, genome in enumerate(genomes):
      G = genome.to_networkx()
      metric[i] = torch.mean([G.out_degree(n) for n in G.nodes()])
   return metric

def avg_degree(genomes):
   """Returns the average degree of the genotype"""
   metric = torch.zeros(len(genomes))
   for i, genome in enumerate(genomes):
      G = genome.to_networkx()
      metric[i] = torch.mean([G.degree(n) for n in G.nodes()])
   return metric

def hierarchy(genomes):
   """Returns the hierarchy of the genotype"""
   metric = torch.zeros(len(genomes))
   for i, genome in enumerate(genomes):
      G = genome.to_networkx()
      metric[i] = nx.algorithms.flow_hierarchy(G, weight='weight')
   return metric

def assortativity(genomes):
   """Returns the assortativity of the genotype"""
   metric = torch.zeros(len(genomes))
   for i, genome in enumerate(genomes):
      G = genome.to_networkx()
      metric[i] = nx.algorithms.assortativity.degree_assortativity_coefficient(G)
   return metric

def planarity(genomes):
   """Returns the planarity of the genotype"""
   metric = torch.zeros(len(genomes))
   for i, genome in enumerate(genomes):
      G = genome.to_networkx()
      metric[i] = nx.algorithms.planarity.check_planarity(G)[0]
   return metric

def radius(genomes):
   """Returns the radius of the genotype. 
   Note this is not technically the radius of G itself, 
   but instead the  largest radius amongst all components
   within G."""
   metric = torch.zeros(len(genomes))
   for i, genome in enumerate(genomes):
      G = genome.to_networkx()
      metric[i] = max([max(j.values())/2.0 for (i,j) in nx.shortest_path_length(G)])
   return metric

def diameter(genomes):
   """Returns the diameter of the genotype. 
   Note this is not technically the diameter of G itself, 
   but instead the  largest diameter amongst all components
   within G."""
   metric = torch.zeros(len(genomes))
   for i, genome in enumerate(genomes):
      G = genome.to_networkx()
      metric[i] = max([max(j.values()) for (i,j) in nx.shortest_path_length(G)])
   return metric

def eccentricity(genomes):
   """Returns the eccentricity of the genotype.
   Note this is not technically the eccentricity of G itself,
   but instead the mean eccentricity amongst all components
   within G."""
   metric = torch.zeros(len(genomes))
   for i, genome in enumerate(genomes):
      G = genome.to_networkx()
      metric[i] = torch.mean([max(j.values()) for (i,j) in nx.shortest_path_length(G)])
   return metric

def std_of_weights(genomes):
   """Returns the standard deviation of the weights of the genotype"""
   metric = torch.zeros(len(genomes))
   for i, genome in enumerate(genomes):
      metric[i] = np.std([c.weight.item() for c in genome.enabled_connections()])
   return metric

def mean_of_weights(genomes):
   """Returns the mean of the weights of the genotype"""
   metric = torch.zeros(len(genomes))
   for i, genome in enumerate(genomes):
      metric[i] = torch.mean([c.weight.item() for c in genome.enabled_connections()])
   return metric

def path_length(genomes):
   """Returns the path length of the genotype 
   Note this is not technically the path length of G itself,
   but instead the mean path length amongst all components
   within G."""
   metric = torch.zeros(len(genomes))
   for i, genome in enumerate(genomes):
      G = genome.to_networkx()
      metric[i] = torch.mean([torch.mean(j.values()) for (i,j) in nx.shortest_path_length(G)])
   return metric

def global_efficiency(genomes):
   """Returns the global efficiency of the genotype"""
   metric = torch.zeros(len(genomes))
   for i, genome in enumerate(genomes):
      G = genome.to_networkx()
      metric[i] = nx.algorithms.efficiency_measures.global_efficiency(G)
   return metric

def num_nodes(genomes):
   """Returns the number of nodes in the genotype"""
   metric = torch.zeros(len(genomes))
   for i, genome in enumerate(genomes):
      metric[i] = genome.count_nodes()
   return metric

def num_edges(genomes):
   """Returns the number of edges in the genotype"""
   metric = torch.zeros(len(genomes))
   for i, genome in enumerate(genomes):
      metric[i] = genome.count_enabled_connections()
   return metric

def num_activation_functions(genomes):
   """Returns the number of unique activation functions in the genotype"""
   metric = torch.zeros(len(genomes))
   for i, genome in enumerate(genomes):
      metric[i] = genome.count_activation_functions()
   return metric

def depth(genomes):
   """Returns the depth of the genotype"""
   metric = torch.zeros(len(genomes))
   for i, genome in enumerate(genomes):
      metric[i] = genome.depth()
   return metric

def max_width(genomes):
   """Returns the max width of the genotype"""
   metric = torch.zeros(len(genomes))
   for i, genome in enumerate(genomes):
      metric[i] = genome.width(agg=max)
   return metric

def min_width(genomes):
   """Returns the min width of the genotype"""
   metric = torch.zeros(len(genomes))
   for i, genome in enumerate(genomes):
      metric[i] = genome.width(agg=min)
   return metric

def avg_width(genomes):
   """Returns the width of the genotype"""
   metric = torch.zeros(len(genomes))
   for i, genome in enumerate(genomes):
      metric[i] = genome.width(agg=torch.mean)
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

def sgd_loss_delta(genomes):
   metric = torch.zeros(len(genomes))
   for i, genome in enumerate(genomes):
      if hasattr(genome, "loss_delta"):
         metric[i] = genome.loss_delta**2
      else:
         logging.warning("Genome does not have loss_delta attribute, returning 0")
   return metric

   
GENOTYPE_FUNCTIONS = [min_nodes, max_nodes, min_connections, max_connections, max_activation_fns, modularity, partition_coverage, partition_performance, partition_quality, max_width, avg_width, depth, age, inv_age, std_of_weights, mean_of_weights, sgd_loss_delta, n_cells]
NO_GRADIENT = GENOTYPE_FUNCTIONS + [compression_ratio]
NO_MEAN = NO_GRADIENT
NO_NORM = GENOTYPE_FUNCTIONS + [compression_ratio]
name_to_fn = {k:v for k,v in locals().items() if callable(v) and k not in ["name_to_function", "GENOTYPE_FUNCTIONS"]}
