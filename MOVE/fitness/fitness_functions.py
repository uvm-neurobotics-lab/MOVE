"""Fitness functions."""
import logging
import warnings
import re
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torchvision.transforms import Resize
import networkx as nx

try:
   import piq
except ImportError:  # pragma: no cover - optional dependency
   piq = None  # type: ignore

try:
   import piqa
   from piqa import HaarPSI
except ImportError:  # pragma: no cover - optional dependency
   piqa = None  # type: ignore
   HaarPSI = None  # type: ignore

from .style_loss import StyleLoss, ContentLoss
from .dists import DISTS
from .lpips import LPIPS
from .dss import dss as piq_dss
from ..util import is_canonical_image_batch
from .backbones import FEATURE_EXTRACTOR

try:  # pragma: no cover - optional dependency
   from ..clip.clip_model import embed_images as clip_embed_images
   from ..clip.clip_model import embed_text as clip_embed_text
   from ..clip.clip_model import cosine_similarity as clip_cosine_similarity
except ImportError:  # pragma: no cover - optional dependency
   clip_embed_images = None  # type: ignore
   clip_embed_text = None  # type: ignore
   clip_cosine_similarity = None  # type: ignore
_CLIP_TEXT_FN_CACHE: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {}
_CLIP_TEXT_EMBED_CACHE: Dict[Tuple[str, torch.device], torch.Tensor] = {}


def _ensure_clip_available() -> None:
   if clip_embed_images is None or clip_embed_text is None or clip_cosine_similarity is None:
      raise ImportError(
         "CLIP-based fitness requires the 'clip' package. Install it with `pip install git+https://github.com/openai/CLIP.git`."
      )


def _normalize_embeddings(embeddings: torch.Tensor) -> torch.Tensor:
   return torch.nn.functional.normalize(embeddings, dim=-1)


def _clip_text_embedding(text: str, device: torch.device) -> torch.Tensor:
   _ensure_clip_available()
   key = (text, device)
   embedding = _CLIP_TEXT_EMBED_CACHE.get(key)
   if embedding is None:
      with torch.no_grad():
         embedding = clip_embed_text(text, device=device).detach()  # type: ignore[arg-type]
      embedding = _normalize_embeddings(embedding)
      _CLIP_TEXT_EMBED_CACHE[key] = embedding
   elif embedding.device != device:
      embedding = embedding.to(device)
      _CLIP_TEXT_EMBED_CACHE[key] = embedding
   return embedding


def _pairwise_cosine(image_embeddings: torch.Tensor, target_embeddings: torch.Tensor) -> torch.Tensor:
   image_norm = _normalize_embeddings(image_embeddings)
   target_norm = _normalize_embeddings(target_embeddings)
   if target_norm.shape[0] == 1 and image_norm.shape[0] > 1:
      target_norm = target_norm.expand(image_norm.shape[0], -1)
   elif target_norm.shape[0] != image_norm.shape[0]:
      target_norm = target_norm[: image_norm.shape[0]]
   score = (image_norm * target_norm).sum(dim=-1)
   return torch.clamp((score + 1.0) / 2.0, 0.0, 1.0)


def _sanitize_clip_candidates(candidates: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
   sanitized, _ = correct_dims(candidates, target)
   assert_images(sanitized, sanitized)
   return sanitized


def _require_piq():
   if piq is None or piqa is None or HaarPSI is None:
      raise ImportError(
         "The 'piq' and 'piqa' packages are required for fitness metrics such as DSS, GMSD, MDSI, MS-SSIM, and HaarPSI."
      )


def control(candidates, target):
   return torch.rand(len(candidates), dtype=torch.float32, device=target.device)


def _ensure_batched_rgb(img: torch.Tensor) -> torch.Tensor:
   if img.ndim == 2:
      img = img.unsqueeze(0).unsqueeze(0)
   elif img.ndim == 3:
      if img.shape[0] in (1, 3) and img.shape[1] != 3 and img.shape[2] != 3:
         img = img.unsqueeze(0)
      elif img.shape[-1] == 3 and img.shape[0] != 3:
         img = img.permute(2, 0, 1).unsqueeze(0)
      else:
         img = img.unsqueeze(1)
   elif img.ndim == 4:
      if img.shape[1] not in (1, 3) and img.shape[-1] == 3:
         img = img.permute(0, 3, 1, 2)
   else:
      raise ValueError(f"Unsupported image tensor with shape {tuple(img.shape)}")

   if img.ndim != 4:
      img = img.unsqueeze(0)

   if img.shape[1] == 3:
      return img.contiguous()
   if img.shape[1] == 1:
      return img.repeat(1, 3, 1, 1).contiguous()
   if img.shape[-1] == 3:
      return img.permute(0, 3, 1, 2).contiguous()
   return img[:, :1].repeat(1, 3, 1, 1).contiguous()


def _resize_to_min(img: torch.Tensor) -> torch.Tensor:
   if img.shape[-2] < 33 or img.shape[-1] < 33:
      img = F.interpolate(img, size=(33, 33), mode="bilinear", align_corners=False)
   return img


@torch.no_grad()
def correct_dims(candidates, target):
   if is_canonical_image_batch(candidates):
      f = candidates.contiguous()
   else:
      f = _ensure_batched_rgb(candidates).to(dtype=torch.float32)
      if not torch.isfinite(f).all():
         raise ValueError("Non-finite values detected in candidate images")
      f = _resize_to_min(f)
      if not torch.isfinite(f).all():
         raise ValueError("Non-finite values detected after resizing candidate images")
      f = torch.clamp(f, 0.0, 1.0)
      if not torch.isfinite(f).all():
         raise ValueError("Non-finite values detected after clamping candidate images")

   if is_canonical_image_batch(target):
      r = target.contiguous()
   else:
      r = _ensure_batched_rgb(target).to(device=f.device, dtype=torch.float32)
      if not torch.isfinite(r).all():
         raise ValueError("Non-finite values detected in target images")
      r = _resize_to_min(r)
      if not torch.isfinite(r).all():
         raise ValueError("Non-finite values detected after resizing target images")
      r = torch.clamp(r, 0.0, 1.0)
      if not torch.isfinite(r).all():
         raise ValueError("Non-finite values detected after clamping target images")

   if f.shape[0] != 1 and r.shape[0] == 1:
      logging.warning(
         "Only one target in batch but %s candidates. Repeating target for comparison.",
         f.shape[0],
      )
      r = r.repeat(f.shape[0], 1, 1, 1)

   return f.contiguous(), r.contiguous()

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
    #   print("Creating DISTS instance")
      # dists_instance = piq_dists(reduction='none').eval().to(candidates.device)
      dists_instance = DISTS(FEATURE_EXTRACTOR).eval().to(candidates.device)
      if "DISTS_INSTANCE" not in globals().keys():
         globals()["DISTS_INSTANCE"] = {}
      globals()["DISTS_INSTANCE"][candidates.device] = dists_instance
   assert_images(candidates, target)
   
   # Ensure inputs are float32 to match model weights
   # AMP may convert inputs to float16, but we need float32 for the shared VGG backbone
   candidates_fp32 = candidates.float() if candidates.dtype != torch.float32 else candidates
   target_fp32 = target.float() if target.dtype != torch.float32 else target
   
   # loss = dists_instance(candidates, target)
   # value = torch.tensor([1.0]*len(candidates)).to(loss) - loss
   # return torch.sub(1.0, dists_instance(candidates, target))
   
   val = dists_instance(candidates_fp32, target_fp32, require_grad=True, batch_average=False)
   if len(candidates) == 1:
      val = val.unsqueeze(0) # batch
   return torch.sub(1.0, val)
   
def lpips(candidates, target):
   # return 1.0 - lpips_instance(candidates, target)   
   if "LPIPS_INSTANCE" in globals().keys() and candidates.device in globals()["LPIPS_INSTANCE"].keys():
      lpips_instance = globals()["LPIPS_INSTANCE"][candidates.device]
   else:
    #   print("Creating LPIPS instance")
      # lpips_instance = piq_lpips(reduction='none').eval()
      lpips_instance = LPIPS(FEATURE_EXTRACTOR, reduction='none').eval().to(candidates.device)
      if "LPIPS_INSTANCE" not in globals().keys():
         globals()["LPIPS_INSTANCE"] = {}
      globals()["LPIPS_INSTANCE"][candidates.device] = lpips_instance
   assert_images(candidates, target)
   
   # Ensure inputs are float32 to match model weights
   # AMP may convert inputs to float16, but we need float32 for the shared VGG backbone
   candidates_fp32 = candidates.float() if candidates.dtype != torch.float32 else candidates
   target_fp32 = target.float() if target.dtype != torch.float32 else target
   
   value = torch.sub(1.0, lpips_instance(candidates_fp32, target_fp32))
   return value


def haarpsi(candidates, target):
   _require_piq()
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
   _require_piq()
   assert_images(candidates, target)
   loss = piq.gmsd(candidates, target, data_range=1., reduction='none')
   return torch.sub(0.35, loss) # 0.35 is max value

def mdsi(candidates, target):
   # TODO NAN IN GRAD
   _require_piq()
   assert_images(candidates, target)
   return 1.0 - piq.mdsi(candidates, target, data_range=1., reduction='none')

def msssim(candidates, target):
   _require_piq()
   assert_images(candidates, target)
   value = piq.multi_scale_ssim(candidates, target, data_range=1., kernel_size=3,k2=0.2,
                                reduction='none')
   return value

def style(candidates, target):
   beta = 1
   if "STYLE_INSTANCE" in globals().keys() and candidates.device in globals()["STYLE_INSTANCE"].keys():
      style_instance = globals()["STYLE_INSTANCE"][candidates.device]
   else:
      style_instance = StyleLoss(FEATURE_EXTRACTOR, candidates.device, target)
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
      content_instance = ContentLoss(candidates.device, target)
      if "CONTENT_INSTANCE" not in globals().keys():
         globals()["CONTENT_INSTANCE"] = {}
      globals()["CONTENT_INSTANCE"][candidates.device] = content_instance
   assert_images(candidates, target)
   loss = content_instance(candidates, target)
   value = torch.ones_like(loss) - loss
   return value

def pieAPP(candidates, target):
   _require_piq()
   assert_images(candidates, target)
   candidates,target = Resize((128,128))(candidates), Resize((128,128))(target)
   loss = piq.PieAPP(reduction='none', stride=32)(candidates, target)
   value = torch.tensor([1.0]*len(candidates)).to(loss) - loss
   return value


def clip_similarity(candidates: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
   """CLIP cosine similarity between candidate images and the target image batch."""

   _ensure_clip_available()
   sanitized = _sanitize_clip_candidates(candidates, target)
   target_sanitized = _sanitize_clip_candidates(target, target)
   image_embeddings = clip_embed_images(sanitized, device=sanitized.device)  # type: ignore[arg-type]
   with torch.no_grad():
      target_embeddings = clip_embed_images(target_sanitized, device=sanitized.device).detach()  # type: ignore[arg-type]
   return _pairwise_cosine(image_embeddings, target_embeddings)


def _make_clip_text_objective(text: str) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:

   def clip_text_objective(candidates: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
      _ensure_clip_available()
      sanitized = _sanitize_clip_candidates(candidates, target)
      image_embeddings = clip_embed_images(sanitized, device=sanitized.device)  # type: ignore[arg-type]
      text_embedding = _clip_text_embedding(text, sanitized.device).unsqueeze(0)
      return _pairwise_cosine(image_embeddings, text_embedding)

   clip_text_objective.__name__ = f"clip_{re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-')}"
   return clip_text_objective


def get_clip_text_objective(name: str) -> Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]:
   """Return a CLIP text objective for ``name`` or ``None`` if unsupported."""

   if not isinstance(name, str) or not name.startswith("clip_"):
      return None
   if name == "clip_similarity":
      return clip_similarity
   text = name[len("clip_"):]
   text = text.replace("_", " ")
   text = text.replace("-", " ")
   text = re.sub(r"\s+", " ", text).strip()
   if not text:
      return clip_similarity
   cached = _CLIP_TEXT_FN_CACHE.get(name)
   if cached is not None:
      return cached
   objective = _make_clip_text_objective(text)
   objective.__name__ = name
   _CLIP_TEXT_FN_CACHE[name] = objective
   FITNESS_FUNCTIONS[name] = objective
   return objective


"""The principle philosophy underlying the original SSIM
approach is that the human visual system is highly adapted to
extract structural information from visual scenes. (https://ece.uwaterloo.ca/~z70wang/publications/SPM09.pdf pg. 105)"""
def ssim(candidates, target):
   _require_piq()
   assert_images(candidates, target)
   value = piq.ssim(candidates, target, data_range=1.0, reduction='none')
   return value

def psnr(candidates, target):
   _require_piq()
   assert_images(candidates, target)
   return torch.div(piq.psnr(candidates, target, data_range=1.0, reduction='none'), 50.0) # max is normally 50 DB

def vif(candidates, target):
   _require_piq()
   assert_images(candidates, target)
   candidates, target = Resize((41,41),antialias=False)(candidates), Resize((41,41),antialias=False)(target)
   value = piq.vif_p(candidates, target, data_range=1.0, reduction='none')
   return value

def vsi(candidates, target):
   _require_piq()
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
   _require_piq()
   assert_images(candidates, target)
   candidates, target = Resize((161,161),antialias=False)(candidates), Resize((161,161),antialias=False)(target)
   value = piq.srsim(candidates, target, data_range=1.0, reduction='none')
   return value

def fsim(candidates, target):
   _require_piq()
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
      metric[i] = 1.0 / genome.count_enabled_connections
   return metric

def max_connections(genomes):
   """Returns the count of enabled connections in the genotype"""
   metric = torch.zeros(len(genomes))
   for i, genome in enumerate(genomes):
      metric[i] = genome.count_enabled_connections / 100.0 # /100 to make it closer to 0-1
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
      metric[i] = np.std([c.weight.item() for c in genome.enabled_connections])
   return metric

def mean_of_weights(genomes):
   """Returns the mean of the weights of the genotype"""
   metric = torch.zeros(len(genomes))
   for i, genome in enumerate(genomes):
      metric[i] = torch.mean([c.weight.item() for c in genome.enabled_connections])
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
      metric[i] = genome.count_enabled_connections
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
# name_to_fn = {k:v for k,v in locals().items() if callable(v) and k not in ["name_to_fn", "GENOTYPE_FUNCTIONS"]}

FITNESS_FUNCTIONS={
    k:v for k,v in locals().items() if callable(v)
}

def register_fitness_function(name, fn, is_genotype=False):
    FITNESS_FUNCTIONS[name] = fn
    if is_genotype:
        GENOTYPE_FUNCTIONS.append(fn)