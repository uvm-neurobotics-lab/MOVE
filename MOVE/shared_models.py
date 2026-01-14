"""
Shared model management for multiprocessing MOVE.

This module provides a way to load expensive models (like VGG for LPIPS) once
in the main process and share them across worker processes via shared memory.

This solves the GPU memory exhaustion problem where each worker would otherwise
load its own copy of large models.
"""

import logging
import multiprocessing as mp
from typing import Dict, Optional, Any
import torch
import torch.nn as nn


# Global registry of shared models
_SHARED_MODELS: Dict[str, Any] = {}
_SHARED_MODEL_LOCK = mp.Lock()


class SharedModelManager:
    """Manages shared model instances across processes"""
    
    def __init__(self):
        self.models = {}
        self.enabled = False
        
    def enable(self):
        """Enable shared model mode"""
        self.enabled = True
        logging.info("Shared model manager enabled")
        
    def disable(self):
        """Disable shared model mode"""
        self.enabled = False
        self.models.clear()
        logging.info("Shared model manager disabled")
        
    def register_model(self, name: str, model: nn.Module):
        """Register a model for sharing"""
        if not self.enabled:
            return
            
        # Move model to CPU and set to eval mode
        model = model.cpu()
        model.eval()
        
        # Disable gradients for all parameters
        for param in model.parameters():
            param.requires_grad = False
            
        self.models[name] = model
        logging.info(f"Registered shared model: {name}")
        
    def get_model(self, name: str, device: str = "cpu") -> Optional[nn.Module]:
        """Get a shared model instance"""
        if not self.enabled or name not in self.models:
            return None
            
        model = self.models[name]
        
        # For GPU, we need to be careful about sharing
        # For now, keep on CPU and let worker copy to GPU if needed
        return model
        
    def has_model(self, name: str) -> bool:
        """Check if a model is registered"""
        return name in self.models


# Global shared model manager instance
_shared_manager = SharedModelManager()


def enable_shared_models():
    """Enable shared model mode globally"""
    _shared_manager.enable()


def disable_shared_models():
    """Disable shared model mode globally"""
    _shared_manager.disable()


def register_shared_model(name: str, model: nn.Module):
    """Register a model for sharing across processes"""
    _shared_manager.register_model(name, model)


def get_shared_model(name: str, device: str = "cpu") -> Optional[nn.Module]:
    """Get a shared model instance"""
    return _shared_manager.get_model(name, device)


def has_shared_model(name: str) -> bool:
    """Check if a model is registered"""
    return _shared_manager.has_model(name)


def is_shared_models_enabled() -> bool:
    """Check if shared models are enabled"""
    return _shared_manager.enabled


# Model loading wrappers for common models
def get_or_load_vgg(device: str = "cpu") -> nn.Module:
    """Get or load VGG model for LPIPS"""
    model_name = "vgg_lpips"
    
    # Check if we have a shared model
    model = get_shared_model(model_name, device)
    if model is not None:
        # Copy to target device if needed
        if str(device) != "cpu":
            model = model.to(device)
        return model
    
    # Load fresh model (normal path when not in shared mode)
    from MOVE.fitness.lpips import LPIPS
    model = LPIPS(net='vgg', pnet_rand=False, pnet_tune=False, use_dropout=True, 
                  spatial=False, version='0.1', lpips=True)
    
    # Move to device
    if str(device) != "cpu":
        model = model.to(device)
    
    # Register for sharing if enabled
    if is_shared_models_enabled():
        register_shared_model(model_name, model.cpu())
        
    return model


def get_or_load_dists_model(device: str = "cpu") -> nn.Module:
    """Get or load DISTS model"""
    model_name = "dists"
    
    model = get_shared_model(model_name, device)
    if model is not None:
        if str(device) != "cpu":
            model = model.to(device)
        return model
    
    # Load fresh model
    try:
        from MOVE.fitness.dists import DISTS
        model = DISTS()
        
        if is_shared_models_enabled():
            register_shared_model(model_name, model.cpu())
            
        if str(device) != "cpu":
            model = model.to(device)
    except ImportError:
        model = None
        
    return model


def preload_shared_models(config, device: str = "cpu"):
    """Preload all models that will be needed by workers"""
    if not is_shared_models_enabled():
        logging.info("Shared models not enabled, skipping preload")
        return
        
    logging.info("Preloading shared models...")
    
    # Get objective functions from config
    objective_fns = getattr(config, 'objective_functions', [])
    
    # Check which models are needed
    needs_lpips = any('lpips' in str(fn).lower() for fn in objective_fns)
    needs_dists = any('dists' in str(fn).lower() for fn in objective_fns)
    needs_style = any('style' in str(fn).lower() for fn in objective_fns)
    
    loaded = []
    
    if needs_lpips:
        try:
            model = get_or_load_vgg(device="cpu")  # Always load to CPU first
            if model is not None:
                loaded.append("VGG/LPIPS")
        except Exception as e:
            logging.warning(f"Failed to preload VGG/LPIPS: {e}")
    
    if needs_dists:
        try:
            model = get_or_load_dists_model(device="cpu")
            if model is not None:
                loaded.append("DISTS")
        except Exception as e:
            logging.warning(f"Failed to preload DISTS: {e}")
    
    if needs_style:
        # Style also uses VGG
        if not needs_lpips:
            try:
                model = get_or_load_vgg(device="cpu")
                if model is not None:
                    loaded.append("VGG/Style")
            except Exception as e:
                logging.warning(f"Failed to preload VGG for style: {e}")
    
    if loaded:
        logging.info(f"Preloaded shared models: {', '.join(loaded)}")
    else:
        logging.warning("No models were preloaded")


def get_shared_model_memory_estimate(config) -> int:
    """Estimate GPU memory needed for shared models (in bytes)"""
    objective_fns = getattr(config, 'objective_functions', [])
    
    total_mb = 0
    
    # VGG for LPIPS/Style: ~500MB
    if any('lpips' in str(fn).lower() or 'style' in str(fn).lower() for fn in objective_fns):
        total_mb += 500
    
    # DISTS: ~100MB
    if any('dists' in str(fn).lower() for fn in objective_fns):
        total_mb += 100
        
    return total_mb * 1024 * 1024  # Convert to bytes


def calculate_safe_gpu_workers(config, device: str = "cuda:0") -> int:
    """Calculate how many GPU workers can safely run given available memory"""
    if device == "cpu" or not torch.cuda.is_available():
        return 0
    
    try:
        device_obj = torch.device(device)
        free_memory, total_memory = torch.cuda.mem_get_info(device_obj.index)
        
        # Estimate memory per worker
        model_memory = get_shared_model_memory_estimate(config)
        cppn_memory = 100 * 1024 * 1024  # ~100MB for CPPN + activations
        safety_margin = 0.2  # Keep 20% free
        
        available = free_memory * (1 - safety_margin)
        per_worker = model_memory + cppn_memory
        
        max_workers = int(available / per_worker)
        
        logging.info(f"GPU memory analysis for {device}:")
        logging.info(f"  Total: {total_memory / 1e9:.2f} GB")
        logging.info(f"  Free: {free_memory / 1e9:.2f} GB")
        logging.info(f"  Available (after safety margin): {available / 1e9:.2f} GB")
        logging.info(f"  Per worker estimate: {per_worker / 1e6:.1f} MB")
        logging.info(f"  Max safe workers: {max_workers}")
        
        return max(0, min(max_workers, 4))  # Cap at 4 workers per GPU
        
    except Exception as e:
        logging.warning(f"Failed to calculate GPU workers: {e}")
        return 0
