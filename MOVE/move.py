import json
import os
import random
import copy
import torch
import torch.nn.functional as F
import logging
import math
import re
from typing import Dict, Tuple
from contextlib import nullcontext

import numpy as np

from tqdm import tqdm


# from cppn_torch import ImageCPPN

# from cppn_torch.graph_util import activate_population
# from cppn.util import visualize_network, initialize_inputs
from .cppn.util import *
# from evolution_torch import CPPNEvolutionaryAlgorithm
from .evolution import CPPNEvolutionaryAlgorithm

from .util import *

from .move_map import MOVEMap
from .run_setup import run_setup
from .sgd_weights import sgd_weights, sgd_weights_imaml
from .record_keeping import Record

from .norm import norm_tensor, read_norm_data
from .fitness import fitness_functions as ff
from .fitness.feature_cache import feature_cache_scope


class MOVE(CPPNEvolutionaryAlgorithm):
    def __init__(self, config, debug_output=False) -> None:
        self.config = copy.deepcopy(config)
        self._show_progress = bool(getattr(self.config, "show_progress", True))
        self._clip_variant_objectives = []
        self._clip_noise_scale_initial = None
        self._clip_noise_scale_current = None
        self.clip_partial_prompts = []
        
        if not hasattr(self.config, "use_amp"):
            self.config.use_amp = True

        
        clip_text = getattr(self.config, "clip_text_target", None)
        if clip_text is None and isinstance(self.config.objective_functions, list) and len(self.config.objective_functions) == 1:
            candidate_fn = self.config.objective_functions[0]
            if isinstance(candidate_fn, str) and candidate_fn not in ff.FITNESS_FUNCTIONS:
                clip_text = candidate_fn
                self.config.clip_text_target = clip_text

        self.clip_mode = clip_text is not None
        self.clip_text = clip_text

        if self.clip_mode:
            self.fns = self._build_clip_objectives(clip_text)
            self.config.objective_functions = self.fns
        elif self.config.objective_functions is None:
            # default: use all from paper
            self.fns = [
                ff.mse,
                ff.psnr,
                ff.lpips,
                ff.dists,
                ff.style,
                ff.vif,
                ff.dss,
                ff.ssim,
                ff.msssim,
                ff.haarpsi,
            ]

            self.config.objective_functions = self.fns
        else:
            resolved_fns = []
            for fn in self.config.objective_functions:
                if isinstance(fn, str):
                    resolved = getattr(ff, fn, None)
                    if resolved is None:
                        resolved = ff.get_clip_text_objective(fn)
                    if resolved is None:
                        raise AttributeError(f"Unknown objective function '{fn}'")
                    resolved_fns.append(resolved)
                else:
                    resolved_fns.append(fn)
            self.fns = resolved_fns
            self.config.objective_functions = self.fns
                
        
        self.allow_multiple_placements = self.config.allow_jumps > 0
        
        super().__init__(self.config, debug_output)
        
        self.map = MOVEMap(self.config, self.fns)
        
        self.n_cells = self.map.n_cells
        self.n_fns = self.map.n_fns
        self.fns_per_cell = self.map.fns_per_cell
        self.use_avg_fit = self.config.get("use_avg_fit", False)
            
        self.record = Record(self.config, self.n_fns, self.n_cells, self.total_batches, self.config.low_mem)
        self.norm = read_norm_data(self.config.norm_df_path, self.config.target_path)
        self._prepare_norm_stats()

        self.init_inputs()

        if self.clip_mode:
            self._init_clip_placeholder_target()
        elif self.config.target is not None:
            self.init_target()

        self._target_base = self.target if hasattr(self, "target") else None
        self._target_cache = {}
        
        if self.config.with_grad:
            self.init_sgd()
        else:
            torch.autograd.set_grad_enabled(False)
        
        if self.config.evolve_every != 1 and self.config.batch_size < self.config.num_cells:
            print("\n\nWARNING: evolve_every != 1 with batch_size < num_cells may cause some cells to never be evolved\n\n")
        if self.config.grad_every != 1 and self.config.batch_size < self.config.num_cells and self.config.sgd_steps > 0:
            print("\n\nWARNING: grad_every != 1 with batch_size < num_cells may cause some cells to never be trained with SGD\n\n")

        print("Initialized MOVE on device:", self.config.device)
        
                            
    def _build_clip_objectives(self, text: str):
        from .clip.semantic_targets import (
            ClipSemanticConfig,
            generate_clip_targets,
            generate_partial_prompts,
            DEFAULT_STOP_WORDS,
        )
        from .clip.clip_objectives import (
            ClipAugmentations,
            ClipSimilarityObjective,
            build_clip_objectives,
        )
        from .clip.clip_model import embed_text, maybe_compile_clip_models, set_clip_model_names

        # Allow either a single prompt string (historical behaviour) or a list
        # of prompt strings (each prompt contributes its own CLIP objectives).
        prompts = text
        if isinstance(text, (list, tuple)):
            prompts_list = [str(t).strip() for t in text if t is not None and str(t).strip()]
            if not prompts_list:
                raise ValueError("clip_text_target list is empty")
        else:
            prompts_list = [str(text).strip()]

        variants = max(1, int(getattr(self.config, "clip_num_variants", 1)))
        microbatch = int(
            getattr(self.config, "clip_embed_microbatch_size", 0)
            or getattr(self.config, "clip_microbatch_size", 0)
            or 0
        )
        augmentations = None
        aug_views = int(getattr(self.config, "clip_augmentations", 4) or 0)
        if aug_views > 0:
            min_scale = float(getattr(self.config, "clip_aug_min_scale", 0.5))
            max_scale = float(getattr(self.config, "clip_aug_max_scale", 1.0))
            flip_prob = float(getattr(self.config, "clip_aug_flip_prob", 0.5))
            jitter_std = float(getattr(self.config, "clip_aug_jitter_std", 0.02))
            augmentations = ClipAugmentations(
                num_random_crops=aug_views,
                min_crop_scale=min_scale,
                max_crop_scale=max_scale,
                flip_prob=flip_prob,
                jitter_std=jitter_std,
            )

        set_clip_model_names(
            getattr(self.config, "clip_vit_model", "ViT-B/32"),
            getattr(self.config, "clip_rn50_model", "RN50"),
        )
        maybe_compile_clip_models(self.config, device=self.config.device)

        objectives = []
        all_embeddings = []
        variant_objectives = []
        partial_meta = []

        for prompt_idx, prompt_text in enumerate(prompts_list):
            config = ClipSemanticConfig(
                text=prompt_text,
                num_variants=variants,
                noise_scale=float(getattr(self.config, "clip_noise_scale", 0.2)),
                seed=getattr(self.config, "clip_random_seed", None),
                device=self.config.device,
            )
            embeddings = generate_clip_targets(config)
            all_embeddings.extend(embeddings)

            prefix = "clip" if len(prompts_list) == 1 else f"clip_p{prompt_idx:02d}"
            these_objectives = build_clip_objectives(
                embeddings,
                prefix=prefix,
                microbatch_size=microbatch,
                augmentations=augmentations,
            )
            objectives.extend(these_objectives)
            variant_objectives.extend(these_objectives[: len(embeddings)])

            include_partials = bool(getattr(self.config, "clip_include_partials", False))
            if bool(getattr(self.config, "clip_disable_partials", False)):
                include_partials = False

            if include_partials:
                stop_words_cfg = getattr(self.config, "clip_partial_stopwords", None)
                if stop_words_cfg is None:
                    stop_words_set = DEFAULT_STOP_WORDS
                else:
                    if isinstance(stop_words_cfg, str):
                        stop_words_iter = [stop_words_cfg]
                    else:
                        stop_words_iter = stop_words_cfg
                    stop_words_set = {str(word).lower() for word in stop_words_iter}
                min_length = max(1, int(getattr(self.config, "clip_partial_min_length", 3)))
                max_partials = getattr(self.config, "clip_max_partial_prompts", None)
                try:
                    max_partials_int = None if max_partials is None else max(0, int(max_partials))
                except (TypeError, ValueError):
                    max_partials_int = None

                partial_prompts = generate_partial_prompts(
                    prompt_text,
                    min_length=min_length,
                    stop_words=stop_words_set,
                    max_partial_prompts=max_partials_int,
                )
                partial_meta.append((prompt_idx, prompt_text, partial_prompts))

                for idx, token_prompt in enumerate(partial_prompts):
                    partial_embedding = embed_text(token_prompt, device=self.config.device).float()
                    normalized_embedding = F.normalize(partial_embedding, dim=0)
                    slug = re.sub(r"[^a-z0-9]+", "_", token_prompt.lower()).strip("_")
                    if not slug:
                        slug = f"token_{idx:02d}"
                    token_prefix = "clip_token" if len(prompts_list) == 1 else f"clip_p{prompt_idx:02d}_token"
                    identifier = f"{token_prefix}_{idx:02d}_{slug}"
                    objectives.append(
                        ClipSimilarityObjective(
                            embedding=normalized_embedding,
                            identifier=identifier,
                            microbatch_size=microbatch,
                        )
                    )

        self.clip_embeddings = all_embeddings
        self._clip_variant_objectives = variant_objectives
        self._clip_noise_scale_initial = float(getattr(self.config, "clip_noise_scale", 0.2))
        self._clip_noise_scale_current = float(getattr(self.config, "clip_noise_scale", 0.2))
        self.config.clip_noise_scale = float(getattr(self.config, "clip_noise_scale", 0.2))

        # Keep the original prompt (or prompt list) for later noise scheduling refresh.
        self.clip_text = text
        # For backwards-compat keep storing partial prompts for the single-prompt case.
        if len(prompts_list) == 1 and partial_meta:
            self.clip_partial_prompts = partial_meta[0][2]
        else:
            self.clip_partial_prompts = []

        return objectives

    def _refresh_clip_noise_embeddings(self, noise_scale: float) -> None:
        if not self._clip_variant_objectives:
            return

        from .clip.semantic_targets import ClipSemanticConfig, generate_clip_targets

        variant_count = len(self._clip_variant_objectives)
        config = ClipSemanticConfig(
            text=self.clip_text,
            num_variants=variant_count,
            noise_scale=float(noise_scale),
            seed=getattr(self.config, "clip_random_seed", None),
            device=self.config.device,
        )
        embeddings = generate_clip_targets(config)
        for objective, embedding in zip(self._clip_variant_objectives, embeddings):
            normalized = F.normalize(embedding.detach(), dim=0)
            objective.embedding = normalized
        self.clip_embeddings = embeddings
        self._clip_noise_scale_current = float(noise_scale)
        self.config.clip_noise_scale = float(noise_scale)

    def _compute_annealed_noise_scale(self) -> float:
        if self._clip_noise_scale_initial is None:
            self._clip_noise_scale_initial = float(getattr(self.config, "clip_noise_scale", 0.2))

        initial = float(self._clip_noise_scale_initial)
        final = float(getattr(self.config, "clip_noise_final_scale", initial))
        start = float(getattr(self.config, "clip_noise_anneal_start", 0.0))
        end = float(getattr(self.config, "clip_noise_anneal_end", 1.0))
        power = float(getattr(self.config, "clip_noise_anneal_power", 1.0))

        total_offspring = float(getattr(self.config, "total_offspring", 0) or 0)
        progress = 0.0
        if total_offspring > 0:
            progress = self.total_offspring / total_offspring
        progress = max(0.0, min(progress, 1.0))

        start = max(0.0, start)
        end = max(start, end)
        if end == start:
            return final if progress >= end else initial

        span = end - start
        t = (progress - start) / span
        t = max(0.0, min(t, 1.0))

        if power != 1.0:
            power = max(power, 1e-8)
            t = t ** power

        return initial + (final - initial) * t

    def _update_clip_noise_schedule(self) -> None:
        if not self.clip_mode:
            return
        if not getattr(self.config, "clip_noise_anneal", False):
            return
        if not self._clip_variant_objectives:
            return

        target_scale = self._compute_annealed_noise_scale()
        if not math.isfinite(target_scale):
            return

        if (
            self._clip_noise_scale_current is None
            or not math.isclose(
                float(self._clip_noise_scale_current),
                float(target_scale),
                rel_tol=1e-3,
                abs_tol=1e-3,
            )
        ):
            logging.debug(
                "Updating CLIP noise scale from %.4f to %.4f",
                float(self._clip_noise_scale_current or target_scale),
                float(target_scale),
            )
            self._refresh_clip_noise_embeddings(target_scale)

    def batch_start(self):
        super().batch_start()
        self._update_clip_noise_schedule()

    def _init_clip_placeholder_target(self) -> None:
        height = getattr(self.config, "res_h", 33)
        width = getattr(self.config, "res_w", 33)
        placeholder = torch.zeros(
            (self.config.initial_batch_size, 3, height, width),
            device=self.config.device,
            dtype=torch.float32,
        )
        self.target = placeholder

    def _prepare_norm_stats(self) -> None:
        self._norm_stats: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._norm_eps = torch.tensor(1e-8, device=self.config.device, dtype=torch.float32)

        norm_df = self.norm
        if norm_df is None or not hasattr(norm_df, "groupby"):
            return
        if not hasattr(norm_df, "columns") or "function" not in norm_df.columns:
            return

        grouped = norm_df.groupby("function").mean(numeric_only=True)
        min_col = "min_fitness_mean" if "min_fitness_mean" in grouped.columns else "min_fitness"
        max_col = "max_fitness_mean" if "max_fitness_mean" in grouped.columns else "max_fitness"
        if min_col not in grouped.columns or max_col not in grouped.columns:
            return

        for fn_name, row in grouped.iterrows():
            min_val = float(row[min_col])
            max_val = float(row[max_col])
            if not math.isfinite(min_val) or not math.isfinite(max_val):
                continue
            span_val = max(max_val - min_val, 1e-8)
            min_tensor = torch.tensor(min_val, device=self.config.device, dtype=torch.float32)
            span_tensor = torch.tensor(span_val, device=self.config.device, dtype=torch.float32)
            self._norm_stats[str(fn_name)] = (min_tensor, span_tensor)


    def _normalize_with_cache(self, fn, fitness: torch.Tensor, *, clamp: bool = False) -> torch.Tensor:
        if fn in ff.NO_NORM or fitness.numel() == 0:
            return fitness

        key = getattr(fn, "__name__", None)
        cache = self._norm_stats.get(key) if key is not None else None

        if cache is not None:
            min_t, span_t = cache
            if min_t.device != fitness.device or min_t.dtype != fitness.dtype:
                min_t = min_t.to(device=fitness.device, dtype=fitness.dtype)
                span_t = span_t.to(device=fitness.device, dtype=fitness.dtype)
                self._norm_stats[key] = (min_t, span_t)

            eps = self._norm_eps.to(device=span_t.device, dtype=span_t.dtype)
            denom = torch.clamp(span_t, min=eps)
            normalized = (fitness - min_t) / denom
            if clamp:
                normalized = torch.clamp(normalized, 0.0, 1.0)
            if torch.isfinite(normalized).all().item():
                return normalized

        if key is None:
            return fitness

        normalized = norm_tensor(fitness, self.norm, key, warn=False, clamp=clamp)
        return normalized


    def _can_skip_correct_dims(self, batch_imgs: torch.Tensor) -> bool:
        return (
            isinstance(batch_imgs, torch.Tensor)
            and batch_imgs.ndim == 4
            and batch_imgs.shape[1] == 3
            and isinstance(self.target, torch.Tensor)
            and self.target.ndim == 4
            and self.target.shape[1] == 3
        )


    def _canonicalize_candidates(self, batch_imgs: torch.Tensor) -> torch.Tensor:
        if is_canonical_image_batch(batch_imgs):
            return batch_imgs.contiguous()

        candidates = ff._ensure_batched_rgb(batch_imgs).to(dtype=torch.float32)
        if not torch.isfinite(candidates).all():
            raise ValueError("Non-finite values detected in candidate images")
        candidates = ff._resize_to_min(candidates)
        if not torch.isfinite(candidates).all():
            raise ValueError("Non-finite values detected after resizing candidate images")
        candidates = torch.clamp(candidates, 0.0, 1.0)
        if not torch.isfinite(candidates).all():
            raise ValueError("Non-finite values detected after clamping candidate images")
        return candidates.contiguous()


    def _get_cached_target(
        self,
        base_target: torch.Tensor,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        spatial: Tuple[int, int],
    ) -> torch.Tensor:
        base = self._target_base if self._target_base is not None else base_target
        cache_key = (id(base), int(batch_size), spatial, dtype, device)
        cached = self._target_cache.get(cache_key)
        if cached is not None:
            return cached

        target = base
        if target.device != device or target.dtype != dtype:
            target = target.to(device=device, dtype=dtype, non_blocking=True)
        if target.shape[-2:] != spatial:
            target = F.interpolate(
                target,
                size=spatial,
                mode="bilinear",
                align_corners=False,
            )

        if target.shape[0] != batch_size:
            if target.shape[0] > batch_size:
                target = target[:batch_size]
            else:
                repeats = batch_size // target.shape[0]
                target = target.repeat(repeats, *([1] * len(target.shape[1:])))

        if not is_canonical_image_batch(target):
            if not torch.isfinite(target).all():
                raise ValueError("Non-finite values detected in target images")
            target = torch.clamp(target, 0.0, 1.0)
            if not torch.isfinite(target).all():
                raise ValueError("Non-finite values detected after clamping target images")
        else:
            target = target.contiguous()

        if target.dtype != torch.float32:
            target = target.to(dtype=torch.float32)

        self._target_cache[cache_key] = target
        return target


    def _finalize_target(self, target: torch.Tensor, batch_imgs: torch.Tensor) -> torch.Tensor:
        batch_size = batch_imgs.shape[0]
        cached_target = self._get_cached_target(
            target,
            batch_size,
            batch_imgs.device,
            batch_imgs.dtype,
            batch_imgs.shape[-2:],
        )
        if cached_target is not self.target:
            self.target = cached_target
        return cached_target


    def init_sgd(self, batch_cell_ids=None):
        if self.config.sgd_steps == 0:
            return
        torch.autograd.set_grad_enabled(True)
        if batch_cell_ids is None:
            # all of them
            batch_cell_ids = torch.arange(self.n_cells, device=self.config.device)    
        exclude = set(ff.NO_GRADIENT).intersection(self.fns)
        self.sgd_fns = set(self.fns).difference(exclude)
        skip_fns = [self.fns.index(f) for f in exclude]
        self.mask = torch.stack([torch.index_select(self.map.fn_mask[i], 0, batch_cell_ids) 
                            for i in range(len(self.map.fn_mask)) if i not in skip_fns])
    

    def run_one_batch(self):
        # reproduce
        self.selection_and_reproduction()
        self.population = self.map.get_population(include_empty=False)
        self.agg_fitnesses = self.map.get_agg_fitnesses()
        self.fitnesses = self.map.get_fitnesses()
    

    def evolve(self, run_number = 1, show_output=False, initial_population=False, resume=None):
        # start evolving, defaults to no initial population because the initial pop is generated during gen 0
        try:
            if resume is not None:
                self.current_batch = self.record.load_checkpoint(resume, self.checkpoints_dir, self.map, self.config)
                logging.info(f"Resumed from checkpoint '{resume}' at batch {self.current_batch}, population: {self.map.get_population()}")
            super().evolve(run_number, show_output, initial_population)
        except KeyboardInterrupt:
            pass # allow user to stop early
        
        
    @torch.no_grad()
    def new_child(self, parent, all_parents):
        if parent is None:
            child = self.genome_type(self.config)
            # child = self.genome_type(self.config.num_inputs, self.config.num_outputs, self.config.hidden_nodes_at_start, self.config.init_connection_probability)
            child.cell_lineage = [-1]
            child.n_cells = 0
            return child
        if self.config.do_crossover:
            # sexual reproduction, choose another parent randomly
            all_parents = list(filter(lambda x: x is not None, all_parents))
            other_parent = np.random.choice(all_parents)
            child = parent.crossover(other_parent, self.config) # crossover
            child.n_cells = parent.n_cells
            
            # mutate
            # child.mutate(self.config)
            # child.reset(self.config)

            # TODO lineage
            return child.to(self.config.device)
        else:
            # asexual reproduction, child is mutated clone of parent
            child = parent.clone(self.config, new_id=True)
            child.parents = (parent.id, parent.id)
            child.cell_lineage = parent.cell_lineage
            child.n_cells = parent.n_cells
            
            # mutate
            # child.mutate(self.config)
            # child.reset(self.config)
            return child.to(self.config.device)
        
  
    @torch.no_grad()
    def correct_target_count(self, count):
        """If the number of children has changed between iterations, 
        adjust the target to match
        """
        if self.target.shape[0]>count :
            logging.warning(
                "Target batch size %s larger than population size %s; truncating (total offspring %s)",
                self.target.shape[0],
                count,
                self.total_offspring,
            )
        elif self.target.shape[0]<count:
            logging.warning(
                "Target batch size %s smaller than population size %s; repeating (total offspring %s)",
                self.target.shape[0],
                count,
                self.total_offspring,
            )
        self.target = self._get_cached_target(
            self.target,
            int(count),
            self.target.device,
            self.target.dtype,
            self.target.shape[-2:],
        )
    

    @torch.no_grad()  # Use no_grad instead of inference_mode to allow tensor reuse in SGD
    def measure_fitness(self, genomes, imgs, skip_genotype=False):
        total = len(genomes)
        if total == 0:
            empty = torch.zeros((0, len(self.fns)), device=self.config.device, requires_grad=False)
            return empty, empty.clone(), empty.new_zeros(0)

        batch_size = self.config.batch_size
        num_batches = math.ceil(total / batch_size)

        fit_children = torch.zeros((total, len(self.fns)), device=self.config.device, requires_grad=False)
        fc_normed = torch.zeros_like(fit_children)

        if self.config.dry_run:
            fit_children = torch.rand((total, len(self.fns)), device=self.config.device, requires_grad=False)
            return fit_children, fit_children.clone()

        pbar = None
        # pbar = tqdm(total=num_batches, desc="Measuring fitness") if self._show_progress else None
        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch_slice = slice(batch_start, batch_end)
            batch_genomes = genomes[batch_slice]

            current_batch_size = batch_end - batch_start
            self.correct_target_count(current_batch_size)

            if imgs is None:
                batch_imgs = self.activate_population([g for _, _, g in batch_genomes])
            else:
                batch_imgs = imgs[batch_start:batch_end]

            if not is_canonical_image_batch(batch_imgs):
                if not torch.isfinite(batch_imgs).all():
                    raise ValueError("Non-finite values detected in candidate batch images")
                batch_imgs = torch.clamp(batch_imgs, 0.0, 1.0)
                if not torch.isfinite(batch_imgs).all():
                    raise ValueError("Non-finite values detected after clamping batch images")
            else:
                batch_imgs = batch_imgs.contiguous()

            if self._can_skip_correct_dims(batch_imgs):
                corrected_target = self._finalize_target(self.target, batch_imgs)
            else:
                batch_imgs = self._canonicalize_candidates(batch_imgs)
                corrected_target = self._finalize_target(self.target, batch_imgs)

            use_amp = bool(getattr(self.config, "use_amp", True))
            use_cuda_amp = use_amp and torch.device(self.config.device).type == "cuda"
            amp_whitelist = {"lpips", "dists"}
            with feature_cache_scope():
                for i, fn in enumerate(self.fns):
                    normed_fitness = torch.full((current_batch_size,), -torch.inf, device=self.config.device)
                    if fn in ff.GENOTYPE_FUNCTIONS:
                        if skip_genotype:
                            continue
                        fitness = fn([g[2] for g in genomes[batch_slice]]).to(
                            device=self.config.device,
                            dtype=fit_children.dtype,
                            non_blocking=True,
                        )
                    else:
                        fn_name = getattr(fn, "__name__", "")
                        amp_ctx = (
                            torch.cuda.amp.autocast(dtype=torch.float16)
                            if use_cuda_amp and fn_name in amp_whitelist
                            else nullcontext()
                        )
                        with amp_ctx:
                            fitness = fn(batch_imgs, corrected_target)
                        if fitness.dim() == 0:
                            fitness = fitness.unsqueeze(0)
                        if not torch.isfinite(fitness).all():
                            disable_ctx = (
                                torch.cuda.amp.autocast(enabled=False)
                                if torch.device(self.config.device).type == "cuda"
                                else nullcontext()
                            )
                            with disable_ctx:
                                fitness = fn(batch_imgs, corrected_target)
                                if fitness.dim() == 0:
                                    fitness = fitness.unsqueeze(0)
                        fitness = fitness.to(device=self.config.device, non_blocking=True)
                        normed_fitness = fitness
                        if fn not in ff.NO_NORM:
                            normed_fitness = self._normalize_with_cache(fn, fitness)

                    if fitness.dtype != fit_children.dtype:
                        fitness = fitness.to(dtype=fit_children.dtype)
                    if normed_fitness.dtype != fc_normed.dtype:
                        normed_fitness = normed_fitness.to(dtype=fc_normed.dtype)

                    if fn not in ff.NO_MEAN:
                        fc_normed[batch_slice, i] = normed_fitness

                    fit_children[batch_slice, i] = normed_fitness if self.use_avg_fit else fitness

            if pbar is not None:
                pbar.update(1)

        if pbar is not None:
            pbar.close()

        if len(fc_normed) == 0:
            fc_normed = fit_children.clone()

        for idx, (g, f) in enumerate(zip(genomes, fc_normed)):
            f = f[f != -torch.inf]
            g[2].fitness = f.mean().detach()

        agg_fc_normed = fc_normed.detach().mean(dim=1)

        return fit_children, fc_normed, agg_fc_normed


    @torch.no_grad()
    def mutation(self, genomes):
        for g in genomes:
            g.mutate(self.config)
            g.reset(self.config)


    @torch.no_grad()
    def reproduction(self, parents, batch_cell_ids, batch_size):
        new_children = []
        for child_i, cell_i in enumerate(batch_cell_ids):
            p = parents[cell_i]
            child = self.new_child(p, parents)

            # extra mutations for children
            for _ in range(self.config.extra_initial_mutations):
                child.mutate(self.config)
                
            new_children.append((child_i, cell_i, child))
            
            self.total_offspring += 1
            if len(new_children) >= batch_size:
                break

        return new_children


    @torch.no_grad()
    def get_batch_size(self):
        initial_pop_done = self.total_offspring >= self.config.num_cells
        batch_size = self.config.batch_size if initial_pop_done else self.config.initial_batch_size
        n_offspring_per_cell = self.config.offspring_per_cell
        return batch_size, initial_pop_done, n_offspring_per_cell
    

    @torch.no_grad()
    def get_next_batch_ids(self):
        batch_size, initial_pop_done, n_offspring_per_cell = self.get_batch_size()
        if initial_pop_done or not self.config.enforce_initial_fill:
            # random parents 
            batch_cell_ids = torch.tensor(np.random.choice(self.map.n_cells, size=batch_size//n_offspring_per_cell, replace=False), device=self.config.device)
            # duplicate n_offspring_per_cell times
            batch_cell_ids = batch_cell_ids.repeat_interleave(n_offspring_per_cell)
        else:
            # insure each cell is used once at first
            batch_cell_ids = torch.arange(start=self.total_offspring, end=min(self.map.n_cells, self.total_offspring+batch_size), device=self.config.device)
        return batch_cell_ids
    

    @torch.no_grad()
    def selection(self):
        parents = self.map.get_population(include_empty=True) # current elite map

        assert len(parents) == self.map.n_cells
        
        batch_size, initial_pop_done, n_offspring_per_cell = self.get_batch_size()
        
        batch_cell_ids = self.get_next_batch_ids()

        if hasattr(self, "target"):
           self.correct_target_count(len(batch_cell_ids))

        return parents, batch_cell_ids, initial_pop_done, batch_size, n_offspring_per_cell

    @torch.no_grad()
    def bloat_population(self, new_children):
        n_bloat = torch.zeros(len(new_children), device=self.config.device, dtype=torch.int32)
        if self.config.bloat_prune_ratio > 0:
            for child_i,_,c in new_children:
                before = len(c.connections)
                this_bloat = int(len(c.connections) * self.config.bloat_prune_ratio)
                for _ in range(this_bloat):
                    c.add_connection(self.config)
                after = len(c.connections)
                n_bloat[child_i] = after - before
        return n_bloat
    

    def sgd_population(self, new_children, batch_cell_ids, *, baseline=None, return_fitness: bool = False):
        steps = 0
        n_passes = [len(new_children), 0]  # fwd, back (will always do 1 fwd)
        after_result = baseline
        before_fit = None

        if self.config.sgd_steps > 0 and self.config.with_grad and (self.current_batch + 1) % self.config.grad_every == 0:
            self.init_sgd(batch_cell_ids)
            if self.config.thread_count > 1:
                raise NotImplementedError("Multiprocessing no longer implemented for MOVE")
            else:
                if baseline is not None:
                    before_fit, _, _ = baseline
                else:
                    before_fit, _, _ = self.measure_fitness(new_children, None)

                sgd_fn = sgd_weights
                if self.config.sgd_strat == 'imaml':
                    sgd_fn = sgd_weights_imaml

                steps = sgd_fn(
                    new_children,
                    mask=self.mask,
                    inputs=self.inputs,
                    target=self.target,
                    fns=self.sgd_fns,
                    norm=self.norm,
                    config=self.config,
                    early_stop=self.config.sgd_early_stop,
                    record_passes=n_passes,
                    normalizer=lambda fn, fitness: self._normalize_with_cache(fn, fitness, clamp=True),
                )

                if steps > 0:
                    after_result = self.measure_fitness(new_children, None)
                    after_fit, _, _ = after_result
                    mean_delta = (after_fit.mean() - before_fit.mean()).item()
                    max_delta = (after_fit - before_fit).max().item()
                    self._update_progress_stat(
                        "sgdΔ",
                        f"μ:{mean_delta:.4f} max:{max_delta:.4f}",
                    )
                else:
                    self._update_progress_stat("sgdΔ", None)
                    after_result = baseline if baseline is not None else (before_fit, None, None)
        else:
            self._update_progress_stat("sgdΔ", None)

        if return_fitness:
            return steps, n_passes, after_result
        return steps, n_passes
    

    @torch.no_grad()
    def prune_population(self, new_children, n_bloat):
        
        if self.config.bloat_prune_ratio > 0:
            for child_i, _, c in new_children:
                num = c.prune_fixed_connections(n_bloat[child_i])
        else:
            # populate the node activations or else they will be pruned for inactivity
            imgs = self.activate_population([g for _,_,g in new_children])
               
        n_pruned, n_pruned_nodes = 0,0
        for _,_,child in new_children:
            cx_pruned,nodes_pruned = child.prune(self.config)
            n_pruned += cx_pruned
            n_pruned_nodes += nodes_pruned
        return n_pruned, n_pruned_nodes


    @torch.no_grad()
    def replace_by_voting(self, fit_child, normed_fit_child):
        """ 
        The meat of MOVE
        """
        # find out if we should replace the current elites (must be better than the elite on average)
        if self.map.using_soft_mask:
            self.map.fitness = torch.where(torch.isinf(self.map.fitness), torch.ones_like(self.map.fitness)*-1000, self.map.fitness) # TODO no
            
            if self.config.soft_replace:
                # use soft replacement
                replaces = self.soft_replacement(normed_fit_child)
            else:
                # simple weighted average, replace if better or same
                replaces = (fit_child * self.map.fn_mask).sum(dim=0) >= (self.map.fitness * self.map.fn_mask).sum(dim=0)
            votes = None
            improvement = None
        else:
            # voting
            # find the cell/function combinations where the child is better than the current elite
            improvement = fit_child > self.map.fitness 
        
            # filter by the functions that are in each cell:
            improvement = improvement * self.map.fn_mask 
        
            # tabulate votes by taking sum over masked functions
            votes = improvement.sum(dim=0) 
            replaces = votes > self.fns_per_cell/2 
        
        return votes, improvement, replaces
    

    @torch.no_grad()
    def replacement(self, new_children, fit_children, fc_normed, batch_cell_ids, agg_fc_normed, initial_pop_done):
        all_replacements = torch.zeros((self.n_cells, self.n_cells), device=self.config.device)
        latest_jump_for_target: Dict[int, int] = {}
        
        random.shuffle(new_children) # random order of children
        for _, c_tuple in enumerate(new_children):
            # repeat for comparison against current map
            child_i, cell_i, child = c_tuple
            fit_child = fit_children[child_i] # (fns)
            fc_child_normed = fc_normed[child_i] # (fns)
            
            fit_child = fit_child.repeat(self.n_cells, 1).T # (fns, cells)
            normed_fit_child = fc_child_normed.repeat(self.n_cells, 1).T # (fns, cells)
            
            # find where this child is better than the current elite
            replaces = None
            if self.use_avg_fit:
                _, _, replaces = self.replace_by_avg_fit(fit_child)
            else:
               _, _, replaces = self.replace_by_voting(fit_child, normed_fit_child)
            
            # update the elites
          
            if not initial_pop_done and self.config.enforce_initial_fill:
                # only replace the selected cells initially, to preserve the initial population diversity
                tmp = torch.zeros_like(replaces) # start all 0s
                use = replaces[torch.isin(torch.arange(self.n_cells, device=self.config.device), batch_cell_ids)]
                tmp[torch.isin(torch.arange(self.n_cells, device=self.config.device), batch_cell_ids)] = use
                replaces = tmp
          
            if not self.config.allow_jumps:
                # can only replace parent
                tmp = torch.zeros_like(replaces) # start all 0s
                tmp[child_i] = replaces[child_i] # keep parent cell the same
                replaces = tmp
                assert torch.sum(replaces) <= 1
            

            elif torch.sum(replaces) > self.config.allow_jumps:
                # only allow_jumps cells can be replaced, sorted by superiority
                indices = torch.nonzero(replaces) # find replaces
                superiority = (fit_child * self.map.fn_mask) - self.map.fitness # (fns, cells)
                superiority = superiority.T # (cells, fns)
                sorted_indices = sorted(indices, key=lambda x: torch.sum(superiority[x]>0), reverse=True) # sort by superiority
                indices = sorted_indices[:self.config.allow_jumps] # take the most dominated allow_jumps cells
                replaces = torch.zeros_like(replaces) # start all 0s
                for idx in indices:
                    replaces[idx] = True # set the chosen cells to 1
            
            child.n_cells = torch.sum(replaces).item()
            
            all_replacements[cell_i] = replaces # all_replacements is (cells, cells) where each row is the replacement for that cell
            
            self.map.fitness[:,replaces] = fit_child[:,replaces].detach() # update fitness values
            self.map.normed_fitness[:,replaces] = normed_fit_child[:,replaces].detach() # update fitness values
            
            idxs_to_replace = torch.nonzero(replaces).squeeze().tolist()
            if isinstance(idxs_to_replace, int):
                idxs_to_replace = [idxs_to_replace]
                
            if self.debug_output:
                logging.debug(f"Replacing cells: {idxs_to_replace} with {child.id}")
                
            source_idx = int(cell_i.item()) if torch.is_tensor(cell_i) else int(cell_i)
            for r in idxs_to_replace:
                target_idx = int(r)
                if target_idx != source_idx:
                    prev_jump = latest_jump_for_target.get(target_idx)
                    if prev_jump is not None:
                        self.record.mark_jump_overridden(prev_jump)
                    jump_index = self.record.log_jump(source_idx, target_idx, self.current_batch)
                    latest_jump_for_target[target_idx] = jump_index
                placed = child.clone(self.config, new_id=False, cpu=True)
                placed.cell_lineage = child.cell_lineage + [r]
                placed.n_cells = child.n_cells
                self.map.map[r] = placed
                self.map.agg_fitness[r] = agg_fc_normed[child_i] 
            del child
            
            if not self.allow_multiple_placements:
                assert torch.sum(replaces) <= 1
                
        total_replacements = int(torch.sum(all_replacements).item())
        self._update_progress_stat("repl", str(total_replacements))

        return all_replacements
    
    
    @torch.no_grad()
    def save_checkpoint(self):
        print("Saving checkpoint")
        self.record.save_checkpoint(self.run_dir, self.checkpoints_dir, self.map, self.config, self.current_batch,
                                    save_data=True)
        self.save_move_info()
        

    @torch.no_grad()
    def selection_and_reproduction(self):
        evolve_this_batch = self.current_batch == 0 or (self.current_batch+1) % self.config.evolve_every == 0
        
        # selection
        if evolve_this_batch:
            parents, batch_cell_ids, initial_pop_done, batch_size, _ = self.selection()
        else: # no need to select
            batch_size, initial_pop_done, _ = self.get_batch_size()
            batch_cell_ids = self.get_next_batch_ids()

        # reproduction
        if evolve_this_batch:
            new_children = self.reproduction(parents, batch_cell_ids, batch_size)
        else: # new genomes are parents
            parents = self.map.get_population(include_empty=True)
            new_children = []
            for i, cell_i in enumerate(batch_cell_ids):
                p = parents[cell_i]
                if p is not None:
                    p = p.to(self.config.device)
                new_children.append((i, cell_i, p))

        # mutation
        self.mutation([g for _,_,g in new_children])

        # Bloat, SGD, prune  
        n_bloat                  = self.bloat_population(new_children)
        steps, n_passes          = self.sgd_population(new_children, batch_cell_ids)
        n_pruned, n_pruned_nodes = self.prune_population(new_children, n_bloat)
                    
        # Measure children
        fit_children, fc_normed, agg_fc_normed = self.measure_fitness(new_children, None)
        
        # Replace elites    
        if evolve_this_batch:
            all_replacements = self.replacement(new_children,
                                                fit_children,
                                                fc_normed,
                                                batch_cell_ids,
                                                agg_fc_normed,
                                                initial_pop_done)
        else:
            all_replacements = torch.zeros((self.n_cells, self.n_cells), device=self.config.device)

        # Record keeping
        self.record_keep(new_children, steps, n_pruned, n_pruned_nodes, all_replacements, n_passes)
            
       
    def replace_by_avg_fit(self, fit_child):
        """For testing without voting, use the average fitness of the child"""
        divisors = self.map.fn_mask.sum(dim=0) # will all be equal to self.fns_per_cell
        # mask out the functions that are not in each cell:
        masked_f = fit_child * self.map.fn_mask 
        # sum over functions that are in cell
        summed_f = masked_f.sum(dim=0) 
        # mean over functions that are in cell
        D = summed_f / divisors 
        # mask elite map (len(self.fns), num_cells) by the functions that are in each cell:
        masked_f = self.map.fitness * self.map.fn_mask 
        summed_f = masked_f.sum(dim=0) 
        E = summed_f / divisors 
        # replace nan with -inf TODO
        E = torch.where(torch.isnan(E), torch.full_like(E, -float('inf')), E)
        # find out if we should replace the current elites
        votes = D > E 
        replaces = votes
        return votes, D, replaces
   
   
    @torch.no_grad()
    def batch_end(self):
        self.solution_fitness = -torch.inf # force to update
        # self.record_keeping(skip_fitness=False)
        self.record.batch_end(self, skip_fitness=False)
        
        self.avg_nodes = sum([len(g.nodes) for g in self.population]) / len(self.population)
        self.avg_enabled_connections = sum([len(g.enabled_connections) for g in self.population]) / len(self.population)
        
        self.gen = self.total_offspring // self.config.num_cells
        
        if self.current_batch in [0, ] or (self.current_batch+1)%10 == 0:
            b = self.get_best()
            if b is not None:
                b.save(os.path.join(self.genomes_dir, f"batch_{self.current_batch:04d}.json"), self.config)
        if self.config.checkpoint_frequency > 0 and self.current_batch % self.config.checkpoint_frequency == 0:
            self.save_checkpoint()
    

    @torch.no_grad()
    def save_move_info(self):
        with open(os.path.join(self.run_dir, "cell_names.csv"), "w") as f:
            f.write(",".join(self.map.cell_names))
        with open(os.path.join(self.run_dir, "function_names.csv"), "w") as f:
            f.write(",".join([fn.__name__ for fn in self.fns]))
                
        torch.save(self.map.fn_mask, os.path.join(self.run_dir, "fn_mask.pt"))
        
        
    
    def on_end(self):
        super().on_end()
        
        if self.config.thread_count > 1:
            for w in self.workers:
                w.close()
                
        # save data
        self.record.save(self.run_dir, plot=True) # save statistics
           
        # save other data
        self.save_move_info()
        
        if not self.config.dry_run:
            self.record.save_map(self.image_dir, self.map, self.config, self.inputs)
        
        # save lineages
        logging.info("Saving lineages")
        lineages = {i:v for i,v in enumerate(self.get_lineages())}
        json.dump(lineages, open(os.path.join(self.run_dir, "lineages.json"), "w"), indent=4)
        
        self.save_checkpoint()
        
    
    def record_keep(self, new_children, steps, n_pruned, n_pruned_nodes, all_replacements, n_passes):
        n_step_fwds = len(new_children) # one guaranteed 
        n_step_evals = len(new_children) * len(self.fns)
        n_step_fwds_incl_sgd = n_step_fwds+n_passes[0]
        n_step_passes = n_passes[0]+n_passes[1]
        n_step_evals_incl_sgd = n_step_evals+(n_passes[0]*len(self.fns)) # 1 per cppn and 1 per cppn per fwd pass per function
        self.record.update_counts(self.current_batch, n_step_fwds, n_step_fwds_incl_sgd, n_step_evals, n_step_evals_incl_sgd, n_pruned, n_pruned_nodes, n_step_passes)
        if self.current_batch % self.config.record_frequency_batch != 0:
            pass # don't record
        else:
            index = self.current_batch // self.config.record_frequency_batch
            self.record.update(index,
                                all_replacements,
                                self.map.fitness,
                                self.map.normed_fitness,
                                self.map.agg_fitness,
                                self.map.map,
                                self.total_offspring
                                )
    
    def get_lineages(self):
        for g in self.map.map:
            if g is None:
                yield None
            else:
                yield g.cell_lineage       
        

    def soft_replacement(self, normed_fit_child):
        # random probability based on difference in fitness
        self.map.normed_fitness = torch.where(torch.isinf(self.map.fitness), torch.ones_like(self.map.normed_fitness)*-1000, self.map.fitness) # TODO FIXME
        diff  = normed_fit_child - self.map.normed_fitness
        diff  = diff * self.map.fn_mask
        diff  = diff.mean(dim=0)
        diff *= self.config.soft_replace_mod

        # random replacement
        return torch.rand(self.n_cells, device=self.config.device) < diff


def main():
    import threading
    import sys
    from . import MOVEGPU

    threads = []
    ci = -1
    for cfg, args in run_setup():
        ci+=1
        if args.condition is not None and ci != int(args.condition):
            print("Skipping condition", ci, "looking for", args.condition)
            
            continue
        
        # alg = MOVE(cfg, debug_output=args.verbose)
        alg = MOVEGPU(cfg, debug_output=args.verbose)
        if cfg.do_profile:
            import cProfile
            prof_path = os.path.join(alg.config.output_dir, f"{cfg.run_id:04d}.prof")
            cProfile.run("alg.evolve(resume=args.resume)", prof_path, sort="cumtime")
            import pstats

            file = open(os.path.join(alg.config.output_dir, f"{cfg.run_id:04d}.prof.txt"), 'w')
            profile = pstats.Stats(prof_path, stream=file)
            profile.sort_stats('cumulative') # Sorts the result according to the supplied criteria
            profile.print_stats(1000) # Prints the first 1000 lines of the sorted report
            file.close() 
        else:
            if args.parallel:
                logging.warning("Parallel processing not implemented for MOVE")
                print("Starting thread")
                # parallel processing
                evolve_thread = threading.Thread(target=alg.evolve, name="Evolve", daemon=False, args=(1, False, False, args.resume,))
                threads.append(evolve_thread)
            else:
                alg.evolve(resume=args.resume)    
        
    
    if threads:
        try:
            print(len(threads), "threads")
            for t in threads:
                t.start()
                print("Started thread")
            # wait for all threads to finish
            for t in threads:
                t.join()
        except (KeyboardInterrupt, SystemExit):
            print("Interrupted")
            for t in threads:
                t.join()
            sys.exit()
            

if __name__ == '__main__':
    main()
