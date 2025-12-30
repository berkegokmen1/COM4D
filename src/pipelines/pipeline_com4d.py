import inspect
import math
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import copy
import os

from tqdm import tqdm
import numpy as np
import PIL
import PIL.Image
import torch
import trimesh
import PIL.ImageFilter
from collections import defaultdict
from diffusers.image_processor import PipelineImageInput
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils import logging
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils.torch_utils import randn_tensor
from transformers import (
    BitImageProcessor,
    Dinov2Model,
)
from ..utils.inference_utils import hierarchical_extract_geometry
from ..utils.inference_utils import hierarchical_extract_fields
from ..utils.inference_utils import eliminate_collisions
from ..utils.inference_utils import field_to_mesh


from ..schedulers import RectifiedFlowScheduler
from ..models.autoencoders import TripoSGVAEModel
from ..models.transformers import COM4DDiTModel
from ..models.attention_processor import COM4DAttnProcessor, TripoSGAttnProcessor2_0
from .pipeline_com4d_output import COM4DPipelineOutput
from .pipeline_utils import TransformerDiffusionMixin
from ..utils.inference import _apply_mask, _combine_masks
from ..utils.render_utils import export_renderings, render_sequence_fixed_camera, render_views_around_mesh

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError(
            "Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values"
        )
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class COM4DInferencePipeline(DiffusionPipeline, TransformerDiffusionMixin):
    """Pipeline that reconstructs a static scene and a dynamic 4D object with history-guided diffusion."""

    def __init__(
        self,
        vae: TripoSGVAEModel,
        transformer: COM4DDiTModel,
        scheduler: FlowMatchEulerDiscreteScheduler,
        image_encoder_dinov2: Dinov2Model,
        feature_extractor_dinov2: BitImageProcessor,
        image_encoder_dinov2_multi: Optional[Dinov2Model] = None,
    ):
        super().__init__()
        self.register_modules(
            vae=vae,
            transformer=transformer,
            scheduler=scheduler,
            image_encoder_dinov2=image_encoder_dinov2,
            feature_extractor_dinov2=feature_extractor_dinov2,
            image_encoder_dinov2_multi=image_encoder_dinov2_multi,
        )

    def encode_image(self, image, device, num_images_per_prompt, use_multi=False):
        dtype = next(self.image_encoder_dinov2.parameters()).dtype
        if not isinstance(image, torch.Tensor):
            image = self.feature_extractor_dinov2(image, return_tensors="pt").pixel_values
        image = image.to(device=device, dtype=dtype)
        image_embeds = self.image_encoder_dinov2(image).last_hidden_state
        image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
        uncond_image_embeds = torch.zeros_like(image_embeds)
        return image_embeds, uncond_image_embeds

    def prepare_latents(
        self,
        batch_size,
        num_tokens,
        num_channels_latents,
        dtype,
        device,
        generator,
        latents: Optional[torch.Tensor] = None,
    ):
        shape = (batch_size, num_tokens, num_channels_latents)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, "
                f"but requested an effective batch size of {batch_size}."
            )
        noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        return noise if latents is None else latents

    def _build_attn_processor_map(self, active_ids: List[int]):
        mapping = {}
        for layer_id in range(self.transformer.config.num_layers):
            for attn_id in [1, 2]:
                key = f"blocks.{layer_id}.attn{attn_id}.processor"
                if layer_id in active_ids:
                    mapping[key] = COM4DAttnProcessor()
                else:
                    mapping[key] = TripoSGAttnProcessor2_0()
        return mapping

    def _set_attention_blocks(self, active_ids: List[int]):
        self.transformer.set_global_attn_block_ids(active_ids)

    def _clone_scheduler(self) -> RectifiedFlowScheduler:
        return RectifiedFlowScheduler.from_config(self.scheduler.config)

    def _run_joint_scene_stage(
        self,
        frame,
        masks,
        num_tokens: int,
        num_static: int,
        num_dynamic: int,
        block_size: int,
        scene_mix_cutoff: int,
        guidance_scale: float,
        num_inference_steps: int,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    ):
        device = self.device
        total_parts = num_static + num_dynamic
        if total_parts == 0:
            return [], torch.empty(0, device=device)

        base_image = frame[0] if isinstance(frame, (list, tuple)) and len(frame) > 0 else frame

        dynamic_masked_images: List[PipelineImageInput] = []
        mask_list = masks or []
        for idx in range(min(len(mask_list), num_dynamic)):
            mask = mask_list[idx]
            masked = _apply_mask(base_image, mask, keep_foreground=True, dilation_radius=1)
            dynamic_masked_images.append(masked)
    
        if num_dynamic > 0 and not dynamic_masked_images:
            dynamic_masked_images = [base_image] * num_dynamic

        static_full_embeds = None
        static_full_uncond = None
        if num_static > 0:
            static_full_embeds, static_full_uncond = self.encode_image([base_image], device, 1)
            static_full_embeds = static_full_embeds.repeat(num_static, 1, 1)
            static_full_uncond = static_full_uncond.repeat(num_static, 1, 1)

        dynamic_embeds = None
        dynamic_uncond = None
        if num_dynamic > 0:
            dynamic_embeds, dynamic_uncond = self.encode_image(dynamic_masked_images, device, 1, use_multi=True)
            if dynamic_embeds.shape[0] != num_dynamic:
                dynamic_embeds = self._tile_history_tensor(dynamic_embeds, num_dynamic)
                dynamic_uncond = self._tile_history_tensor(dynamic_uncond, num_dynamic)

        dtype = (
            static_full_embeds.dtype
            if static_full_embeds is not None
            else dynamic_embeds.dtype
        )

        scheduler = self._clone_scheduler()
        timesteps_ref, num_inference_steps = retrieve_timesteps(
            scheduler, num_inference_steps, device
        )

        matrix_steps = min(scene_mix_cutoff, len(timesteps_ref))
        remaining_timesteps = timesteps_ref[matrix_steps:]

        num_channels_latents = self.transformer.config.in_channels
        static_latents = (
            self.prepare_latents(
                num_static,
                num_tokens,
                num_channels_latents,
                dtype,
                device,
                generator,
                None,
            )
            if num_static > 0
            else torch.empty(0, num_tokens, num_channels_latents, device=device, dtype=dtype)
        )
        dynamic_latents = (
            self.prepare_latents(
                num_dynamic,
                num_tokens,
                num_channels_latents,
                dtype,
                device,
                generator,
                None,
            )
            if num_dynamic > 0
            else torch.empty(0, num_tokens, num_channels_latents, device=device, dtype=dtype)
        )

        do_cfg = guidance_scale > 1.0

        attention_kwargs = {"num_parts": total_parts, "num_frames": 1}

        progress_disable = (
            self._progress_bar_config.get("disable", False)
            if hasattr(self, "_progress_bar_config")
            else False
        )

        static_blocks: List[List[int]] = []
        if num_static > 0:
            block = max(1, block_size)
            static_blocks = [
                list(range(start, min(start + block, num_static)))
                for start in range(0, num_static, block)
            ]

        static_iters = len(remaining_timesteps) * len(static_blocks)
        dynamic_iters = len(remaining_timesteps) * num_dynamic if num_dynamic > 0 else 0

        total_iters = matrix_steps + static_iters + dynamic_iters
        self.set_progress_bar_config(
            desc="Joint Scene Denoising",
            ncols=125,
            disable=progress_disable,
        )

        static_matrix_embeds = (
            static_full_embeds.unsqueeze(1) if static_full_embeds is not None else None
        )
        dynamic_matrix_embeds = (
            dynamic_embeds.unsqueeze(1) if dynamic_embeds is not None else None
        )

        with self.progress_bar(total=total_iters) as progress_bar:
            for step_idx in range(matrix_steps):
                t = timesteps_ref[step_idx]

                latents_blocks: List[torch.Tensor] = []
                encoder_blocks: List[torch.Tensor] = []

                if num_static > 0:
                    latents_blocks.append(static_latents.unsqueeze(1))
                    encoder_blocks.append(static_matrix_embeds)
                if num_dynamic > 0:
                    latents_blocks.append(dynamic_latents.unsqueeze(1))
                    encoder_blocks.append(dynamic_matrix_embeds)

                latents_matrix = torch.cat(latents_blocks, dim=0)
                encoder_hidden_states_matrix = torch.cat(encoder_blocks, dim=0)

                timestep = t.expand(1)

                transformer_output = self.transformer.forward_mixing(
                    latents_matrix,
                    timestep,
                    encoder_hidden_states_matrix,
                    static_count=num_static,
                    dynamic_count=num_dynamic,
                    return_dict=False,
                    cutoff=False,
                )[0]

                transformer_output_uncond = (
                    self.transformer.forward_mixing(
                        latents_matrix,
                        timestep,
                        torch.zeros_like(encoder_hidden_states_matrix),
                        static_count=num_static,
                        dynamic_count=num_dynamic,
                        return_dict=False,
                        cutoff=False,
                    )[0]
                    if do_cfg
                    else None
                )

                if do_cfg:
                    noise_pred = transformer_output_uncond + guidance_scale * (
                        transformer_output - transformer_output_uncond
                    )
                else:
                    noise_pred = transformer_output

                noise_pred_flat = noise_pred[:, 0]
                latents_cat = torch.cat([static_latents, dynamic_latents], dim=0)

                latents_next = scheduler.step(
                    noise_pred_flat, t, latents_cat, return_dict=False
                )[0]

                if latents_next.dtype != latents_cat.dtype:
                    latents_next = latents_next.to(latents_cat.dtype)

                if num_static > 0:
                    static_latents = latents_next[:num_static]
                if num_dynamic > 0:
                    dynamic_latents = latents_next[num_static:]

                progress_bar.update()

            if len(remaining_timesteps) > 0 and num_static > 0:
                static_attention_kwargs = dict(attention_kwargs or {})
                static_attention_kwargs.setdefault("num_parts", num_static)
                static_attention_kwargs.setdefault("num_frames", 1)

                static_encoder = static_full_embeds
                static_encoder_uncond = static_full_uncond

                for block_indices in static_blocks:
                    for t in remaining_timesteps:
                        update_mask = torch.zeros(
                            (static_latents.shape[0], 1, 1),
                            dtype=torch.bool,
                            device=device,
                        )
                        update_mask[block_indices] = True

                        latents_prev = static_latents.clone()
                        latents_feed = static_latents.clone()

                        latent_model_input = (
                            torch.cat([latents_feed, latents_feed], dim=0)
                            if do_cfg
                            else latents_feed
                        )
                        timestep = t.expand(latent_model_input.shape[0])

                        encoder_states = static_encoder
                        encoder_states_cfg = (
                            torch.cat(
                                [static_encoder_uncond, static_encoder], dim=0
                            )
                            if do_cfg
                            else encoder_states
                        )

                        noise_pred = self.transformer.forward(
                            latent_model_input,
                            timestep,
                            encoder_hidden_states=encoder_states_cfg
                            if do_cfg
                            else encoder_states,
                            attention_kwargs=static_attention_kwargs,
                            return_dict=False,
                        )[0].to(static_latents.dtype)

                        if do_cfg:
                            noise_uncond, noise_cond = noise_pred.chunk(2)
                            noise_pred = noise_uncond + guidance_scale * (
                                noise_cond - noise_uncond
                            )

                        latents_next = scheduler.step(
                            noise_pred, t, latents_feed, return_dict=False
                        )[0]

                        static_latents = torch.where(
                            update_mask, latents_next, latents_prev
                        )
                        progress_bar.update()

            if len(remaining_timesteps) > 0 and num_dynamic > 0:
                dynamic_attention_kwargs = dict(attention_kwargs or {})
                dynamic_attention_kwargs.setdefault("num_parts", 1)
                dynamic_attention_kwargs.setdefault("num_frames", 1)

                new_dynamic_latents = []
                for idx in range(num_dynamic):
                    scheduler_single = self._clone_scheduler()
                    timesteps_single, _ = retrieve_timesteps(
                        scheduler_single, len(remaining_timesteps), device
                    )

                    latents_single = dynamic_latents[idx : idx + 1].clone()
                    encoder_single = dynamic_embeds[idx : idx + 1]
                    encoder_single_uncond = (
                        dynamic_uncond[idx : idx + 1] if do_cfg else None
                    )

                    for t in timesteps_single:
                        latent_model_input = (
                            torch.cat([latents_single, latents_single], dim=0)
                            if do_cfg
                            else latents_single
                        )
                        timestep = t.expand(latent_model_input.shape[0])

                        encoder_states = encoder_single
                        encoder_states_cfg = (
                            torch.cat(
                                [encoder_single_uncond, encoder_single], dim=0
                            )
                            if do_cfg
                            else encoder_states
                        )

                        noise_pred = self.transformer.forward(
                            latent_model_input,
                            timestep,
                            encoder_hidden_states=encoder_states_cfg
                            if do_cfg
                            else encoder_states,
                            attention_kwargs=dynamic_attention_kwargs,
                            return_dict=False,
                        )[0].to(latents_single.dtype)

                        if do_cfg:
                            noise_uncond, noise_cond = noise_pred.chunk(2)
                            noise_pred = noise_uncond + guidance_scale * (
                                noise_cond - noise_uncond
                            )

                        latents_single = scheduler_single.step(
                            noise_pred, t, latents_single, return_dict=False
                        )[0]
                        progress_bar.update()

                    new_dynamic_latents.append(latents_single.squeeze(0))

                dynamic_latents = torch.stack(new_dynamic_latents, dim=0)

        self.vae.set_flash_decoder()
        scene_meshes: List[trimesh.Trimesh] = []
        dynamic_meshes: List[trimesh.Trimesh] = []

        self.set_progress_bar_config(
            desc="Joint Scene Decoding",
            ncols=125,
            disable=progress_disable,
        )

        with self.progress_bar(total=len(static_latents)) as progress_bar:
            for i in range(len(static_latents)):
                geometric_func = lambda x, idx=i: self.vae.decode(
                    static_latents[idx].unsqueeze(0), sampled_points=x
                ).sample
                try:
                    mesh_v_f = hierarchical_extract_geometry(
                        geometric_func,
                        device,
                        dtype=static_latents.dtype,
                        bounds=(-1.005, -1.005, -1.005, 1.005, 1.005, 1.005),
                        dense_octree_depth=8,
                        hierarchical_octree_depth=9,
                        max_num_expanded_coords=1e8,
                    )
                    mesh = trimesh.Trimesh(
                        mesh_v_f[0].astype(np.float32), mesh_v_f[1]
                    )
                except Exception as exc:
                    print(f"Warning: joint mesh extraction failed for item {i}: {exc}")
                    mesh_v_f = None
                    mesh = None
                scene_meshes.append(mesh)
                progress_bar.update()
        
        with self.progress_bar(total=len(dynamic_latents)) as progress_bar:
            for i in range(len(dynamic_latents)):
                geometric_func = lambda x, idx=i: self.vae.decode(
                    dynamic_latents[idx].unsqueeze(0), sampled_points=x
                ).sample
                try:
                    mesh_v_f = hierarchical_extract_geometry(
                        geometric_func,
                        device,
                        dtype=dynamic_latents.dtype,
                        bounds=(-1.005, -1.005, -1.005, 1.005, 1.005, 1.005),
                        dense_octree_depth=8,
                        hierarchical_octree_depth=9,
                        max_num_expanded_coords=1e8,
                    )
                    mesh = trimesh.Trimesh(
                        mesh_v_f[0].astype(np.float32), mesh_v_f[1]
                    )
                except Exception as exc:
                    print(f"Warning: joint mesh extraction failed for item {i}: {exc}")
                    mesh_v_f = None
                    mesh = None
                dynamic_meshes.append(mesh)
                progress_bar.update()

        return scene_meshes, dynamic_meshes, static_latents, dynamic_latents

    def _tile_history_tensor(self, tensor: torch.Tensor, target: int) -> torch.Tensor:
        if tensor.shape[0] == target:
            return tensor
        repeats = (target + tensor.shape[0] - 1) // tensor.shape[0]
        return tensor.repeat(repeats, 1, 1)[:target]

    def _run_dynamic_stage(
        self,
        frames: List[PipelineImageInput],
        masks: Optional[List[PipelineImageInput]],
        scene_latents: torch.Tensor,
        initial_dynamic_latents: Optional[torch.Tensor],
        num_tokens: int,
        num_static: int,
        num_dynamic: int,
        block_size: int,
        dynamic_mix_cutoff: int,
        guidance_scale: float,
        num_inference_steps: int,
        generator,
        dynamic_max_memory_frames: int = 6,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ):
        device = self.device

        foreground_frames_per_object: List[List[PipelineImageInput]] = []

        mask_list = (masks or [])[:num_dynamic]
        for obj_masks in mask_list:
            masked_frames = []
            for frame_idx, mask in enumerate(obj_masks):
                masked_frames.append(_apply_mask(frames[frame_idx], mask, keep_foreground=True, dilation_radius=1))
 
            foreground_frames_per_object.append(masked_frames)

        dynamic_embeds_per_object = [
            self.encode_image(masked_frames, device, 1, use_multi=True)[0]
            for masked_frames in foreground_frames_per_object
        ]
        if len(dynamic_embeds_per_object) == 0:
            dynamic_embeds = torch.empty((0, len(frames), 0, 0), device=device)
        elif len(dynamic_embeds_per_object) == 1:
            dynamic_embeds = dynamic_embeds_per_object[0].unsqueeze(0)
        else:
            dynamic_embeds = torch.cat([demb.unsqueeze(0) for demb in dynamic_embeds_per_object], dim=0)

        static_frames_full = list(frames)
        static_embeds = self.encode_image(static_frames_full, device, 1)[0]

        # N is the number of dynamic parts (objects), F is the number of frames
        N, F, _, _ = dynamic_embeds.shape
        
        if N == 0:
            return [], torch.empty(0, device=device)

        scheduler = self._clone_scheduler()
        timesteps_ref, num_inference_steps = retrieve_timesteps(scheduler, num_inference_steps, device)

        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            N,
            num_tokens,
            num_channels_latents,
            dynamic_embeds.dtype,
            device,
            generator,
            None,
        )
        # Use the same starting noise for the frames of each object
        latents = latents.unsqueeze(1).repeat(1, F, 1, 1)  # (N, F, num_tokens, C)
        latents = latents.view(N, F, num_tokens, num_channels_latents)

        block_size = max(1, int(block_size))
        blocks = [
            list(range(start, min(start + block_size, F)))
            for start in range(0, F, block_size)
        ]

        history_latents_first_block: Optional[torch.Tensor] = None
        history_dynamic_embeds_first_block: Optional[torch.Tensor] = None
        history_static_embed_first_block: Optional[torch.Tensor] = None
        if initial_dynamic_latents is not None and initial_dynamic_latents.numel() > 0:
            history_latents_first_block = initial_dynamic_latents.to(
                device=device, dtype=latents.dtype
            ).unsqueeze(1)
            if history_latents_first_block.shape[0] != N:
                if history_latents_first_block.shape[0] == 1 and N > 1:
                    history_latents_first_block = history_latents_first_block.repeat(
                        N, 1, 1, 1
                    )
                elif history_latents_first_block.shape[0] < N:
                    repeats = (N + history_latents_first_block.shape[0] - 1) // history_latents_first_block.shape[0]
                    history_latents_first_block = history_latents_first_block.repeat(
                        repeats, 1, 1, 1
                    )[:N]
                else:
                    history_latents_first_block = history_latents_first_block[:N]

            history_dynamic_embeds_first_block = dynamic_embeds[:, :1].clone()
            history_static_embed_first_block = static_embeds[:1].clone()
        
        initial_history_consumed = False

        spatial_layers = list(getattr(self.transformer, "spatial_global_attn_block_ids", []))
        temporal_layers = list(getattr(self.transformer, "temporal_global_attn_block_ids", []))

        print(f"Dynamic stage with {F} frames, block_size={block_size}, blocks={len(blocks)}")
        print(f"  Spatial attn layers: {spatial_layers}")
        print(f"  Temporal attn layers: {temporal_layers}")

        self._set_attention_blocks(temporal_layers)

        do_cfg = guidance_scale > 1.0

        static_latents_all = (
            scene_latents.to(device=device, dtype=latents.dtype)
            if scene_latents is not None and scene_latents.numel() > 0
            else latents.new_zeros((0, latents.shape[2], latents.shape[3]))
        )

        static_count = static_latents_all.shape[0]

        base_attention_kwargs = dict(attention_kwargs or {})
        base_attention_kwargs = base_attention_kwargs or None

        progress_disable = (
            self._progress_bar_config.get("disable", False)
            if hasattr(self, "_progress_bar_config")
            else False
        )
        total_iters = len(timesteps_ref) * len(blocks)
        self.set_progress_bar_config(desc="4D Denoising", ncols=125, disable=progress_disable)

        MAX_FRAMES_IN_MEMORY = dynamic_max_memory_frames

        with self.progress_bar(total=total_iters) as progress_bar:
            for block_indices in blocks:
                block_dynamic_count = len(block_indices)
                timesteps_block, _ = retrieve_timesteps(scheduler, num_inference_steps, device)

                history_start = max(0, block_indices[0] - block_size)
                history_indices = list(range(history_start, block_indices[0]))
                
                use_initial_history = (
                    history_latents_first_block is not None
                    and not initial_history_consumed
                    and block_indices
                    and block_indices[0] == 0
                )
                extra_history = 1 if use_initial_history else 0
                total_frames = block_dynamic_count + len(history_indices) + extra_history
                max_frames_allowed = MAX_FRAMES_IN_MEMORY + extra_history

                if total_frames > max_frames_allowed:
                    overflow = total_frames - max_frames_allowed
                    if overflow > 0 and len(history_indices) > 0:
                        trim = min(overflow, len(history_indices))
                        history_indices = history_indices[trim:]
                    total_frames = block_dynamic_count + len(history_indices) + extra_history
                    overflow = total_frames - max_frames_allowed
                    if overflow > 0 and use_initial_history:
                        use_initial_history = False
                        extra_history = 0
                        total_frames = block_dynamic_count + len(history_indices)
                        max_frames_allowed = MAX_FRAMES_IN_MEMORY
                        overflow = total_frames - max_frames_allowed
                        if overflow > 0 and len(history_indices) > 0:
                            trim = min(overflow, len(history_indices))
                            history_indices = history_indices[trim:]
                            total_frames = block_dynamic_count + len(history_indices)
                    if total_frames > max_frames_allowed:
                        print(
                            "Warning: dynamic block exceeds memory budget even after trimming history; "
                            "consider reducing block_size."
                        )

                block_dynamic_history_count = len(history_indices) + (1 if use_initial_history else 0)

                if use_initial_history and not initial_history_consumed:
                    print("Using initial dynamic latents as history for first dynamic block")

                for t_index, t in enumerate(timesteps_block):
                    cutoff = t_index > dynamic_mix_cutoff

                    static_embed_slices: List[torch.Tensor] = []
                    if use_initial_history and history_static_embed_first_block is not None:
                        static_embed_slices.append(history_static_embed_first_block)
                    if history_indices:
                        static_embed_slices.append(static_embeds[history_indices])
                    static_embed_slices.append(static_embeds[block_indices])
                    encoder_hidden_states_static = torch.cat(static_embed_slices, dim=0).to(device=device)

                    dynamic_embed_slices: List[torch.Tensor] = []
                    if use_initial_history and history_dynamic_embeds_first_block is not None:
                        dynamic_embed_slices.append(history_dynamic_embeds_first_block)
                    if history_indices:
                        dynamic_embed_slices.append(dynamic_embeds[:, history_indices])
                    dynamic_embed_slices.append(dynamic_embeds[:, block_indices])
                    encoder_hidden_states_dynamic = torch.cat(dynamic_embed_slices, dim=1).to(device=device)

                    if not cutoff:
                        # replace dynamic hidden states with static
                        encoder_hidden_states_dynamic[:, :] = encoder_hidden_states_static.unsqueeze(0).repeat(N, 1, 1, 1)

                    # Use all dynamic objects
                    dynamic_latent_slices: List[torch.Tensor] = []
                    if use_initial_history and history_latents_first_block is not None:
                        dynamic_latent_slices.append(history_latents_first_block)
                    if history_indices:
                        dynamic_latent_slices.append(latents[:, history_indices].clone())
                    dynamic_latent_slices.append(latents[:, block_indices].clone())
                    dynamic_latents = torch.cat(dynamic_latent_slices, dim=1).to(device=device)

                    dynamic_len = dynamic_latents.shape[0]
                    dynamic_frame_len = dynamic_latents.shape[1]
                    static_len = scene_latents.shape[0]
                    _, _, T, C = dynamic_latents.shape
                    _, _, K, D = encoder_hidden_states_dynamic.shape

                    latents_matrix = torch.zeros((dynamic_len + static_len, dynamic_frame_len, T, C)).to(dynamic_latents.device).to(dynamic_latents.dtype)
                    encoder_hidden_states_matrix = torch.zeros((dynamic_len + static_len, dynamic_frame_len, K, D)).to(dynamic_latents.device).to(dynamic_latents.dtype)

                    latents_matrix[:static_len] = static_latents_all.unsqueeze(1).repeat(1, dynamic_frame_len, 1, 1)
                    latents_matrix[static_len:] = dynamic_latents

                    encoder_hidden_states_matrix[:static_len] = encoder_hidden_states_static.unsqueeze(0).repeat(static_len, 1, 1, 1)
                    encoder_hidden_states_matrix[static_len:] = encoder_hidden_states_dynamic

                    timestep = t.expand(1)

                    transformer_output = self.transformer.forward_mixing(
                        latents_matrix,
                        timestep,
                        encoder_hidden_states_matrix,
                        static_count=static_count,
                        dynamic_count=block_dynamic_history_count + block_dynamic_count,
                        return_dict=False,
                        cutoff=cutoff
                    )[0]

                    transformer_output_uncond = self.transformer.forward_mixing(
                        latents_matrix,
                        timestep,
                        torch.zeros_like(encoder_hidden_states_matrix),
                        static_count=static_count,
                        dynamic_count=block_dynamic_history_count + block_dynamic_count,
                        return_dict=False,
                        cutoff=cutoff
                    )[0] if do_cfg else None

                    if do_cfg:
                        noise_uncond, noise_cond = transformer_output_uncond, transformer_output
                        noise_pred = noise_uncond + guidance_scale * (
                            noise_cond - noise_uncond
                        )
                    else:
                        noise_pred = transformer_output

                    noise_dynamic = noise_pred[static_len:]

                    current_dynamic_count = noise_dynamic.shape[0]
                    current_dynamic_frame_count = noise_dynamic.shape[1]

                    noise_dynamic = noise_dynamic.view(current_dynamic_count * current_dynamic_frame_count, T, C)
                    dynamic_latents = dynamic_latents.view(current_dynamic_count * current_dynamic_frame_count, T, C)

                    latents_next = scheduler.step(noise_dynamic, t, dynamic_latents, return_dict=False)[0]
                    if latents_next.dtype != latents.dtype:
                        latents_next = latents_next.to(latents.dtype)

                    latents_next = latents_next.view(current_dynamic_count, current_dynamic_frame_count, T, C)
                    
                    latents[:, block_indices] = latents_next[:, -block_dynamic_count:]
                    progress_bar.update()

                if use_initial_history:
                    initial_history_consumed = True
        
        self.vae.set_flash_decoder()

        dynamic_fields = []
        for i in tqdm(range(latents.shape[0]), desc="Extracting dynamic fields", disable=progress_disable):
            dynamic_fields_per_part = []
            for j in tqdm(range(latents.shape[1]), desc=f" Part {i} frames", disable=progress_disable):
                if torch.isnan(latents[i, j]).any():
                    raise ValueError(f"NaN detected in latents at dynamic part {i}, frame {j}")
                
                geometric_func = lambda x, idx=j: self.vae.decode(latents[i, idx].unsqueeze(0), sampled_points=x).sample
                try:
                    field = hierarchical_extract_fields(
                        geometric_func,
                        device,
                        dtype=latents.dtype,
                        bounds=(-1.005, -1.005, -1.005, 1.005, 1.005, 1.005),
                        dense_octree_depth=8,
                        hierarchical_octree_depth=9,
                        max_num_expanded_coords=1e8,
                    )

                    dynamic_fields_per_part.append(field)
                except Exception as e:
                    print(f"Warning: dynamic field extraction failed for item {i}: {e}")
                    dynamic_fields_per_part.append(None)
            
            dynamic_fields.append(dynamic_fields_per_part)

        dynamic_meshes_per_frame = []
            
        for frame in range(F):
            frame_dynamic_meshes = []
            for obj_idx in range(len(dynamic_fields)):
                frame_dynamic_meshes.append(field_to_mesh(
                    dynamic_fields[obj_idx][frame],
                    bounds=(-1.005, -1.005, -1.005, 1.005, 1.005, 1.005),
                    octree_depth=9,
                    device=device,
                ))
            dynamic_meshes_per_frame.append(frame_dynamic_meshes)

        return dynamic_meshes_per_frame

    @torch.no_grad()
    def __call__(
        self,
        frames: List[PipelineImageInput],
        masks: Optional[List[PipelineImageInput]] = None,
        num_tokens: int = 2048,
        scene_num_parts: Optional[int] = None,
        dynamic_num_parts: Optional[int] = None,
        scene_block_size: int = 4,
        dynamic_block_size: int = 8,
        dynamic_max_memory_frames: int = 8,
        scene_mix_cutoff: int = 10,
        dynamic_mix_cutoff: int = 10,
        scene_inference_steps: int = 50,
        dynamic_inference_steps: int = 50,
        guidance_scale_scene: float = 7.0,
        guidance_scale_dynamic: float = 7.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        return_dict: bool = True,

    ):

        print("Starting COM4D inference"
            f"\n  Number of tokens: {num_tokens}"
            f"\n  Scene parts: {scene_num_parts}"
            f"\n  Dynamic parts: {dynamic_num_parts}"
            f"\n  Scene block size: {scene_block_size}"
            f"\n  Dynamic block size: {dynamic_block_size}"
            f"\n  Dynamic max memory frames: {dynamic_max_memory_frames}"
            f"\n  Scene mix cutoff: {scene_mix_cutoff}"
            f"\n  Dynamic mix cutoff: {dynamic_mix_cutoff}"
        )
        
        device = next(self.transformer.parameters()).device
        scene_part_count = int(scene_num_parts or 0)
        dynamic_part_count = int(dynamic_num_parts) if dynamic_num_parts is not None else len(masks or [])

        if not frames:
            raise ValueError("At least one frame is required for joint scene stage")

        if scene_part_count != 0:

            scene_meshes, dynamic_meshes, static_latents, dynamic_latents = self._run_joint_scene_stage(
                frames[0],
                [masks[i][0] for i in range(dynamic_part_count)],
                num_tokens=num_tokens,
                num_static=scene_part_count,
                num_dynamic=dynamic_part_count,
                block_size=scene_block_size,
                scene_mix_cutoff=scene_mix_cutoff,
                guidance_scale=guidance_scale_scene,
                num_inference_steps=scene_inference_steps,
                generator=generator,
            )

        else:
            scene_meshes = []
            static_latents = torch.empty(0, device=device)
            dynamic_latents = torch.empty(0, device=device)

        dynamic_meshes_per_frame = self._run_dynamic_stage(
            frames,
            masks,
            static_latents,
            dynamic_latents,
            num_tokens=num_tokens,
            num_static=scene_part_count,
            num_dynamic=dynamic_part_count,
            block_size=dynamic_block_size,
            dynamic_mix_cutoff=dynamic_mix_cutoff,
            guidance_scale=guidance_scale_dynamic,
            num_inference_steps=dynamic_inference_steps,            
            generator=generator,
            dynamic_max_memory_frames=dynamic_max_memory_frames,
        )
        # animation_file = self._render_animation(
        #     scene_meshes,
        #     static_meshes_per_frame,
        #     dynamic_meshes_per_frame,
        #     animation_path,
        #     animation_fps,
        #     insert_rotation_every,
        #     render_kwargs,
        #     frames,
        # )
        output = COM4DPipelineOutput(
            scene_meshes=scene_meshes,
            dynamic_meshes_per_frame=dynamic_meshes_per_frame,
        )
        if not return_dict:
            return output.scene_meshes, output.dynamic_meshes_per_frame
        
        return output
