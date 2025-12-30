import argparse
import inspect
import json
import os
import shutil
import sys
from glob import glob
import time
from pathlib import Path
from typing import Any, List, Optional, Union

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from accelerate.utils import set_seed
from PIL import Image
from src.models.transformers import COM4DDiTModel
from src.pipelines.pipeline_com4d import COM4DInferencePipeline

from src.utils.inference import (_gather_images, _gather_all_masks, _combine_masks, _resolve_repo_or_dir, _parse_id_string, build_pipeline, _invert_mask)

from src.utils.data_utils import get_colored_mesh_composition
from src.utils.render_utils import render_views_around_mesh_and_export, render_dynamic_animation


def seed_all(seed: int):
    set_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    parser = argparse.ArgumentParser(description="COM4D Inference")
    parser.add_argument(
        "--frames_dir",
        type=str,
        default=None,
        help="Directory containing ordered RGB frames of the scene/video",
    )
    parser.add_argument(
        "--masks_dir",
        type=str,
        default=None,
        help="Optional directory containing binary masks aligned with frames",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results_3d4d",
        help="Directory to save meshes and renders",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Run identifier appended to output directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--num_tokens",
        type=int,
        default=2048,
        help="VAE token count",
    )
    parser.add_argument(
        "--scene_steps",
        type=int,
        default=50,
        help="Diffusion steps for static scene reconstruction",
    )
    parser.add_argument(
        "--dynamic_steps",
        type=int,
        default=50,
        help="Diffusion steps for dynamic object reconstruction",
    )
    parser.add_argument(
        "--scene_num_parts",
        type=int,
        default=1,
        help="Number of static scene parts to reconstruct",
    )
    parser.add_argument(
        "--dynamic_num_parts",
        type=int,
        default=1,
        help="Number of dynamic object parts to reconstruct",
    )
    parser.add_argument(
        "--scene_block_size",
        type=int,
        default=1,
        help="Block size for static scene parts to reconstruct",
    )
    parser.add_argument(
        "--scene_guidance",
        type=float,
        default=7.0,
        help="Guidance scale during scene reconstruction",
    )
    parser.add_argument(
        "--dynamic_guidance",
        type=float,
        default=7.0,
        help="Guidance scale during dynamic reconstruction",
    )
    parser.add_argument(
        "--dynamic_block_size",
        type=int,
        default=8,
        help="Autoregressive block size for dynamic stage",
    )
    parser.add_argument(
        "--animation",
        action="store_true",
        help="Export animated GIF with reconstructed scene",
    )
    parser.add_argument(
        "--animation_fps",
        type=int,
        default=8,
        help="FPS for exported animation",
    )
    parser.add_argument(
        "--animation_insert_rotation_every",
        type=int,
        default=0,
        help="Insert a 360 render every N frames (0 disables)",
    )
    parser.add_argument(
        "--render_size",
        type=int,
        default=1024,
        help="Render resolution (square)",
    )
    parser.add_argument(
        "--base_weights_dir",
        type=str,
        default="pretrained_weights/TripoSG",
        help="HF repo-id or local directory for base weights",
    )
    parser.add_argument(
        "--transformer_dir",
        type=str,
        default=None,
        help="Directory containing transformer weights (config.json + diffusion model)",
    )
    parser.add_argument(
        "--scene_attn_ids",
        type=str,
        default=None,
        help="Optional space/comma separated list of scene attention block ids",
    )
    parser.add_argument(
        "--dynamic_attn_ids",
        type=str,
        default=None,
        help="Optional space/comma separated list of dynamic attention block ids",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Computation device",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float32", "float16", "bfloat16"],
        help="Computation dtype",
    )
    parser.add_argument(
        "--frames_start_idx",
        type=int,
        default=0,
        help="Start index for frames (inclusive)",
    )
    parser.add_argument(
        "--frames_end_idx",
        type=int,
        default=None,
        help="End index for frames (exclusive)",
    )
    parser.add_argument(
        "--frame_stride",
        type=int,
        default=1,
        help="Stride for sampling frames and masks (1 keeps every frame)",
    )
    parser.add_argument(
        "--scene_mix_cutoff",
        type=int,
        default=10,
        help="Cutoff for mixing static and dynamic scenes",
    )
    parser.add_argument(
        "--dynamic_mix_cutoff",
        type=int,
        default=50,
        help="Cutoff for mixing static and dynamic scenes",
    )
    parser.add_argument(
        "--dynamic_max_memory_frames",
        type=int,
        default=8,
        help="Maximum number of frames to keep in memory for dynamic objects",
    )
    args, _ = parser.parse_known_args()

    seed_all(args.seed)

    device = torch.device(args.device)
    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]

    scene_num_parts = max(1, int(args.scene_num_parts))
    dynamic_num_parts = max(1, int(args.dynamic_num_parts))

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    export_dir = output_root / args.tag
    export_dir.mkdir(parents=True, exist_ok=True)

    frames_dir = Path(args.frames_dir)
    masks_dir = Path(args.masks_dir)

    frames: List[Image.Image] = _gather_images(frames_dir)
    masks: List[List[Image.Image]] = _gather_all_masks(masks_dir, dynamic_num_parts)

    print(f"Loaded {len(frames)} frames from {frames_dir}")
    print(f"Loaded {len(masks)} sets of masks from {masks_dir} each with {len(masks[0])} masks.")

    frame_slice = slice(args.frames_start_idx, args.frames_end_idx, args.frame_stride)
    start_idx, stop_idx, stride_step = frame_slice.indices(len(frames))
    selected_indices = list(range(start_idx, stop_idx, stride_step))

    frames = [frames[i] for i in selected_indices]
    masks = ([[mask_list[i] for i in selected_indices] for mask_list in masks] if masks else None)

    pipe = build_pipeline(
        COM4DDiTModel, 
        COM4DInferencePipeline, 
        Path(args.base_weights_dir), 
        Path(args.transformer_dir),
        device, 
        dtype, 
    )
    pipe.set_progress_bar_config(disable=False)
    
    output = pipe(
        frames=frames,
        masks=masks,
        num_tokens=args.num_tokens,
        scene_num_parts=scene_num_parts,
        dynamic_num_parts=dynamic_num_parts,
        scene_block_size=args.scene_block_size,
        dynamic_block_size=args.dynamic_block_size,
        dynamic_max_memory_frames=args.dynamic_max_memory_frames,
        scene_mix_cutoff=args.scene_mix_cutoff,
        dynamic_mix_cutoff=args.dynamic_mix_cutoff,
        scene_inference_steps=args.scene_steps,
        dynamic_inference_steps=args.dynamic_steps,
        guidance_scale_scene=args.scene_guidance,
        guidance_scale_dynamic=args.dynamic_guidance,
    )

    scene_meshes = output.scene_meshes
    dynamic_meshes_per_frame = output.dynamic_meshes_per_frame

    merged_scene_mesh = get_colored_mesh_composition(scene_meshes)
    merged_scene_mesh.export(export_dir / "scene.glb")

    # save dynamic meshes per frame
    for frame_idx, dynamic_meshes in enumerate(dynamic_meshes_per_frame):
        merged_dynamic_mesh = get_colored_mesh_composition(dynamic_meshes, is_sorted=True)
        merged_dynamic_mesh.export(export_dir / f"dynamic_frame_{frame_idx:04d}.glb")

    if args.animation:
        render_views_around_mesh_and_export(
            merged_scene_mesh,
            str(export_dir / "scene.gif"),
            fps=args.animation_fps,
            render_kwargs={"image_size": (args.render_size, args.render_size)},
        )

        if dynamic_meshes_per_frame:
            render_dynamic_animation(
                merged_scene_mesh,
                dynamic_meshes_per_frame,
                str(export_dir / "dynamic.gif"),
                fps=args.animation_fps,
                insert_rotation_every=args.animation_insert_rotation_every,
                render_kwargs={"image_size": (args.render_size, args.render_size)},
                input_frames=frames,
                vis_path=str(export_dir / "vis.gif"),
            )
        else:
            print("Warning: dynamic meshes are empty; skipping dynamic animation.")

    try:
        with open(export_dir / "args.json", "w") as f:
            json.dump(vars(args), f, indent=2, sort_keys=True)
        with open(export_dir / "cmd.txt", "w") as f:
            f.write(" ".join(sys.argv) + "\n")
    except Exception as exc:
        print(f"Warning: failed to save run metadata: {exc}")

if __name__ == "__main__":
    main()
