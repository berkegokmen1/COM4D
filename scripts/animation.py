"""
python scripts/animation.py \
    --output_dir assets/demo \
    --frames_dir assets/demo/frames \
    --insert_rotation_every 20
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import trimesh
from PIL import Image

from src.utils.inference import _gather_images
from src.utils.render_utils import render_dynamic_animation


def _load_scene(path: Path) -> trimesh.Scene:
    if not path.exists():
        raise FileNotFoundError(f"Missing mesh file: {path}")
    return trimesh.load(str(path), force="scene")

def _load_args_defaults(output_dir: Path) -> dict:
    args_path = output_dir / "args.json"
    if not args_path.exists():
        return {}
    try:
        with args_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {args_path}: {exc}") from exc

def _coalesce(value, fallback):
    return fallback if value is None else value


def main() -> None:
    parser = argparse.ArgumentParser(description="Render COM4D dynamic mesh animations")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory containing scene.glb and dynamic_frame_*.glb files",
    )
    parser.add_argument(
        "--frames_dir",
        type=str,
        default=None,
        help="Directory containing input frames",
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=None,
        help="Start index for frames (inclusive)",
    )
    parser.add_argument(
        "--end_idx",
        type=int,
        default=None,
        help="End index for frames (exclusive)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Stride for sampling frames",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="FPS for exported animations",
    )
    parser.add_argument(
        "--render_size",
        type=int,
        default=None,
        help="Render resolution (square)",
    )
    parser.add_argument(
        "--insert_rotation_every",
        type=int,
        default=None,
        help="Insert a 360 render every N frames (0 disables)",
    )
    parser.add_argument(
        "--num_views",
        type=int,
        default=36,
        help="Number of views for each 360 rotation",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=3.5,
        help="Camera radius for 360 rotation renders",
    )
    parser.add_argument(
        "--mesh_gif",
        type=str,
        default="animation.gif",
        help="Output filename for the mesh-only animation",
    )
    parser.add_argument(
        "--vis_gif",
        type=str,
        default="vis.gif",
        help="Output filename for the side-by-side animation",
    )
    parser.add_argument(
        "--mp4",
        action="store_true",
        help="Export MP4 instead of GIF (overrides output extensions)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    defaults = _load_args_defaults(output_dir)
    frames_dir_value = _coalesce(args.frames_dir, defaults.get("frames_dir"))
    if frames_dir_value is None:
        raise ValueError("Missing frames_dir; provide --frames_dir or set frames_dir in args.json.")
    frames_dir = Path(frames_dir_value)

    args.start_idx = _coalesce(args.start_idx, defaults.get("frames_start_idx", 0))
    args.end_idx = _coalesce(args.end_idx, defaults.get("frames_end_idx"))
    args.stride = _coalesce(args.stride, defaults.get("frame_stride", 1))
    args.fps = _coalesce(args.fps, defaults.get("animation_fps", 8))
    args.insert_rotation_every = _coalesce(
        args.insert_rotation_every,
        defaults.get("animation_insert_rotation_every", 0),
    )
    args.render_size = _coalesce(args.render_size, defaults.get("render_size", 1024))

    scene_path = output_dir / "scene.glb"
    dynamic_paths = sorted(output_dir.glob("dynamic_frame_*.glb"))
    if not dynamic_paths:
        raise FileNotFoundError(f"No dynamic_frame_*.glb found in {output_dir}")

    scene_mesh = _load_scene(scene_path)

    frames_all: List[Image.Image] = _gather_images(frames_dir)
    frame_slice = slice(args.start_idx, args.end_idx, args.stride)
    start_idx, stop_idx, stride_step = frame_slice.indices(len(frames_all))
    selected_indices = list(range(start_idx, stop_idx, stride_step))
    frames = [frames_all[i] for i in selected_indices]

    if len(dynamic_paths) != len(frames):
        max_len = min(len(dynamic_paths), len(frames))
        if max_len == 0:
            raise ValueError("No overlapping frames between meshes and input frames.")
        if len(dynamic_paths) != max_len:
            print(f"Warning: trimming dynamic meshes to {max_len} frames for alignment.")
            dynamic_paths = dynamic_paths[:max_len]
        if len(frames) != max_len:
            print(f"Warning: trimming input frames to {max_len} frames for alignment.")
            frames = frames[:max_len]

    dynamic_meshes_per_frame = [_load_scene(path) for path in dynamic_paths]

    mesh_name = args.mesh_gif
    vis_name = args.vis_gif
    if args.mp4:
        mesh_name = str(Path(mesh_name).with_suffix(".mp4"))
        vis_name = str(Path(vis_name).with_suffix(".mp4"))

    animation_path = output_dir / mesh_name
    vis_path = output_dir / vis_name

    render_dynamic_animation(
        scene_mesh,
        dynamic_meshes_per_frame,
        str(animation_path),
        fps=args.fps,
        insert_rotation_every=args.insert_rotation_every,
        render_kwargs={
            "image_size": (args.render_size, args.render_size),
            "num_views": args.num_views,
            "radius": args.radius,
        },
        input_frames=frames,
        vis_path=str(vis_path),
    )

    print(f"Wrote {animation_path}")
    print(f"Wrote {vis_path}")


if __name__ == "__main__":
    main()
