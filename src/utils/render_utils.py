from src.utils.typing_utils import *
from src.utils.data_utils import RGB

import os
import numpy as np
from PIL import Image
import trimesh
from trimesh.transformations import rotation_matrix
import pyrender
from diffusers.utils import export_to_video
from diffusers.utils.loading_utils import load_video
import torch
from torchvision.utils import make_grid

os.environ['PYOPENGL_PLATFORM'] = 'egl'

def render(
    scene: pyrender.Scene,
    renderer: pyrender.Renderer,
    camera: pyrender.Camera,
    pose: np.ndarray,
    light: Optional[pyrender.Light] = None,
    normalize_depth: bool = False,
    flags: int = pyrender.constants.RenderFlags.NONE,
    return_type: Literal['pil', 'ndarray'] = 'pil'
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[Image.Image, Image.Image]]:
    camera_node = scene.add(camera, pose=pose)
    if light is not None:
        light_node = scene.add(light, pose=pose)
    image, depth = renderer.render(
        scene, 
        flags=flags
    )
    scene.remove_node(camera_node)
    if light is not None:
        scene.remove_node(light_node)
    if normalize_depth or return_type == 'pil':
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    if return_type == 'pil':
        image = Image.fromarray(image)
        depth = Image.fromarray(depth.astype(np.uint8))
    return image, depth

def rotation_matrix_from_vectors(vec1, vec2):
    a, b = vec1 / np.linalg.norm(vec1), vec2 / np.linalg.norm(vec2)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    if s == 0:
        return np.eye(3) if c > 0 else -np.eye(3)
    kmat = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])
    return np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s ** 2))

def create_circular_camera_positions(
    num_views: int,
    radius: float,
    axis: np.ndarray = np.array([0.0, 1.0, 0.0]),
    height: float = 0.0,
    center: Optional[np.ndarray] = None,
) -> List[np.ndarray]:
    # Create a list of positions for a circular camera trajectory
    # around the given axis with the given radius.
    positions = []
    axis = axis / np.linalg.norm(axis)
    if center is None:
        center = np.zeros(3)
    center = np.asarray(center, dtype=float)
    for i in range(num_views):
        theta = 2 * np.pi * i / num_views
        position = np.array([
            np.sin(theta) * radius,
            0.0,
            np.cos(theta) * radius
        ])
        if not np.allclose(axis, np.array([0.0, 1.0, 0.0])):
            R = rotation_matrix_from_vectors(np.array([0.0, 1.0, 0.0]), axis)
            position = R @ position
        position = position + axis * height + center
        positions.append(position)
    return positions

def create_circular_camera_poses(
    num_views: int,
    radius: float,
    axis: np.ndarray = np.array([0.0, 1.0, 0.0]),
    height: float = 0.0,
    target: Optional[np.ndarray] = None,
) -> List[np.ndarray]:
    # Create a list of poses for a circular camera trajectory
    # around the given axis with the given radius.
    # The camera always looks at the provided target (defaults to origin).
    axis = axis / np.linalg.norm(axis)
    if target is None:
        target = np.zeros(3)
    target = np.asarray(target, dtype=float)
    positions = create_circular_camera_positions(
        num_views=num_views,
        radius=radius,
        axis=axis,
        height=height,
        center=target,
    )
    poses: List[np.ndarray] = []
    for position in positions:
        forward = target - position
        forward_norm = np.linalg.norm(forward)
        if forward_norm < 1e-8:
            raise ValueError("Camera position coincides with the target point.")
        forward /= forward_norm

        up_hint = axis
        up_hint = up_hint / np.linalg.norm(up_hint)
        right = np.cross(forward, up_hint)
        if np.linalg.norm(right) < 1e-8:
            # Fallback to canonical up vectors if the camera is aligned with the axis
            for fallback_up in (
                np.array([0.0, 0.0, 1.0]),
                np.array([0.0, 1.0, 0.0]),
                np.array([1.0, 0.0, 0.0]),
            ):
                fallback_up = fallback_up / np.linalg.norm(fallback_up)
                right = np.cross(forward, fallback_up)
                if np.linalg.norm(right) >= 1e-8:
                    up_hint = fallback_up
                    break
        right_norm = np.linalg.norm(right)
        if right_norm < 1e-8:
            raise ValueError("Unable to compute a stable camera orientation basis.")
        right /= right_norm
        true_up = np.cross(right, forward)
        true_up /= np.linalg.norm(true_up)

        pose = np.eye(4)
        pose[:3, :3] = np.stack([right, true_up, -forward], axis=1)
        pose[:3, 3] = position
        poses.append(pose)
    return poses

def render_views_around_mesh(
    mesh: Union[trimesh.Trimesh, trimesh.Scene],
    num_views: int = 36,
    radius: float = 3.5,
    axis: np.ndarray = np.array([0.0, 1.0, 0.0]),
    camera_height: float = 0.0,
    image_size: tuple = (512, 512),
    fov: float = 40.0,
    light_intensity: Optional[float] = 5.0,
    znear: float = 0.1,
    zfar: float = 10.0, 
    normalize_depth: bool = False,
    flags: int = pyrender.constants.RenderFlags.NONE,
    return_depth: bool = False, 
    return_type: Literal['pil', 'ndarray'] = 'pil',
    bg_color: Optional[Tuple[float, float, float, float]] = None
) -> Union[
        List[Image.Image], 
        List[np.ndarray], 
        Tuple[List[Image.Image], List[Image.Image]], 
        Tuple[List[np.ndarray], List[np.ndarray]]
    ]:
    
    if not isinstance(mesh, (trimesh.Trimesh, trimesh.Scene)):
        raise ValueError("mesh must be a trimesh.Trimesh or trimesh.Scene object")
    if isinstance(mesh, trimesh.Trimesh):
        mesh = trimesh.Scene(mesh)

    bounds = getattr(mesh, "bounds", None)
    if bounds is not None:
        center = bounds.mean(axis=0)
    else:
        center = np.zeros(3)
    center = np.asarray(center, dtype=float)

    scene = pyrender.Scene.from_trimesh_scene(mesh)
    # Optionally set background color (default: black)
    if bg_color is not None:
        scene.bg_color = np.array(bg_color)
    light = pyrender.DirectionalLight(
        color=np.ones(3), 
        intensity=light_intensity
    ) if light_intensity is not None else None
    camera = pyrender.PerspectiveCamera(
        yfov=np.deg2rad(fov),
        aspectRatio=image_size[0]/image_size[1],
        znear=znear,
        zfar=zfar
    )
    renderer = pyrender.OffscreenRenderer(*image_size)

    camera_poses = create_circular_camera_poses(
        num_views, 
        radius, 
        axis=axis,
        height=camera_height,
        target=center,
    )

    images, depths = [], []
    for pose in camera_poses:
        image, depth = render(
            scene, renderer, camera, pose, light, 
            normalize_depth=normalize_depth,
            flags=flags,
            return_type=return_type
        )
        images.append(image)
        depths.append(depth)

    renderer.delete()

    if return_depth:
        return images, depths
    return images

def render_normal_views_around_mesh(
    mesh: Union[trimesh.Trimesh, trimesh.Scene],
    num_views: int = 36,
    radius: float = 3.5,
    axis: np.ndarray = np.array([0.0, 1.0, 0.0]),
    image_size: tuple = (512, 512),
    fov: float = 40.0,
    light_intensity: Optional[float] = 5.0,
    znear: float = 0.1,
    zfar: float = 10.0,
    normalize_depth: bool = False,
    flags: int = pyrender.constants.RenderFlags.NONE,
    return_depth: bool = False, 
    return_type: Literal['pil', 'ndarray'] = 'pil',
    bg_color: Optional[Tuple[float, float, float, float]] = (0.0, 0.0, 0.0, 1.0)
) -> Union[
        List[Image.Image], 
        List[np.ndarray], 
        Tuple[List[Image.Image], List[Image.Image]], 
        Tuple[List[np.ndarray], List[np.ndarray]]
    ]:
    
    if not isinstance(mesh, (trimesh.Trimesh, trimesh.Scene)):
        raise ValueError("mesh must be a trimesh.Trimesh or trimesh.Scene object")
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.to_geometry()
    normals = mesh.vertex_normals
    colors = ((normals + 1.0) / 2.0 * 255).astype(np.uint8)
    mesh.visual = trimesh.visual.ColorVisuals(
        mesh=mesh,
        vertex_colors=colors
    )
    mesh = trimesh.Scene(mesh)
    return render_views_around_mesh(
        mesh=mesh,
        num_views=num_views,
        radius=radius,
        axis=axis,
        camera_height=0.0,
        image_size=image_size,
        fov=fov,
        light_intensity=light_intensity,
        znear=znear,
        zfar=zfar,
        normalize_depth=normalize_depth,
        flags=flags,
        return_depth=return_depth,
        return_type=return_type,
        bg_color=bg_color,
    )

def create_camera_pose_on_sphere(
    azimuth: float = 0.0,  # in degrees
    elevation: float = 0.0,  # in degrees
    radius: float = 3.5,
) -> np.ndarray:
    """Return a camera pose located on a sphere that keeps world-up stable."""
    azimuth_rad = np.deg2rad(azimuth)
    elevation_rad = np.deg2rad(elevation)

    direction = np.array([
        np.cos(elevation_rad) * np.sin(azimuth_rad),
        np.sin(elevation_rad),
        np.cos(elevation_rad) * np.cos(azimuth_rad),
    ])
    direction_norm = np.linalg.norm(direction)
    if direction_norm < 1e-8:
        raise ValueError("Camera direction is ill-defined.")
    backward = direction / direction_norm  # Points from target to camera.

    world_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(world_up, backward)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-6:
        # Camera direction is too close to world_up; pick an alternate up hint.
        for up_hint in (
            np.array([0.0, 0.0, 1.0]),
            np.array([1.0, 0.0, 0.0]),
        ):
            right = np.cross(up_hint, backward)
            right_norm = np.linalg.norm(right)
            if right_norm >= 1e-6:
                break
        else:
            raise ValueError("Failed to construct a stable camera basis.")
    right /= right_norm
    up = np.cross(backward, right)
    up /= np.linalg.norm(up)

    pose = np.eye(4)
    pose[:3, 0] = right
    pose[:3, 1] = up
    pose[:3, 2] = backward
    pose[:3, 3] = backward * radius
    return pose

def render_single_view(
    mesh: Union[trimesh.Trimesh, trimesh.Scene],
    azimuth: float = 0.0, # in degrees
    elevation: float = 0.0, # in degrees
    radius: float = 3.5,
    image_size: tuple = (512, 512),
    fov: float = 40.0,
    light_intensity: Optional[float] = 5.0,
    num_env_lights: int = 0,
    env_light_intensity: Optional[float] = None,
    znear: float = 0.1,
    zfar: float = 10.0,
    normalize_depth: bool = False,
    flags: int = pyrender.constants.RenderFlags.NONE,
    return_depth: bool = False, 
    return_type: Literal['pil', 'ndarray'] = 'pil',
    bg_color: Optional[Tuple[float, float, float, float]] = None,
    target: Optional[np.ndarray] = None,
) -> Union[
        Image.Image, 
        np.ndarray, 
        Tuple[Image.Image, Image.Image], 
        Tuple[np.ndarray, np.ndarray]
    ]:
    
    if not isinstance(mesh, (trimesh.Trimesh, trimesh.Scene)):
        raise ValueError("mesh must be a trimesh.Trimesh or trimesh.Scene object")
    if isinstance(mesh, trimesh.Trimesh):
        mesh = trimesh.Scene(mesh)

    scene = pyrender.Scene.from_trimesh_scene(mesh)
    # Optionally set background color (default: black)
    if bg_color is not None:
        scene.bg_color = np.array(bg_color)
    light = pyrender.DirectionalLight(
        color=np.ones(3), 
        intensity=light_intensity
    ) if light_intensity is not None else None
    camera = pyrender.PerspectiveCamera(
        yfov=np.deg2rad(fov),
        aspectRatio=image_size[0]/image_size[1],
        znear=znear,
        zfar=zfar
    )
    renderer = pyrender.OffscreenRenderer(*image_size)

    target_vec: Optional[np.ndarray]
    if target is not None:
        target_arr = np.asarray(target, dtype=np.float64).reshape(-1)
        if target_arr.size != 3:
            raise ValueError("target must be a 3D vector")
        target_vec = target_arr.astype(np.float64)
    else:
        target_vec = None

    camera_pose = create_camera_pose_on_sphere(
        azimuth,
        elevation,
        radius
    ).copy()
    if target_vec is not None:
        camera_pose[:3, 3] += target_vec

    if num_env_lights > 0:
        env_intensity = env_light_intensity if env_light_intensity is not None else light_intensity
        if env_intensity is not None and env_intensity > 0.0:
            env_light_poses = create_circular_camera_poses(
                num_env_lights,
                radius,
                axis = np.array([0.0, 1.0, 0.0])
            )
            for pose in env_light_poses:
                pose_to_add = pose.copy()
                if target_vec is not None:
                    pose_to_add[:3, 3] += target_vec
                scene.add(pyrender.DirectionalLight(
                    color=np.ones(3),
                    intensity=env_intensity
                ), pose=pose_to_add)
            # set light to None
            light = None
        else:
            # No usable env light intensity -> skip env lights entirely
            num_env_lights = 0

    image, depth = render(
        scene, renderer, camera, camera_pose, light,
        normalize_depth=normalize_depth,
        flags=flags,
        return_type=return_type
    )
    renderer.delete()

    if return_depth:
        return image, depth
    return image

def render_normal_single_view(
    mesh: Union[trimesh.Trimesh, trimesh.Scene],
    azimuth: float = 0.0, # in degrees
    elevation: float = 0.0, # in degrees
    radius: float = 3.5,
    image_size: tuple = (512, 512),
    fov: float = 40.0,
    light_intensity: Optional[float] = 5.0,
    znear: float = 0.1,
    zfar: float = 10.0,
    normalize_depth: bool = False,
    flags: int = pyrender.constants.RenderFlags.NONE,
    return_depth: bool = False,
    return_type: Literal['pil', 'ndarray'] = 'pil',
    bg_color: Optional[Tuple[float, float, float, float]] = (0.0, 0.0, 0.0, 1.0)
) -> Union[
        Image.Image,
        np.ndarray,
        Tuple[Image.Image, Image.Image],
        Tuple[np.ndarray, np.ndarray]
    ]:

    if not isinstance(mesh, (trimesh.Trimesh, trimesh.Scene)):
        raise ValueError("mesh must be a trimesh.Trimesh or trimesh.Scene object")
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.to_geometry()
    normals = mesh.vertex_normals
    colors = ((normals + 1.0) / 2.0 * 255).astype(np.uint8)
    mesh.visual = trimesh.visual.ColorVisuals(
        mesh=mesh,
        vertex_colors=colors
    )
    mesh = trimesh.Scene(mesh)
    return render_single_view(
        mesh, azimuth, elevation, radius, 
        image_size, fov, light_intensity, znear, zfar,
        normalize_depth, flags, 
        return_depth, return_type,
        bg_color
    )

def export_renderings(
    images: List[Image.Image],
    export_path: str,
    fps: int = 36,
    loop: int = 0,
    frame_durations: Optional[List[int]] = None,
):
    export_type = export_path.split('.')[-1]
    if export_type == 'mp4':
        export_to_video(
            images,
            export_path,
            fps=fps,
        )
    elif export_type == 'gif':
        if frame_durations is not None:
            # Use per-frame durations if provided (milliseconds per frame)
            durations = [int(max(1, d)) for d in frame_durations]
            if len(durations) != len(images):
                raise ValueError(
                    f"frame_durations length ({len(durations)}) must match images length ({len(images)})"
                )
            images[0].save(
                export_path,
                save_all=True,
                append_images=images[1:],
                duration=durations,
                loop=loop,
            )
        else:
            duration = int(max(1, round(1000 / fps)))
            images[0].save(
                export_path,
                save_all=True,
                append_images=images[1:],
                duration=duration,
                loop=loop,
            )
    else:
        raise ValueError(f'Unknown export type: {export_type}')
    

# ==========================
# Fixed-camera rendering util
# ==========================
def _mesh_bounds(mesh: Union[trimesh.Trimesh, trimesh.Scene]) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(mesh, trimesh.Scene):
        bounds = mesh.bounds
    else:
        bounds = mesh.bounds
    return bounds[0].astype(float), bounds[1].astype(float)

def compute_global_center_and_radius(meshes: List[Union[trimesh.Trimesh, trimesh.Scene]]) -> Tuple[np.ndarray, float]:
    """Compute the union AABB center and a radius (half-diagonal) across meshes.
    Returns (center, radius).
    """
    if len(meshes) == 0:
        raise ValueError("compute_global_center_and_radius: empty meshes list")
    mins = []
    maxs = []
    for m in meshes:
        mn, mx = _mesh_bounds(m)
        mins.append(mn)
        maxs.append(mx)
    mn = np.minimum.reduce(mins)
    mx = np.maximum.reduce(maxs)
    center = (mn + mx) / 2.0
    radius = 0.5 * np.linalg.norm(mx - mn)
    # Avoid zero radius
    radius = float(max(radius, 1e-4))
    return center, radius

def render_sequence_fixed_camera(
    meshes: List[Union[trimesh.Trimesh, trimesh.Scene]],
    azimuth: float = 0.0,
    elevation: float = 0.0,
    distance: Optional[float] = None,
    fit_scale: float = 2.0,
    image_size: tuple = (512, 512),
    fov: float = 55.0,
    light_intensity: Optional[float] = 5.0,
    znear: float = 0.1,
    zfar: float = 100.0,
    flags: int = pyrender.constants.RenderFlags.NONE,
    bg_color: Optional[Tuple[float, float, float, float]] = None,
    return_type: Literal['pil', 'ndarray'] = 'pil',
) -> List[Union[Image.Image, np.ndarray]]:
    """Render a list of meshes from a single, fixed camera across frames.
    - Computes a global center/radius across all meshes
    - Uses the same camera pose for every frame
    - Translates each mesh so the global center is at the origin, preserving relative motion
    """
    if len(meshes) == 0:
        return []

    center, rad = compute_global_center_and_radius(meshes)
    cam_radius = float(distance) if (distance is not None and distance > 0) else float(fit_scale * rad)
    cam_radius = max(cam_radius, 1e-4)

    # Build renderer, camera, and optional light once
    camera = pyrender.PerspectiveCamera(
        yfov=np.deg2rad(fov),
        aspectRatio=image_size[0] / image_size[1],
        znear=znear,
        zfar=zfar,
    )
    renderer = pyrender.OffscreenRenderer(*image_size)

    # Prepare fixed camera pose looking at origin, placed on sphere at distance
    cam_pose = create_camera_pose_on_sphere(azimuth=azimuth, elevation=elevation, radius=cam_radius)
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=light_intensity) if light_intensity is not None else None

    out_frames: List[Union[Image.Image, np.ndarray]] = []
    for m in meshes:
        # Translate mesh so that global center appears at origin (fixed camera sees consistent world)
        if isinstance(m, trimesh.Scene):
            scene_trimesh = m.copy()
            scene_trimesh.apply_translation(-center)
        else:
            geom = m.copy()
            geom.apply_translation(-center)
            scene_trimesh = trimesh.Scene(geom)

        scene = pyrender.Scene.from_trimesh_scene(scene_trimesh)
        if bg_color is not None:
            scene.bg_color = np.array(bg_color)

        img, _ = render(scene, renderer, camera, cam_pose, light, normalize_depth=False, flags=flags, return_type=return_type)
        out_frames.append(img)

    renderer.delete()
    return out_frames

def make_grid_for_images_or_videos(
    images_or_videos: Union[List[Image.Image], List[List[Image.Image]]],
    nrow: int = 4, 
    padding: int = 0, 
    pad_value: int = 0, 
    image_size: tuple = (512, 512),
    return_type: Literal['pil', 'ndarray'] = 'pil'
) -> Union[Image.Image, List[Image.Image], np.ndarray]:
    if isinstance(images_or_videos[0], Image.Image):
        images = [np.array(image.resize(image_size).convert('RGB')) for image in images_or_videos]
        images = np.stack(images, axis=0).transpose(0, 3, 1, 2) # [N, C, H, W]
        images = torch.from_numpy(images)
        image_grid = make_grid(
            images,
            nrow=nrow,
            padding=padding,
            pad_value=pad_value,
            normalize=False
        ) # [C, H', W']
        image_grid = image_grid.cpu().numpy()
        if return_type == 'pil':
            image_grid = Image.fromarray(image_grid.transpose(1, 2, 0))
        return image_grid
    elif isinstance(images_or_videos[0], list) and isinstance(images_or_videos[0][0], Image.Image):
        image_grids = []
        for i in range(len(images_or_videos[0])):
            images = [video[i] for video in images_or_videos]
            image_grid = make_grid_for_images_or_videos(
                images,
                nrow=nrow,
                padding=padding,
                return_type=return_type
            )
            image_grids.append(image_grid)
        if return_type == 'ndarray':
            image_grids = np.stack(image_grids, axis=0)
        return image_grids
    else:
        raise ValueError(f'Unknown input type: {type(images_or_videos[0])}')

def render_views_around_mesh_and_export(
    mesh: trimesh.Scene,
    path: str,
    fps: int = 12,
    render_kwargs: Optional[Dict[str, Any]] = None,
) -> List[Image.Image]:
    frames = render_views_around_mesh(
        mesh,
        num_views=render_kwargs.get('num_views', 36),
        radius=render_kwargs.get('radius', 3.5),
        image_size=render_kwargs.get('image_size', (512, 512)),
    )

    export_renderings(frames, path, fps=fps)

def render_dynamic_animation(
    scene_mesh: Optional[Union[trimesh.Trimesh, trimesh.Scene]],
    dynamic_meshes_per_frame: List[Union[trimesh.Trimesh, trimesh.Scene, List[trimesh.Trimesh]]],
    animation_path: Optional[str],
    fps: int = 12,
    insert_rotation_every: int = 0,
    render_kwargs: Optional[Dict[str, Any]] = None,
    input_frames: Optional[List[Image.Image]] = None,
    vis_path: Optional[str] = None,
) -> Optional[str]:
    if animation_path is None:
        return None
    if not dynamic_meshes_per_frame:
        return None

    render_kwargs = render_kwargs or {}
    target_size = render_kwargs.get("image_size", (512, 512))
    if isinstance(target_size, (int, float)):
        target_size = (int(target_size), int(target_size))
    else:
        target_size = tuple(int(v) for v in target_size)

    def _to_pil_image(image: Image.Image) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, torch.Tensor):
            tensor = image.detach().cpu()
            if tensor.ndim == 4:
                tensor = tensor[0]
            if tensor.ndim == 3:
                if tensor.shape[0] == 1:
                    tensor = tensor.repeat(3, 1, 1)
                elif tensor.shape[0] != 3:
                    raise ValueError(f"Unsupported tensor shape for image conversion: {tensor.shape}")
                array = (
                    tensor.float()
                    .clamp(0, 1)
                    .mul(255)
                    .round()
                    .to(torch.uint8)
                    .permute(1, 2, 0)
                    .numpy()
                )
                return Image.fromarray(array)
            if tensor.ndim == 2:
                array = (
                    tensor.float()
                    .clamp(0, 1)
                    .mul(255)
                    .round()
                    .to(torch.uint8)
                    .numpy()
                )
                array = np.repeat(array[..., None], 3, axis=2)
                return Image.fromarray(array)
            raise ValueError(f"Unsupported tensor ndim for image conversion: {tensor.ndim}")
        if isinstance(image, np.ndarray):
            array = image
            if array.ndim == 2:
                array = np.repeat(array[..., None], 3, axis=2)
            elif array.ndim == 3:
                if array.shape[2] == 1:
                    array = np.repeat(array, 3, axis=2)
                elif array.shape[2] not in (3, 4):
                    raise ValueError(f"Unsupported ndarray channel count: {array.shape[2]}")
            else:
                raise ValueError(f"Unsupported ndarray ndim for image conversion: {array.ndim}")
            if array.dtype != np.uint8:
                array = np.clip(array, 0.0, 1.0)
                array = (array * 255.0).round().astype(np.uint8)
            if array.shape[-1] == 4:
                array = array[..., :3]
            return Image.fromarray(array)
        raise TypeError(f"Unsupported image type for conversion: {type(image)}")

    def _compose_side_by_side(left: Image.Image, right: Image.Image) -> Image.Image:
        if left.size != target_size:
            left = left.resize(target_size, Image.LANCZOS)
        if right.size != target_size:
            right = right.resize(target_size, Image.LANCZOS)
        canvas = Image.new("RGB", (target_size[0] * 2, target_size[1]))
        canvas.paste(left, (0, 0))
        canvas.paste(right, (target_size[0], 0))
        return canvas

    def _generate_palette(num_colors: int) -> List[np.ndarray]:
        if num_colors <= 0:
            return []
        palette: List[np.ndarray] = []
        for idx in range(num_colors):
            color_rgb = RGB[idx % len(RGB)]
            palette.append(np.array(color_rgb, dtype=np.uint8))
        return palette

    def _recolor_scene(
        scene: trimesh.Scene,
        palette: List[np.ndarray],
        start_index: int = 0,
        key_order: Optional[List[str]] = None,
    ) -> trimesh.Scene:
        recolored = trimesh.Scene()
        if key_order is not None and len(key_order) < len(scene.geometry):
            keys = sorted(scene.geometry.keys())
        else:
            keys = key_order or sorted(scene.geometry.keys())
        for idx, key in enumerate(keys):
            geom = scene.geometry.get(key)
            if geom is None:
                continue
            color_idx = start_index + idx
            if color_idx >= len(palette):
                break
            color_rgb = palette[color_idx]
            geom_copy = geom.copy()
            if geom_copy.vertices.size > 0:
                rgba = np.array([color_rgb[0], color_rgb[1], color_rgb[2], 255], dtype=np.uint8)
                vertex_colors = np.tile(rgba, (geom_copy.vertices.shape[0], 1))
                face_colors = np.tile(rgba, (geom_copy.faces.shape[0], 1)) if geom_copy.faces.size > 0 else None
                geom_copy.visual = trimesh.visual.ColorVisuals(
                    mesh=geom_copy,
                    vertex_colors=vertex_colors,
                    face_colors=face_colors,
                )
            recolored.add_geometry(geom_copy)
        return recolored

    def _add_to_scene(scene: trimesh.Scene, mesh: Union[trimesh.Trimesh, trimesh.Scene, None]) -> None:
        if mesh is None:
            return
        if isinstance(mesh, trimesh.Scene):
            for geom in mesh.dump():
                scene.add_geometry(geom.copy())
        elif isinstance(mesh, trimesh.Trimesh):
            scene.add_geometry(mesh.copy())
        else:
            raise TypeError(f"Unsupported mesh type: {type(mesh)}")

    dynamic_scenes: List[trimesh.Scene] = []
    for frame_dynamic in dynamic_meshes_per_frame:
        if isinstance(frame_dynamic, (list, tuple)):
            scene_dynamic = trimesh.Scene()
            for geom in frame_dynamic:
                if geom is None:
                    continue
                scene_dynamic.add_geometry(geom.copy())
        elif isinstance(frame_dynamic, trimesh.Scene):
            scene_dynamic = frame_dynamic.copy()
        elif isinstance(frame_dynamic, trimesh.Trimesh):
            scene_dynamic = trimesh.Scene(frame_dynamic.copy())
        else:
            raise TypeError(f"Unsupported dynamic mesh type: {type(frame_dynamic)}")
        dynamic_scenes.append(scene_dynamic)

    static_scene = None
    if scene_mesh is not None:
        if isinstance(scene_mesh, trimesh.Scene):
            static_scene = scene_mesh.copy()
        elif isinstance(scene_mesh, trimesh.Trimesh):
            static_scene = trimesh.Scene(scene_mesh.copy())
        else:
            raise TypeError(f"Unsupported scene mesh type: {type(scene_mesh)}")

    static_count = len(static_scene.geometry) if static_scene is not None else 0
    dynamic_max_count = max((len(scene.geometry) for scene in dynamic_scenes), default=0)
    palette = _generate_palette(static_count + dynamic_max_count)

    static_key_order = (
        sorted(static_scene.geometry.keys()) if static_scene is not None else []
    )
    dynamic_key_order = sorted(dynamic_scenes[0].geometry.keys()) if dynamic_scenes else []

    if static_scene is not None:
        static_scene = _recolor_scene(static_scene, palette, start_index=0, key_order=static_key_order)

    composite_meshes: List[trimesh.Scene] = []
    for scene_dynamic in dynamic_scenes:
        recolored_dynamic = _recolor_scene(
            scene_dynamic,
            palette,
            start_index=static_count,
            key_order=dynamic_key_order if dynamic_key_order else None,
        )
        scene = trimesh.Scene()
        _add_to_scene(scene, static_scene)
        _add_to_scene(scene, recolored_dynamic)
        composite_meshes.append(scene)

    if not composite_meshes:
        return None

    base_frames = render_sequence_fixed_camera(
        composite_meshes,
        azimuth=render_kwargs.get("azimuth", 0.0),
        elevation=render_kwargs.get("elevation", 0.0),
        distance=render_kwargs.get("distance", None),
        fit_scale=render_kwargs.get("fit_scale", 2.0),
        image_size=target_size,
        fov=render_kwargs.get("fov", 55.0),
        light_intensity=render_kwargs.get("light_intensity", 5.0),
        znear=render_kwargs.get("znear", 0.1),
        zfar=render_kwargs.get("zfar", 100.0),
        flags=render_kwargs.get("flags", pyrender.constants.RenderFlags.NONE),
        bg_color=render_kwargs.get("bg_color", None),
        return_type="pil",
    )

    final_frames: List[Image.Image] = []
    vis_frames: List[Image.Image] = []
    input_frames = input_frames or []

    for idx, frame in enumerate(base_frames):
        final_frames.append(frame)
        if input_frames and vis_path is not None:
            left = _to_pil_image(input_frames[min(idx, len(input_frames) - 1)])
            vis_frames.append(_compose_side_by_side(left, frame))

        if insert_rotation_every and (idx + 1) % insert_rotation_every == 0:
            rotation_frames = render_views_around_mesh(
                composite_meshes[idx],
                num_views=render_kwargs.get("num_views", 36),
                radius=render_kwargs.get("radius", 3.5),
                axis=render_kwargs.get("axis", np.array([0.0, 1.0, 0.0])),
                camera_height=render_kwargs.get("camera_height", 0.0),
                image_size=target_size,
                fov=render_kwargs.get("fov", 55.0),
                light_intensity=render_kwargs.get("light_intensity", 5.0),
                znear=render_kwargs.get("znear", 0.1),
                zfar=render_kwargs.get("zfar", 10.0),
                flags=render_kwargs.get("flags", pyrender.constants.RenderFlags.NONE),
                bg_color=render_kwargs.get("bg_color", None),
                return_type="pil",
            )
            for rot_frame in rotation_frames:
                final_frames.append(rot_frame)
                if input_frames and vis_path is not None:
                    left = _to_pil_image(input_frames[min(idx, len(input_frames) - 1)])
                    vis_frames.append(_compose_side_by_side(left, rot_frame))

    export_renderings(final_frames, animation_path, fps=fps)

    if vis_path is not None and vis_frames:
        export_renderings(vis_frames, vis_path, fps=fps)

    return animation_path
