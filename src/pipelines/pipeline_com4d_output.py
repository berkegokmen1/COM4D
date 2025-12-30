from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import torch
import trimesh
from diffusers.utils import BaseOutput
from PIL import Image


@dataclass
class COM4DPipelineOutput(BaseOutput):
    scene_meshes: List[trimesh.Trimesh]
    dynamic_meshes_per_frame: List[List[trimesh.Trimesh]]
    