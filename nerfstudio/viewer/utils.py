# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, List, Literal, Optional, Tuple, Type, Union

import numpy as np
import torch
from jaxtyping import Float
from torch import nn

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.models.base_model import Model


@dataclass
class CameraState:
    """A dataclass for storing the camera state."""

    fov: float
    """The field of view of the camera."""
    aspect: float
    """The aspect ratio of the image. """
    c2w: Float[torch.Tensor, "3 4"]
    """The camera matrix."""
    camera_type: Literal[CameraType.PERSPECTIVE, CameraType.EQUIRECTANGULAR, CameraType.FISHEYE]
    """Type of camera to render."""
    time: float = 0.0
    """The rendering time of the camera state."""
    idx: int = 0
    """The index of the current camera."""
    fx: Optional[float] = None
    """Optional focal length x. If None, computed from FOV."""
    fy: Optional[float] = None
    """Optional focal length y. If None, computed from FOV."""
    cx: Optional[float] = None
    """Optional principal point x. If None, computed from image center."""
    cy: Optional[float] = None
    """Optional principal point y. If None, computed from image center."""
    # Radial distortion coefficients (k1-k6 for perspective, k1-k4 for fisheye)
    k1: float = 0.0
    k2: float = 0.0
    k3: float = 0.0
    k4: float = 0.0
    k5: float = 0.0
    k6: float = 0.0
    # Tangential distortion coefficients (p1, p2)
    p1: float = 0.0
    p2: float = 0.0
    # Thin prism distortion coefficients (s1-s4)
    s1: float = 0.0
    s2: float = 0.0
    s3: float = 0.0
    s4: float = 0.0


def get_camera(
    camera_state: CameraState, image_height: int, image_width: Optional[Union[int, float]] = None
) -> Cameras:
    """Returns the camera intrinsics matrix and the camera to world homogeneous matrix.

    Args:
        camera_state: the camera state
        image_height: the height of the image
        image_width: the width of the image (if None, computed from aspect ratio)
    """
    # intrinsics
    fov = camera_state.fov
    aspect = camera_state.aspect
    if image_width is None:
        image_width = aspect * image_height
    pp_w = image_width / 2.0
    pp_h = image_height / 2.0
    
    # Use custom intrinsics if provided, otherwise compute from FOV
    if camera_state.fx is not None:
        fx = float(camera_state.fx)
    else:
        focal_length = pp_h / np.tan(fov / 2.0)
        fx = float(focal_length)
    
    if camera_state.fy is not None:
        fy = float(camera_state.fy)
    else:
        if camera_state.fx is not None:
            # If fx is provided but fy is not, use fx as fy (square pixels)
            fy = float(camera_state.fx)
        else:
            focal_length = pp_h / np.tan(fov / 2.0)
            fy = float(focal_length)
    
    if camera_state.cx is not None:
        cx = float(camera_state.cx)
    else:
        cx = pp_w
    
    if camera_state.cy is not None:
        cy = float(camera_state.cy)
    else:
        cy = pp_h

    if camera_state.camera_type is CameraType.EQUIRECTANGULAR:
        fx = float(image_width / 2)
        fy = float(image_height)

    # Handle distortion parameters
    # Construct distortion_params tensor based on camera type
    # Cameras class expects [*num_cameras 6] shape, so we pad/truncate as needed
    distortion_params = None
    metadata = {"cam_idx": camera_state.idx}
    
    if camera_state.camera_type == CameraType.FISHEYE:
        # Fisheye: 4 radial coefficients [k1, k2, k3, k4], pad to 6
        distortion_params = torch.tensor(
            [camera_state.k1, camera_state.k2, camera_state.k3, camera_state.k4, 0.0, 0.0],
            dtype=torch.float32,
        )[None, ...]
        # Store full distortion parameters in metadata for rasterization
        # Fisheye uses 4 radial coefficients
        metadata["distortion_radial"] = torch.tensor(
            [camera_state.k1, camera_state.k2, camera_state.k3, camera_state.k4],
            dtype=torch.float32,
        )[None, ...]
    elif camera_state.camera_type == CameraType.PERSPECTIVE:
        # Perspective: use first 6 for compatibility [k1, k2, k3, k4, p1, p2]
        distortion_params = torch.tensor(
            [camera_state.k1, camera_state.k2, camera_state.k3, camera_state.k4, camera_state.p1, camera_state.p2],
            dtype=torch.float32,
        )[None, ...]
        # Store full distortion parameters in metadata for rasterization
        # Perspective: 6 radial (k1-k6), 2 tangential (p1, p2), 4 thin prism (s1-s4)
        metadata["distortion_radial"] = torch.tensor(
            [camera_state.k1, camera_state.k2, camera_state.k3, camera_state.k4, camera_state.k5, camera_state.k6],
            dtype=torch.float32,
        )[None, ...]
        metadata["distortion_tangential"] = torch.tensor(
            [camera_state.p1, camera_state.p2],
            dtype=torch.float32,
        )[None, ...]
        metadata["distortion_thin_prism"] = torch.tensor(
            [camera_state.s1, camera_state.s2, camera_state.s3, camera_state.s4],
            dtype=torch.float32,
        )[None, ...]
    # For EQUIRECTANGULAR, distortion_params remains None
    
    camera = Cameras(
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        camera_type=camera_state.camera_type,
        camera_to_worlds=camera_state.c2w.to(torch.float32)[None, ...],
        times=torch.tensor([camera_state.time], dtype=torch.float32),
        distortion_params=distortion_params,
        metadata=metadata,
    )
    return camera


def update_render_aabb(
    crop_viewport: bool, crop_min: Tuple[float, float, float], crop_max: Tuple[float, float, float], model: Model
):
    """
    update the render aabb box for the viewer:

    Args:
        crop_viewport: whether to crop the viewport
        crop_min: min of the crop box
        crop_max: max of the crop box
        model: the model to render
    """

    if crop_viewport:
        crop_min_tensor = torch.tensor(crop_min, dtype=torch.float32)
        crop_max_tensor = torch.tensor(crop_max, dtype=torch.float32)

        if isinstance(model.render_aabb, SceneBox):
            model.render_aabb.aabb[0] = crop_min_tensor
            model.render_aabb.aabb[1] = crop_max_tensor
        else:
            model.render_aabb = SceneBox(aabb=torch.stack([crop_min_tensor, crop_max_tensor], dim=0))
    else:
        model.render_aabb = None


def parse_object(
    obj: Any,
    type_check: Type[Any],
    tree_stub: str,
) -> List[Tuple[str, Any]]:
    """
    obj: the object to parse
    type_check: recursively adds instances of this type to the output
    tree_stub: the path down the object tree to this object

    Returns:
        a list of (path/to/object, obj), which represents the path down the object tree
        along with the object itself
    """

    def add(ret: List[Tuple[str, Any]], ts: str, v: Any):
        """
        helper that adds to ret, and if v exists already keeps the tree stub with
        the shortest path
        """
        for i, (t, o) in enumerate(ret):
            if o == v:
                if len(t.split("/")) > len(ts.split("/")):
                    ret[i] = (ts, v)
                return
        ret.append((ts, v))

    if not hasattr(obj, "__dict__"):
        return []
    ret = []
    # get a list of the properties of the object, sorted by whether things are instances of type_check
    # we skip cached properties, which can be expensive to call `getattr()` on!
    obj_props = [(k, getattr(obj, k)) for k in dir(obj) if not isinstance(getattr(type(obj), k, None), cached_property)]
    for k, v in obj_props:
        if k[0] == "_":
            continue
        new_tree_stub = f"{tree_stub}/{k}"
        if isinstance(v, type_check):
            add(ret, new_tree_stub, v)
        elif isinstance(v, nn.Module):
            if v is obj:
                # some nn.Modules might contain infinite references, e.g. consider foo = nn.Module(), foo.bar = foo
                # to stop infinite recursion, we skip such attributes
                continue
            lower_rets = parse_object(v, type_check, new_tree_stub)
            # check that the values aren't already in the tree
            for ts, o in lower_rets:
                add(ret, ts, o)
    return ret
