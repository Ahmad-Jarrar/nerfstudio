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

"""Manage the state of the viewer"""

from __future__ import annotations

import contextlib
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
import viser
import viser.theme
import viser.transforms as vtf
from typing_extensions import assert_never

from nerfstudio.cameras.camera_optimizers import CameraOptimizer
from nerfstudio.cameras.cameras import CameraType
from nerfstudio.configs import base_config as cfg
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.models.base_model import Model
from nerfstudio.models.splatfacto import SplatfactoModel
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils.decorators import check_main_thread, decorate_all
from nerfstudio.utils.writer import GLOBAL_BUFFER, EventName
from nerfstudio.viewer.control_panel import ControlPanel
from nerfstudio.viewer.export_panel import populate_export_tab
from nerfstudio.viewer.render_panel import populate_render_tab
from nerfstudio.viewer.render_state_machine import RenderAction, RenderStateMachine
from nerfstudio.viewer.utils import CameraState, parse_object
from nerfstudio.viewer.viewer_elements import ViewerControl, ViewerElement, ViewerClick
from nerfstudio.viewer_legacy.server import viewer_utils

if TYPE_CHECKING:
    from nerfstudio.engine.trainer import Trainer


VISER_NERFSTUDIO_SCALE_RATIO: float = 10.0


@decorate_all([check_main_thread])
class Viewer:
    """Class to hold state for viewer variables

    Args:
        config: viewer setup configuration
        log_filename: filename to log viewer output to
        datapath: path to data
        pipeline: pipeline object to use
        trainer: trainer object to use
        share: print a shareable URL

    Attributes:
        viewer_info: information string for the viewer
        viser_server: the viser server
    """

    viewer_info: List[str]
    viser_server: viser.ViserServer

    def __init__(
        self,
        config: cfg.ViewerConfig,
        log_filename: Path,
        datapath: Path,
        pipeline: Pipeline,
        trainer: Optional[Trainer] = None,
        train_lock: Optional[threading.Lock] = None,
        share: bool = False,
    ):
        self.ready = False  # Set to True at end of constructor.
        self.config = config
        self.trainer = trainer
        self.last_step = 0
        self.last_gaussian_points_update = -1000  # Initialize to allow first update
        self.train_lock = train_lock
        self.pipeline = pipeline
        self.log_filename = log_filename
        self.datapath = datapath.parent if datapath.is_file() else datapath
        self.include_time = self.pipeline.datamanager.includes_time
        self.train_dataset: Optional[InputDataset] = None  # Will be set in init_scene

        if self.config.websocket_port is None:
            websocket_port = viewer_utils.get_free_port(default_port=self.config.websocket_port_default)
        else:
            websocket_port = self.config.websocket_port
        self.log_filename.parent.mkdir(exist_ok=True)

        # viewer specific variables
        self.output_type_changed = True
        self.output_split_type_changed = True
        self.step = 0
        self.train_btn_state: Literal["training", "paused", "completed"] = (
            "training" if self.trainer is None else self.trainer.training_state
        )
        self._prev_train_state: Literal["training", "paused", "completed"] = self.train_btn_state
        self.last_move_time = 0
        # track the camera index that last being clicked
        self.current_camera_idx = 0
        # Selection mode state
        self.selection_mode = False
        self.selected_pixels: List[Dict[str, Any]] = []  # List of {x, y, is_positive, id, patch_data, feature_vector}
        self._selection_counter = 0  # For unique IDs
        self.last_rendered_image: Optional[np.ndarray] = None  # Store last rendered image for patch extraction
        self.last_rendered_features: Optional[torch.Tensor] = None  # Store last rendered features for pixel selection

        self.viser_server = viser.ViserServer(host=config.websocket_host, port=websocket_port)
        # Set the name of the URL either to the share link if available, or the localhost
        share_url = None
        if share:
            share_url = self.viser_server.request_share_url()
            if share_url is None:
                print("Couldn't make share URL!")

        if share_url is not None:
            self.viewer_info = [f"Viewer at: http://localhost:{websocket_port} or {share_url}"]
        elif config.websocket_host == "0.0.0.0":
            # 0.0.0.0 is not a real IP address and was confusing people, so
            # we'll just print localhost instead. There are some security
            # (and IPv6 compatibility) implications here though, so we should
            # note that the server is bound to 0.0.0.0!
            self.viewer_info = [f"Viewer running locally at: http://localhost:{websocket_port} (listening on 0.0.0.0)"]
        else:
            self.viewer_info = [f"Viewer running locally at: http://{config.websocket_host}:{websocket_port}"]

        buttons = (
            viser.theme.TitlebarButton(
                text="Getting Started",
                icon=None,
                href="https://nerf.studio",
            ),
            viser.theme.TitlebarButton(
                text="Github",
                icon="GitHub",
                href="https://github.com/nerfstudio-project/nerfstudio",
            ),
            viser.theme.TitlebarButton(
                text="Documentation",
                icon="Description",
                href="https://docs.nerf.studio",
            ),
        )
        image = viser.theme.TitlebarImage(
            image_url_light="https://docs.nerf.studio/_static/imgs/logo.png",
            image_url_dark="https://docs.nerf.studio/_static/imgs/logo-dark.png",
            image_alt="NerfStudio Logo",
            href="https://docs.nerf.studio/",
        )
        titlebar_theme = viser.theme.TitlebarConfig(buttons=buttons, image=image)
        self.viser_server.gui.configure_theme(
            titlebar_content=titlebar_theme,
            control_layout="collapsible",
            dark_mode=True,
            brand_color=(255, 211, 105),
        )

        self.render_statemachines: Dict[int, RenderStateMachine] = {}
        self.viser_server.on_client_disconnect(self.handle_disconnect)
        self.viser_server.on_client_connect(self.handle_new_client)

        # Populate the header, which includes the pause button, train cam button, and stats
        self.pause_train = self.viser_server.gui.add_button(
            label="Pause Training", disabled=False, icon=viser.Icon.PLAYER_PAUSE_FILLED
        )
        self.pause_train.on_click(lambda _: self.toggle_pause_button())
        self.pause_train.on_click(lambda han: self._toggle_training_state(han))
        self.resume_train = self.viser_server.gui.add_button(
            label="Resume Training", disabled=False, icon=viser.Icon.PLAYER_PLAY_FILLED
        )
        self.resume_train.on_click(lambda _: self.toggle_pause_button())
        self.resume_train.on_click(lambda han: self._toggle_training_state(han))
        if self.train_btn_state == "training":
            self.resume_train.visible = False
        else:
            self.pause_train.visible = False

        # Add buttons to toggle training image visibility
        self.hide_images = self.viser_server.gui.add_button(
            label="Hide Train Cams", disabled=False, icon=viser.Icon.EYE_OFF, color=None
        )
        self.hide_images.on_click(lambda _: self.set_camera_visibility(False))
        self.hide_images.on_click(lambda _: self.toggle_cameravis_button())
        self.show_images = self.viser_server.gui.add_button(
            label="Show Train Cams", disabled=False, icon=viser.Icon.EYE, color=None
        )
        self.show_images.on_click(lambda _: self.set_camera_visibility(True))
        self.show_images.on_click(lambda _: self.toggle_cameravis_button())
        self.show_images.visible = False
        mkdown = self.make_stats_markdown(0, "0x0px")
        self.stats_markdown = self.viser_server.gui.add_markdown(mkdown)
        tabs = self.viser_server.gui.add_tab_group()
        control_tab = tabs.add_tab("Control", viser.Icon.SETTINGS)
        with control_tab:
            self.control_panel = ControlPanel(
                self.viser_server,
                self.include_time,
                VISER_NERFSTUDIO_SCALE_RATIO,
                self._trigger_rerender,
                self._output_type_change,
                self._output_split_type_change,
                default_composite_depth=self.config.default_composite_depth,
                viewer=self,
            )
        config_path = self.log_filename.parents[0] / "config.yml"
        with tabs.add_tab("Render", viser.Icon.CAMERA):
            self.render_tab_state = populate_render_tab(
                self.viser_server, config_path, self.datapath, self.control_panel
            )

        with tabs.add_tab("Export", viser.Icon.PACKAGE_EXPORT):
            populate_export_tab(self.viser_server, self.control_panel, config_path, self.pipeline.model)
        
        # Initialize selection list UI
        self.control_panel.initialize_selection_list()

        # Keep track of the pointers to generated GUI folders, because each generated folder holds a unique ID.
        viewer_gui_folders = dict()

        def prev_cb_wrapper(prev_cb):
            # We wrap the callbacks in the train_lock so that the callbacks are thread-safe with the
            # concurrently executing render thread. This may block rendering, however this can be necessary
            # if the callback uses get_outputs internally.
            def cb_lock(element):
                with self.train_lock if self.train_lock is not None else contextlib.nullcontext():
                    prev_cb(element)

            return cb_lock

        def nested_folder_install(folder_labels: List[str], prev_labels: List[str], element: ViewerElement):
            if len(folder_labels) == 0:
                element.install(self.viser_server)
                # also rewire the hook to rerender
                prev_cb = element.cb_hook
                element.cb_hook = lambda element: [prev_cb_wrapper(prev_cb)(element), self._trigger_rerender()]
            else:
                # recursively create folders
                # If the folder name is "Custom Elements/a/b", then:
                #   in the beginning: folder_path will be
                #       "/".join([] + ["Custom Elements"]) --> "Custom Elements"
                #   later, folder_path will be
                #       "/".join(["Custom Elements"] + ["a"]) --> "Custom Elements/a"
                #       "/".join(["Custom Elements", "a"] + ["b"]) --> "Custom Elements/a/b"
                #  --> the element will be installed in the folder "Custom Elements/a/b"
                #
                # Note that the gui_folder is created only when the folder is not in viewer_gui_folders,
                # and we use the folder_path as the key to check if the folder is already created.
                # Otherwise, use the existing folder as context manager.
                folder_path = "/".join(prev_labels + [folder_labels[0]])
                if folder_path not in viewer_gui_folders:
                    viewer_gui_folders[folder_path] = self.viser_server.gui.add_folder(folder_labels[0])
                with viewer_gui_folders[folder_path]:
                    nested_folder_install(folder_labels[1:], prev_labels + [folder_labels[0]], element)

        with control_tab:
            from nerfstudio.viewer_legacy.server.viewer_elements import ViewerElement as LegacyViewerElement

            if len(parse_object(pipeline, LegacyViewerElement, "Custom Elements")) > 0:
                from nerfstudio.utils.rich_utils import CONSOLE

                CONSOLE.print(
                    "Legacy ViewerElements detected in model, please import nerfstudio.viewer.viewer_elements instead",
                    style="bold yellow",
                )
            self.viewer_elements = []
            self.viewer_elements.extend(parse_object(pipeline, ViewerElement, "Custom Elements"))
            for param_path, element in self.viewer_elements:
                folder_labels = param_path.split("/")[:-1]
                nested_folder_install(folder_labels, [], element)

            # scrape the trainer/pipeline for any ViewerControl objects to initialize them
            self.viewer_controls: List[ViewerControl] = [
                e for (_, e) in parse_object(pipeline, ViewerControl, "Custom Elements")
            ]
        for c in self.viewer_controls:
            c._setup(self)

        # Diagnostics for Gaussian Splatting: where the points are at the start of training.
        # This is hidden by default, it can be shown from the Viser UI's scene tree table.
        if isinstance(pipeline.model, SplatfactoModel):
            # Store initial means for comparison
            with torch.no_grad():
                initial_means = pipeline.model.means.data.clone().detach()
                if initial_means.is_cuda:
                    initial_means = initial_means.cpu()
                initial_points = initial_means.numpy() * VISER_NERFSTUDIO_SCALE_RATIO
            
            self.viser_server.scene.add_point_cloud(
                "/gaussian_splatting_initial_points",
                points=initial_points,
                colors=(255, 0, 0),
                point_size=0.01,
                point_shape="circle",
                visible=False,  # Hidden by default.
            )
            # Current gaussian points (green circles) - will be updated during training
            # Initialize with same points, but will be updated
            self.viser_server.scene.add_point_cloud(
                "/gaussian_splatting_current_points",
                points=initial_points,  # Start with initial points, will be updated
                colors=(0, 255, 0),
                point_size=0.01,
                point_shape="circle",
                visible=False,  # Hidden by default.
            )
        self.ready = True

    def toggle_pause_button(self) -> None:
        self.pause_train.visible = not self.pause_train.visible
        self.resume_train.visible = not self.resume_train.visible

    def toggle_cameravis_button(self) -> None:
        self.hide_images.visible = not self.hide_images.visible
        self.show_images.visible = not self.show_images.visible

    def make_stats_markdown(self, step: Optional[int], res: Optional[str]) -> str:
        # if either are None, read it from the current stats_markdown content
        if step is None:
            step = int(self.stats_markdown.content.split("\n")[0].split(": ")[1])
        if res is None:
            res = (self.stats_markdown.content.split("\n")[1].split(": ")[1]).strip()
        return f"Step: {step}  \nResolution: {res}"

    def update_step(self, step):
        """
        Args:
            step: the train step to set the model to
        """
        self.stats_markdown.content = self.make_stats_markdown(step, None)

    def get_camera_state(self, client: viser.ClientHandle) -> CameraState:
        R = vtf.SO3(wxyz=client.camera.wxyz)
        R = R @ vtf.SO3.from_x_radians(np.pi)
        R = torch.tensor(R.as_matrix())
        pos = torch.tensor(client.camera.position, dtype=torch.float64) / VISER_NERFSTUDIO_SCALE_RATIO
        c2w = torch.concatenate([R, pos[:, None]], dim=1)
        
        # Get custom intrinsics and distortion from render tab state if available
        fx = None
        fy = None
        cx = None
        cy = None
        # Distortion parameters - will be set as separate attributes in CameraState
        k1 = k2 = k3 = k4 = k5 = k6 = 0.0
        p1 = p2 = 0.0
        s1 = s2 = s3 = s4 = 0.0
        if self.ready and hasattr(self, 'render_tab_state'):
            if self.render_tab_state.use_custom_intrinsics:
                fx = self.render_tab_state.fx
                fy = self.render_tab_state.fy
                cx = self.render_tab_state.cx
                cy = self.render_tab_state.cy
            if self.render_tab_state.use_distortion:
                # Determine camera type for distortion parameters
                if self.render_tab_state.preview_render:
                    cam_type_str = self.render_tab_state.preview_camera_type
                else:
                    cam_type_str = "Perspective"  # Default for non-preview
                
                if cam_type_str == "Fisheye":
                    # Fisheye: 4 radial coefficients
                    k1 = self.render_tab_state.fisheye_k1
                    k2 = self.render_tab_state.fisheye_k2
                    k3 = self.render_tab_state.fisheye_k3
                    k4 = self.render_tab_state.fisheye_k4
                else:
                    # Perspective: 12 parameters (6 radial, 2 tangential, 4 thin prism)
                    k1 = self.render_tab_state.k1
                    k2 = self.render_tab_state.k2
                    k3 = self.render_tab_state.k3
                    k4 = self.render_tab_state.k4
                    k5 = self.render_tab_state.k5
                    k6 = self.render_tab_state.k6
                    p1 = self.render_tab_state.p1
                    p2 = self.render_tab_state.p2
                    s1 = self.render_tab_state.s1
                    s2 = self.render_tab_state.s2
                    s3 = self.render_tab_state.s3
                    s4 = self.render_tab_state.s4
        
        if self.ready and self.render_tab_state.preview_render:
            camera_type = self.render_tab_state.preview_camera_type
            camera_state = CameraState(
                fov=self.render_tab_state.preview_fov,
                aspect=self.render_tab_state.preview_aspect,
                c2w=c2w,
                time=self.render_tab_state.preview_time,
                camera_type=CameraType.PERSPECTIVE
                if camera_type == "Perspective"
                else CameraType.FISHEYE
                if camera_type == "Fisheye"
                else CameraType.EQUIRECTANGULAR
                if camera_type == "Equirectangular"
                else assert_never(camera_type),
                idx=self.current_camera_idx,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                k1=k1,
                k2=k2,
                k3=k3,
                k4=k4,
                k5=k5,
                k6=k6,
                p1=p1,
                p2=p2,
                s1=s1,
                s2=s2,
                s3=s3,
                s4=s4,
            )
        else:
            camera_state = CameraState(
                fov=client.camera.fov,
                aspect=client.camera.aspect,
                c2w=c2w,
                camera_type=CameraType.PERSPECTIVE,
                idx=self.current_camera_idx,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                k1=k1,
                k2=k2,
                k3=k3,
                k4=k4,
                k5=k5,
                k6=k6,
                p1=p1,
                p2=p2,
                s1=s1,
                s2=s2,
                s3=s3,
                s4=s4,
            )
        return camera_state

    def _load_camera_parameters_to_ui(self, camera_idx: int) -> None:
        """Load camera parameters from dataset into the custom intrinsics/distortions UI fields.
        
        Args:
            camera_idx: Index of the camera in the training dataset
        """
        if not self.ready or not hasattr(self, 'render_tab_state') or self.render_tab_state._ui_elements is None:
            return
        
        if self.train_dataset is None or camera_idx >= len(self.train_dataset):
            return
        
        ui = self.render_tab_state._ui_elements
        camera = self.train_dataset.cameras[camera_idx].reshape(())
        
        # Extract intrinsics
        fx_val = float(camera.fx[0].cpu())
        fy_val = float(camera.fy[0].cpu())
        cx_val = float(camera.cx[0].cpu())
        cy_val = float(camera.cy[0].cpu())
        
        # Calculate FOV from intrinsics (using height)
        height = float(camera.height[0].cpu())
        fov_rad = 2 * np.arctan(height / (2 * fy_val))
        fov_degrees = fov_rad * 180.0 / np.pi
        
        # Update FOV slider
        ui["fov_degrees"].value = fov_degrees
        
        # Enable and update custom intrinsics
        ui["use_custom_intrinsics"].value = True
        ui["fx_number"].value = fx_val
        ui["fy_number"].value = fy_val
        ui["cx_number"].value = cx_val
        ui["cy_number"].value = cy_val
        
        # Extract distortion parameters
        camera_type_enum = CameraType(int(camera.camera_type[0].cpu()))
        has_distortion = camera.distortion_params is not None and not torch.all(camera.distortion_params == 0)
        
        if has_distortion and camera.distortion_params is not None:
            # Enable distortion
            ui["use_distortion"].value = True
            
            # Set camera type based on dataset camera type
            if camera_type_enum == CameraType.FISHEYE:
                ui["camera_type"].value = "Fisheye"
                dist_params_raw = camera.distortion_params[0].cpu()
                # Convert to list safely, handling 0-dim tensors
                if dist_params_raw.dim() == 0:
                    dist_list = [float(dist_params_raw.item())]
                else:
                    dist_list = dist_params_raw.flatten().tolist()
                # Fisheye: 4 radial coefficients [k1, k2, k3, k4]
                ui["fisheye_k1_number"].value = float(dist_list[0]) if len(dist_list) > 0 else 0.0
                ui["fisheye_k2_number"].value = float(dist_list[1]) if len(dist_list) > 1 else 0.0
                ui["fisheye_k3_number"].value = float(dist_list[2]) if len(dist_list) > 2 else 0.0
                ui["fisheye_k4_number"].value = float(dist_list[3]) if len(dist_list) > 3 else 0.0
            else:
                # Perspective: 6 radial (k1-k6), 2 tangential (p1, p2), 4 thin prism (s1-s4)
                ui["camera_type"].value = "Perspective"
                dist_params_raw = camera.distortion_params[0].cpu()
                # Convert to list safely, handling 0-dim tensors
                if dist_params_raw.dim() == 0:
                    dist_list = [float(dist_params_raw.item())]
                else:
                    dist_list = dist_params_raw.flatten().tolist()
                # Standard format: [k1, k2, k3, k4, p1, p2] for first 6
                ui["k1_number"].value = float(dist_list[0]) if len(dist_list) > 0 else 0.0
                ui["k2_number"].value = float(dist_list[1]) if len(dist_list) > 1 else 0.0
                ui["k3_number"].value = float(dist_list[2]) if len(dist_list) > 2 else 0.0
                ui["k4_number"].value = float(dist_list[3]) if len(dist_list) > 3 else 0.0
                ui["p1_number"].value = float(dist_list[4]) if len(dist_list) > 4 else 0.0
                ui["p2_number"].value = float(dist_list[5]) if len(dist_list) > 5 else 0.0
                
                # Check metadata for extended distortion parameters (k5, k6, s1-s4)
                if camera.metadata is not None:
                    if "distortion_radial" in camera.metadata:
                        radial_raw = camera.metadata["distortion_radial"][0].cpu()
                        # Convert to list safely, handling 0-dim tensors
                        if radial_raw.dim() == 0:
                            radial_list = [float(radial_raw.item())]
                        else:
                            radial_list = radial_raw.flatten().tolist()
                        # Radial: [k1, k2, k3, k4, k5, k6] for perspective
                        if len(radial_list) >= 5:
                            ui["k5_number"].value = float(radial_list[4])
                        if len(radial_list) >= 6:
                            ui["k6_number"].value = float(radial_list[5])
                    if "distortion_tangential" in camera.metadata:
                        tangential_raw = camera.metadata["distortion_tangential"][0].cpu()
                        # Convert to list safely, handling 0-dim tensors
                        if tangential_raw.dim() == 0:
                            tangential_list = [float(tangential_raw.item())]
                        else:
                            tangential_list = tangential_raw.flatten().tolist()
                        # Tangential: [p1, p2]
                        if len(tangential_list) >= 1:
                            ui["p1_number"].value = float(tangential_list[0])
                        if len(tangential_list) >= 2:
                            ui["p2_number"].value = float(tangential_list[1])
                    if "distortion_thin_prism" in camera.metadata:
                        thin_prism_raw = camera.metadata["distortion_thin_prism"][0].cpu()
                        # Convert to list safely, handling 0-dim tensors
                        if thin_prism_raw.dim() == 0:
                            thin_prism_list = [float(thin_prism_raw.item())]
                        else:
                            thin_prism_list = thin_prism_raw.flatten().tolist()
                        # Thin prism: [s1, s2, s3, s4]
                        if len(thin_prism_list) >= 1:
                            ui["s1_number"].value = float(thin_prism_list[0])
                        if len(thin_prism_list) >= 2:
                            ui["s2_number"].value = float(thin_prism_list[1])
                        if len(thin_prism_list) >= 3:
                            ui["s3_number"].value = float(thin_prism_list[2])
                        if len(thin_prism_list) >= 4:
                            ui["s4_number"].value = float(thin_prism_list[3])
                    else:
                        # Set thin prism to 0 if not in metadata
                        ui["s1_number"].value = 0.0
                        ui["s2_number"].value = 0.0
                        ui["s3_number"].value = 0.0
                        ui["s4_number"].value = 0.0
                else:
                    # No metadata, set extended parameters to 0
                    ui["k5_number"].value = 0.0
                    ui["k6_number"].value = 0.0
                    ui["s1_number"].value = 0.0
                    ui["s2_number"].value = 0.0
                    ui["s3_number"].value = 0.0
                    ui["s4_number"].value = 0.0
        else:
            # Disable distortion if no distortion parameters
            ui["use_distortion"].value = False

    def handle_disconnect(self, client: viser.ClientHandle) -> None:
        if client.client_id in self.render_statemachines:
            self.render_statemachines[client.client_id].running = False
            self.render_statemachines.pop(client.client_id)

    def handle_new_client(self, client: viser.ClientHandle) -> None:
        self.render_statemachines[client.client_id] = RenderStateMachine(self, VISER_NERFSTUDIO_SCALE_RATIO, client)
        self.render_statemachines[client.client_id].start()

        @client.camera.on_update
        def _(_: viser.CameraHandle) -> None:
            if not self.ready:
                return
            # Skip camera updates when in selection mode
            if self.selection_mode:
                return
            # Check if render state machine exists for this client
            if client.client_id not in self.render_statemachines:
                return
            self.last_move_time = time.time()
            with self.viser_server.atomic():
                camera_state = self.get_camera_state(client)
                self.render_statemachines[client.client_id].action(RenderAction("move", camera_state))

    def set_camera_visibility(self, visible: bool) -> None:
        """Toggle the visibility of the training cameras."""
        with self.viser_server.atomic():
            for idx in self.camera_handles:
                self.camera_handles[idx].visible = visible
    
    def set_selection_mode(self, enabled: bool) -> None:
        """Enable or disable selection mode.
        
        Args:
            enabled: If True, enables selection mode and disables camera navigation.
                     If False, disables selection mode and enables camera navigation.
        """
        # Always ensure callback is removed first when disabling, even if we think it's already removed
        if not enabled and hasattr(self, '_selection_callback'):
            try:
                self.viser_server.scene.remove_pointer_callback()
            except Exception:
                pass
            if hasattr(self, '_selection_callback'):
                delattr(self, '_selection_callback')
        
        self.selection_mode = enabled
        
        if enabled:
            # Register pointer callback for selection directly with viser
            # Use the ViewerControl wrapper to avoid interfering with other handlers
            
            def wrapped_cb(scene_pointer_msg):
                try:
                    # Debug: Check if callback is being triggered
                    # print(f"Selection callback triggered: event_type={scene_pointer_msg.event_type}, selection_mode={self.selection_mode}")
                    
                    # Only handle click events, not drag or other events
                    # This allows camera drag to work normally
                    if scene_pointer_msg.event_type != "click":
                        return
                    
                    origin = scene_pointer_msg.ray_origin
                    direction = scene_pointer_msg.ray_direction
                    if origin is None or direction is None:
                        return
                    
                    # Access screen_pos safely
                    screen_pos = scene_pointer_msg.screen_pos
                    if not screen_pos or len(screen_pos) == 0:
                        return
                    screen_pos = screen_pos[0] if isinstance(screen_pos, (list, tuple)) else screen_pos
                    
                    origin = tuple([x / VISER_NERFSTUDIO_SCALE_RATIO for x in origin])
                    
                    # Get button info - default to "left" if not available
                    button = getattr(scene_pointer_msg, 'button', None)
                    if button is not None and isinstance(button, int):
                        button_map = {0: "left", 1: "middle", 2: "right"}
                        button = button_map.get(button, "left")  # Default to left if unknown
                    else:
                        # If button is not available, assume left click
                        button = "left"
                    
                    click = ViewerClick(origin, direction, screen_pos, button=button)
                    # Handle left clicks for pixel selection (or if button is None/unknown, treat as left)
                    if button == "left" or button is None:
                        # Debug: Verify we're calling the handler
                        # print(f"Calling _handle_pixel_selection: screen_pos={screen_pos}, button={button}")
                        self._handle_pixel_selection(click)
                except Exception as e:
                    # Log error but don't break the viewer
                    import traceback
                    print(f"Error in selection callback: {e}")
                    print(traceback.format_exc())
            
            # Register callback only for click events (not drag/move)
            # This should not interfere with camera drag navigation
            try:
                self.viser_server.scene.on_pointer_event(event_type="click")(wrapped_cb)
                self._selection_callback = wrapped_cb
                print(f"DEBUG: Selection callback registered successfully, selection_mode={self.selection_mode}")
            except Exception as e:
                print(f"ERROR: Failed to register selection callback: {e}")
                import traceback
                traceback.print_exc()
    
    def _handle_pixel_selection(self, click: "ViewerClick") -> None:
        """Handle pixel selection click event.
        
        Args:
            click: The click event containing screen position and button info
        """
        try:
            # Debug: Check if handler is called
            print(f"DEBUG: _handle_pixel_selection called: selection_mode={self.selection_mode}, screen_pos={click.screen_pos}")
            
            if not self.selection_mode:
                print("DEBUG: Selection mode is False, returning early")
                return
            
            # Get screen position (normalized [0, 1])
            screen_x, screen_y = click.screen_pos
            
            # Extract image patch if available
            patch_data = None
            if self.last_rendered_image is not None:
                try:
                    patch_data = self._extract_image_patch(screen_x, screen_y, self.last_rendered_image)
                except Exception:
                    patch_data = None
            
            # Extract feature vector at selected pixel if available
            # This is optional - selection should work even if feature extraction fails
            feature_vector = None
            if self.last_rendered_features is not None:
                try:
                    feature_vector = self._extract_feature_vector(screen_x, screen_y, self.last_rendered_features)
                except Exception as e:
                    # Feature extraction failed - log but don't break selection
                    import traceback
                    print(f"Warning: Failed to extract feature vector: {e}")
                    feature_vector = None
            
            # Sync with model's viewer_utils if available (for pc-feature-splatting)
            # This allows the model to use the stored feature vector for similarity computation
            # This is optional - selection should work even if model sync fails
            try:
                # Only try to sync if we have a pipeline (viewer is ready)
                if hasattr(self, 'pipeline') and self.pipeline is not None:
                    model = self.get_model()
                    # Check if this is a PCFeatureSplattingModel with viewer_utils
                    if hasattr(model, 'viewer_utils') and hasattr(model.viewer_utils, 'set_selected_pixel'):
                        # Update viewer_utils with pixel location and feature vector
                        model.viewer_utils.set_selected_pixel(screen_x, screen_y, feature_vector)
            except Exception as e:
                # Silently fail if model doesn't have viewer_utils (not pc-feature-splatting)
                # This is expected for non-pc-feature-splatting models
                # Don't log errors here as this is expected for many models
                pass
            
            # Add to selection list (only left clicks supported)
            self._selection_counter += 1
            pixel_id = self._selection_counter
            pixel_data = {
                "id": pixel_id,
                "x": screen_x,
                "y": screen_y,
                "is_positive": True,  # Default to positive (green +)
                "patch_data": patch_data,  # Base64 encoded patch image (not displayed in UI)
                "feature_vector": feature_vector,  # Feature vector at selected pixel (torch.Tensor or None, stored but not displayed in UI)
            }
            self.selected_pixels.append(pixel_data)
            
            # Update UI
            if hasattr(self, 'control_panel') and self.control_panel is not None:
                try:
                    self.control_panel.update_selection_list()
                except Exception:
                    pass  # Silently fail if UI update fails
        except Exception as e:
            # Log error but don't break the viewer
            import traceback
            print(f"Error in pixel selection: {e}")
            print(traceback.format_exc())
    
    def _extract_feature_vector(self, screen_x: float, screen_y: float, features: torch.Tensor) -> Optional[torch.Tensor]:
        """Extract feature vector at the clicked pixel location.
        
        Args:
            screen_x: Normalized x coordinate [0, 1]
            screen_y: Normalized y coordinate [0, 1]
            features: The rendered features as torch tensor (H, W, C) or (C, H, W)
            
        Returns:
            Feature vector at the pixel location (C,) or None if extraction fails
        """
        try:
            # Handle different feature tensor formats
            if len(features.shape) == 3:
                if features.shape[0] < features.shape[2]:  # Likely (H, W, C)
                    h, w, c = features.shape
                    pixel_x = int(screen_x * w)
                    pixel_y = int(screen_y * h)
                    pixel_x = max(0, min(pixel_x, w - 1))
                    pixel_y = max(0, min(pixel_y, h - 1))
                    feature_vector = features[pixel_y, pixel_x, :].clone()  # [C]
                else:  # Likely (C, H, W)
                    c, h, w = features.shape
                    pixel_x = int(screen_x * w)
                    pixel_y = int(screen_y * h)
                    pixel_x = max(0, min(pixel_x, w - 1))
                    pixel_y = max(0, min(pixel_y, h - 1))
                    feature_vector = features[:, pixel_y, pixel_x].clone()  # [C]
            else:
                return None
            
            return feature_vector
        except Exception:
            return None
    
    def _extract_image_patch(self, screen_x: float, screen_y: float, image: np.ndarray) -> Optional[str]:
        """Extract a 16x16 pixel patch centered at the clicked location.
        
        Args:
            screen_x: Normalized x coordinate [0, 1]
            screen_y: Normalized y coordinate [0, 1]
            image: The rendered image as numpy array (H, W, 3)
            
        Returns:
            Base64 encoded image string or None if extraction fails
        """
        import base64
        from io import BytesIO
        from PIL import Image
        
        try:
            h, w = image.shape[:2]
            # Convert normalized coordinates to pixel coordinates
            pixel_x = int(screen_x * w)
            pixel_y = int(screen_y * h)
            
            # Extract 16x16 patch centered at the pixel
            patch_size = 16
            half_size = patch_size // 2
            
            # Calculate bounds with clamping
            x_start = max(0, pixel_x - half_size)
            x_end = min(w, pixel_x + half_size)
            y_start = max(0, pixel_y - half_size)
            y_end = min(h, pixel_y + half_size)
            
            # Extract patch
            patch = image[y_start:y_end, x_start:x_end, :]
            
            # If patch is smaller than 16x16 (near edges), pad it
            if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
                padded_patch = np.zeros((patch_size, patch_size, 3), dtype=patch.dtype)
                pad_y = (patch_size - patch.shape[0]) // 2
                pad_x = (patch_size - patch.shape[1]) // 2
                padded_patch[pad_y:pad_y+patch.shape[0], pad_x:pad_x+patch.shape[1], :] = patch
                patch = padded_patch
            
            # Convert to PIL Image and encode as base64
            patch_uint8 = patch.astype(np.uint8)
            pil_image = Image.fromarray(patch_uint8)
            buffer = BytesIO()
            pil_image.save(buffer, format='PNG')
            img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
            return f"data:image/png;base64,{img_str}"
        except Exception:
            return None
    
    def _patch_to_svg_icon(self, patch_data: str) -> Optional[str]:
        """Convert a base64 image patch to an SVG icon string.
        
        Args:
            patch_data: Base64 encoded image data URL (e.g., "data:image/png;base64,...")
            
        Returns:
            SVG string with embedded image, or None if conversion fails
        """
        try:
            if not patch_data or not patch_data.startswith("data:image"):
                return None
            
            # Extract the base64 part
            base64_data = patch_data.split(",")[1] if "," in patch_data else None
            if not base64_data:
                return None
            
            # Create an SVG with the embedded image
            # Size: 32x32 pixels to match icon size
            svg = f'''<svg width="32" height="32" xmlns="http://www.w3.org/2000/svg">
  <image href="{patch_data}" width="32" height="32" preserveAspectRatio="none"/>
  <rect width="32" height="32" fill="none" stroke="#ccc" stroke-width="1"/>
</svg>'''
            return svg
        except Exception:
            return None
    
    def toggle_pixel_sign(self, pixel_id: int) -> None:
        """Toggle the positive/negative sign of a selected pixel.
        
        Args:
            pixel_id: The ID of the pixel to toggle
        """
        for pixel_data in self.selected_pixels:
            if pixel_data["id"] == pixel_id:
                pixel_data["is_positive"] = not pixel_data["is_positive"]
                break
    
    def clear_selected_pixels(self) -> None:
        """Clear all selected pixels."""
        self.selected_pixels.clear()
        
        # Clear model's viewer_utils selection
        try:
            model = self.get_model()
            if hasattr(model, 'viewer_utils') and hasattr(model.viewer_utils, 'set_selected_pixel'):
                model.viewer_utils.set_selected_pixel(None, None)
        except Exception:
            pass
    
    def remove_selected_pixel(self, pixel_id: int) -> None:
        """Remove a specific selected pixel by ID.
        
        Args:
            pixel_id: The ID of the pixel to remove
        """
        # Find the pixel to remove
        pixel_to_remove = None
        for p in self.selected_pixels:
            if p["id"] == pixel_id:
                pixel_to_remove = p
                break
        
        # Remove from list
        self.selected_pixels = [p for p in self.selected_pixels if p["id"] != pixel_id]
        
        # If this was the last selected pixel, clear model's viewer_utils selection
        if len(self.selected_pixels) == 0:
            try:
                model = self.get_model()
                if hasattr(model, 'viewer_utils') and hasattr(model.viewer_utils, 'set_selected_pixel'):
                    model.viewer_utils.set_selected_pixel(None, None)
            except Exception:
                pass
    
    def get_pixel_sign(self, pixel_id: int) -> Optional[bool]:
        """Get the positive/negative sign of a selected pixel.
        
        Args:
            pixel_id: The ID of the pixel
            
        Returns:
            True if positive, False if negative, None if pixel not found
        """
        for pixel_data in self.selected_pixels:
            if pixel_data["id"] == pixel_id:
                return pixel_data["is_positive"]
        return None

    def update_camera_poses(self):
        # TODO this fn accounts for like ~5% of total train time
        # Update the train camera locations based on optimization
        assert self.camera_handles is not None
        if hasattr(self.pipeline.model, "camera_optimizer"):
            camera_optimizer = self.pipeline.model.camera_optimizer
        else:
            return
        idxs = list(self.camera_handles.keys())
        with torch.no_grad():
            assert isinstance(camera_optimizer, CameraOptimizer)
            c2ws_delta = camera_optimizer(torch.tensor(idxs, device=camera_optimizer.device)).cpu().numpy()
        for i, key in enumerate(idxs):
            # both are numpy arrays
            c2w_orig = self.original_c2w[key]
            c2w_delta = c2ws_delta[i, ...]
            c2w = c2w_orig @ np.concatenate((c2w_delta, np.array([[0, 0, 0, 1]])), axis=0)
            R = vtf.SO3.from_matrix(c2w[:3, :3])  # type: ignore
            R = R @ vtf.SO3.from_x_radians(np.pi)
            self.camera_handles[key].position = c2w[:3, 3] * VISER_NERFSTUDIO_SCALE_RATIO
            self.camera_handles[key].wxyz = R.wxyz

    def _trigger_rerender(self) -> None:
        """Interrupt current render."""
        if not self.ready:
            return
        clients = self.viser_server.get_clients()
        for id in clients:
            if id not in self.render_statemachines:
                continue
            camera_state = self.get_camera_state(clients[id])
            self.render_statemachines[id].action(RenderAction("move", camera_state))

    def _toggle_training_state(self, _) -> None:
        """Toggle the trainer's training state."""
        if self.trainer is not None:
            if self.trainer.training_state == "training":
                self.trainer.training_state = "paused"
            elif self.trainer.training_state == "paused":
                self.trainer.training_state = "training"

    def _output_type_change(self, _):
        self.output_type_changed = True

    def _output_split_type_change(self, _):
        self.output_split_type_changed = True


    def _pick_drawn_image_idxs(self, total_num: int) -> list[int]:
        """Determine indicies of images to display in viewer.

        Args:
            total_num: total number of training images.

        Returns:
            List of indices from [0, total_num-1].
        """
        if self.config.max_num_display_images < 0:
            num_display_images = total_num
        else:
            num_display_images = min(self.config.max_num_display_images, total_num)
        # draw indices, roughly evenly spaced
        return np.linspace(0, total_num - 1, num_display_images, dtype=np.int32).tolist()

    def init_scene(
        self,
        train_dataset: InputDataset,
        train_state: Literal["training", "paused", "completed"],
        eval_dataset: Optional[InputDataset] = None,
    ) -> None:
        """Draw some images and the scene aabb in the viewer.

        Args:
            dataset: dataset to render in the scene
            train_state: Current status of training
        """
        # Store train_dataset reference for camera parameter loading
        self.train_dataset = train_dataset
        
        # draw the training cameras and images
        self.camera_handles: Dict[int, viser.CameraFrustumHandle] = {}
        self.original_c2w: Dict[int, np.ndarray] = {}
        image_indices = self._pick_drawn_image_idxs(len(train_dataset))
        for idx in image_indices:
            image = train_dataset[idx]["image"]
            camera = train_dataset.cameras[idx]
            image_uint8 = (image * 255).detach().type(torch.uint8)
            image_uint8 = image_uint8.permute(2, 0, 1)

            # torchvision can be slow to import, so we do it lazily.
            import torchvision

            image_uint8 = torchvision.transforms.functional.resize(image_uint8, 100, antialias=None)  # type: ignore
            image_uint8 = image_uint8.permute(1, 2, 0)
            image_uint8 = image_uint8.cpu().numpy()
            c2w = camera.camera_to_worlds.cpu().numpy()
            R = vtf.SO3.from_matrix(c2w[:3, :3])
            R = R @ vtf.SO3.from_x_radians(np.pi)
            camera_handle = self.viser_server.scene.add_camera_frustum(
                name=f"/cameras/camera_{idx:05d}",
                fov=float(2 * np.arctan((camera.cx / camera.fx[0]).cpu())),
                scale=self.config.camera_frustum_scale,
                aspect=float((camera.cx[0] / camera.cy[0]).cpu()),
                image=image_uint8,
                wxyz=R.wxyz,
                position=c2w[:3, 3] * VISER_NERFSTUDIO_SCALE_RATIO,
            )

            def create_on_click_callback(capture_idx):
                def on_click_callback(event: viser.SceneNodePointerEvent[viser.CameraFrustumHandle]) -> None:
                    with event.client.atomic():
                        event.client.camera.position = event.target.position
                        event.client.camera.wxyz = event.target.wxyz
                        self.current_camera_idx = capture_idx
                        # Load camera parameters into custom intrinsics/distortions fields
                        self._load_camera_parameters_to_ui(capture_idx)

                return on_click_callback

            camera_handle.on_click(create_on_click_callback(idx))

            self.camera_handles[idx] = camera_handle
            self.original_c2w[idx] = c2w

        self.train_state = train_state
        self.train_util = 0.9

    def update_scene(self, step: int, num_rays_per_batch: Optional[int] = None) -> None:
        """updates the scene based on the graph weights

        Args:
            step: iteration step of training
            num_rays_per_batch: number of rays per batch, used during training
        """
        self.step = step

        if len(self.render_statemachines) == 0:
            return
        # this stops training while moving to make the response smoother
        while time.time() - self.last_move_time < 0.1:
            time.sleep(0.05)
        if self.trainer is not None and self.trainer.training_state == "training" and self.train_util != 1:
            if (
                EventName.TRAIN_RAYS_PER_SEC.value in GLOBAL_BUFFER["events"]
                and EventName.VIS_RAYS_PER_SEC.value in GLOBAL_BUFFER["events"]
            ):
                train_s = GLOBAL_BUFFER["events"][EventName.TRAIN_RAYS_PER_SEC.value]["avg"]
                vis_s = GLOBAL_BUFFER["events"][EventName.VIS_RAYS_PER_SEC.value]["avg"]
                train_util = self.train_util
                vis_n = self.control_panel.max_res**2
                train_n = num_rays_per_batch
                train_time = train_n / train_s
                vis_time = vis_n / vis_s

                render_freq = train_util * vis_time / (train_time - train_util * train_time)
            else:
                render_freq = 30
            if step > self.last_step + render_freq:
                self.last_step = step
                clients = self.viser_server.get_clients()
                for id in clients:
                    if id not in self.render_statemachines:
                        continue
                    camera_state = self.get_camera_state(clients[id])
                    if camera_state is not None:
                        self.render_statemachines[id].action(RenderAction("step", camera_state))
                self.update_camera_poses()
                self.update_step(step)
                # Update current gaussian points visualization (update less frequently to avoid performance issues)
                # Update every 50 steps to be more responsive to changes
                if isinstance(self.pipeline.model, SplatfactoModel) and step - self.last_gaussian_points_update >= 50:
                    try:
                        # Get current means - access directly from gauss_params to get actual current values
                        model = self.pipeline.model
                        assert isinstance(model, SplatfactoModel)
                        
                        # Check if means are frozen - if so, they won't change
                        freeze_means = getattr(model.config, 'freeze_means', False)
                        if freeze_means:
                            # Means are frozen, so current points should match initial points
                            # Skip update to avoid unnecessary work
                            self.last_gaussian_points_update = step
                        else:
                            with torch.no_grad():
                                # Access the parameter data directly - this gets the current optimized values
                                # Use .clone() to ensure we get a fresh copy of the data
                                current_means = model.gauss_params["means"].data.clone().detach()
                                if current_means.is_cuda:
                                    current_means = current_means.cpu()
                                current_points = current_means.numpy() * VISER_NERFSTUDIO_SCALE_RATIO
                                
                                # Always update (remove change detection to ensure updates happen)
                                # Remove and re-add to update (viser doesn't have direct update method)
                                self.viser_server.scene.remove_point_cloud("/gaussian_splatting_current_points")
                                self.viser_server.scene.add_point_cloud(
                                    "/gaussian_splatting_current_points",
                                    points=current_points,
                                    colors=(0, 255, 0),
                                    point_size=0.01,
                                    point_shape="circle",
                                    visible=False,  # Hidden by default.
                                )
                            self.last_gaussian_points_update = step
                    except Exception as e:
                        # Log error for debugging instead of silently failing
                        from nerfstudio.utils.rich_utils import CONSOLE
                        CONSOLE.log(f"[yellow]Failed to update gaussian points at step {step}: {e}")
                        import traceback
                        CONSOLE.log(f"[yellow]Traceback: {traceback.format_exc()}")

    def update_colormap_options(self, dimensions: int, dtype: type) -> None:
        """update the colormap options based on the current render

        Args:
            dimensions: the number of dimensions of the render
            dtype: the data type of the render
        """
        if self.output_type_changed:
            self.control_panel.update_colormap_options(dimensions, dtype)
            self.output_type_changed = False

    def update_split_colormap_options(self, dimensions: int, dtype: type) -> None:
        """update the colormap options based on the current render

        Args:
            dimensions: the number of dimensions of the render
            dtype: the data type of the render
        """
        if self.output_split_type_changed:
            self.control_panel.update_split_colormap_options(dimensions, dtype)
            self.output_split_type_changed = False

    def get_model(self) -> Model:
        """Returns the model."""
        return self.pipeline.model

    def training_complete(self) -> None:
        """Called when training is complete."""
        self.training_state = "completed"
