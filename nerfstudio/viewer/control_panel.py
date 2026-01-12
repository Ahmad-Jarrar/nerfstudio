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

"""Control panel for the viewer"""

from collections import defaultdict
from typing import Any, Callable, DefaultDict, Dict, List, Tuple, get_args, Optional

import numpy as np
import torch
import viser
import viser.transforms as vtf
from viser import ViserServer

from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.utils.colormaps import ColormapOptions, Colormaps
from nerfstudio.viewer.viewer_elements import (  # ViewerButtonGroup,
    ViewerButtonGroup,
    ViewerCheckbox,
    ViewerDropdown,
    ViewerElement,
    ViewerNumber,
    ViewerRGB,
    ViewerSlider,
    ViewerVec3,
)


class ControlPanel:
    """
    Initializes the control panel with all the elements
    Args:
        time_enabled: whether or not the time slider should be enabled
        rerender_cb: a callback that will be called when the user changes a parameter that requires a rerender
            (eg train speed, max res, etc)
        update_output_cb: a callback that will be called when the user changes the output render
        default_composite_depth: whether to default to compositing depth or not
    """

    def __init__(
        self,
        server: ViserServer,
        time_enabled: bool,
        scale_ratio: float,
        rerender_cb: Callable[[], None],
        update_output_cb: Callable,
        update_split_output_cb: Callable,
        default_composite_depth: bool = True,
        viewer: Optional[Any] = None,
    ):
        self.viser_scale_ratio = scale_ratio
        # elements holds a mapping from tag: [elements]
        self.server = server
        self._elements_by_tag: DefaultDict[str, List[ViewerElement]] = defaultdict(lambda: [])
        self.default_composite_depth = default_composite_depth
        self.viewer = viewer  # Reference to Viewer for selection mode

        self._train_speed = ViewerButtonGroup(
            name="Train Speed",
            default_value="Mid",
            options=["Slow", "Mid", "Fast"],
            cb_hook=lambda han: self._train_speed_cb(),
        )
        self._output_render = ViewerDropdown(
            "Output type",
            "not set",
            ["not set"],
            cb_hook=lambda han: [self.update_control_panel(), update_output_cb(han), rerender_cb()],
            hint="The output to render",
        )
        self._colormap = ViewerDropdown[Colormaps](
            "Colormap", "default", ["default"], cb_hook=lambda _: rerender_cb(), hint="The colormap to use"
        )
        self._invert = ViewerCheckbox("Invert", False, cb_hook=lambda _: rerender_cb(), hint="Invert the colormap")
        self._normalize = ViewerCheckbox(
            "Normalize", True, cb_hook=lambda _: rerender_cb(), hint="Normalize the colormap"
        )
        self._min = ViewerNumber("Min", 0.0, cb_hook=lambda _: rerender_cb(), hint="Min value of the colormap")
        self._max = ViewerNumber("Max", 1.0, cb_hook=lambda _: rerender_cb(), hint="Max value of the colormap")

        self._split = ViewerCheckbox(
            "Enable",
            False,
            cb_hook=lambda han: [self.update_control_panel(), rerender_cb()],
            hint="Render two outputs",
        )
        self._split_percentage = ViewerSlider(
            "Split percentage", 0.5, 0.0, 1.0, 0.01, cb_hook=lambda _: rerender_cb(), hint="Where to split"
        )
        self._split_output_render = ViewerDropdown(
            "Output render split",
            "not set",
            ["not set"],
            cb_hook=lambda han: [self.update_control_panel(), update_split_output_cb(han), rerender_cb()],
            hint="The second output",
        )
        # Hack: spaces are after at the end of the names to make them unique
        self._split_colormap = ViewerDropdown[Colormaps](
            "Colormap ", "default", ["default"], cb_hook=lambda _: rerender_cb(), hint="Colormap of the second output"
        )
        self._split_invert = ViewerCheckbox(
            "Invert ", False, cb_hook=lambda _: rerender_cb(), hint="Invert the colormap of the second output"
        )
        self._split_normalize = ViewerCheckbox(
            "Normalize ", True, cb_hook=lambda _: rerender_cb(), hint="Normalize the colormap of the second output"
        )
        self._split_min = ViewerNumber(
            "Min ", 0.0, cb_hook=lambda _: rerender_cb(), hint="Min value of the colormap of the second output"
        )
        self._split_max = ViewerNumber(
            "Max ", 1.0, cb_hook=lambda _: rerender_cb(), hint="Max value of the colormap of the second output"
        )

        self._train_util = ViewerSlider(
            "Train Util",
            default_value=0.85,
            min_value=0.0,
            max_value=1,
            step=0.05,
            hint="Target training utilization, 0.0 is slow, 1.0 is fast. Doesn't affect final render quality",
        )
        self._layer_depth = ViewerCheckbox(
            "Composite depth",
            self.default_composite_depth,
            cb_hook=lambda _: rerender_cb(),
            hint="Allow NeRF to occlude 3D browser objects",
        )
        self._max_res = ViewerSlider(
            "Max res",
            512,
            64,
            2048,
            100,
            cb_hook=lambda _: rerender_cb(),
            hint="Maximum resolution to render in viewport",
        )
        self._crop_viewport = ViewerCheckbox(
            "Enable ",
            False,
            cb_hook=lambda han: [self.update_control_panel(), rerender_cb()],
            hint="Crop the scene to a specified box",
        )
        self._background_color = ViewerRGB(
            "Background color", (38, 42, 55), cb_hook=lambda _: rerender_cb(), hint="Color of the background"
        )
        self._crop_handle = self.server.scene.add_transform_controls("Crop", depth_test=False, line_width=4.0)

        def update_center(han):
            self._crop_handle.position = tuple(p * self.viser_scale_ratio for p in han.value)  # type: ignore

        self._crop_center = ViewerVec3(
            "Crop center",
            (0.0, 0.0, 0.0),
            step=0.01,
            cb_hook=lambda e: [rerender_cb(), update_center(e)],
            hint="Center of the crop box",
        )

        def update_rot(han):
            self._crop_handle.wxyz = vtf.SO3.from_rpy_radians(*han.value).wxyz

        self._crop_rot = ViewerVec3(
            "Crop rotation",
            (0.0, 0.0, 0.0),
            step=0.01,
            cb_hook=lambda e: [rerender_cb(), update_rot(e)],
            hint="Rotation of the crop box",
        )

        self._crop_scale = ViewerVec3(
            "Crop scale", (1.0, 1.0, 1.0), step=0.01, cb_hook=lambda _: rerender_cb(), hint="Size of the crop box."
        )

        @self._crop_handle.on_update
        def _update_crop_handle(han):
            pos = self._crop_handle.position
            self._crop_center.value = tuple(p / self.viser_scale_ratio for p in pos)  # type: ignore
            rpy = vtf.SO3(self._crop_handle.wxyz).as_rpy_radians()
            self._crop_rot.value = (float(rpy.roll), float(rpy.pitch), float(rpy.yaw))

        self._time = ViewerSlider("Time", 0.0, 0.0, 1.0, 0.01, cb_hook=lambda _: rerender_cb(), hint="Time to render")
        self._time_enabled = time_enabled

        self.add_element(self._train_speed)
        self.add_element(self._train_util)

        with self.server.gui.add_folder("Render Options"):
            self.add_element(self._max_res)
            self.add_element(self._output_render)
            self.add_element(self._colormap)
            self.add_element(self._layer_depth)
            # colormap options
            self.add_element(self._invert, additional_tags=("colormap",))
            self.add_element(self._normalize, additional_tags=("colormap",))
            self.add_element(self._min, additional_tags=("colormap",))
            self.add_element(self._max, additional_tags=("colormap",))

        # split options
        with self.server.gui.add_folder("Split Screen"):
            self.add_element(self._split)

            self.add_element(self._split_percentage, additional_tags=("split",))
            self.add_element(self._split_output_render, additional_tags=("split",))
            self.add_element(self._split_colormap, additional_tags=("split",))

            self.add_element(self._split_invert, additional_tags=("split_colormap",))
            self.add_element(self._split_normalize, additional_tags=("split_colormap",))
            self.add_element(self._split_min, additional_tags=("split_colormap",))
            self.add_element(self._split_max, additional_tags=("split_colormap",))

        with self.server.gui.add_folder("Crop Viewport"):
            self.add_element(self._crop_viewport)
            # Crop options
            self.add_element(self._background_color, additional_tags=("crop",))
            self.add_element(self._crop_center, additional_tags=("crop",))
            self.add_element(self._crop_scale, additional_tags=("crop",))
            self.add_element(self._crop_rot, additional_tags=("crop",))

        self.add_element(self._time, additional_tags=("time",))
        self._reset_camera = server.gui.add_button(
            label="Reset Up Direction",
            icon=viser.Icon.ARROW_BIG_UP_LINES,
            color="gray",
            hint="Set the up direction of the camera orbit controls to the camera's current up direction.",
        )
        self._reset_camera.on_click(self._reset_camera_cb)
        
        # Selection mode controls
        with self.server.gui.add_folder("Pixel Selection"):
            self._selection_mode = ViewerCheckbox(
                "Selection Mode",
                False,
                cb_hook=lambda _: self._selection_mode_cb(),
                hint="Toggle between navigation mode and pixel selection mode",
            )
            self.add_element(self._selection_mode)
            
            # Reset button (always visible, but disabled when no selections)
            self._reset_selection_button = self.server.gui.add_button(
                label="Reset Selection",
                icon=viser.Icon.TRASH,
                color="red",
                hint="Clear all selected pixels",
            )
            self._reset_selection_button.on_click(lambda _: self._reset_selection_cb())
            self._reset_selection_button.disabled = True  # Disabled when no selections
            
            # Create a container for selection list items within the folder
            # This will be updated dynamically but created here to maintain folder context
            self._selection_list_container = self.server.gui.add_folder("Selected Pixels")
            self._selection_list_items: Dict[int, Dict[str, Any]] = {}  # id -> {markdown, remove_button}

    def _train_speed_cb(self) -> None:
        pass

        """Callback for when the train speed is changed"""
        if self._train_speed.value == "Fast":
            self._train_util.value = 0.95
            self._max_res.value = 256
        elif self._train_speed.value == "Mid":
            self._train_util.value = 0.85
            self._max_res.value = 512
        elif self._train_speed.value == "Slow":
            self._train_util.value = 0.5
            self._max_res.value = 1024

    def _reset_camera_cb(self, _) -> None:
        for client in self.server.get_clients().values():
            client.camera.up_direction = vtf.SO3(client.camera.wxyz) @ np.array([0.0, -1.0, 0.0])
    
    def _selection_mode_cb(self) -> None:
        """Callback when selection mode is toggled"""
        if self.viewer is None:
            return
        self.viewer.set_selection_mode(self._selection_mode.value)
        # Update selection list (always visible, regardless of mode)
        self.update_selection_list()
    
    def update_selection_list(self) -> None:
        """Update the selection list UI - always visible regardless of selection mode"""
        if self.viewer is None:
            return
        
        # Remove old list items (including empty state message)
        for item_data in self._selection_list_items.values():
            if "markdown" in item_data and item_data["markdown"] is not None:
                try:
                    item_data["markdown"].remove()
                except Exception:
                    pass  # Element may already be removed
            if "toggle_button" in item_data and item_data["toggle_button"] is not None:
                try:
                    item_data["toggle_button"].remove()
                except Exception:
                    pass  # Element may already be removed
            if "remove_button" in item_data and item_data["remove_button"] is not None:
                try:
                    item_data["remove_button"].remove()
                except Exception:
                    pass  # Element may already be removed
        self._selection_list_items.clear()
        
        # Update reset button state based on whether there are selections
        has_selections = len(self.viewer.selected_pixels) > 0
        self._reset_selection_button.disabled = not has_selections
        
        # Add items within the container folder to maintain proper ordering
        with self._selection_list_container:
            # If no selections, show empty state message
            if not has_selections:
                empty_markdown = self.server.gui.add_markdown(
                    "*No pixels selected. Enable Selection Mode and click on the rendered image to select pixels.*"
                )
                self._selection_list_items[-1] = {"markdown": empty_markdown}  # Use -1 as placeholder ID
                return
            
            # Add items for each selection
            for pixel_data in self.viewer.selected_pixels:
                pixel_id = pixel_data["id"]
                x = pixel_data["x"]
                y = pixel_data["y"]
                is_positive = pixel_data.get("is_positive", True)  # Default to True if not set
                patch_data = pixel_data.get("patch_data", None)
                
                # Convert normalized coordinates to pixel coordinates
                # We need image dimensions - for now use a reasonable default or get from render
                # For display, we'll show normalized coordinates
                x_display = f"{x:.4f}"
                y_display = f"{y:.4f}"
                
                # Create compact display with coordinates
                # Use the absolute simplest markdown - just plain text
                coordinates_text = f"({x_display}, {y_display})"
                
                # Extract pixel color from the rendered image for text display
                pixel_color_rgb = None
                if self.viewer.last_rendered_image is not None:
                    try:
                        h, w = self.viewer.last_rendered_image.shape[:2]
                        pixel_x = int(x * w)
                        pixel_y = int(y * h)
                        if 0 <= pixel_x < w and 0 <= pixel_y < h:
                            color_rgb = self.viewer.last_rendered_image[pixel_y, pixel_x, :]
                            pixel_color_rgb = (int(color_rgb[0]), int(color_rgb[1]), int(color_rgb[2]))
                    except Exception:
                        pass
                
                # Use the simplest possible markdown - just plain text, no formatting
                if pixel_color_rgb:
                    color_text = f"RGB({pixel_color_rgb[0]}, {pixel_color_rgb[1]}, {pixel_color_rgb[2]})"
                    markdown_text = f"Pixel: {coordinates_text} - {color_text}"
                else:
                    markdown_text = f"Pixel: {coordinates_text}"
                
                # Create markdown - this should always work
                # Don't catch exceptions here - let them propagate so we can see what's wrong
                markdown = self.server.gui.add_markdown(markdown_text)
                patch_markdown = None  # No longer using separate patch markdown
                
                # Create toggle button for positive/negative sign
                # Use text label with appropriate color
                toggle_btn = self.server.gui.add_button(
                    label="+" if is_positive else "-",
                    color="green" if is_positive else "red",
                )
                
                def create_toggle_callback(pid: int):
                    def toggle_cb(_):
                        if self.viewer is not None:
                            self.viewer.toggle_pixel_sign(pid)
                            self.update_selection_list()
                    return toggle_cb
                
                toggle_btn.on_click(create_toggle_callback(pixel_id))
                
                # Create remove button
                remove_btn = self.server.gui.add_button(
                    label="Remove",
                    icon=viser.Icon.X,
                    color="gray",
                )
                
                def create_remove_callback(pid: int):
                    def remove_cb(_):
                        if self.viewer is not None:
                            self.viewer.remove_selected_pixel(pid)
                            self.update_selection_list()
                    return remove_cb
                
                remove_btn.on_click(create_remove_callback(pixel_id))
                
                self._selection_list_items[pixel_id] = {
                    "markdown": markdown,
                    "toggle_button": toggle_btn,
                    "remove_button": remove_btn,
                }
    
    def _reset_selection_cb(self) -> None:
        """Callback to reset all selections"""
        if self.viewer is not None:
            self.viewer.clear_selected_pixels()
            self.update_selection_list()
    
    def initialize_selection_list(self) -> None:
        """Initialize the selection list UI on startup"""
        if self.viewer is not None:
            self.update_selection_list()

    def update_output_options(self, new_options: List[str]):
        """
        Args:
            new_options: a list of new output options
        """
        self._output_render.set_options(new_options)
        self._split_output_render.set_options(new_options)
        self._split_output_render.value = new_options[-1]

    def add_element(self, e: ViewerElement, additional_tags: Tuple[str, ...] = tuple()) -> None:
        """Adds an element to the control panel

        Args:
            e: the element to add
            additional_tags: additional tags to add to the element for selection
        """
        self._elements_by_tag["all"].append(e)
        for t in additional_tags:
            self._elements_by_tag[t].append(e)
        e.install(self.server)

    def update_control_panel(self) -> None:
        """
        Sets elements to be hidden or not based on the current state of the control panel
        """
        self._colormap.set_disabled(self.output_render == "rgb")
        for e in self._elements_by_tag["colormap"]:
            e.set_hidden(self.output_render == "rgb")
        for e in self._elements_by_tag["split_colormap"]:
            e.set_hidden(not self._split.value or self.split_output_render == "rgb")
        for e in self._elements_by_tag["crop"]:
            e.set_hidden(not self.crop_viewport)
        self._time.set_hidden(not self._time_enabled)
        self._split_percentage.set_hidden(not self._split.value)
        self._split_output_render.set_hidden(not self._split.value)
        self._split_colormap.set_hidden(not self._split.value)
        self._split_colormap.set_disabled(self.split_output_render == "rgb")
        self._crop_handle.visible = self.crop_viewport

    def update_colormap_options(self, dimensions: int, dtype: type) -> None:
        """update the colormap options based on the current render

        Args:
            dimensions: the number of dimensions of the render
            dtype: the data type of the render
        """
        self._colormap.set_options(_get_colormap_options(dimensions, dtype))

    def update_split_colormap_options(self, dimensions: int, dtype: type) -> None:
        """update the split colormap options based on the current render

        Args:
            dimensions: the number of dimensions of the render
            dtype: the data type of the render
        """
        self._split_colormap.set_options(_get_colormap_options(dimensions, dtype))

    @property
    def output_render(self) -> str:
        """Returns the current output render"""
        return self._output_render.value

    @property
    def split_output_render(self) -> str:
        """Returns the current output for the split render"""
        return self._split_output_render.value

    @property
    def split(self) -> bool:
        """Returns whether the split is enabled"""
        return self._split.value

    @property
    def split_percentage(self) -> float:
        """Returns the percentage of the screen to split"""
        return self._split_percentage.value

    @property
    def train_util(self) -> float:
        """Returns the current train util setting"""
        return self._train_util.value

    @property
    def max_res(self) -> int:
        """Returns the current max res setting"""
        return self._max_res.value

    @property
    def crop_viewport(self) -> bool:
        """Returns the current crop viewport setting"""
        return self._crop_viewport.value

    @crop_viewport.setter
    def crop_viewport(self, value: bool):
        """Sets the crop viewport setting"""
        self._crop_viewport.value = value

    @property
    def crop_obb(self):
        """Returns the current crop obb setting"""
        rxyz = self._crop_rot.value
        R = torch.tensor(vtf.SO3.from_rpy_radians(rxyz[0], rxyz[1], rxyz[2]).as_matrix())
        obb = OrientedBox(R, torch.tensor(self._crop_center.value), torch.tensor(self._crop_scale.value))
        return obb

    @property
    def background_color(self) -> Tuple[int, int, int]:
        """Returns the current background color"""
        return self._background_color.value

    @background_color.setter
    def background_color(self, value: Tuple[int, int, int]):
        """Sets the background color"""
        self._background_color.value = value

    @property
    def time(self) -> float:
        """Returns the current background color"""
        return self._time.value

    @time.setter
    def time(self, value: float):
        """Sets the background color"""
        self._time.value = value

    @property
    def colormap_options(self) -> ColormapOptions:
        """Returns the current colormap options"""
        return ColormapOptions(
            colormap=self._colormap.value,
            normalize=self._normalize.value,
            colormap_min=self._min.value,
            colormap_max=self._max.value,
            invert=self._invert.value,
        )

    @property
    def split_colormap_options(self) -> ColormapOptions:
        """Returns the current colormap options"""
        return ColormapOptions(
            colormap=self._split_colormap.value,
            normalize=self._split_normalize.value,
            colormap_min=self._split_min.value,
            colormap_max=self._split_max.value,
            invert=self._split_invert.value,
        )

    @property
    def layer_depth(self):
        return self._layer_depth.value


def _get_colormap_options(dimensions: int, dtype: type) -> List[Colormaps]:
    """
    Given the number of dimensions and data type, returns a list of available colormap options
    to use with the visualize() function.

    Args:
        dimensions: the number of dimensions of the render
        dtype: the data type of the render
    Returns:
        a list of available colormap options
    """
    colormap_options: List[Colormaps] = []
    if dimensions == 3:
        colormap_options = ["default"]
    if dimensions == 1 and dtype in [torch.float64, torch.float32, torch.float16, torch.bfloat16]:
        colormap_options = [c for c in list(get_args(Colormaps)) if c not in ("default", "pca")]
    if dimensions > 3:
        colormap_options = ["pca"]
    return colormap_options
