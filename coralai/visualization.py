import time
import torch
import taichi as ti
from dataclasses import dataclass
from typing import Callable, List, Tuple
from .substrate.substrate import Substrate

@ti.dataclass
class VisualizationTiData:
    """Vis data to be sent to Taichi Kernels"""
    mem_w: ti.i32
    mem_h: ti.i32
    img_w: ti.i32
    img_h: ti.i32
    scale: ti.i32


@dataclass
class VisualizationData:
    """"""
    ti_data: VisualizationTiData
    substrate: Substrate
    chinds: torch.Tensor
    name: str
    image: ti.types.ndarray()
    window: ti.ui.Window
    canvas: ti.ui.Canvas
    gui: ti.ui.Gui
    escaped: bool

    def __init__(self, substrate: Substrate, chids: list = None, window_w=800, name="Default Visualization"):
        self.substrate = substrate

        max_dim = max(self.substrate.w, self.substrate.h)
        desired_max_dim = window_w
        scale = desired_max_dim // max_dim

        self.ti_data = VisualizationTiData(
            mem_w = substrate.w,
            mem_h = substrate.h,
            img_w = substrate.w * scale,
            img_h = substrate.h * scale,
            scale = scale
        )
        self.name = name
        if not chids:
            self.chinds = torch.tensor([0,1,2], device = substrate.torch_device)
        else:
            self.chinds = torch.tensor(list(substrate.get_inds_tivec(chids)), device = substrate.torch_device)

        self.image = ti.Vector.field(n=3, dtype=ti.f32, shape=(self.ti_data.img_w, self.ti_data.img_h))

        self.window = ti.ui.Window(
            f"{self.name}", (self.ti_data.img_w, self.ti_data.img_h), fps_limit=200, vsync=True
        )

        self.canvas = self.window.get_canvas()
        self.gui = self.window.get_gui()
        self.escaped = False


@ti.dataclass
class DrawableTiData:
    brush_radius: ti.i32
    channel_to_paint: ti.i32
    val_to_paint: ti.f32
    pos_x: ti.f32
    pos_y: ti.f32


@dataclass
class DrawableVisData(VisualizationData):
    draw_ti_data: DrawableTiData
    drawing: bool
    prev_time: float
    prev_pos: Tuple[float, float]

    def __init__(self, substrate: Substrate, chids: list = None, window_w=800, name="Default Visualization"):
        super().__init__(substrate, chids, window_w, name)
        self.drawing = False
        self.prev_time = time.time()
        self.prev_pos = self.window.get_cursor_pos()
        self.draw_ti_data = DrawableTiData(brush_radius=4, channel_to_paint=0, val_to_paint=0.1,
                                           pos_x=self.prev_pos[0], pos_y=self.prev_pos[1])


@ti.kernel
def write_to_renderer(vis_data: VisualizationTiData, mem: ti.types.ndarray(), image: ti.template(),
                      max_vals: ti.types.ndarray(), chinds: ti.types.ndarray()):
    for i, j in image:
        xind = (i//vis_data.scale) % vis_data.mem_w
        yind = (j//vis_data.scale) % vis_data.mem_h
        for k in ti.static(range(3)):
            chid = chinds[k]
            image[i, j][k] = mem[0, chid, xind, yind] / max_vals[k]


@ti.kernel
def add_val_to_loc(
        draw_data: DrawableTiData,
        vis_data: VisualizationTiData,
        mem: ti.types.ndarray()
    ):
    ind_x = int(draw_data.pos_x * vis_data.mem_w)
    ind_y = int(draw_data.pos_y * vis_data.mem_h)
    # offset = int(drawData.pos_x) * 3
    for i, j in ti.ndrange((-draw_data.brush_radius, draw_data.brush_radius), (-draw_data.brush_radius, draw_data.brush_radius)):
        if (i**2) + j**2 < draw_data.brush_radius**2:
            mem[0, draw_data.channel_to_paint, (i + ind_x) % vis_data.mem_w, (j + ind_y) % vis_data.mem_h] += draw_data.val_to_paint


def add_channel_controls(vis_data: VisualizationData, sub_window):
    """Adds sliders for each channel to the subwindow."""
    vis_data.chinds[0] = sub_window.slider_int(
        f"R: {vis_data.substrate.index_to_chname(vis_data.chinds[0])}", 
        vis_data.chinds[0], 0, vis_data.substrate.mem.shape[1]-1)
    vis_data.chinds[1] = sub_window.slider_int(
        f"G: {vis_data.substrate.index_to_chname(vis_data.chinds[1])}", 
        vis_data.chinds[1], 0, vis_data.substrate.mem.shape[1]-1)
    vis_data.chinds[2] = sub_window.slider_int(
        f"B: {vis_data.substrate.index_to_chname(vis_data.chinds[2])}", 
        vis_data.chinds[2], 0, vis_data.substrate.mem.shape[1]-1)


def render_sub_window(vis_data: VisualizationData,
                      content_adders: List[Callable[..., None]],
                      x = 0.05,
                      y = 0.05,
                      w = 200,
                      h = 300):
    """Renders a subwindow in the Taichi visualization containing the content specified by the content_adders."""
    vis_data.canvas.set_background_color((1, 1, 1))
    subw_w = min(500/vis_data.ti_data.img_w, vis_data.ti_data.img_w)
    subw_h = min(200/vis_data.ti_data.img_h, vis_data.ti_data.img_h)
    with vis_data.gui.sub_window("Options", x, y, subw_w, subw_h) as sub_w:
        for add_content in content_adders:
            add_content(vis_data, sub_w)


def render_default_window(vis_data: VisualizationData):
    return render_sub_window(vis_data, [add_channel_controls])


def check_default_events(vis_data: VisualizationData):
    for e in vis_data.window.get_events(ti.ui.PRESS):
        if e.key in [ti.ui.ESCAPE]:
            vis_data.escaped = True


def check_drawable_events(draw_data: DrawableVisData):
    for e in draw_data.window.get_events(ti.ui.PRESS):
        if e.key == ti.ui.LMB and draw_data.window.is_pressed(ti.ui.SHIFT):
            draw_data.drawing = True
        elif e.key == ti.ui.SPACE:
            draw_data.substrate.mem *= 0.0
    for e in draw_data.window.get_events(ti.ui.RELEASE):
        if e.key == ti.ui.LMB:
            draw_data.drawing = False


def compose_visualization(vis_data: VisualizationData, subwindow_adders: List[Callable[..., None]] = None,
                          event_handlers: List[Callable[..., None]] = None):
    """Creates an update method which renders the substrate, subwindows and handles events.

    Args:
        vis_data: VisualizationData
        render_sub_window: (optional) renders a subwindow in the visualization. See render_default_window for an example.
        event_handlers: (optional) List of event handlers. See check_default_events for an example (always appended).
    """
    if event_handlers is None:
        event_handlers = []
    event_handlers += [check_default_events]
    def update():
        for event_checker in event_handlers:
            event_checker(vis_data)

        render_sub_window(vis_data, subwindow_adders)

        max_vals = torch.tensor([vis_data.substrate.mem[0, ch].max() for ch in vis_data.chinds])
        write_to_renderer(vis_data.ti_data, vis_data.substrate.mem, vis_data.image, max_vals, vis_data.chinds)
        vis_data.canvas.set_image(vis_data.image)
        vis_data.window.show()
    return update


def compose_drawable_vis(draw_data: DrawableVisData, render_sub_window: Callable[..., None] = None,
                         event_checkers: List[Callable[..., None]] = None):
    if event_checkers is None:
        event_checkers = []
    event_checkers += [check_drawable_events, check_default_events]
    default_update = compose_visualization(draw_data, render_sub_window, event_checkers)
    def update():
        default_update()
        current_time = time.time()
        current_pos = draw_data.window.get_cursor_pos()
        draw_data.ti_data.pos_x = current_pos[0]
        draw_data.ti_data.pos_y = current_pos[1]
        if draw_data.drawing and ((current_time - draw_data.prev_time) > 0.1): # or (current_pos != draw_data.prev_pos)):
            add_val_to_loc(draw_data.draw_ti_data, draw_data.ti_data, draw_data.substrate.mem)
    return update
