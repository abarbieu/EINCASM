import torch
import taichi as ti
import numpy as np
import matplotlib.pyplot as plt

ti.init(ti.metal)

window = ti.ui.Window(
            f"asd", (100, 100), fps_limit=200, vsync=True
        )

canvas = window.get_canvas()
gui = window.get_gui()

exit()
@ti.dataclass
class VisData:
    chinds: ti.f32
    scale: ti.i32

@ti.dataclass
class DrawableData:
    brush_radius: ti.i32
    channel_to_paint: ti.i32
    val_to_paint: ti.f32

@ti.dataclass
class DrawableVisData:
    visData: VisData
    drawData: DrawableData

    def __init__(self):
        self.visData = VisData(chinds=3, scale=1)
        self.drawData = DrawableData(brush_radius=1, channel_to_paint=0, val_to_paint=1)

@ti.kernel
def test_v(visdata: VisData) -> VisData:
    visdata.chinds = 1
    return visdata

v = VisData(chinds=3, scale=1)
print(test_v(v))

@ti.kernel
def tst_dv(visdata: DrawableVisData) -> DrawableVisData:
    visdata.visData.chinds = 1
    return visdata

dv = DrawableVisData()
print(tst_dv(dv))

exit()

t = torch.rand(1,1,4,4) - 0.5
print(t)
def ch_norm(input_tensor):
    # Calculate the mean across batch and channel dimensions
    mean = input_tensor.mean(dim=(0, 2, 3), keepdim=True)

    # Calculate the variance across batch and channel dimensions
    var = input_tensor.var(dim=(0, 2, 3), keepdim=True, unbiased=False)

    # Normalize the input tensor
    input_tensor.sub_(mean).div_(torch.sqrt(var + 1e-5))

    return input_tensor

print(ch_norm(t))

exit()

@ti.kernel
def a(mem: ti.types.ndarray()):
    for ox, oy, k in ti.ndrange((-1,1), (-1,1), (0,1)):
        x = (2+ox) % mem.shape[2]
        y = (2+oy) % mem.shape[3]
        mem[0, k, x,y] *= 0
        mem[0, k, x,y] += 99
a(t)

print(t)

exit()
t = torch.tensor([[[[0.0,0,1],
                   [0,10,0],
                   [0,0,1]],
                  [[1,0,3],
                   [3,0,0],
                   [2,0,10]],
                  [[0,0,0],
                   [0,100,0],
                   [0,0,0]]
                  ]])
print(torch.softmax(t[0,[0,1]],dim=1))



exit()



# Predefined sequence of offsets in a clockwise order starting from [1, 0]
dir_kernel = torch.tensor([[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]])

def produce_order(kernel_len):
    order = []
    ind = 0
    i = 0
    while i < kernel_len:
        order.append(ind)
        if ind > 0:
            ind = -ind
        else:
            ind = (-ind + 1)
        i += 1
    return order

dir_order = produce_order(dir_kernel.shape[0])
print(dir_order)
kernel_len = len(dir_kernel)
rot = 3
max_act_i = 2

# Set up the plot
fig, ax = plt.subplots()
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
import matplotlib.cm as cm

# Plot the origin
ax.plot(0, 0, 'ok')  # 'ok' plots a black dot

colors = cm.rainbow(np.linspace(0, 1, dir_kernel.shape[0]))

print(f"Rot: {rot}, Dir: {dir_kernel[rot]}")
print(f"max_act_i: {max_act_i}, max_act_rot: {(rot + dir_order[max_act_i]) % dir_kernel.shape[0]}," +
      f"max_act_dir: {dir_kernel[(rot + dir_order[max_act_i]) % dir_kernel.shape[0]]}")
for i in range(dir_kernel.shape[0]):
    ind = (rot+dir_order[i]) % dir_kernel.shape[0]
    print(f"i {i}, dir_ind: {ind}, dir: {dir_kernel[ind]}")
    dx, dy = dir_kernel[ind].numpy()
    color = colors[i]
    ax.arrow(0, 0, dx, dy, head_width=0.1, head_length=0.2, fc=color, ec=color)
    ax.text(dx * 1.1, dy * 1.1, str(i), color='black')
plt.grid(True)
plt.show()
exit()



# Function to map a given direction to its starting index in the offset_sequence
def get_starting_index_for_direction(direction):
    # Calculate angle for the given direction
    angle = np.arctan2(direction[1], direction[0])
    
    # Normalize the angle to be within [0, 2*pi)
    if angle < 0:
        angle += 2 * np.pi

    # Each segment in the circle corresponds to 45 degrees (pi/4 radians)
    # Calculate the index by dividing the angle by the segment size
    index = int(round(4 * angle / np.pi)) % len(offset_sequence)
    return index

def get_alternating_offsets(offsets):
    i = 0
    new_offsets = []
    ind = 0
    while i < len(offsets):
        new_offsets.append(offsets[ind])
        if ind >= 0:
            ind = -(ind + 1)
        else:
            ind = -ind
        i += 1
    return new_offsets

new_dir = [-1,-1]
starting_index = get_starting_index_for_direction(new_dir)

def rotate_sequence(starting_index, sequence):
    return sequence[starting_index:] + sequence[:starting_index]

rotated_sequence = rotate_sequence(starting_index, offset_sequence)

print(offset_sequence)
print(f"new_dir: {new_dir},\nstarting_index: {starting_index},\nrotated_sequence: {rotated_sequence}")
print(get_alternating_offsets(rotated_sequence))
