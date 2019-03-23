""" !!! Not fully ready !!! """
""" Work in progress demo utility for showing 
live (realtime) latent space interpolations of trained models """


import torch as th
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from MSG_GAN.GAN import Generator
from generate_multi_scale_samples import progressive_upscaling
from torchvision.utils import make_grid
from math import ceil, sqrt

# ==========================================================================
# Tweakable parameters
# ==========================================================================
generator_file_path = "models/celebahq_testing/GAN_GEN_SHADOW_720.pth"
depth = 8
latent_size = 512
num_points = 30
transition_points = 15
# ==========================================================================

# create the device for running the demo:
device = th.device("cuda" if th.cuda.is_available() else "cpu")

# load the model for the demo
gen = th.nn.DataParallel(Generator(depth=depth + 1, latent_size=latent_size))
gen.load_state_dict(th.load(generator_file_path, map_location=str(device)))


# function to generate an image given a latent_point
def get_image(point):
    images = list(map(lambda x: x.detach(), gen(point)))
    images = progressive_upscaling(images)
    images = list(map(lambda x: x.squeeze(dim=0), images))
    image = make_grid(
        images,
        nrow=int(ceil(sqrt(len(images)))),
        normalize=True,
        scale_each=True
    )
    return image.cpu().numpy().transpose(1, 2, 0)


# generate the set of points:
fixed_points = th.randn(num_points, 512).to(device)
# fixed_points = (fixed_points / fixed_points.norm(dim=1, keepdim=True)) * (512 ** 0.5)
points = []  # start with an empty list
for i in range(len(fixed_points) - 1):
    pt_1 = fixed_points[i].view(1, -1)
    pt_2 = fixed_points[i + 1].view(1, -1)
    direction = pt_2 - pt_1
    for j in range(transition_points):
        pt = pt_1 + ((direction / transition_points) * j)
        # pt = (pt / pt.norm()) * (512 ** 0.5)
        points.append(pt)
    # also append the final point:
    points.append(pt_2)

start_point = points[0]
points = points[1:]

fig, ax = plt.subplots()
plt.axis("off")
shower = plt.imshow(get_image(start_point))


def init():
    return shower,


def update(point):
    shower.set_data(get_image(point))
    return shower,


ani = FuncAnimation(fig, update, frames=points,
                    init_func=init, blit=False)
plt.show()
