""" Utility to downsample real images (half every resolution)
Used to create the images shown in /diagrams/images folder """

import argparse
import os
import numpy as np
import torch as th
from PIL import Image
from scipy.misc import imsave
from torch.nn.functional import avg_pool2d
from generate_multi_scale_samples import progressive_upscaling


def parse_arguments():
    """
    default command line argument parser
    :return: args => parsed command line arguments
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("--image_path", action="store", type=str,
                        help="path to the image",
                        default="./", required=True)

    parser.add_argument("--npz_file", action="store", type=bool,
                        default=True,
                        help="Whether it is an npz file or not")

    parser.add_argument("--out_dir", action="store", type=str,
                        default="../diagrams/images",
                        help="directory to save the generated images")

    args = parser.parse_args()

    return args


def main(args):
    """
    Main function for the script
    :param args: parsed command line arguments
    :return: None
    """

    img_file = os.path.join(args.image_path)
    if args.npz_file:
        img = np.load(img_file)
    else:
        img = np.array(Image.open(img_file))
        img = np.expand_dims(img, axis=0).transpose(0, 3, 1, 2)

    img = th.from_numpy(img).float()  # make a tensor out of image

    # progressively downsample the image and save it at the given location
    dim = img.shape[-1]  # gives the highest dimension
    ds_img = img  # start from the highest resolution image
    images = [ds_img]  # original image already in
    while dim > 4:
        ds_img = avg_pool2d(ds_img, kernel_size=2, stride=2)
        images.append(ds_img)
        dim //= 2

    images = progressive_upscaling(list(reversed(images)))

    # save the images:
    for count in range(len(images)):
        imsave(os.path.join(args.out_dir, str(count + 1) + ".png"),
               images[count].squeeze(0).permute(1, 2, 0))


if __name__ == '__main__':
    main(parse_arguments())
