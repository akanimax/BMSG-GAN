""" Script to show any single image from the dataset
It is not required usually if the dataset is in Image
It is useful to view the CelebA-HQ npz files :) """

import argparse
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


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
                        help="Whether it is an npz file or not", required=True)

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
        img = img.squeeze(0).transpose(1, 2, 0)
    else:
        img = np.array(Image.open(img_file))

    # show the image on screen:
    plt.figure().suptitle(args.image_path.split("/")[-1])
    plt.imshow(img)
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    plt.show()


if __name__ == '__main__':
    main(parse_arguments())
