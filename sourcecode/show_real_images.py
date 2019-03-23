""" One by one show real images in the CelebA-HQ npy files' directory
    Very useful for analyzing the data. press Q to move to the next image """

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

    parser.add_argument("--images_path", action="store", type=str,
                        help="path to the directory containing the images",
                        default="./", required=True)

    parser.add_argument("--npz_files", action="store", type=bool,
                        default=True,
                        help="Whether it contains npz files or not", required=True)

    args = parser.parse_args()

    return args


def main(args):
    """
    Main function for the script
    :param args: parsed command line arguments
    :return: None
    """
    # go over the image files in the directory
    for img_file_name in os.listdir(args.images_path):

        img_file = os.path.join(args.images_path, img_file_name)
        if args.npz_files:
            img = np.load(img_file)
            img = img.squeeze(0).transpose(1, 2, 0)
        else:
            img = np.array(Image.open(img_file))

        # show the image on screen:
        plt.figure().suptitle(img_file_name)
        plt.imshow(img)
        mng = plt.get_current_fig_manager()
        mng.resize(*mng.window.maxsize())
        plt.show()


if __name__ == '__main__':
    main(parse_arguments())
