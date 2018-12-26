""" demo script for sampling the trained model"""

import argparse

import numpy as np
import torch as th
import os
import matplotlib.pyplot as plt
from torch.backends import cudnn

# define the device for the training script
device = th.device("cuda" if th.cuda.is_available() else "cpu")

# enable fast training
cudnn.benchmark = True


def parse_arguments():
    """
    command line arguments parser
    :return: args => parsed command line arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--generator_file", action="store", type=str,
                        default="models/Celeba/3/GAN_GEN_5.pth",
                        help="pretrained weights file for generator")

    parser.add_argument("--output_dir", action="store", type=str,
                        default="samples/generated_samples/3",
                        help="path for the generated samples directory")

    parser.add_argument("--depth", action="store", type=int,
                        default=6,
                        help="Depth of the GAN")

    parser.add_argument("--latent_size", action="store", type=int,
                        default=256,
                        help="latent size for the generator")

    parser.add_argument("--num_samples", action="store", type=int,
                        default=36,
                        help="number of samples to generate for creating the grid" +
                             " should be a square number preferably")

    parser.add_argument("--show_samples", action="store", type=bool,
                        default=False,
                        help="Whether to show the generated samples in windows")

    args = parser.parse_args()

    return args


def main(args):
    """
    Main function for the script
    :param args: parsed command line arguments
    :return: None
    """
    from MSG_GAN.TeacherGAN import Generator, TeacherGAN
    from torch.nn import DataParallel

    # create a generator:
    msg_gan_generator = Generator(depth=args.depth, latent_size=args.latent_size).to(device)

    if device == th.device("cuda"):
        msg_gan_generator = DataParallel(msg_gan_generator)

    if args.generator_file is not None:
        # load the weights into generator
        msg_gan_generator.load_state_dict(th.load(args.generator_file))

    print("Loaded Generator Configuration: ")
    print(msg_gan_generator)

    # generate all the samples in a list of lists:
    samples = []  # start with an empty list
    for _ in range(args.num_samples):
        gen_samples = msg_gan_generator(th.randn(1, args.latent_size))
        samples.append(gen_samples)

        if args.show_samples:
            for gen_sample in gen_samples:
                plt.figure()
                plt.imshow(th.squeeze(gen_sample.detach()).permute(1, 2, 0) / 2 + 0.5)
            plt.show()

    # create a grid of the generated samples:
    file_names = []  # initialize to empty list
    for res_val in range(args.depth):
        res_dim = np.power(2, res_val + 2)
        file_name = os.path.join(args.output_dir,
                                 str(res_dim) + "_" + str(res_dim) + ".png")
        file_names.append(file_name)

    images = list(map(lambda x: th.cat(x, dim=0), zip(*samples)))
    TeacherGAN.create_grid(images, file_names)

    print("samples have been generated. Please check:", args.output_dir)


if __name__ == '__main__':
    # invoke the main function of the script
    main(parse_arguments())
