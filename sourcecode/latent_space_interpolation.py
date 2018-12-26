""" script for generating samples from a trained model """

import torch as th
import argparse
import matplotlib.pyplot as plt
import os

from torch.backends import cudnn

# define the device for the training script
device = th.device("cuda" if th.cuda.is_available() else "cpu")


def parse_arguments():
    """
    command line arguments parser
    :return: args => parsed command line arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--generator_file", action="store", type=str,
                        help="pretrained weights file for generator", required=True)

    parser.add_argument("--latent_size", action="store", type=int,
                        default=256,
                        help="latent size for the generator")

    parser.add_argument("--depth", action="store", type=int,
                        default=6,
                        help="latent size for the generator")

    parser.add_argument("--time", action="store", type=float,
                        default=5,
                        help="Number of minutes for the video to make")

    parser.add_argument("--std", action="store", type=float, default=1,
                        help="Truncated standard deviation fo the drawn samples")

    parser.add_argument("--traversal_time", action="store", type=float,
                        default=3,
                        help="Number of seconds to go from one point to another")

    parser.add_argument("--static_time", action="store", type=float,
                        default=1,
                        help="Number of seconds to display a sample")

    parser.add_argument("--fps", action="store", type=int,
                        default=30, help="Frames per second in the video")

    parser.add_argument("--out_dir", action="store", type=str,
                        default="interp_animation_frames/",
                        help="path to the output directory for the frames")

    args = parser.parse_args()

    return args


def main(args):
    """
    Main function of the script
    :param args: parsed commandline arguments
    :return: None
    """
    from MSG_GAN.GAN import Generator

    # create generator object:
    print("Creating a generator object ...")
    generator = th.nn.DataParallel(
        Generator(depth=args.depth,
                  latent_size=args.latent_size).to(device))

    # load the trained generator weights
    print("loading the trained generator weights ...")
    generator.load_state_dict(th.load(args.generator_file))

    # total_frames in the video:
    total_time_for_one_transition = args.traversal_time + args.static_time
    total_frames_for_one_transition = (total_time_for_one_transition * args.fps)
    number_of_transitions = int((args.time * 60) / total_time_for_one_transition)
    total_frames = int(number_of_transitions * total_frames_for_one_transition)

    # Let's create the animation video from the latent space interpolation
    # I save the frames required for making the video here
    point_1 = th.randn(1, args.latent_size).to(device) * args.std

    # create output directory
    os.makedirs(args.out_dir, exist_ok=True)

    # Run the main loop for the interpolation:
    global_frame_counter = 1  # counts number of frames
    while global_frame_counter <= total_frames:
        point_2 = th.randn(1, args.latent_size).to(device) * args.std
        direction = point_2 - point_1

        # create the points for images in this space:
        number_of_points = int(args.traversal_time * args.fps)
        for i in range(number_of_points):
            point = point_1 + ((direction / number_of_points) * i)

            # generate the image for this point:
            generator.load_state_dict(th.load(args.generator_file))
            img = th.squeeze(generator(point)[-1].detach(), dim=0).permute(1, 2, 0) / 2 + 0.5

            # save the image:
            plt.imsave(os.path.join(args.out_dir, str(global_frame_counter) + ".png"), img)

            # increment the counter:
            global_frame_counter += 1

        # at point_2, now add static frames:
        generator.load_state_dict(th.load(args.generator_file))
        img = th.squeeze(generator(point_2)[-1].detach(), dim=0).permute(1, 2, 0) / 2 + 0.5

        # now save the same image a number of times:
        for _ in range(args.static_time * args.fps):
            plt.imsave(os.path.join(args.out_dir, str(global_frame_counter) + ".png"), img)
            global_frame_counter += 1

        # set the point_1 := point_2
        point_1 = point_2

        print("Generated %d frames ..." % global_frame_counter)

    # video frames have been generated
    print("Video frames have been generated at:", args.out_dir)


if __name__ == "__main__":
    main(parse_arguments())
