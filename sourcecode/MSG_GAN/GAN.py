""" Module implementing GAN which will be trained using the Progressive growing
    technique -> https://arxiv.org/abs/1710.10196
"""
import datetime
import os
import time
import timeit
import numpy as np
import torch as th


class Generator(th.nn.Module):
    """ Generator of the GAN network """

    def __init__(self, depth=7, latent_size=512, dilation=1, use_spectral_norm=True):
        """
        constructor for the Generator class
        :param depth: required depth of the Network
        :param latent_size: size of the latent manifold
        :param dilation: amount of dilation to be used by the 3x3 convs
                         in the Generator module.
        :param use_spectral_norm: whether to use spectral normalization
        """
        from torch.nn import ModuleList, Conv2d
        from MSG_GAN.CustomLayers import GenGeneralConvBlock, GenInitialBlock

        super().__init__()

        assert latent_size != 0 and ((latent_size & (latent_size - 1)) == 0), \
            "latent size not a power of 2"
        if depth >= 4:
            assert latent_size >= np.power(2, depth - 4), "latent size will diminish to zero"

        # state of the generator:
        self.depth = depth
        self.latent_size = latent_size
        self.spectral_norm_mode = None
        self.dilation = dilation

        # register the modules required for the GAN Below ...
        # create the ToRGB layers for various outputs:
        def to_rgb(in_channels):
            return Conv2d(in_channels, 3, (1, 1), bias=True)

        # create a module list of the other required general convolution blocks
        self.layers = ModuleList([GenInitialBlock(self.latent_size)])
        self.rgb_converters = ModuleList([to_rgb(self.latent_size)])

        # create the remaining layers
        for i in range(self.depth - 1):
            if i <= 2:
                layer = GenGeneralConvBlock(self.latent_size, self.latent_size,
                                            dilation=dilation)
                rgb = to_rgb(self.latent_size)
            else:
                layer = GenGeneralConvBlock(
                    int(self.latent_size // np.power(2, i - 3)),
                    int(self.latent_size // np.power(2, i - 2)),
                    dilation=dilation
                )
                rgb = to_rgb(int(self.latent_size // np.power(2, i - 2)))
            self.layers.append(layer)
            self.rgb_converters.append(rgb)

        # if spectral normalization is on:
        if use_spectral_norm:
            self.turn_on_spectral_norm()

    def turn_on_spectral_norm(self):
        """
        private helper for turning on the spectral normalization
        :return: None (has side effect)
        """
        from torch.nn.utils import spectral_norm

        if self.spectral_norm_mode is not None:
            assert self.spectral_norm_mode is False, \
                "can't apply spectral_norm. It is already applied"

        # apply the same to the remaining relevant blocks
        for module in self.layers:
            module.conv_1 = spectral_norm(module.conv_1)
            module.conv_2 = spectral_norm(module.conv_2)

        # toggle the state variable:
        self.spectral_norm_mode = True

    def turn_off_spectral_norm(self):
        """
        private helper for turning off the spectral normalization
        :return: None (has side effect)
        """
        from torch.nn.utils import remove_spectral_norm

        if self.spectral_norm_mode is not None:
            assert self.spectral_norm_mode is True, \
                "can't remove spectral_norm. It is not applied"

        # remove the applied spectral norm
        for module in self.layers:
            remove_spectral_norm(module.conv_1)
            remove_spectral_norm(module.conv_2)

        # toggle the state variable:
        self.spectral_norm_mode = False

    def forward(self, x):
        """
        forward pass of the Generator
        :param x: input noise
        :return: *y => output of the generator at various scales
        """
        from torch import tanh
        outputs = []  # initialize to empty list

        y = x  # start the computational pipeline
        for block, converter in zip(self.layers, self.rgb_converters):
            y = block(y)
            outputs.append(tanh(converter(y)))

        return outputs


class Discriminator(th.nn.Module):
    """ Discriminator of the GAN """

    def __init__(self, depth=7, feature_size=512, dilation=1, use_spectral_norm=True):
        """
        constructor for the class
        :param depth: total depth of the discriminator
                       (Must be equal to the Generator depth)
        :param feature_size: size of the deepest features extracted
                             (Must be equal to Generator latent_size)
        :param dilation: amount of dilation to be applied to
                         the 3x3 convolutional blocks of the discriminator
        :param use_spectral_norm: whether to use spectral_normalization
        """
        from torch.nn import ModuleList
        from MSG_GAN.CustomLayers import DisGeneralConvBlock, DisFinalBlock
        from torch.nn import Conv2d

        super().__init__()

        assert feature_size != 0 and ((feature_size & (feature_size - 1)) == 0), \
            "latent size not a power of 2"
        if depth >= 4:
            assert feature_size >= np.power(2, depth - 4), \
                "feature size cannot be produced"

        # create state of the object
        self.depth = depth
        self.feature_size = feature_size
        self.spectral_norm_mode = None
        self.dilation = dilation

        # create the fromRGB layers for various inputs:
        def from_rgb(out_channels):
            return Conv2d(3, out_channels, (1, 1), bias=True)

        self.rgb_to_features = ModuleList([from_rgb(self.feature_size // 2)])

        # create a module list of the other required general convolution blocks
        self.layers = ModuleList([DisFinalBlock(self.feature_size)])

        # create the remaining layers
        for i in range(self.depth - 1):
            if i > 2:
                layer = DisGeneralConvBlock(
                    int(self.feature_size // np.power(2, i - 2)),
                    int(self.feature_size // np.power(2, i - 2)),
                    dilation=dilation
                )
                rgb = from_rgb(int(self.feature_size // np.power(2, i - 1)))
            else:
                layer = DisGeneralConvBlock(self.feature_size, self.feature_size // 2,
                                            dilation=dilation)
                rgb = from_rgb(self.feature_size // 2)

            self.layers.append(layer)
            self.rgb_to_features.append(rgb)

        # just replace the last converter
        self.rgb_to_features[self.depth - 1] = \
            from_rgb(self.feature_size // np.power(2, i - 2))

        # if spectral normalization is on:
        if use_spectral_norm:
            self.turn_on_spectral_norm()

    def turn_on_spectral_norm(self):
        """
        private helper for turning on the spectral normalization
        :return: None (has side effect)
        """
        from torch.nn.utils import spectral_norm

        if self.spectral_norm_mode is not None:
            assert self.spectral_norm_mode is False, \
                "can't apply spectral_norm. It is already applied"

        # apply the same to the remaining relevant blocks
        for module in self.layers:
            module.conv_1 = spectral_norm(module.conv_1)
            module.conv_2 = spectral_norm(module.conv_2)

        # toggle the state variable:
        self.spectral_norm_mode = True

    def turn_off_spectral_norm(self):
        """
        private helper for turning off the spectral normalization
        :return: None (has side effect)
        """
        from torch.nn.utils import remove_spectral_norm

        if self.spectral_norm_mode is not None:
            assert self.spectral_norm_mode is True, \
                "can't remove spectral_norm. It is not applied"

        # remove the applied spectral norm
        for module in self.layers:
            remove_spectral_norm(module.conv_1)
            remove_spectral_norm(module.conv_2)

        # toggle the state variable:
        self.spectral_norm_mode = False

    def forward(self, inputs):
        """
        forward pass of the discriminator
        :param inputs: (multi-scale input images) to the network list[Tensors]
        :return: out => raw prediction values
        """

        assert len(inputs) == self.depth, \
            "Mismatch between input and Network scales"

        y = self.rgb_to_features[self.depth - 1](inputs[self.depth - 1])
        y = self.layers[self.depth - 1](y)
        for x, block, converter in \
                zip(reversed(inputs[:-1]),
                    reversed(self.layers[:-1]),
                    reversed(self.rgb_to_features[:-1])):
            input_part = converter(x)  # convert the input:
            y = th.cat((input_part, y), dim=1)  # concatenate the inputs:
            y = block(y)  # apply the block

        return y


class MSG_GAN:
    """ Unconditional TeacherGAN

        args:
            depth: depth of the GAN (will be used for each generator and discriminator)
            latent_size: latent size of the manifold used by the GAN
            gen_dilation: amount of dilation for generator
            dis_dilation: amount of dilation for discriminator
            use_spectral_norm: whether to use spectral normalization to the convolutional
                               blocks.
            device: device to run the GAN on (GPU / CPU)
    """

    def __init__(self, depth=7, latent_size=512, gen_dilation=1,
                 dis_dilation=1, use_spectral_norm=True, device=th.device("cpu")):
        """ constructor for the class """
        from torch.nn import DataParallel

        self.gen = Generator(depth, latent_size, dilation=gen_dilation,
                             use_spectral_norm=use_spectral_norm).to(device)
        self.dis = Discriminator(depth, latent_size, dilation=dis_dilation,
                                 use_spectral_norm=use_spectral_norm).to(device)

        # Create the Generator and the Discriminator
        if device == th.device("cuda"):
            self.gen = DataParallel(self.gen)
            self.dis = DataParallel(self.dis)

        # state of the object
        self.latent_size = latent_size
        self.depth = depth
        self.device = device

        # by default the generator and discriminator are in eval mode
        self.gen.eval()
        self.dis.eval()

    def generate_samples(self, num_samples):
        """
        generate samples using this gan
        :param num_samples: number of samples to be generated
        :return: generated samples tensor: list[ Tensor(B x H x W x C)]
        """
        noise = th.randn(num_samples, self.latent_size).to(self.device)
        generated_images = self.gen(noise)

        # reshape the generated images
        generated_images = list(map(lambda x: (x.detach().permute(0, 2, 3, 1) / 2) + 0.5,
                                    generated_images))

        return generated_images

    def optimize_discriminator(self, dis_optim, noise, real_batch, loss_fn):
        """
        performs one step of weight update on discriminator using the batch of data
        :param dis_optim: discriminator optimizer
        :param noise: input noise of sample generation
        :param real_batch: real samples batch
                           should contain a list of tensors at different scales
        :param loss_fn: loss function to be used (object of GANLoss)
        :return: current loss
        """

        # generate a batch of samples
        fake_samples = self.gen(noise)
        fake_samples = list(map(lambda x: x.detach(), fake_samples))

        loss = loss_fn.dis_loss(real_batch, fake_samples)

        # optimize discriminator
        dis_optim.zero_grad()
        loss.backward()
        dis_optim.step()

        return loss.item()

    def optimize_generator(self, gen_optim, noise, real_batch, loss_fn):
        """
        performs one step of weight update on generator using the batch of data
        :param gen_optim: generator optimizer
        :param noise: input noise of sample generation
        :param real_batch: real samples batch
                           should contain a list of tensors at different scales
        :param loss_fn: loss function to be used (object of GANLoss)
        :return: current loss
        """

        # generate a batch of samples
        fake_samples = self.gen(noise)

        loss = loss_fn.gen_loss(real_batch, fake_samples)

        # optimize discriminator
        gen_optim.zero_grad()
        loss.backward()
        gen_optim.step()

        return loss.item()

    @staticmethod
    def create_grid(samples, img_files):
        """
        utility function to create a grid of GAN samples
        :param samples: generated samples for storing list[Tensors]
        :param img_files: list of names of files to write
        :return: None (saves multiple files)
        """
        from torchvision.utils import save_image
        from numpy import sqrt
        
        # save the images:
        for sample, img_file in zip(samples, img_files):
            sample = th.clamp((sample.detach() / 2) + 0.5, min=0, max=1)
            save_image(sample, img_file, nrow=int(sqrt(sample.shape[0])))

    def train(self, data, gen_optim, dis_optim, loss_fn,
              start=1, num_epochs=12, feedback_factor=10, checkpoint_factor=1,
              data_percentage=100, num_samples=64,
              log_dir=None, sample_dir="./samples",
              save_dir="./models"):

        # TODOcomplete write the documentation for this method
        # no more procrastination ... HeHe
        """
        Method for training the network
        :param data: pytorch dataloader which iterates over images
        :param gen_optim: Optimizer for generator.
                          please wrap this inside a Scheduler if you want to
        :param dis_optim: Optimizer for discriminator.
                          please wrap this inside a Scheduler if you want to
        :param loss_fn: Object of GANLoss
        :param start: starting epoch number
        :param num_epochs: total number of epochs to run for (ending epoch number)
                           note this is absolute and not relative to start
        :param feedback_factor: number of logs generated and samples generated
                                during training per epoch
        :param checkpoint_factor: save model after these many epochs
        :param data_percentage: amount of data to be used
        :param num_samples: number of samples to be drawn for feedback grid
        :param log_dir: path to directory for saving the loss.log file
        :param sample_dir: path to directory for saving generated samples' grids
        :param save_dir: path to directory for saving the trained models
        :return: None (writes multiple files to disk)
        """

        from torch.nn.functional import avg_pool2d

        # turn the generator and discriminator into train mode
        self.gen.train()
        self.dis.train()

        assert isinstance(gen_optim, th.optim.Optimizer), \
            "gen_optim is not an Optimizer"
        assert isinstance(dis_optim, th.optim.Optimizer), \
            "dis_optim is not an Optimizer"

        print("Starting the training process ... ")

        # create fixed_input for debugging
        fixed_input = th.randn(num_samples, self.latent_size).to(self.device)

        # create a global time counter
        global_time = time.time()

        for epoch in range(start, num_epochs + 1):
            start = timeit.default_timer()  # record time at the start of epoch

            print("\nEpoch: %d" % epoch)
            total_batches = len(iter(data))

            limit = int((data_percentage / 100) * total_batches)

            for (i, batch) in enumerate(data, 1):

                # extract current batch of data for training
                images = batch.to(self.device)
                extracted_batch_size = images.shape[0]

                # create a list of downsampled images from the real images:
                images = [images] + [avg_pool2d(images, int(np.power(2, i)))
                                     for i in range(1, self.depth)]
                images = list(reversed(images))

                gan_input = th.randn(
                    extracted_batch_size, self.latent_size).to(self.device)

                # optimize the discriminator:
                dis_loss = self.optimize_discriminator(dis_optim, gan_input,
                                                       images, loss_fn)

                # optimize the generator:
                # resample from the latent noise
                gan_input = th.randn(
                    extracted_batch_size, self.latent_size).to(self.device)
                gen_loss = self.optimize_generator(gen_optim, gan_input,
                                                   images, loss_fn)

                # provide a loss feedback
                if i % int(limit / feedback_factor) == 0 or i == 1:
                    elapsed = time.time() - global_time
                    elapsed = str(datetime.timedelta(seconds=elapsed))
                    print("Elapsed [%s] batch: %d  d_loss: %f  g_loss: %f"
                          % (elapsed, i, dis_loss, gen_loss))

                    # also write the losses to the log file:
                    if log_dir is not None:
                        log_file = os.path.join(log_dir, "loss.log")
                        os.makedirs(os.path.dirname(log_file), exist_ok=True)
                        with open(log_file, "a") as log:
                            log.write(str(dis_loss) + "\t" + str(gen_loss) + "\n")

                    # create a grid of samples and save it
                    reses = [str(int(np.power(2, dep))) + "_x_"
                             + str(int(np.power(2, dep)))
                             for dep in range(2, self.depth + 2)]
                    gen_img_files = [os.path.join(sample_dir, res, "gen_" +
                                                  str(epoch) + "_" +
                                                  str(i) + ".png")
                                     for res in reses]

                    # Make sure all the required directories exist
                    # otherwise make them
                    os.makedirs(sample_dir, exist_ok=True)
                    for gen_img_file in gen_img_files:
                        os.makedirs(os.path.dirname(gen_img_file), exist_ok=True)

                    dis_optim.zero_grad()
                    gen_optim.zero_grad()
                    with th.no_grad():
                        self.create_grid(self.gen(fixed_input), gen_img_files)

                if i > limit:
                    break

            # calculate the time required for the epoch
            stop = timeit.default_timer()
            print("Time taken for epoch: %.3f secs" % (stop - start))

            if epoch % checkpoint_factor == 0 or epoch == 1 or epoch == num_epochs:
                os.makedirs(save_dir, exist_ok=True)
                gen_save_file = os.path.join(save_dir, "GAN_GEN_" + str(epoch) + ".pth")
                dis_save_file = os.path.join(save_dir, "GAN_DIS_" + str(epoch) + ".pth")
                gen_optim_save_file = os.path.join(save_dir,
                                                   "GAN_GEN_OPTIM_" + str(epoch) + ".pth")
                dis_optim_save_file = os.path.join(save_dir,
                                                   "GAN_DIS_OPTIM_" + str(epoch) + ".pth")

                th.save(self.gen.state_dict(), gen_save_file)
                th.save(self.dis.state_dict(), dis_save_file)
                th.save(gen_optim.state_dict(), gen_optim_save_file)
                th.save(dis_optim.state_dict(), dis_optim_save_file)

        print("Training completed ...")

        # return the generator and discriminator back to eval mode
        self.gen.eval()
        self.dis.eval()
