""" Module containing custom layers """
import torch as th
import copy


# ==========================================================
# Layers required for Building The generator and
# discriminator
# ==========================================================
class GenInitialBlock(th.nn.Module):
    """ Module implementing the initial block of the Generator
        Takes in whatever latent size and generates output volume
        of size 4 x 4
    """

    def __init__(self, in_channels):
        """
        constructor for the inner class
        :param in_channels: number of input channels to the block
        """
        from torch.nn import LeakyReLU
        from torch.nn import Conv2d, ConvTranspose2d
        super().__init__()

        self.conv_1 = ConvTranspose2d(in_channels, in_channels, (4, 4), bias=True)
        self.conv_2 = Conv2d(in_channels, in_channels, (3, 3), padding=(1, 1), bias=True)

        # leaky_relu:
        self.lrelu = LeakyReLU(0.2)

    def forward(self, x):
        """
        forward pass of the block
        :param x: input to the module
        :return: y => output
        """
        # convert the tensor shape:
        y = x.view(*x.shape, 1, 1)  # add two dummy dimensions for
        # convolution operation

        # perform the forward computations:
        y = self.lrelu(self.conv_1(y))
        y = self.lrelu(self.conv_2(y))

        return y


class GenGeneralConvBlock(th.nn.Module):
    """ Module implementing a general convolutional block """

    def __init__(self, in_channels, out_channels, dilation=1):
        """
        constructor for the class
        :param in_channels: number of input channels to the block
        :param out_channels: number of output channels required
        """
        from torch.nn import LeakyReLU

        super().__init__()

        from torch.nn import Conv2d
        self.conv_1 = Conv2d(in_channels, out_channels, (3, 3),
                             dilation=dilation, padding=dilation, bias=True)
        self.conv_2 = Conv2d(out_channels, out_channels, (3, 3),
                             dilation=dilation, padding=dilation, bias=True)

        # leaky_relu:
        self.lrelu = LeakyReLU(0.2)

    def forward(self, x):
        """
        forward pass of the block
        :param x: input
        :return: y => output
        """
        from torch.nn.functional import interpolate

        y = interpolate(x, scale_factor=2)
        y = self.lrelu(self.conv_1(y))
        y = self.lrelu(self.conv_2(y))

        return y


class MinibatchStdDev(th.nn.Module):
    def __init__(self, averaging='all'):
        """
        constructor for the class
        :param averaging: the averaging mode used for calculating the MinibatchStdDev
        """
        super().__init__()

        # lower case the passed parameter
        self.averaging = averaging.lower()

        if 'group' in self.averaging:
            self.n = int(self.averaging[5:])
        else:
            assert self.averaging in \
                   ['all', 'flat', 'spatial', 'none', 'gpool'], \
                   'Invalid averaging mode %s' % self.averaging

        # calculate the std_dev in such a way that it doesn't result in 0
        # otherwise 0 norm operation's gradient is nan
        self.adjusted_std = lambda x, **kwargs: th.sqrt(
            th.mean((x - th.mean(x, **kwargs)) ** 2, **kwargs) + 1e-8)

    def forward(self, x):
        """
        forward pass of the Layer
        :param x: input
        :return: y => output
        """
        shape = list(x.size())
        target_shape = copy.deepcopy(shape)

        # compute the std's over the minibatch
        vals = self.adjusted_std(x, dim=0, keepdim=True)

        # perform averaging
        if self.averaging == 'all':
            target_shape[1] = 1
            vals = th.mean(vals, dim=1, keepdim=True)

        elif self.averaging == 'spatial':
            if len(shape) == 4:
                vals = th.mean(th.mean(vals, 2, keepdim=True), 3, keepdim=True)

        elif self.averaging == 'none':
            target_shape = [target_shape[0]] + [s for s in target_shape[1:]]

        elif self.averaging == 'gpool':
            if len(shape) == 4:
                vals = th.mean(th.mean(th.mean(x, 2, keepdim=True),
                                       3, keepdim=True), 0, keepdim=True)
        elif self.averaging == 'flat':
            target_shape[1] = 1
            vals = th.FloatTensor([self.adjusted_std(x)])

        else:  # self.averaging == 'group'
            target_shape[1] = self.n
            vals = vals.view(self.n, self.shape[1] /
                             self.n, self.shape[2], self.shape[3])
            vals = th.mean(vals, 0, keepdim=True).view(1, self.n, 1, 1)

        # spatial replication of the computed statistic
        vals = vals.expand(*target_shape)

        # concatenate the constant feature map to the input
        y = th.cat([x, vals], 1)

        # return the computed value
        return y


class DisFinalBlock(th.nn.Module):
    """ Final block for the Discriminator """

    def __init__(self, in_channels):
        """
        constructor of the class
        :param in_channels: number of input channels
        """
        from torch.nn import LeakyReLU
        from torch.nn import Conv2d

        super().__init__()

        # declare the required modules for forward pass
        self.batch_discriminator = MinibatchStdDev()

        # modules required:
        self.conv_1 = Conv2d(in_channels + 1, in_channels, (3, 3), padding=1, bias=True)
        self.conv_2 = Conv2d(in_channels, in_channels, (4, 4), bias=True)

        # final conv layer emulates a fully connected layer
        self.conv_3 = Conv2d(in_channels, 1, (1, 1), bias=True)

        # leaky_relu:
        self.lrelu = LeakyReLU(0.2)

    def forward(self, x):
        """
        forward pass of the FinalBlock
        :param x: input
        :return: y => output
        """
        # minibatch_std_dev layer
        y = self.batch_discriminator(x)

        # define the computations
        y = self.lrelu(self.conv_1(y))
        y = self.lrelu(self.conv_2(y))

        # fully connected layer
        y = self.conv_3(y)  # This layer has linear activation

        # flatten the output raw discriminator scores
        return y.view(-1)


class DisGeneralConvBlock(th.nn.Module):
    """ General block in the discriminator  """

    def __init__(self, in_channels, out_channels, dilation=1):
        """
        constructor of the class
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        """
        from torch.nn import AvgPool2d, LeakyReLU
        from torch.nn import Conv2d

        super().__init__()

        # convolutional modules
        self.conv_1 = Conv2d(in_channels, in_channels, (3, 3),
                             dilation=dilation, padding=dilation, bias=True)
        self.conv_2 = Conv2d(in_channels, out_channels, (3, 3),
                             dilation=dilation, padding=dilation, bias=True)
        self.downSampler = AvgPool2d(2)  # downsampler

        # leaky_relu:
        self.lrelu = LeakyReLU(0.2)

    def forward(self, x):
        """
        forward pass of the module
        :param x: input
        :return: y => output
        """
        # define the computations
        y = self.lrelu(self.conv_1(x))
        y = self.lrelu(self.conv_2(y))
        y = self.downSampler(y)

        return y
