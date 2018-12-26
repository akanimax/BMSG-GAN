""" Module implementing various loss functions """

import torch as th


# TODO_complete Major rewrite: change the interface to use only predictions
# for real and fake samples
# The interface doesn't need to change to only use predictions for real and fake samples
# because for loss such as WGAN-GP requires the samples to calculate gradient penalty

class GANLoss:
    """
    Base class for all losses
    Note that the gen_loss also has
    """

    def __init__(self, device, dis):
        self.device = device
        self.dis = dis

    def dis_loss(self, real_samps, fake_samps):
        raise NotImplementedError("dis_loss method has not been implemented")

    def gen_loss(self, real_samps, fake_samps):
        raise NotImplementedError("gen_loss method has not been implemented")

    def conditional_dis_loss(self, real_samps, fake_samps, conditional_vectors):
        raise NotImplementedError("conditional_dis_loss method has not been implemented")

    def conditional_gen_loss(self, real_samps, fake_samps, conditional_vectors):
        raise NotImplementedError("conditional_gen_loss method has not been implemented")


class StandardGAN(GANLoss):

    def __init__(self, dev, dis):
        from torch.nn import BCEWithLogitsLoss

        super().__init__(dev, dis)

        # define the criterion object
        self.criterion = BCEWithLogitsLoss()

    def dis_loss(self, real_samps, fake_samps):
        # calculate the real loss:
        real_loss = self.criterion(th.squeeze(self.dis(real_samps)),
                                   th.ones(real_samps.shape[0]).to(self.device))
        # calculate the fake loss:
        fake_loss = self.criterion(th.squeeze(self.dis(fake_samps)),
                                   th.zeros(fake_samps.shape[0]).to(self.device))

        # return final loss as average of the two:
        return (real_loss + fake_loss) / 2

    def gen_loss(self, _, fake_samps):
        return self.criterion(th.squeeze(self.dis(fake_samps)),
                              th.ones(fake_samps.shape[0]).to(self.device))

    def conditional_dis_loss(self, real_samps, fake_samps, conditional_vectors):
        # calculate the real loss:
        real_loss = self.criterion(th.squeeze(self.dis(real_samps, conditional_vectors)),
                                   th.ones(real_samps.shape[0]).to(self.device))
        # calculate the fake loss:
        fake_loss = self.criterion(th.squeeze(self.dis(fake_samps, conditional_vectors)),
                                   th.zeros(fake_samps.shape[0]).to(self.device))

        # return final loss as average of the two:
        return (real_loss + fake_loss) / 2

    def conditional_gen_loss(self, real_samps, fake_samps, conditional_vectors):
        return self.criterion(th.squeeze(self.dis(fake_samps, conditional_vectors)),
                              th.ones(fake_samps.shape[0]).to(self.device))


class LSGAN(GANLoss):

    def __init__(self, device, dis):
        super().__init__(device, dis)

    def dis_loss(self, real_samps, fake_samps):
        return 0.5 * (((th.mean(self.dis(real_samps)) - 1) ** 2)
                      + (th.mean(self.dis(fake_samps))) ** 2)

    def gen_loss(self, _, fake_samps):
        return 0.5 * ((th.mean(self.dis(fake_samps)) - 1) ** 2)

    def conditional_dis_loss(self, real_samps, fake_samps, conditional_vectors):
        return 0.5 * (((th.mean(self.dis(real_samps, conditional_vectors)) - 1) ** 2)
                      + (th.mean(self.dis(fake_samps, conditional_vectors))) ** 2)

    def conditional_gen_loss(self, real_samps, fake_samps, conditional_vectors):
        return 0.5 * ((th.mean(self.dis(fake_samps, conditional_vectors)) - 1) ** 2)


class HingeGAN(GANLoss):

    def __init__(self, device, dis):
        super().__init__(device, dis)

    def dis_loss(self, real_samps, fake_samps):
        return (th.mean(th.nn.ReLU()(1 - self.dis(real_samps))) +
                th.mean(th.nn.ReLU()(1 + self.dis(fake_samps))))

    def gen_loss(self, real_samps, fake_samps):
        return -th.mean(self.dis(fake_samps))

    def conditional_dis_loss(self, real_samps, fake_samps, conditional_vectors):
        return (th.mean(th.nn.ReLU()(1 - self.dis(real_samps, conditional_vectors))) +
                th.mean(th.nn.ReLU()(1 + self.dis(fake_samps, conditional_vectors))))

    def conditional_gen_loss(self, real_samps, fake_samps, conditional_vectors):
        return -th.mean(self.dis(fake_samps, conditional_vectors))


class RelativisticAverageHingeGAN(GANLoss):

    def __init__(self, device, dis):
        super().__init__(device, dis)

    def dis_loss(self, real_samps, fake_samps):
        # difference between real and fake:
        r_f_diff = self.dis(real_samps) - th.mean(self.dis(fake_samps))

        # difference between fake and real samples
        f_r_diff = self.dis(fake_samps) - th.mean(self.dis(real_samps))

        # return the loss
        return (th.mean(th.nn.ReLU()(1 - r_f_diff))
                + th.mean(th.nn.ReLU()(1 + f_r_diff)))

    def gen_loss(self, real_samps, fake_samps):
        # difference between real and fake:
        r_f_diff = self.dis(real_samps) - th.mean(self.dis(fake_samps))

        # difference between fake and real samples
        f_r_diff = self.dis(fake_samps) - th.mean(self.dis(real_samps))

        # return the loss
        return (th.mean(th.nn.ReLU()(1 + r_f_diff))
                + th.mean(th.nn.ReLU()(1 - f_r_diff)))

    def conditional_dis_loss(self, real_samps, fake_samps, conditional_vectors):
        # difference between real and fake:
        r_f_diff = (self.dis(real_samps, conditional_vectors)
                    - th.mean(self.dis(fake_samps, conditional_vectors)))

        # difference between fake and real samples
        f_r_diff = (self.dis(fake_samps, conditional_vectors)
                    - th.mean(self.dis(real_samps, conditional_vectors)))

        # return the loss
        return (th.mean(th.nn.ReLU()(1 - r_f_diff))
                + th.mean(th.nn.ReLU()(1 + f_r_diff)))

    def conditional_gen_loss(self, real_samps, fake_samps, conditional_vectors):
        # difference between real and fake:
        r_f_diff = (self.dis(real_samps, conditional_vectors)
                    - th.mean(self.dis(fake_samps, conditional_vectors)))

        # difference between fake and real samples
        f_r_diff = (self.dis(fake_samps, conditional_vectors)
                    - th.mean(self.dis(real_samps, conditional_vectors)))

        # return the loss
        return (th.mean(th.nn.ReLU()(1 + r_f_diff))
                + th.mean(th.nn.ReLU()(1 - f_r_diff)))
