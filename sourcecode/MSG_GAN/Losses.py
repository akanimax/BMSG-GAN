""" Module implementing various loss functions """

import torch as th


# =============================================================
# Interface for the losses
# =============================================================

class GANLoss:
    """ Base class for all losses
        @args:
            dis: Discriminator used for calculating the loss
                 Note this must be a part of the GAN framework
    """

    def __init__(self, dis):
        self.dis = dis

    def dis_loss(self, real_samps, fake_samps, height, alpha):
        """
        calculate the discriminator loss using the following data
        :param real_samps: batch of real samples
        :param fake_samps: batch of generated (fake) samples
        :param height: current height at which training is going on
        :param alpha: current value of the fader alpha
        :return: loss => calculated loss Tensor
        """
        raise NotImplementedError("dis_loss method has not been implemented")

    def gen_loss(self, real_samps, fake_samps, height, alpha):
        """
        calculate the generator loss
        :param real_samps: batch of real samples
        :param fake_samps: batch of generated (fake) samples
        :param height: current height at which training is going on
        :param alpha: current value of the fader alpha
        :return: loss => calculated loss Tensor
        """
        raise NotImplementedError("gen_loss method has not been implemented")


class ConditionalGANLoss:
    """ Base class for all conditional losses """

    def __init__(self, dis):
        self.dis = dis

    def dis_loss(self, real_samps, fake_samps, labels, height, alpha):
        raise NotImplementedError("dis_loss method has not been implemented")

    def gen_loss(self, real_samps, fake_samps, labels, height, alpha):
        raise NotImplementedError("gen_loss method has not been implemented")


# =============================================================
# Normal versions of the Losses:
# =============================================================

class StandardGAN(GANLoss):

    def __init__(self, dis):
        from torch.nn import BCEWithLogitsLoss

        super().__init__(dis)

        # define the criterion and activation used for object
        self.criterion = BCEWithLogitsLoss()

    def dis_loss(self, real_samps, fake_samps, height, alpha):
        # small assertion:
        assert real_samps.device == fake_samps.device, \
            "Real and Fake samples are not on the same device"

        # device for computations:
        device = fake_samps.device

        # predictions for real images and fake images separately :
        r_preds = self.dis(real_samps, height, alpha)
        f_preds = self.dis(fake_samps, height, alpha)

        # calculate the real loss:
        real_loss = self.criterion(
            th.squeeze(r_preds),
            th.ones(real_samps.shape[0]).to(device))

        # calculate the fake loss:
        fake_loss = self.criterion(
            th.squeeze(f_preds),
            th.zeros(fake_samps.shape[0]).to(device))

        # return final losses
        return (real_loss + fake_loss) / 2

    def gen_loss(self, _, fake_samps, height, alpha):
        preds, _, _ = self.dis(fake_samps, height, alpha)
        return self.criterion(th.squeeze(preds),
                              th.ones(fake_samps.shape[0]).to(fake_samps.device))


class WGAN_GP(GANLoss):

    def __init__(self, dis, drift=0.001, use_gp=False):
        super().__init__(dis)
        self.drift = drift
        self.use_gp = use_gp

    def __gradient_penalty(self, real_samps, fake_samps,
                           height, alpha, reg_lambda=10):
        """
        private helper for calculating the gradient penalty
        :param real_samps: real samples
        :param fake_samps: fake samples
        :param height: current depth in the optimization
        :param alpha: current alpha for fade-in
        :param reg_lambda: regularisation lambda
        :return: tensor (gradient penalty)
        """

        batch_size = real_samps.shape[0]

        # generate random epsilon
        epsilon = th.rand((batch_size, 1, 1, 1)).to(fake_samps.device)

        # create the merge of both real and fake samples
        merged = (epsilon * real_samps) + ((1 - epsilon) * fake_samps)
        merged.requires_grad = True

        # forward pass
        op = self.dis(merged, height, alpha)

        # perform backward pass from op to merged for obtaining the gradients
        op.backward(gradient=th.ones_like(op), create_graph=True)
        gradient = merged.grad  # this is the gradient of the op wrt. merged

        # calculate the penalty using these gradients
        gradient = gradient.view(gradient.shape[0], -1)
        penalty = reg_lambda * ((gradient.norm(p=2, dim=1) - 1) ** 2).mean()

        # return the calculated penalty:
        return penalty

    def dis_loss(self, real_samps, fake_samps, height, alpha):
        # define the (Wasserstein) loss
        fake_out = self.dis(fake_samps, height, alpha)
        real_out = self.dis(real_samps, height, alpha)

        loss = (th.mean(fake_out) - th.mean(real_out)
                + (self.drift * th.mean(real_out ** 2)))

        if self.use_gp:
            # calculate the WGAN-GP (gradient penalty)
            gp = self.__gradient_penalty(real_samps, fake_samps, height, alpha)
            loss += gp

        return loss

    def gen_loss(self, _, fake_samps, height, alpha):
        # calculate the WGAN loss for generator
        loss = -th.mean(self.dis(fake_samps, height, alpha))

        return loss


class LSGAN(GANLoss):

    def __init__(self, dis):
        super().__init__(dis)

    def dis_loss(self, real_samps, fake_samps, height, alpha):
        return 0.5 * (((th.mean(self.dis(real_samps, height, alpha)) - 1) ** 2)
                      + (th.mean(self.dis(fake_samps, height, alpha))) ** 2)

    def gen_loss(self, _, fake_samps, height, alpha):
        return 0.5 * ((th.mean(self.dis(fake_samps, height, alpha)) - 1) ** 2)


class LSGAN_SIGMOID(GANLoss):

    def __init__(self, dis):
        super().__init__(dis)

    def dis_loss(self, real_samps, fake_samps, height, alpha):
        from torch.nn.functional import sigmoid
        real_scores = th.mean(sigmoid(self.dis(real_samps, height, alpha)))
        fake_scores = th.mean(sigmoid(self.dis(fake_samps, height, alpha)))
        return 0.5 * (((real_scores - 1) ** 2) + (fake_scores ** 2))

    def gen_loss(self, _, fake_samps, height, alpha):
        from torch.nn.functional import sigmoid
        scores = th.mean(sigmoid(self.dis(fake_samps, height, alpha)))
        return 0.5 * ((scores - 1) ** 2)


class HingeGAN(GANLoss):

    def __init__(self, dis):
        super().__init__(dis)

    def dis_loss(self, real_samps, fake_samps, height, alpha):
        r_preds, r_mus, r_sigmas = self.dis(real_samps, height, alpha)
        f_preds, f_mus, f_sigmas = self.dis(fake_samps, height, alpha)

        loss = (th.mean(th.nn.ReLU()(1 - r_preds)) +
                th.mean(th.nn.ReLU()(1 + f_preds)))

        return loss

    def gen_loss(self, _, fake_samps, height, alpha):
        return -th.mean(self.dis(fake_samps, height, alpha))


class RelativisticAverageHingeGAN(GANLoss):

    def __init__(self, dis):
        super().__init__(dis)

    def dis_loss(self, real_samps, fake_samps, height, alpha):
        # Obtain predictions
        r_preds = self.dis(real_samps, height, alpha)
        f_preds = self.dis(fake_samps, height, alpha)

        # difference between real and fake:
        r_f_diff = r_preds - th.mean(f_preds)

        # difference between fake and real samples
        f_r_diff = f_preds - th.mean(r_preds)

        # return the loss
        loss = (th.mean(th.nn.ReLU()(1 - r_f_diff))
                + th.mean(th.nn.ReLU()(1 + f_r_diff)))

        return loss

    def gen_loss(self, real_samps, fake_samps, height, alpha):
        # Obtain predictions
        r_preds = self.dis(real_samps, height, alpha)
        f_preds = self.dis(fake_samps, height, alpha)

        # difference between real and fake:
        r_f_diff = r_preds - th.mean(f_preds)

        # difference between fake and real samples
        f_r_diff = f_preds - th.mean(r_preds)

        # return the loss
        return (th.mean(th.nn.ReLU()(1 + r_f_diff))
                + th.mean(th.nn.ReLU()(1 - f_r_diff)))


# =============================================================
# Conditional versions of the Losses:
# =============================================================

class CondStandardGAN(ConditionalGANLoss):

    def __init__(self, dis):
        from torch.nn import BCEWithLogitsLoss

        super().__init__(dis)

        # define the criterion and activation used for object
        self.criterion = BCEWithLogitsLoss()

    def dis_loss(self, real_samps, fake_samps, labels, height, alpha):
        # small assertion:
        assert real_samps.device == fake_samps.device, \
            "Real and Fake samples are not on the same device"

        # device for computations:
        device = fake_samps.device

        # predictions for real images and fake images separately:
        r_preds = self.dis(real_samps, labels, height, alpha)
        f_preds = self.dis(fake_samps, labels, height, alpha)

        # calculate the real loss:
        real_loss = self.criterion(
            th.squeeze(r_preds),
            th.ones(real_samps.shape[0]).to(device))

        # calculate the fake loss:
        fake_loss = self.criterion(
            th.squeeze(f_preds),
            th.zeros(fake_samps.shape[0]).to(device))

        # return final loss
        return (real_loss + fake_loss) / 2

    def gen_loss(self, _, fake_samps, labels, height, alpha):
        preds, _, _ = self.dis(fake_samps, labels, height, alpha)
        return self.criterion(th.squeeze(preds),
                              th.ones(fake_samps.shape[0]).to(fake_samps.device))


class CondWGAN_GP(ConditionalGANLoss):

    def __init__(self, dis, drift=0.001, use_gp=False):
        super().__init__(dis)
        self.drift = drift
        self.use_gp = use_gp

    def __gradient_penalty(self, real_samps, fake_samps, labels,
                           height, alpha, reg_lambda=10):
        """
        private helper for calculating the gradient penalty
        :param real_samps: real samples
        :param fake_samps: fake samples
        :param labels: used for conditional loss calculation
                       Note that this is just [Batch x 1] plain integer labels
        :param height: current depth in the optimization
        :param alpha: current alpha for fade-in
        :param reg_lambda: regularisation lambda
        :return: tensor (gradient penalty)
        """
        from torch.autograd import grad

        batch_size = real_samps.shape[0]

        # generate random epsilon
        epsilon = th.rand((batch_size, 1, 1, 1)).to(fake_samps.device)

        # create the merge of both real and fake samples
        merged = (epsilon * real_samps) + ((1 - epsilon) * fake_samps)

        # forward pass
        op = self.dis(merged, labels, height, alpha)

        # obtain gradient of op wrt. merged
        gradient = grad(outputs=op, inputs=merged, create_graph=True,
                        grad_outputs=th.ones_like(op), only_inputs=True)[0]

        # calculate the penalty using these gradients
        gradient = gradient.view(batch_size, -1)
        penalty = reg_lambda * ((gradient.norm(p=2, dim=1) - 1) ** 2).mean()

        # return the calculated penalty:
        return penalty

    def dis_loss(self, real_samps, fake_samps, labels, height, alpha):
        # define the (Wasserstein) loss
        fake_out = self.dis(fake_samps, labels, height, alpha)
        real_out = self.dis(real_samps, labels, height, alpha)

        loss = (th.mean(fake_out) - th.mean(real_out)
                + (self.drift * th.mean(real_out ** 2)))

        if self.use_gp:
            # calculate the WGAN-GP (gradient penalty)
            fake_samps.requires_grad = True  # turn on gradients for penalty calculation
            gp = self.__gradient_penalty(real_samps, fake_samps,
                                         labels, height, alpha)
            loss += gp

        return loss

    def gen_loss(self, _, fake_samps, labels, height, alpha):
        # calculate the WGAN loss for generator
        loss = -th.mean(self.dis(fake_samps, labels, height, alpha))

        return loss


class CondLSGAN(ConditionalGANLoss):

    def __init__(self, dis):
        super().__init__(dis)

    def dis_loss(self, real_samps, fake_samps, labels, height, alpha):
        return 0.5 * (((th.mean(self.dis(real_samps, labels, height, alpha)) - 1) ** 2)
                      + (th.mean(self.dis(fake_samps, labels, height, alpha))) ** 2)

    def gen_loss(self, _, fake_samps, labels, height, alpha):
        return 0.5 * ((th.mean(self.dis(fake_samps, labels, height, alpha)) - 1) ** 2)


class CondLSGAN_SIGMOID(ConditionalGANLoss):

    def __init__(self, dis):
        super().__init__(dis)

    def dis_loss(self, real_samps, fake_samps, labels, height, alpha):
        from torch.nn.functional import sigmoid
        real_scores = th.mean(sigmoid(self.dis(real_samps, labels, height, alpha)))
        fake_scores = th.mean(sigmoid(self.dis(fake_samps, labels, height, alpha)))
        return 0.5 * (((real_scores - 1) ** 2) + (fake_scores ** 2))

    def gen_loss(self, _, fake_samps, labels, height, alpha):
        from torch.nn.functional import sigmoid
        scores = th.mean(sigmoid(self.dis(fake_samps, labels, height, alpha)))
        return 0.5 * ((scores - 1) ** 2)


class CondHingeGAN(ConditionalGANLoss):

    def __init__(self, dis):
        super().__init__(dis)

    def dis_loss(self, real_samps, fake_samps, labels, height, alpha):
        r_preds, r_mus, r_sigmas = self.dis(real_samps, labels, height, alpha)
        f_preds, f_mus, f_sigmas = self.dis(fake_samps, labels, height, alpha)

        loss = (th.mean(th.nn.ReLU()(1 - r_preds)) +
                th.mean(th.nn.ReLU()(1 + f_preds)))

        return loss

    def gen_loss(self, _, fake_samps, labels, height, alpha):
        return -th.mean(self.dis(fake_samps, labels, height, alpha))


class CondRelativisticAverageHingeGAN(ConditionalGANLoss):

    def __init__(self, dis):
        super().__init__(dis)

    def dis_loss(self, real_samps, fake_samps, labels, height, alpha):
        # Obtain predictions
        r_preds = self.dis(real_samps, labels, height, alpha)
        f_preds = self.dis(fake_samps, labels, height, alpha)

        # difference between real and fake:
        r_f_diff = r_preds - th.mean(f_preds)

        # difference between fake and real samples
        f_r_diff = f_preds - th.mean(r_preds)

        # return the loss
        loss = (th.mean(th.nn.ReLU()(1 - r_f_diff))
                + th.mean(th.nn.ReLU()(1 + f_r_diff)))

        return loss

    def gen_loss(self, real_samps, fake_samps, labels, height, alpha):
        # Obtain predictions
        r_preds = self.dis(real_samps, labels, height, alpha)
        f_preds = self.dis(fake_samps, labels, height, alpha)

        # difference between real and fake:
        r_f_diff = r_preds - th.mean(f_preds)

        # difference between fake and real samples
        f_r_diff = f_preds - th.mean(r_preds)

        # return the loss
        return (th.mean(th.nn.ReLU()(1 + r_f_diff))
                + th.mean(th.nn.ReLU()(1 - f_r_diff)))
