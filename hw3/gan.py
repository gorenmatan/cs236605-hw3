from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.distributions import uniform
from .autoencoder import EncoderCNN, DecoderCNN


class Discriminator(nn.Module):
    def __init__(self, in_size):
        """
        :param in_size: The size of on input image (without batch dimension).
        """
        super().__init__()
        self.in_size = in_size
        # TODO: Create the discriminator model layers.
        # To extract image features you can use the EncoderCNN from the VAE
        # section or implement something new.
        # You can then use either an affine layer or another conv layer to
        # flatten the features.
        # ====== YOUR CODE: ======
        self.kernel_sz = 5
        self.pool_sz = 2
        # TODO: Implement a CNN. Save the layers in the modules list.
        # The input shape is an image batch: (N, in_channels, H_in, W_in).
        # The output shape should be (N, out_channels, H_out, W_out).
        # You can assume H_in, W_in >= 64.
        # Architecture is up to you, but you should use at least 3 Conv layers.
        # You can use any Conv layer parameters, use pooling or only strides,
        # use any activation functions, use BN or Dropout, etc.
        # ====== YOUR CODE: ======
        K = [64, 128, 256]
        in_channels = self.in_size[0]
        num_pooling_layers = 0
        
        fa_modules = []
        
        for input_chnl, output_chnl in zip([in_channels] + K, K):
            fa_modules.extend([nn.Conv2d(input_chnl, output_chnl, self.kernel_sz, padding=2, stride=1), 
                               nn.BatchNorm2d(output_chnl),
                               nn.MaxPool2d(self.pool_sz),
                               nn.LeakyReLU()])
            num_pooling_layers += 1
        self.feature_extractor = nn.Sequential(*fa_modules)
        # ========================
        
        h, w = self.in_size[1:]
        ds_factor = self.pool_sz ** num_pooling_layers
        h_ds, w_ds = h // ds_factor, w // ds_factor
        
        classifier_modules = [nn.Linear(h_ds * w_ds * K[-1], h_ds * w_ds // 4),
                              nn.ReLU(),
                              nn.Dropout(0.2),
                              nn.Linear(h_ds * w_ds // 4, 1)]
        self.classifier = nn.Sequential(*classifier_modules)
        # ========================

    def forward(self, x):
        """
        :param x: Input of shape (N,C,H,W) matching the given in_size.
        :return: Discriminator class score (aka logits, not probability) of
        shape (N,).
        """
        # TODO: Implement discriminator forward pass.
        # No need to apply sigmoid to obtain probability - we'll combine it
        # with the loss due to improved numerical stability.
        # ====== YOUR CODE: ======
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        y = self.classifier(x)
        # ========================
        return y


class Generator(nn.Module):
    def __init__(self, z_dim, featuremap_size=4, out_channels=3):
        """
        :param z_dim: Dimension of latent space.
        :featuremap_size: Spatial size of first feature map to create
        (determines output size). For example set to 4 for a 4x4 feature map.
        :out_channels: Number of channels in the generated image.
        """
        super().__init__()
        self.z_dim = z_dim

        # TODO: Create the generator model layers.
        # To combine image features you can use the DecoderCNN from the VAE
        # section or implement something new.
        # You can assume a fixed image size.
        # ====== YOUR CODE: ======
        modules = []
        K = [512, 256, 128, 64]
        first_layer = False
        for in_channel, out_channel in zip([self.z_dim] + K, K + [out_channels]):
            if not first_layer:
                first_layer = True
                padding = 0
            else:
                padding = 1

            block = [nn.ConvTranspose2d(in_channel, out_channel, featuremap_size, 2, padding, bias=False), nn.Tanh()] if out_channel == out_channels \
                else [nn.ConvTranspose2d(in_channel, out_channel, featuremap_size, 2, padding, bias=False),
                      nn.BatchNorm2d(out_channel),
                      nn.ReLU()]
            modules.extend(block)

        self.seq = nn.Sequential(*modules)
        # ========================

    def sample(self, n, with_grad=False):
        """
        Samples from the Generator.
        :param n: Number of instance-space samples to generate.
        :param with_grad: Whether the returned samples should have
        gradients or not.
        :return: A batch of samples, shape (N,C,H,W).
        """
        device = next(self.parameters()).device
        # TODO: Sample from the model.
        # Generate n latent space samples and return their reconstructions.
        # Don't use a loop.
        # ====== YOUR CODE: ======
        # enable or disable autograd
        torch.autograd.set_grad_enabled(with_grad)

        # generate random samples from the latent space
        z = torch.randn([n, self.z_dim], device=device, requires_grad=with_grad)

        # forward the batch and get the results
        samples = self.forward(z)

        # enable back the autograd
        torch.autograd.set_grad_enabled(True)
        # ========================
        return samples

    def forward(self, z):
        """
        :param z: A batch of latent space samples of shape (N, latent_dim).
        :return: A batch of generated images of shape (N,C,H,W) which should be
        the shape which the Discriminator accepts.
        """
        # TODO: Implement the Generator forward pass.
        # Don't forget to make sure the output instances have the same scale
        # as the original (real) images.
        # ====== YOUR CODE: ======
        z = torch.unsqueeze(z, dim=2)
        z = torch.unsqueeze(z, dim=3)
        # ========================
        return self.seq(z)


def discriminator_loss_fn(y_data, y_generated, data_label=0, label_noise=0.0):
    """
    Computes the combined loss of the discriminator given real and generated
    data using a binary cross-entropy metric.
    This is the loss used to update the Discriminator parameters.
    :param y_data: Discriminator class-scores of instances of data sampled
    from the dataset, shape (N,).
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :param label_noise: The range of the noise to add. For example, if
    data_label=0 and label_noise=0.2 then the labels of the real data will be
    uniformly sampled from the range [-0.1,+0.1].
    :return: The combined loss of both.
    """
    assert data_label == 1 or data_label == 0
    # TODO: Implement the discriminator loss.
    # See torch's BCEWithLogitsLoss for a numerically stable implementation.
    # ====== YOUR CODE: ======
    # create a tensor of positive (real) and negative (fake) labels
    pos_label = torch.full(y_data.size(), data_label, device=y_data.device)
    neg_label = 1 - pos_label

    # create the margin of the noise
    a, b = -label_noise / 2, label_noise / 2

    # add uniform noise to the labels
    pos_label_noisy = pos_label + uniform.Uniform(a, b).sample(pos_label.size()).to(pos_label.device)
    neg_label_noisy = neg_label + uniform.Uniform(a, b).sample(neg_label.size()).to(neg_label.device)

    # calculate the BCE for the real and fake classifications
    loss_data = F.binary_cross_entropy_with_logits(y_data, pos_label_noisy)
    loss_generated = F.binary_cross_entropy_with_logits(y_generated, neg_label_noisy)
    # ========================
    return loss_data + loss_generated


def generator_loss_fn(y_generated, data_label=0):
    """
    Computes the loss of the generator given generated data using a
    binary cross-entropy metric.
    This is the loss used to update the Generator parameters.
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :return: The generator loss.
    """
    # TODO: Implement the Generator loss.
    # Think about what you need to compare the input to, in order to
    # formulate the loss in terms of Binary Cross Entropy.
    # ====== YOUR CODE: ======
    negative_label = torch.full(y_generated.size(), data_label, device=y_generated.device)
    loss = F.binary_cross_entropy_with_logits(y_generated, negative_label)
    # ========================
    return loss


def train_batch(dsc_model: Discriminator, gen_model: Generator,
                dsc_loss_fn: Callable, gen_loss_fn: Callable,
                dsc_optimizer: Optimizer, gen_optimizer: Optimizer,
                x_data: Tensor):
    """
    Trains a GAN for over one batch, updating both the discriminator and
    generator.
    :return: The discriminator and generator losses.
    """

    # TODO: Discriminator update
    # 1. Show the discriminator real and generated data
    # 2. Calculate discriminator loss
    # 3. Update discriminator parameters
    # ====== YOUR CODE: ======
    dsc_optimizer.zero_grad()

    # forward
    real_batch = x_data
    fake_batch = gen_model.sample(x_data.shape[0], with_grad=True)

    assert real_batch.size() == fake_batch.size()

    # get the result for the real and fake images
    real_targets = dsc_model(real_batch)
    fake_targets = dsc_model(fake_batch.detach())

    # calculate d loss and make a backward calculation to calculate the gradients
    dsc_loss = dsc_loss_fn(real_targets, fake_targets)

    # train the weights using the optimizer
    dsc_loss.backward()
    dsc_optimizer.step()

    # ========================

    # TODO: Generator update
    # 1. Show the discriminator generated data
    # 2. Calculate generator loss
    # 3. Update generator parameters
    # ====== YOUR CODE: ======
    gen_optimizer.zero_grad()

    # forward
    fake_targets = dsc_model(fake_batch)

    # calculate g and make a backward calculation to calculate the gradients
    gen_loss = gen_loss_fn(fake_targets)

    # train the weights using the optimizer
    gen_loss.backward()
    gen_optimizer.step()
    # ========================

    return dsc_loss.item(), gen_loss.item()

