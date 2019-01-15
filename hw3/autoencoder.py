import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        modules = []
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
        for input_chnl, output_chnl in zip([in_channels] + K, K + [out_channels]):
            modules.extend([nn.Conv2d(input_chnl, output_chnl, self.kernel_sz, padding=2, stride=1), 
                            nn.BatchNorm2d(output_chnl),
                            nn.MaxPool2d(self.pool_sz),
                            nn.ReLU()])
        # ========================
        self.cnn = nn.Sequential(*modules)

    def forward(self, x):
        return self.cnn(x)


class DecoderCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        modules = []

        self.unpool_sz = 2
        self.kernel_sz = 5
        # TODO: Implement the "mirror" CNN of the encoder.
        # For example, instead of Conv layers use transposed convolutions,
        # instead of pooling do unpooling (if relevant) and so on.
        # You should have the same number of layers as in the Encoder,
        # and they should produce the same volumes, just in reverse order.
        # Output should be a batch of images, with same dimensions as the
        # inputs to the Encoder were.
        # ====== YOUR CODE: ======
        K = [256, 128, 64]
        for input_chnl, output_chnl in zip([in_channels] + K, K + [out_channels]):
            modules.extend([nn.ConvTranspose2d(input_chnl, output_chnl, self.kernel_sz, padding=2, stride=1), 
                            nn.BatchNorm2d(output_chnl),
                            nn.UpsamplingBilinear2d(scale_factor=self.unpool_sz),
                            nn.ReLU()])
        # ========================
        self.cnn = nn.Sequential(*modules)

    def forward(self, h):
        # Tanh to scale to [-1, 1] (same dynamic range as original images).
        return torch.tanh(self.cnn(h))


class VAE(nn.Module):
    def __init__(self, features_encoder, features_decoder, in_size, z_dim):
        """
        :param features_encoder: Instance of an encoder the extracts features
        from an input.
        :param features_decoder: Instance of a decoder that reconstructs an
        input from it's features.
        :param in_size: The size of one input (without batch dimension).
        :param z_dim: The latent space dimension.
        """
        super().__init__()
        self.features_encoder = features_encoder
        self.features_decoder = features_decoder
        self.z_dim = z_dim

        self.features_shape, n_features = self._check_features(in_size)

        # TODO: Add parameters needed for encode() and decode().
        # ====== YOUR CODE: ======
        # get the dimension of the feature output
        #import ipdb
        #ipdb.set_trace()
        device = next(self.parameters()).device
        x = torch.randn(1, *in_size, device=device)
        h = self.features_encoder(x)
        self.h_shape = h.shape[1:]
        h_dim = torch.zeros(self.h_shape).numel()
        
        # create the affine transformation layers for the encode() and decode() operations
        self.fc_u, self.fc_logvar = nn.Linear(h_dim ,z_dim).to(device), nn.Linear(h_dim, z_dim).to(device)
        self.fc_rec = nn.Linear(z_dim, h_dim).to(device)
        # ========================

    def _check_features(self, in_size):
        device = next(self.parameters()).device
        with torch.no_grad():
            # Make sure encoder and decoder are compatible
            x = torch.randn(1, *in_size, device=device)
            h = self.features_encoder(x)
            xr = self.features_decoder(h)
            assert xr.shape == x.shape
            # Return the shape and number of encoded features
            return h.shape[1:], torch.numel(h)//h.shape[0]

    def encode(self, x):
        # TODO: Sample a latent vector z given an input x.
        # 1. Use the features extracted from the input to obtain mu and
        # log_sigma2 (mean and log variance) of the posterior p(z|x).
        # 2. Apply the reparametrization trick.
        # ====== YOUR CODE: ======
        #import ipdb
        #ipdb.set_trace()
        h = self.features_encoder(x)
        h = h.view(h.size(0), -1)
        mu, log_sigma2 = self.fc_u(h), self.fc_logvar(h)
        std = log_sigma2.mul(0.5).exp_()
        z = mu + torch.randn_like(std) * std
        # ========================

        return z, mu, log_sigma2

    def decode(self, z):
        # TODO: Convert a latent vector back into a reconstructed input.
        # 1. Convert latent to features.
        # 2. Apply features decoder.
        # ====== YOUR CODE: ======
        #import ipdb
        #ipdb.set_trace()
        h_rec = self.fc_rec(z)
        h_rec = h_rec.view(h_rec.size(0), *self.h_shape) # creates a tensor of size (BATCH_SZ, ORIG_DIMENSIONS)
        x_rec = self.features_decoder(h_rec)
        # ========================

        # Scale to [-1, 1] (same dynamic range as original images).
        return torch.tanh(x_rec)

    def sample(self, n):
        samples = []
        device = next(self.parameters()).device
        with torch.no_grad():
            # TODO: Sample from the model.
            # Generate n latent space samples and return their reconstructions.
            # Remember that for the model, this is like inference.
            # ====== YOUR CODE: ======
            raise NotImplementedError()
            # ========================
        return samples

    def forward(self, x):
        z, mu, log_sigma2 = self.encode(x)
        return self.decode(z), mu, log_sigma2


def vae_loss(x, xr, z_mu, z_log_sigma2, x_sigma2):
    """
    Pointwise loss function of a VAE with latent space of dimension z_dim.
    :param x: Input image batch of shape (N,C,H,W).
    :param xr: Reconstructed (output) image batch.
    :param z_mu: Posterior mean (batch) of shape (N, z_dim).
    :param z_log_sigma2: Posterior log-variance (batch) of shape (N, z_dim).
    :param x_sigma2: Likelihood variance (scalar).
    :return:
        - The VAE loss
        - The data loss term
        - The KL divergence loss term
    all three are scalars, averaged over the batch dimension.
    """
    loss, data_loss, kldiv_loss = None, None, None
    # TODO: Implement the VAE pointwise loss calculation.
    # Remember that the covariance matrix of the posterior is diagonal.
    # ====== YOUR CODE: ======
    z_dim = z_mu.shape[1]
    log_sigma = z_log_sigma2
    sigma = torch.exp(log_sigma)
    x = x.view(x.shape[0], -1)
    xr = xr.view(xr.shape[0], -1)

    data_loss = torch.sum((x - xr) ** 2, dim=1) / x_sigma2
    tr_sigma = torch.sum(sigma, dim=1)
    norm2_mu = torch.sum(z_mu ** 2, dim=1)
    log_det_sigma = torch.sum(log_sigma, dim=1)
    kldiv_loss = tr_sigma + norm2_mu - z_dim - log_det_sigma

    # additional average over the 2nd dimension (of instance space for
    # data loss, and of latent space for kldiv loss) actually makes it
    # work, but don't ask me why. Time to hit the books.
    data_loss = torch.mean(data_loss) / x.shape[1]
    kldiv_loss = torch.mean(kldiv_loss) / z_dim
    loss = data_loss + kldiv_loss
    # ========================

    return loss, data_loss, kldiv_loss
