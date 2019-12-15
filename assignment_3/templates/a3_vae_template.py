import argparse
import os

import numpy as np
from scipy.stats import norm

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from datasets.bmnist import bmnist


class Encoder(nn.Module):

    def __init__(self, input_dim=784, hidden_dim=500, z_dim=20):
        super().__init__()

        self.embed_trans = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh()
        )
        self.linear_mu = nn.Linear(hidden_dim, z_dim)
        self.linear_var = nn.Linear(hidden_dim, z_dim)

    def forward(self, input):
        """
        Perform forward pass of encoder.

        Returns mean and std with shape [batch_size, z_dim]. Make sure
        that any constraints are enforced.
        """

        embed = self.embed_trans(input)
        # Log variance due to definitions in Kingma & Welling
        mean, log_var = self.linear_mu(embed), self.linear_var(embed)

        return mean, torch.sqrt(torch.exp(log_var)), log_var


class Decoder(nn.Module):

    def __init__(self, output_dim=784, hidden_dim=500, z_dim=20):
        super().__init__()

        self.reconstruct = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )

    def forward(self, input):
        """
        Perform forward pass of encoder.

        Returns mean with shape [batch_size, 784].
        """
        mean = self.reconstruct(input)
        return mean


class VAE(nn.Module):

    def __init__(self, input_dim, hidden_dim=500, z_dim=20, device='cuda:0'):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.device = device
        self.encoder = Encoder(input_dim, hidden_dim, z_dim)
        self.decoder = Decoder(input_dim, hidden_dim, z_dim)

    def forward(self, input):
        """
        Given input, perform an encoding and decoding step and return the
        negative average elbo for the given batch.
        """

        mu, std, log_var = self.encoder(input)

        # randn => N(0, I)
        epsilon = torch.randn(mu.size()).to(self.device)
        latent_z = mu + std * epsilon

        x_hat = self.decoder(latent_z)

        average_negative_elbo = self.calc_neg_elbo_loss(x_hat, input, mu, log_var)

        return average_negative_elbo

    def calc_neg_elbo_loss(self, x_hat, x_target, mu, log_var):
        eps = 1e-12
        log_bernoulli_loss = -torch.sum(x_target * torch.log(x_hat + eps) + (1 - x_target) * torch.log(1 - x_hat + eps),
                                        dim=1)
        KL_loss = 0.5 * torch.sum(log_var.exp() + mu.pow(2) - log_var, dim=1) - 1
        return torch.mean(log_bernoulli_loss + KL_loss, dim=0)

    def sample(self, n_samples):
        """
        Sample n_samples from the model. Return both the sampled images
        (from bernoulli) and the means for these bernoullis (as these are
        used to plot the data manifold).
        """

        # Random latent space vectors of shape (n_samples, z_dim) sampled from N(0, I)
        random_latents = torch.randn((n_samples, self.z_dim))

        with torch.no_grad():
            im_means = self.decoder(random_latents.to(self.device))
        # Sample from Bernoulli by sampling from Uniform and use im_means as a threshold
        sampled_ims = torch.rand(im_means.size()).to(self.device) < im_means.to(self.device)

        return sampled_ims, im_means


def epoch_iter(model, data, optimizer, device):
    """
    Perform a single epoch for either the training or validation.
    use model.training to determine if in 'training mode' or not.

    Returns the average elbo for the complete epoch.
    """

    elbos = []

    for step, batch_inputs in enumerate(data):
        elbo = model(batch_inputs.view(-1, model.input_dim).to(device))
        elbos.append(elbo.item())

        if model.training:
            optimizer.zero_grad()
            elbo.backward()
            optimizer.step()

    average_epoch_elbo = np.mean(elbos)
    return average_epoch_elbo


def run_epoch(model, data, optimizer, device):
    """
    Run a train and validation epoch and return average elbo for each.
    """
    traindata, valdata = data

    model.train()
    train_elbo = epoch_iter(model, traindata, optimizer, device)

    model.eval()
    val_elbo = epoch_iter(model, valdata, optimizer, device)

    return train_elbo, val_elbo


def save_elbo_plot(train_curve, val_curve, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(train_curve, label='train elbo')
    plt.plot(val_curve, label='validation elbo')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('ELBO')
    plt.tight_layout()

    if not os.path.exists('figures/'):
        os.makedirs('figures/')

    plt.savefig("figures/{}".format(filename))


def plot_grid(generated_grids, z_dim, n_plots):
    fig = plt.figure(figsize=(20, 10), dpi=150)

    for i, (grid, epoch) in enumerate(generated_grids):
        ax = fig.add_subplot(1, n_plots, i + 1)
        ax.imshow(grid.detach().numpy())
        ax.set_xlabel('Epoch: {}'.format(epoch), fontsize=24)

    if not os.path.exists('figures/'):
        os.makedirs('figures/')

    plt.savefig("figures/sample_model_z{}.png".format(z_dim))
    plt.show()


def compute_grid(model, n_samples):
    sampled_ims, im_means = model.sample(n_samples)

    im_means_rs = im_means.view(n_samples, 1, 28, 28)
    return make_grid(im_means_rs, nrow=int(np.sqrt(n_samples))).permute(1, 2, 0)


def plot_manifold(model, n_samples, device):
    nrows = int(np.sqrt(n_samples))

    ppf_ = norm.ppf(np.linspace(0, 1, nrows + 2))[1:-1]
    ppf_latents = torch.FloatTensor(np.array([[m, n] for m in ppf_ for n in ppf_])).to(device)

    with torch.no_grad():
        im_means = model.decoder(ppf_latents)
    im_means_rs = im_means.view(n_samples, 1, 28, 28)

    manifold_grid = make_grid(im_means_rs, nrow=nrows).permute(1, 2, 0)

    fig = plt.figure(figsize=(20, 10), dpi=150)

    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(manifold_grid.detach().numpy())
    ax.set_xlabel('Data manifold over 2-D latent space', fontsize=24)

    if not os.path.exists('figures/'):
        os.makedirs('figures/')

    plt.savefig("figures/manifold.png")
    plt.show()


def main():
    device = ARGS.device

    data = bmnist()[:2]  # ignore test split
    input_dim = np.prod(next(iter(data[0])).size()[1:])

    model = VAE(input_dim, z_dim=ARGS.zdim, device=device).to(device)

    optimizer = torch.optim.Adam(model.parameters())

    n_plots = 3
    n_samples = ARGS.n_samples
    grids = [(compute_grid(model, n_samples), 0)]

    train_curve, val_curve = [], []
    for epoch in range(ARGS.epochs):
        elbos = run_epoch(model, data, optimizer, device)
        train_elbo, val_elbo = elbos
        train_curve.append(train_elbo)
        val_curve.append(val_elbo)
        print(f"[Epoch {epoch}] train elbo: {train_elbo} val_elbo: {val_elbo}")

        # --------------------------------------------------------------------
        #  Add functionality to plot samples from model during training.
        #  You can use the make_grid functioanlity that is already imported.
        # --------------------------------------------------------------------

        if epoch == int(ARGS.epochs / 2) - 1 or epoch == ARGS.epochs - 1:
            grids.append((compute_grid(model, n_samples), epoch + 1))

    # --------------------------------------------------------------------
    #  Add functionality to plot plot the learned data manifold after
    #  if required (i.e., if zdim == 2). You can use the make_grid
    #  functionality that is already imported.
    # --------------------------------------------------------------------

    if ARGS.plot_manifold:
        plot_manifold(model, n_samples, device)
    else:
        save_elbo_plot(train_curve, val_curve, 'elbo.png')
        plot_grid(grids, ARGS.zdim, n_plots)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=40, type=int,
                        help='max number of epochs')
    parser.add_argument('--zdim', default=20, type=int,
                        help='dimensionality of latent space')
    parser.add_argument('--n_samples', default=36, type=int,
                        help='number of samples')
    parser.add_argument('--plot_manifold', default=False, type=bool,
                        help='plot latent space manifold')
    parser.add_argument('--device', type=str, default="cpu", help="Training device 'cpu' or 'cuda:0'")

    ARGS = parser.parse_args()

    main()
