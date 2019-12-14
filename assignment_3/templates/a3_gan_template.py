import argparse
import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


class Generator(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Generator, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Construct generator. You are free to experiment with your model,
        # but the following is a good start:
        #   Linear args.latent_dim -> 128
        #   LeakyReLU(0.2)
        #   Linear 128 -> 256
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 256 -> 512
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 512 -> 1024
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 1024 -> 768
        #   Output non-linearity

        self.generator = nn.Sequential(
            # l1
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2),
            # l2
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            # l3
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            # l4
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            # l5
            nn.Linear(1024, input_dim),
            nn.Tanh()
        )

    def forward(self, z):
        # Generate images from z
        return self.generator(z)


class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()

        # Construct distriminator. You are free to experiment with your model,
        # but the following is a good start:
        #   Linear 784 -> 512
        #   LeakyReLU(0.2)
        #   Linear 512 -> 256
        #   LeakyReLU(0.2)
        #   Linear 256 -> 1
        #   Output non-linearity

        self.discriminator = nn.Sequential(
            # l1
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            # l2
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            # l3
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        # return discriminator score for img
        return self.discriminator(img)


def interpolate(generator, N=5, steps=7):
    interpolations = []
    for n in range(N):
        position_a = torch.randn((1, generator.latent_dim)).to(args.device)
        position_b = torch.randn((1, generator.latent_dim)).to(args.device)

        # Even worse with ppf
        steps_ = torch.FloatTensor(norm.ppf(np.linspace(0, 1, steps + 2))[1:-1])[:, None].to(args.device)

        # Maybe something wrong here
        _int_points = steps_ * (position_b - position_a) + position_a
        _int_points = torch.cat([position_a, _int_points, position_b], dim=0)
        with torch.no_grad():
            interpolations.append(generator(_int_points))

    interpolations = torch.stack(interpolations)
    save_image(interpolations.view(N * (steps + 2), 1, 28, 28), 'figures/interpolations.png', nrow=9, normalize=True)


def plot_losses(losses):
    fig = plt.figure(figsize=(15, 10), dpi=150)
    fig.suptitle('GAN losses', fontsize=36)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(losses[0], linewidth=2, color="tomato", label="Generator")
    ax.plot(losses[1], linewidth=2, color="darkblue", label="Discriminator")
    ax.tick_params(labelsize=16)

    ax.set_xlabel('Epoch', fontsize=24)
    ax.set_ylabel('Loss', fontsize=24)
    ax.legend(prop={'size': 16})

    if not os.path.exists('figures/'):
        os.makedirs('figures/')

    plt.savefig("figures/Gan_losses.png")
    plt.show()


def train(dataloader, discriminator, generator, optimizer_G, optimizer_D):
    # Loss across epochs
    losses = [[], []]

    for epoch in range(args.n_epochs):
        # Losses over batches
        losses_batches = [[], []]
        for i, (imgs, _) in enumerate(dataloader):

            imgs = imgs.view(-1, generator.input_dim).to(args.device)

            # Train Generator
            # ---------------
            optimizer_G.zero_grad()

            # sample from Z and complete forward pass
            z_noise = torch.randn((args.batch_size, generator.latent_dim)).to(args.device)
            generated_imgs = generator(z_noise)
            prediction_fake = discriminator(generated_imgs)

            # Generator loss
            loss_g = - torch.mean(torch.log(prediction_fake))

            # Optimize Generator
            loss_g.backward()
            optimizer_G.step()

            # Train Discriminator
            # -------------------
            optimizer_D.zero_grad()

            # Second forward pass
            z_noise = torch.randn((args.batch_size, generator.latent_dim)).to(args.device)
            generated_imgs = generator(z_noise)
            prediction_fake = discriminator(generated_imgs)
            dis_loss_g = torch.mean(torch.log(1 - prediction_fake))

            prediction_real = discriminator(imgs)
            dis_loss_d = torch.mean(torch.log(prediction_real))

            # Discriminator loss
            loss_d = - dis_loss_d - dis_loss_g

            # Optimize Discriminator
            loss_d.backward()
            optimizer_D.step()

            losses_batches[0].append(loss_g.item())
            losses_batches[1].append(loss_d.item())

            # Save Images
            # -----------
            batches_done = epoch * len(dataloader) + i
            if batches_done % args.save_interval == 0:
                # You can use the function save_image(Tensor (shape Bx1x28x28),
                # filename, number of rows, normalize) to save the generated
                # images, e.g.:

                # As descrinbed in question 2.6 generate 25 images
                z_noise = torch.randn((25, generator.latent_dim)).to(args.device)
                generated_imgs = generator(z_noise)
                save_image(generated_imgs.view(25, 1, 28, 28), 'images/gen_imgs_{}.png'.format(batches_done), nrow=5,
                           normalize=True)
            print(
                f"[Epoch {epoch}] batches done: {batches_done}, generator loss: {loss_g.item()} discriminator loss: {loss_d.item()}")

        losses[0].append(np.mean(losses_batches[0]))
        losses[1].append(np.mean(losses_batches[1]))

    torch.save(generator, "trained_generator.pth")
    torch.save(discriminator, "trained_discriminator.pth")

    plot_losses(losses)


def main():
    if args.interpolate:
        generator = torch.load("trained_generator.pth", map_location='cpu')
        print(generator)
        interpolate(generator)
    else:
        # Create output image directory
        os.makedirs('images', exist_ok=True)

        # load data
        dataloader = torch.utils.data.DataLoader(
            datasets.MNIST('./data/mnist', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.5,), (0.5,))])),
            batch_size=args.batch_size, shuffle=True)

        input_dim = np.prod(next(iter(dataloader))[0].size()[1:])

        # Initialize models and optimizers
        generator = Generator(input_dim=input_dim, latent_dim=args.latent_dim).to(args.device)
        discriminator = Discriminator(input_dim=input_dim).to(args.device)
        optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr)
        optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr)

        # Start training
        train(dataloader, discriminator, generator, optimizer_G, optimizer_D)

        # You can save your generator here to re-use it to generate images for your
        # report, e.g.:
        # torch.save(generator.state_dict(), "mnist_generator.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=200,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='learning rate')
    parser.add_argument('--latent_dim', type=int, default=100,
                        help='dimensionality of the latent space')
    parser.add_argument('--save_interval', type=int, default=500,
                        help='save every SAVE_INTERVAL iterations')
    parser.add_argument('--device', type=str, default="cpu", help="Training device 'cpu' or 'cuda:0'")
    parser.add_argument('--interpolate', type=bool, default=False, help="Loads pretrained generator and interpolates "
                                                                        "between two points in the latent space")

    args = parser.parse_args()

    main()
