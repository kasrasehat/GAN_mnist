import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.datasets import MNIST # Training dataset
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from Networks import Generator, Discriminator
from IPython.display import clear_output

torch.manual_seed(0) # Set for testing purposes, please do not change!

def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28)):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in a uniform grid.
    '''
    image_unflat = image_tensor.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()


def get_noise(n_samples, z_dim, device='cpu'):
    '''
    Function for creating noise vectors: Given the dimensions (n_samples, z_dim),
    creates a tensor of that shape filled with random numbers from the normal distribution.
    Parameters:
        n_samples: the number of samples to generate, a scalar
        z_dim: the dimension of the noise vector, a scalar
        device: the device type
    '''
    return torch.randn((n_samples, z_dim), device=device)


criterion = nn.BCEWithLogitsLoss()
n_epochs = 20
z_dim = 64
display_step = 500
batch_size = 128
lr = 0.00001

# Load MNIST dataset as tensors
dataloader = DataLoader(
    MNIST('.', download=False, transform=transforms.ToTensor()),
    batch_size=batch_size,
    shuffle=True)

device = 'cuda'

gen = Generator(z_dim).to(device)
gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)
disc = Discriminator().to(device)
disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)


def get_disc_loss(gen, disc, criterion, real, num_images, z_dim, device):
    '''
    Return the loss of the discriminator given inputs.
    Parameters:
        gen: the generator model, which returns an image given z-dimensional noise
        disc: the discriminator model, which returns a single-dimensional prediction of real/fake
        criterion: the loss function, which should be used to compare
               the discriminator's predictions to the ground truth reality of the images
               (e.g. fake = 0, real = 1)
        real: a batch of real images
        num_images: the number of images the generator should produce,
                which is also the length of the real images
        z_dim: the dimension of the noise vector, a scalar
        device: the device type
    Returns:
        disc_loss: a torch scalar loss value for the current batch
    '''
    #     These are the steps you will need to complete:
    #       1) Create noise vectors and generate a batch (num_images) of fake images.
    #            Make sure to pass the device argument to the noise.
    #       2) Get the discriminator's prediction of the fake image
    #            and calculate the loss. Don't forget to detach the generator!
    #            (Remember the loss function you set earlier -- criterion. You need a
    #            'ground truth' tensor in order to calculate the loss.
    #            For example, a ground truth tensor for a fake image is all zeros.)
    #       3) Get the discriminator's prediction of the real image and calculate the loss.
    #       4) Calculate the discriminator's loss by averaging the real and fake loss
    #            and set it to disc_loss.



    noise_vec = get_noise(num_images, z_dim, device=device)
    fake_images = gen(noise_vec).detach()
    pred_of_fake = disc(fake_images)
    label_of_fake = torch.zeros_like(pred_of_fake)
    loss_fake = criterion(pred_of_fake, label_of_fake)
    pred_of_real = disc(real.to(device))
    label_of_real = torch.ones_like(pred_of_real)
    loss_real = criterion(pred_of_real, label_of_real)
    disc_loss = (loss_fake + loss_real) / 2

    return disc_loss


def get_gen_loss(gen, disc, criterion, num_images, z_dim, device):
    '''
    Return the loss of the generator given inputs.
    Parameters:
        gen: the generator model, which returns an image given z-dimensional noise
        disc: the discriminator model, which returns a single-dimensional prediction of real/fake
        criterion: the loss function, which should be used to compare
               the discriminator's predictions to the ground truth reality of the images
               (e.g. fake = 0, real = 1)
        num_images: the number of images the generator should produce,
                which is also the length of the real images
        z_dim: the dimension of the noise vector, a scalar
        device: the device type
    Returns:
        gen_loss: a torch scalar loss value for the current batch
    '''
    #     These are the steps you will need to complete:
    #       1) Create noise vectors and generate a batch of fake images.
    #           Remember to pass the device argument to the get_noise function.
    #       2) Get the discriminator's prediction of the fake image.
    #       3) Calculate the generator's loss. Remember the generator wants
    #          the discriminator to think that its fake images are real

    noise_vec = get_noise(num_images, z_dim, device = device)
    fake_images = gen(noise_vec)
    pred_of_fake = disc(fake_images)
    label_of_fake = torch.ones_like(pred_of_fake)
    gen_loss = criterion(pred_of_fake, label_of_fake)
    return gen_loss


mean_generator_loss = 0
mean_discriminator_loss = 0

for epoch in range(n_epochs):
    for i , data in enumerate(dataloader):

        real = data[0].view(len(data[0]), -1).to(device)
        num_images = len(data[0])
        disc_loss = get_disc_loss(gen, disc, criterion= criterion, real= real, num_images= num_images, z_dim= z_dim, device=device)
        disc_opt.zero_grad()
        disc_loss.backward(retain_graph = True)
        disc_opt.step()

        gen_loss = get_gen_loss(gen, disc, criterion= criterion, num_images= num_images, z_dim= z_dim, device=device)
        gen_opt.zero_grad()
        gen_loss.backward()
        gen_opt.step()

        # Keep track of the average discriminator loss
        mean_discriminator_loss += disc_loss.item() / display_step

        # Keep track of the average generator loss
        mean_generator_loss += gen_loss.item() / display_step

        if i % display_step == 0 and (epoch % 2 == 0) :

            im_num = 25
            noise_vec = get_noise(im_num, z_dim, device=device)
            generated_images = gen(noise_vec)
            show_tensor_images(real, im_num, size=(1, 28, 28))
            show_tensor_images(generated_images, im_num, size=(1, 28, 28))
            print(f"epoch {epoch}: Generator loss: {mean_generator_loss}, discriminator loss: {mean_discriminator_loss}")
            mean_generator_loss = 0
            mean_discriminator_loss = 0



