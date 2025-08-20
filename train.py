import torch
from torch import nn
from torch import optim
from Modules.PerceptualLoss import PerceptualLoss

def train_afas_net(model, dataloader, num_epochs=100, device='cuda'):
    """Training function for AFAS-Net"""

    # Optimizers
    optimizer_G = optim.Adam(
        list(model.encoder.parameters()) + list(model.decoder.parameters()),
        lr=0.0002, betas=(0.5, 0.999)
    )
    optimizer_D = optim.Adam(
        model.discriminator.parameters(),
        lr=0.0002, betas=(0.5, 0.999)
    )

    # Loss functions
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCELoss()
    perceptual_loss = PerceptualLoss().to(device)

    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        for batch_idx, (cover, secret) in enumerate(dataloader):
            cover, secret = cover.to(device), secret.to(device)
            batch_size = cover.size(0)

            # Real and fake labels
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            # Forward pass
            outputs = model(cover, secret)
            stego = outputs['stego']
            revealed_secret = outputs['revealed_secret']

            # Train Generator (Encoder + Decoder)
            optimizer_G.zero_grad()

            # Reconstruction losses
            cover_loss = mse_loss(stego, cover)
            secret_loss = mse_loss(revealed_secret, secret)

            # Perceptual losses
            perceptual_cover_loss = perceptual_loss(stego, cover)
            perceptual_secret_loss = perceptual_loss(revealed_secret, secret)

            # Adversarial loss
            disc_fake_for_gen = model.discriminator(stego)
            adversarial_loss = bce_loss(disc_fake_for_gen, real_labels)

            # Frequency regularization loss
            freq_reg_loss = torch.mean(torch.abs(outputs['band_weights'] - 0.5))

            # Total generator loss
            gen_loss = (cover_loss * 10 + secret_loss * 10 +
                        perceptual_cover_loss * 0.1 + perceptual_secret_loss * 0.1 +
                        adversarial_loss * 0.01 + freq_reg_loss * 0.1)

            gen_loss.backward()
            optimizer_G.step()

            # Train Discriminator
            optimizer_D.zero_grad()

            # Real images
            disc_real = model.discriminator(cover)
            disc_real_loss = bce_loss(disc_real, real_labels)

            # Fake images
            disc_fake = model.discriminator(stego.detach())
            disc_fake_loss = bce_loss(disc_fake, fake_labels)

            # Total discriminator loss
            disc_loss = (disc_real_loss + disc_fake_loss) / 2

            disc_loss.backward()
            optimizer_D.step()

            epoch_loss += gen_loss.item()

            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}/{num_epochs}, Batch {batch_idx}/{len(dataloader)}, '
                      f'Gen Loss: {gen_loss.item():.4f}, Disc Loss: {disc_loss.item():.4f}')

        print(f'Epoch {epoch}/{num_epochs}, Average Loss: {epoch_loss / len(dataloader):.4f}')

        # Save model checkpoint
        if epoch % 10 == 0:
            torch.save(model.state_dict(), f'afas_net_epoch_{epoch}.pth')
