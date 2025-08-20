from torch import nn
from Modules.Encoder import EncoderNetwork
from Modules.Decoder import DecoderNetwork
from Modules.Discriminator import DiscriminatorNetwork

class AFASNet(nn.Module):
    """Complete AFAS-Net model"""

    def __init__(self):
        super(AFASNet, self).__init__()
        self.encoder = EncoderNetwork()
        self.decoder = DecoderNetwork()
        self.discriminator = DiscriminatorNetwork()

    def forward(self, cover, secret):
        # Encoding phase
        stego, band_weights, attention_map = self.encoder(cover, secret)

        # Decoding phase
        revealed_secret = self.decoder(stego)

        # Discrimination
        disc_real = self.discriminator(cover)
        disc_fake = self.discriminator(stego.detach())

        return {
            'stego': stego,
            'revealed_secret': revealed_secret,
            'band_weights': band_weights,
            'attention_map': attention_map,
            'disc_real': disc_real,
            'disc_fake': disc_fake
        }
