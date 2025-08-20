from AFASNet import AFASNet
from Modules.eval import evaluate_metrics
from Data.Dataset import Dataset, SteganographyDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from train import train_afas_net
import torch
model = AFASNet()
cover = torch.randn(2, 3, 256, 256)
secret = torch.randn(2, 3, 256, 256)
with torch.no_grad():
    outputs = model(cover, secret)
    print("Forward pass successful!")
    print(f"Stego image shape: {outputs['stego'].shape}")
    print(f"Revealed secret shape: {outputs['revealed_secret'].shape}")
    metrics = evaluate_metrics(cover, outputs['stego'], secret, outputs['revealed_secret'])
    print(f"Metrics: {metrics}")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Uncomment to train with actual data
dataset = SteganographyDataset(transform=transform, root_dir="Data/Animals-10")
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
train_afas_net(model, dataloader, num_epochs=100)

print("AFAS-Net implementation complete!")