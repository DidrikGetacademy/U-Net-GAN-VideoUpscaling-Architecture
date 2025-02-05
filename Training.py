import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)
import torch
import os
import sys
import deepspeed
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, project_root)
from UNET_Discriminator import UNET, Discriminator  
from Datasets import Vimeo90KDataset  # Ensure this dataset class is correctly implemented

#NOTES---> GAN Loss: Vurder å bruke MSE Loss eller Perceptual Loss for mer stabil trening.  
#DeepSpeed: Sjekk konfigurasjonen i ds_config.json for korrekt innstilling av mikrobatching og gradientakkumulering.





# ==========================
# CONFIGURATION
# ==========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
NUM_EPOCHS = 10
DS_CONFIG = "ds_config.json"  # Ensure this config file exists

# ==========================
# DATASET & DATALOADER
# ==========================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load the Vimeo90K dataset
train_dataset = Vimeo90KDataset(root_dir="data/Vimeo-90K-Septuplet", mode="train", transform=transform)
test_dataset = Vimeo90KDataset(root_dir="data/Vimeo-90K-Septuplet", mode="test", transform=transform)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# Debugging: Check sample data
for lr, hr in train_loader:
    print("Low-Res Image Shape:", lr.shape)
    print("High-Res Image Shape:", hr.shape)
    break

# ==========================
# INITIALIZE MODELS
# ==========================
generator = UNET(in_channels=3, out_channels=3).to(DEVICE)
discriminator = Discriminator(in_channels=3).to(DEVICE)

# Define loss function (Binary Cross-Entropy with Logits)
criterion = torch.nn.BCEWithLogitsLoss()

# ==========================
# TRAINING FUNCTION
# ==========================
def train_gan(generator, discriminator, dataloader, criterion, num_epochs=NUM_EPOCHS):

    # Wrap generator with DeepSpeed
    generator, optimizer_g, _, _ = deepspeed.initialize(
        model=generator,
        model_parameters=generator.parameters(),
        config=DS_CONFIG
    )

    # Wrap discriminator with DeepSpeed
    discriminator, optimizer_d, _, _ = deepspeed.initialize(
        model=discriminator,
        model_parameters=discriminator.parameters(),
        config=DS_CONFIG
    )

    for epoch in range(num_epochs):
        for i, (low_res, high_res) in enumerate(dataloader):
            low_res, high_res = low_res.to(DEVICE), high_res.to(DEVICE)

            # =====================
            # TRAIN DISCRIMINATOR
            # =====================
            optimizer_d.zero_grad()
            
            real_labels = torch.ones((low_res.size(0), 1), device=DEVICE)
            fake_labels = torch.zeros((low_res.size(0), 1), device=DEVICE)

            # Discriminator on Real Data
            real_output = discriminator(high_res)
            d_loss_real = criterion(real_output, real_labels)

            # Discriminator on Fake Data
            fake_images = generator(low_res)
            fake_output = discriminator(fake_images.detach())  # Detach to prevent generator gradients here
            d_loss_fake = criterion(fake_output, fake_labels)

            # Total Discriminator Loss
            d_loss = d_loss_real + d_loss_fake
            discriminator.backward(d_loss)
            optimizer_d.step()

            # =====================
            # TRAIN GENERATOR
            # =====================
            optimizer_g.zero_grad()

            # Generator tries to fool the Discriminator
            output = discriminator(fake_images)
            g_loss = criterion(output, real_labels)  # Generator wants Discriminator to classify fake as real

            generator.backward(g_loss)
            optimizer_g.step()

            # Print progress
            if i % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], "
                      f"D Loss: {d_loss.item()}, G Loss: {g_loss.item()}")

# ==========================
# RUN TRAINING
# ==========================
if __name__ == "__main__":
    train_gan(generator, discriminator, train_loader, criterion, num_epochs=NUM_EPOCHS)
