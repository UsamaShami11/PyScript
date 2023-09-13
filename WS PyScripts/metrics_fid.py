import os
import time
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, ToTensor
# from torchvision import transforms
from PIL import Image
from piq import FID

start = time.time()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Set up the paths for the real and generated image directories
test_path = "C:/Users/User/Documents/Test_Data"

# # CycleGAN
# suffix1 = "Test_HR_960"
# suffix2 = "Test_SR_CycleGAN"

# # ESRGAN (with pretrained RRDBNet)
# suffix1 = "Test_HR_960"
# suffix2 = "Test_SR_ESRGAN"

# ESRGAN (without pretrained RRDBNet)
suffix1 = "Test_HR_960"
suffix2 = "Test_SR_ESRGAN_woPT"

real_dir = os.path.join(test_path, suffix1)
fake_dir = os.path.join(test_path, suffix2)

# Custom dataset loader to load images from directories without class-specific subfolders
class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform = None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_list = os.listdir(self.root_dir)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        image_path = os.path.join(self.root_dir, image_name)
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        return {'images': image}  # Return the image as a dictionary with the key 'images'

# Set batch size
batch_size = 32

# Transforms for the images
transform = Compose([ToTensor()])

# # Transforms for the images
# transform = transforms.Compose([
#     # transforms.Resize((960, 960)),
#     transforms.ToTensor(),
# ])

# Real images data loader
real_dataset = CustomImageDataset(real_dir, transform = transform)
real_loader = DataLoader(real_dataset, batch_size = batch_size, shuffle = False)

# Fake images data loader
fake_dataset = CustomImageDataset(fake_dir, transform = transform)
fake_loader = DataLoader(fake_dataset, batch_size = batch_size, shuffle = False)

# Calculate features for the real and fake images using InceptionV3
fid_metric = FID()
real_feats = fid_metric.compute_feats(real_loader, device = device)
fake_feats = fid_metric.compute_feats(fake_loader, device = device)

# Compute the FID Score
fid_value = fid_metric(fake_feats, real_feats)

print('Samples: {}'.format(len(real_dataset)))
print(f"FID Score: {fid_value.item():0.4f}")
# print("FID Score: ", fid_value)

end = time.time()
diff = end - start
print("Running time: %.2f seconds" %diff)

# python -m pytorch_fid C:/Users/User/Documents/Test_Data/Test_HR_960 C:/Users/User/Documents/Test_Data/Test_SR_CycleGAN --device cuda:0 