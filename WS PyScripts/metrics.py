import os
import time
import torch
from torchvision.transforms import ToTensor
from PIL import Image
from piq import psnr, ssim, multi_scale_ssim, brisque

start = time.time()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## Define the directory path and the suffix to add to geneate path
test_path = "C:/Users/User/Documents/Test_Data"
suffix1 = "Test_HR_960"
suffix2 = "Test_SR_CycleGAN"

# suffix1 = "Test_HR_960"
# suffix2 = "Test_SR_ESRGAN"

# suffix1 = "Test_HR_960"
# suffix2 = "Test_SR_ESRGAN_woPT"

path1 = os.path.join(test_path, suffix1)
path2 = os.path.join(test_path, suffix2)

transform = ToTensor()

sum_psnr = 0
sum_ssim = 0
sum_ms_ssim = 0
sum_brisque = 0

count = 0

# Loop through each file to find relevant image pairs
for file_name in os.listdir(path1):

    # Read Image, Convert to torch Tensor, Add 4th Dimension and Send to GPU if available
    x = Image.open(os.path.join(path1, file_name)).convert('RGB')
    x = transform(x).unsqueeze(0).to(device)

    y = Image.open(os.path.join(path2, file_name)).convert('RGB')
    y = transform(y).unsqueeze(0).to(device)

    # Compute PSNR
    psnr_index = psnr(x, y, data_range=1.)
    # print(f"PSNR index: {psnr_index.item():0.4f}")

    # Compute SSIM
    ssim_index = ssim(x, y, data_range=1.)
    # print(f"SSIM index: {ssim_index.item():0.4f}")

    # Compute MS-SSIM
    ms_ssim_index = multi_scale_ssim(x, y, data_range=1.)
    # print(f"MS-SSIM index: {ssim_index.item():0.4f}")

    # Compute BRISQUE
    brisque_index = brisque(y, data_range=1.)
    # print(f"BRISQUE index: {brisque_index.item():0.4f}")

    count += 1
    sum_psnr += psnr_index
    sum_ssim += ssim_index
    sum_ms_ssim += ms_ssim_index
    sum_brisque += brisque_index

avg_psnr = sum_psnr/count
avg_ssim = sum_ssim/count
avg_msssim = sum_ms_ssim/count
avg_brisque = sum_brisque/count


print('Samples: {}'.format(count))
print('Avg PSNR: %.4f | Avg SSIM: %.4f | Avg MS-SSIM: %.4f' % (avg_psnr, avg_ssim, avg_msssim))
print('Avg BRISQUE: %.4f' % avg_brisque)

end = time.time()
diff = end - start
print("Running time: %.2f seconds" %diff)