import os
import cv2
import time

start = time.time()

# Set the path to the input directory containing original images
# lr
# input_dir = "D:/Data/WorldStrat_Improved/lr"
# hr
input_dir = "D:/Data/WorldStrat_Improved/hr"

# Set the path to the output directory for the down-scaled images
# lr 
output_dir = "D:/Data/WorldStrat_960/lr"
# hr 
# output_dir = "D:/Data/WorldStrat_960/hr"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Define the output image size
output_size = (960, 960)

# Loop through the images in the input directory
for filename in os.listdir(input_dir):
    # Check if the file is a png image
    if filename.endswith(".png"):
        # Read the image
        img = cv2.imread(os.path.join(input_dir, filename))
        # Downsample the hr image using cv2.resize
        downscaled_img = cv2.resize(img, output_size, interpolation=cv2.INTER_AREA)
        # Upsample the lr image using cv2.resize
        # upscaled_img = cv2.resize(img, output_size, interpolation=cv2.INTER_CUBIC)
        # Save the down-scaled image to the output directory
        cv2.imwrite(os.path.join(output_dir, filename), downscaled_img)

end = time.time()
diff = end - start
print("Running time: %.2f seconds" %diff)