# import os

# # Define the directory path and the suffix to add to the image names
# dir_path = "/path/to/directory"
# suffix = "_ESRGAN_x6_WORLDSTRAT_192.png"

# # Get a list of file names in the directory
# file_names = os.listdir(dir_path)

# # Loop through each file name and rename it
# for file_name in file_names:
#     # Check if the file is a PNG image
#     if file_name.endswith(".png"):
#         # Create the new file name by adding the suffix to the original name
#         new_file_name = file_name.split(".")[0] + suffix + ".png"
#         # Use os.rename() function to rename the file
#         os.rename(os.path.join(dir_path, file_name), os.path.join(dir_path, new_file_name))

# ===================================================

import os

def rename_images(directory, suffix):
    for filename in os.listdir(directory):
        if filename.endswith(suffix):
            original_name = filename.replace(suffix, ".png")
            os.rename(os.path.join(directory, filename), os.path.join(directory, original_name))

if __name__ == "__main__":

    # directory_path = "D:/Remote Server vs code/VSCode_Projects/Test_Data/Test_SR_ESRGAN"
    # suffix = "_ESRGAN_x6_WORLDSTRAT_192.png"

    # directory_path = "D:/Remote Server vs code/VSCode_Projects/Test_Data/Test_SR_ESRGAN_woPT"
    # suffix = "_ESRGAN_x6_WORLDSTRAT_192_woPT.png"

    directory_path = "D:/Remote Server vs code/VSCode_Projects/Test_Data/Test_SR_CycleGAN"
    suffix = "_fake_B.png"
    rename_images(directory_path, suffix)
