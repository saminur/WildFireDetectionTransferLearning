import os
import random
import shutil

train_images_fire = "flame-2_dataset/train/fire"
train_images_no_fire = "flame-2_dataset/train/no_fire"
test_images_fire = "flame-2_dataset/test/fire"
test_images_no_fire = "flame-2_dataset/test/no_fire"

# Create the validation folders if they don't exist
os.makedirs(test_images_fire, exist_ok=True)
os.makedirs(test_images_no_fire, exist_ok=True)

# Get the list of image files in the train set
image_files_fire = os.listdir(train_images_fire)
image_files_no_fire = os.listdir(train_images_no_fire)

# Calculate the number of images to move to the test set
#num_test_images_fire = int(0.1 * len(image_files_fire))
num_test_images_fire = 2740 * 2
num_test_images_no_fire = int(0.1 * len(image_files_no_fire))

# Randomly select the images to move
test_image_files_fire = random.sample(image_files_fire, num_test_images_fire)
test_image_files_no_fire = random.sample(image_files_no_fire, num_test_images_no_fire)

# Move the selected images and their corresponding labels to the test set
for image_file in test_image_files_fire:
  # Move image file
  image_src = os.path.join(train_images_fire, image_file)
  image_dst = os.path.join(test_images_fire, image_file)
  shutil.move(image_src, image_dst)

# Move the selected images and their corresponding labels to the test set
for image_file in test_image_files_no_fire:
  # Move image file
  image_src = os.path.join(train_images_no_fire, image_file)
  image_dst = os.path.join(test_images_no_fire, image_file)
  shutil.move(image_src, image_dst)
