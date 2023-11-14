import cv2
import os
import sys

# Function to resize images
def resize_images(input_folder, output_folder, width, height):
    # Check if output folder exists, if not, create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate through all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
            # Read the image from input folder
            img_path = os.path.join(input_folder, filename)
            img = cv2.imread(img_path)

            # Check if image was successfully read
            if img is None:
                print(f"Error loading image {filename}")
                continue

            # Resize the image
            resized_img = cv2.resize(img, (width, height))

            # Save the resized image to output folder
            output_img_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_img_path, resized_img)
            print(f"Resized and saved {filename} to {output_folder}")

# Define your input and output folders and desired image dimensions
input_folder_path = 'sample_images'  # e.g., 'C:/images'
output_folder_path = 'reized_images'  # e.g., 'C:/resized_images'
new_width = 1600 # e.g., 800 pixels
new_height =  800 # e.g., 600 pixels

# Run the function
resize_images(input_folder_path, output_folder_path, new_width, new_height)
