import os
import shutil
from PIL import Image
from argparse import ArgumentParser
from rich.progress import Progress

def check_image_sizes(image_folder: str, scale: float):
    """
    Check the original sizes of images in a folder and print the first image's size and scaled size.
    
    :param image_folder: str, path to the folder containing images.
    :param scale: float, scaling factor for downsampling.
    """
    try:
        images = [f for f in os.listdir(image_folder) if f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'tiff'))]
        if not images:
            print("No valid image files found in the folder.")
            return
        
        # Open the first image
        first_image_path = os.path.join(image_folder, images[0])
        img = Image.open(first_image_path)
        original_size = img.size  # (width, height)
        scaled_size = (int(original_size[0] * scale), int(original_size[1] * scale))
        
        print(f"First image: {images[0]}")
        print(f"Original size: {original_size}")
        print(f"Scaled size (scale={scale}): {scaled_size}")
    except Exception as e:
        print(f"Error checking image sizes: {e}")

def downsample_image(image_path: str, scale: float, output_folder: str):
    """
    Downsample the input image by the specified scale and save it to the output folder.
    
    :param image_path: str, path to the input image.
    :param scale: float, scale factor for downsampling (e.g., 0.5 for half the size).
    :param output_folder: str, folder where the downsampled image will be saved.
    """
    try:
        # Check if the file is a CSV
        if image_path.endswith('.csv'):
            # Ensure the output folder exists
            os.makedirs(output_folder, exist_ok=True)
            # Copy the file to the output folder
            shutil.copy(image_path, output_folder)
            return

        # Open the input image
        img = Image.open(image_path)

        # Calculate new size based on the scale
        original_size = img.size  # (width, height)
        target_size = (int(original_size[0] * scale), int(original_size[1] * scale))

        # Resize the image
        downsampled_img = img.resize(target_size, Image.Resampling.LANCZOS)

        # Ensure the output folder exists
        os.makedirs(output_folder, exist_ok=True)

        # Prepare the output file path
        file_name = os.path.basename(image_path)
        output_path = os.path.join(output_folder, file_name)

        # Save the downsampled image
        downsampled_img.save(output_path)
    except Exception as e:
        print(f"Error processing {image_path}: {e}")

if __name__ == "__main__":
    # Command-line arguments
    arg = ArgumentParser()
    arg.add_argument("--image_folder", type=str, required=True, help="Path to the folder containing images")
    arg.add_argument("--scale", type=float, required=True, help="Scaling factor for downsampling (e.g., 0.5 for half the size)")
    arg.add_argument("--output_folder", type=str, required=True, help="Path to the folder where results will be saved")
    args = arg.parse_args()

    # Check original image sizes
    check_image_sizes(args.image_folder, args.scale)

    # Process all files in the input folder with progress bar
    images = [f for f in os.listdir(args.image_folder) if f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'tiff', 'csv'))]
    total_files = len(images)
    
    with Progress() as progress:
        task = progress.add_task("Processing images...", total=total_files)
        
        for file_name in images:
            file_path = os.path.join(args.image_folder, file_name)
            downsample_image(file_path, args.scale, args.output_folder)
            progress.update(task, advance=1)
            
            
            
            # example :
            # python downsample.py --image_folder data/images --scale 0.5 --output_folder data/output_images