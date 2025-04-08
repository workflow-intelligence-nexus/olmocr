import os
from PIL import Image
import argparse

def convert_images_to_pdfs(input_folder, output_folder):
    # Create the output folder if it doesn't exist.
    os.makedirs(output_folder, exist_ok=True)
    
    # Supported image extensions.
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')
    
    # Loop through all files in the input folder.
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(image_extensions):
            input_path = os.path.join(input_folder, filename)
            # Open the image file.
            try:
                with Image.open(input_path) as img:
                    # Convert image to RGB (required for PDF conversion)
                    rgb_img = img.convert("RGB")
                    # Define the output PDF filename.
                    base_name, _ = os.path.splitext(filename)
                    output_pdf = os.path.join(output_folder, base_name + ".pdf")
                    # Save the image as PDF.
                    rgb_img.save(output_pdf, "PDF")
                    print(f"Converted {filename} -> {output_pdf}")
            except Exception as e:
                print(f"Failed to convert {filename}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert all images in a folder to PDFs.")
    parser.add_argument("input_folder", help="Folder containing image files.")
    parser.add_argument("output_folder", help="Folder to save converted PDFs.")
    args = parser.parse_args()
    
    convert_images_to_pdfs(args.input_folder, args.output_folder)