import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
from PIL import Image
from torchvision.utils import save_image
from train_first import Generator, Discriminator  # Assuming your model is in this file

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Gray(object):
    def __call__(self, img):
        gray = img.convert('L')
        return gray

# Load the trained weights for the Generator and Discriminator
def load_model_weights(generator_path, discriminator_path):
    model_G = Generator().to(device)
    model_D = Discriminator().to(device)

    # Wrap models in DataParallel to match the saved weights
    model_G = torch.nn.DataParallel(model_G)
    model_D = torch.nn.DataParallel(model_D)

    # Load the state_dict for Generator
    model_G.load_state_dict(torch.load(generator_path, map_location=device))
    model_G.eval()  # Set to evaluation mode
    
    # Load state_dict for Discriminator
    model_D.load_state_dict(torch.load(discriminator_path, map_location=device))
    model_D.eval()  # Set to evaluation mode

    return model_G, model_D

# Prepare the input image (single SAR image)
def preprocess_input_image(image_path):
    transform = transforms.Compose([
        Gray(),  # Assuming you have the Gray class defined
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,)),
    ])
    
    image = Image.open(image_path).convert('RGB')  # Assuming the input image is a color image
    image = transform(image).unsqueeze(0)  # Add batch dimension (1, C, H, W)
    return image.to(device)

# Function to run inference
def generate_image(model_G, input_image):
    with torch.no_grad():  # Disable gradient calculation for inference
        generated_image = model_G(input_image)
    return generated_image

# Function to show and save the generated image
def show_generated_image(generated_image, save_path=None):
    # Convert the tensor back to an image
    generated_image = generated_image.squeeze(0).cpu()  # Remove batch dimension and move to CPU
    save_image(generated_image, save_path, normalize=True)  # Save to disk (optional)
    
    # Display the image
    plt.imshow(generated_image.permute(1, 2, 0).numpy())  # Convert to HxWxC format for display
    plt.axis('off')  # Hide axes
    plt.show()

# Main function to load models, run inference on all files in a folder, and display/save results
def process_images_in_folders(main_folder, generator_weight_path, discriminator_weight_path, output_folder=None):
    # Load the models with the trained weights
    model_G, model_D = load_model_weights(generator_weight_path, discriminator_weight_path)

    # Walk through all subfolders in the main folder
    for root, _, files in os.walk(main_folder):
        # Get all image files in the current folder (e.g., .png, .jpg files)
        image_paths = [os.path.join(root, file) for file in files if file.endswith(('.png', '.jpg', '.jpeg'))]
        
        # Loop through each image file
        for image_path in image_paths:
            print(f"Processing {image_path}...")

            # Preprocess the input image
            input_image = preprocess_input_image(image_path)

            # Generate the image
            generated_image = generate_image(model_G, input_image)

            # Construct output file name and save the generated image
            if output_folder:
                # Maintain directory structure
                relative_path = os.path.relpath(root, main_folder)  # Get relative path of the current subfolder
                output_dir = os.path.join(output_folder, relative_path)  # Create output path with subfolder structure
                os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists
                
                # Get the filename and create the output file path
                filename = os.path.basename(image_path)
                output_image_path = os.path.join(output_dir, filename)
            else:
                output_image_path = None

            # Show and save the generated image
            show_generated_image(generated_image, save_path=output_image_path)

if __name__ == "__main__":
    main_folder = '/home/anojha/test_datasets_from_summer/s1_t'  # Main folder containing subfolders
    generator_weight_path = '/home/anojha/first_trained_result_copy/models/gen_099.pt'  # Path to the generator weights
    discriminator_weight_path = '/home/anojha/first_trained_result_copy/models/dis_099.pt'  # Path to the discriminator weights
    output_folder = '/home/anojha/test_generated_results/100_epoch'  # Main folder to save the output images

    # Run the inference on all images in all subfolders within the main folder
    process_images_in_folders(main_folder, generator_weight_path, discriminator_weight_path, output_folder)