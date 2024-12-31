import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
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

# Function to show the generated image
def show_generated_image(generated_image, save_path=None):
    # Convert the tensor back to an image
    generated_image = generated_image.squeeze(0).cpu()  # Remove batch dimension and move to CPU
    save_image(generated_image, save_path, normalize=True)  # Save to disk (optional)
    
    # Display the image
    plt.imshow(generated_image.permute(1, 2, 0).numpy())  # Convert to HxWxC format for display
    plt.axis('off')  # Hide axes
    plt.show()

# Main function to load models, run inference, and display results
def main(image_path, generator_weight_path, discriminator_weight_path, output_path=None):
    # Load the models with the trained weights
    model_G, model_D = load_model_weights(generator_weight_path, discriminator_weight_path)

    # Preprocess the input image
    input_image = preprocess_input_image(image_path)

    # Generate the image
    generated_image = generate_image(model_G, input_image)

    # Show and save the generated image
    show_generated_image(generated_image, save_path=output_path)

if __name__ == "__main__":
    image_path = '/home/anojha/test_datasets_from_summer/s1_0/ROIs1868_summer_s1_0_p99.png'  # Path to the single SAR image for testing
    generator_weight_path = '/home/anojha/SARtoOpt/models/gen_072.pt'  # Path to the generator weights
    discriminator_weight_path = '/home/anojha/SARtoOpt/models/dis_072.pt'  # Path to the discriminator weights
    output_image_path = '/home/anojha/test_generated_results/chunk15.png'  # Path to save the output image (optional)

    # Run the inference
    main(image_path, generator_weight_path, discriminator_weight_path, output_image_path)