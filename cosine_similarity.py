import torch
import torchvision.transforms as tr
from torchvision import models
import torch.nn as nn
from PIL import Image
import numpy as np
from tqdm import tqdm
import os
import csv

class ImageComparer:
    def __init__(self, device=None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def model(self):
        """Instantiates the feature extracting model 
        
        Returns
        -------
        Vision Transformer model object
        """
        wt = models.ViT_H_14_Weights.DEFAULT
        model = models.vit_h_14(weights=wt)
        model.heads = nn.Sequential(*list(model.heads.children())[:-1])  # Remove final classification layer
        model = model.to(self.device)
        return model

    def process_test_image(self, image_path):
        """Processing images
        
        Parameters
        ----------
        image_path : str
        Returns
        -------
        Processed image
        """
        img = Image.open(image_path)

        # If the image is grayscale (1 channel), convert it to RGB (3 channels)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        transformations = tr.Compose([tr.ToTensor(),
                                      tr.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                      tr.Resize((518, 518))])  # Resize to fit ViT input size
        img = transformations(img).float()
        img = img.unsqueeze_(0)  # Add batch dimension
        
        img = img.to(self.device)
        return img

    def get_embeddings(self, image_path_1, image_path_2):
        """Computes embeddings given images
        
        Returns
        -------
        embeddings: np.ndarray
        """
        img1 = self.process_test_image(image_path_1)
        img2 = self.process_test_image(image_path_2)
        model = self.model()

        # Get embeddings for both images
        emb_one = model(img1).detach().cpu()
        emb_two = model(img2).detach().cpu()

        return emb_one, emb_two

    def compute_scores(self, image_path_1, image_path_2):
        """Computes cosine similarity between two image embeddings."""
        emb_one, emb_two = self.get_embeddings(image_path_1, image_path_2)
        scores = torch.nn.functional.cosine_similarity(emb_one, emb_two)
        return scores.numpy().tolist()


def compare_images_in_folders(folder_1, folder_2, output_csv_path,epoch):
    comparer = ImageComparer()

    # Open the CSV file for writing
    with open(output_csv_path, mode='w', newline='') as csvfile:
        fieldnames = ['image_path_1', 'image_path_2', 'cosine_similarity','epoch']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write the header to the CSV
        writer.writeheader()

        # Loop through all images in folder_1
        for img1_name in tqdm(os.listdir(folder_1), desc="Processing images", unit="image"):
            img1_path = os.path.join(folder_1, img1_name)

            if not img1_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue  # Skip non-image files

            # Check if the corresponding image exists in folder_2
            img2_path = os.path.join(folder_2, img1_name)
            if os.path.exists(img2_path):
                # Compute cosine similarity score between the images
                similarity_score = comparer.compute_scores(img1_path, img2_path)
                print(similarity_score)
                # Write the result to the CSV file
                writer.writerow({
                    'image_path_1': img1_path,
                    'image_path_2': img2_path,
                    'cosine_similarity': similarity_score[0],
                    'epoch':epoch
                })
            else:
                print(f"Warning: No matching file found for {img1_name} in {folder_2}")

    print(f"Similarity scores have been saved to {output_csv_path}")


# Example usage
folder_1 = '/home/anojha/test_datasets_from_summer/s2_t'
folder_2_template = '/home/anojha/test_generated_results/{epoch}_epoch'
output_csv_template = '/home/anojha/metrics/similarity_scores{epoch}.csv'

# List of epochs you want to process
epochs = [0, 20, 40, 60, 80, 100]

for ii in epochs:
    folder_2 = folder_2_template.format(epoch=ii)
    output_csv_path = output_csv_template.format(epoch=ii)
    compare_images_in_folders(folder_1, folder_2, output_csv_path, epoch=ii)
