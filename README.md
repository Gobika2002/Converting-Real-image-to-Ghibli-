# Converting-Real-image-to-Ghibli-
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt

# =========================
# 🔧 Configuration
# =========================
CONTENT_PATH = "woman.jpg"
STYLE_PATH = "ghibli_model.jpg"
IMAGE_SIZE = 256  # Size for input images
STEPS = 100  # Number of optimization steps
STYLE_WEIGHT = 1e6
CONTENT_WEIGHT = 1

# =========================
# 📦 Utility Functions
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_and_preprocess(image_path, size=IMAGE_SIZE):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(), # Convert image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]) # Normalize using ImageNet stats
    ])
    image = transform(image).unsqueeze(0) # Add batch dimension
    return image.to(device)
# =========================
# 🖼️ Image Conversion
def tensor_to_image(tensor):
    image = tensor.clone().detach().cpu().squeeze(0)
    # Denormalize the image
    image = image * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) # Scale back to original range
    # Convert to RGB format
    image = image + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1) # Add mean back
    image = image.numpy().transpose(1, 2, 0) # Change from CHW to HWC format
    image = (image * 255).clip(0, 255).astype('uint8') # Convert to uint8 format
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # Convert back to BGR for OpenCV compatibility

# =========================
# 🧠 VGG Feature Extractor
# =========================
class VGGExtractor(nn.Module):
    def __init__(self):
        super(VGGExtractor, self).__init__() # Initialize the VGG model
        # Load the VGG19 model and extract features from specific layers
        self.model = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features[:36].eval() # Use the first 36 layers of VGG19
        # Freeze the model parameters to prevent training
        for param in self.model.parameters():
            param.requires_grad = False 
    # Move the model to the appropriate device
    def forward(self, x):
        features = []
        for i, layer in enumerate(self.model):
            x = layer(x)
            if i in {1, 6, 11, 20, 29}:
                features.append(x)
        return features

# =========================
# 🧮 Gram Matrix
# =========================
def gram_matrix(tensor):
    # Calculate the Gram matrix for a given tensor
    # The Gram matrix is used to capture the style of an image
    b, c, h, w = tensor.size()
    features = tensor.view(b, c, h * w)
    G = torch.bmm(features, features.transpose(1, 2)) 
    return G / (c * h * w)

# =========================
# 🎨 Style Transfer Logic
# =========================
# This function performs the style transfer by optimizing the target image to match the content and style of the input images
# It uses the VGG feature extractor to compute content and style features, and then optimizes
def stylize(content, style, extractor, steps=STEPS, style_weight=STYLE_WEIGHT, content_weight=CONTENT_WEIGHT):
    target = content.clone().requires_grad_(True).to(device)
    optimizer = optim.Adam([target], lr=0.003)

    with torch.no_grad():
        content_features = extractor(content)
        style_features = extractor(style)

    for step in range(steps):
        target_features = extractor(target)

        content_loss = torch.mean((target_features[2] - content_features[2])**2)
        style_loss = sum(torch.mean((gram_matrix(tf) - gram_matrix(sf))**2)
                         for tf, sf in zip(target_features, style_features))

        total_loss = content_weight * content_loss + style_weight * style_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print(f"[Step {step}] Loss: {total_loss.item():.2f}")

    return target

# =========================
# 🚀 Main Execution
# =========================
# This is the main execution block that runs the style transfer process
# It loads the content and style images, processes them, and applies the style transfer
if __name__ == "__main__":
    try:
        print("🔹 Loading images...")
        content_img = load_and_preprocess(CONTENT_PATH)
        style_img = load_and_preprocess(STYLE_PATH)

        print("🔹 Running style transfer...")
        extractor = VGGExtractor().to(device)
        output_tensor = stylize(content_img, style_img, extractor)

        print("🔹 Saving and displaying result...")
        output_image = tensor_to_image(output_tensor)
        cv2.imwrite("ghibli_output.jpg", output_image)

        # Display using matplotlib
        plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
        plt.title("Ghibli Stylized Image")
        plt.axis("off")
        plt.show()

    except Exception as e:
        print(f"❌ Error: {e}")
