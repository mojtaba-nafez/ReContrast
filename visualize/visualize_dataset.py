import torch
import random
from torchvision import transforms
import matplotlib.pyplot as plt
import os
import zipfile

def show_images(images, labels, dataset_name):
    num_images = len(images)
    rows = int(num_images / 5) + 1

    fig, axes = plt.subplots(rows, 5, figsize=(15, rows * 3))

    for i, ax in enumerate(axes.flatten()):
        if i < num_images:
            ax.imshow(images[i].permute(1, 2, 0))  # permute to (H, W, C) for displaying RGB images
            ax.set_title(f"Label: {labels[i]}")
        ax.axis("off")

    plt.savefig(f'{dataset_name}_visualization.png')


def visualize_random_samples_from_clean_dataset(dataset, dataset_name):
    print(f"Start visualization of clean dataset: {dataset_name}")
    # Choose 20 random indices from the dataset
    if len(dataset) > 20:
        random_indices = random.sample(range(len(dataset)), 20)
    else:
        random_indices = [i for i in range(len(dataset))]

    # Retrieve corresponding samples
    random_samples = [dataset[i] for i in random_indices]

    # Separate images and labels
    images, labels = zip(*random_samples)

    # # Convert PIL images to PyTorch tensors
    # transform = transforms.ToTensor()
    # images = [transform(image) for image in images]

    # Convert labels to PyTorch tensor
    labels = torch.tensor(labels)

    # Show the 20 random samples
    show_images(images, labels, dataset_name)

def zip_all_visualization_results():

    # Get the current directory
    current_dir = os.getcwd()

    # List all files in the current directory
    files = os.listdir(current_dir)

    # Filter out only the .png files
    png_files = [file for file in files if file.endswith('.png')]

    if not png_files:
        print("No PNG files found in the current directory.")
        return

    # Create a zip file
    with zipfile.ZipFile('png_files.zip', 'w') as zipf:
        for file in png_files:
            file_path = os.path.join(current_dir, file)
            zipf.write(file_path, os.path.basename(file_path))

    print("Zip file created successfully.")