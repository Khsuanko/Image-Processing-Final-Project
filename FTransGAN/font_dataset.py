import os
import random

import h5py
import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2

# Define root folder
ROOT_DIR = './datasets/font/train/chinese'
DEFAULT_ROOT_DIR = './datasets/font/train/source'
FILE_PATH = './fonts_chinese_v2.h5'


def create_font_dataset(root_dir, default_dir, file_path):
    with h5py.File(file_path, 'w') as file:
        # Initialize metadata
        font_metadata = []
        current_idx = 0

        # Estimate total image count for pre-allocation
        total_images = sum(len([f for f in os.scandir(font_entry.path) if f.is_file() and f.name.endswith('.png')])
                           for font_entry in os.scandir(root_dir) if font_entry.is_dir())

        characters = dict()

        # Preallocate the dataset with chunking
        images = file.create_dataset(
            'images',
            shape=(total_images, 64, 64),
            dtype=np.uint8,
            chunks=(1000, 64, 64)  # Chunking for better I/O performance
            # compression='gzip'  # Enable compression
        )
        idx2char = []

        # Process images incrementally
        for font_entry in os.scandir(root_dir):
            if not font_entry.is_dir():
                continue

            start_idx = current_idx
            for image_entry in os.scandir(font_entry.path):
                if not image_entry.is_file() or not image_entry.name.endswith('.png'):
                    continue

                char = os.path.splitext(image_entry.name)[0]
                if char not in characters:
                    characters[char] = len(characters)
                    print(char)

                # Open image and write directly into the dataset
                with Image.open(image_entry.path) as image:
                    images[current_idx] = np.array(image.convert('L'), dtype=np.uint8)
                    # char_map[current_idx] = characters[char]
                    idx2char.append(characters[char])
                    current_idx += 1

            # End index for this font
            end_idx = current_idx
            if end_idx >= start_idx:
                # font_metadata.append((font_entry.name, start_idx, end_idx))
                font_metadata.append((font_entry.name, start_idx))
            print(f'{font_entry.name}: Start {start_idx}, End {end_idx}')

        # Save metadata as separate datasets
        font_names = np.array([f[0] for f in font_metadata], dtype=h5py.string_dtype(encoding='utf-8'))
        # font_indices = np.array([(f[1], f[2]) for f in font_metadata], dtype=np.int32)
        font_indices = np.array([f[1] for f in font_metadata] + [end_idx], dtype=np.int32)

        file.create_dataset('fonts', data=font_names)
        file.create_dataset('indices', data=font_indices)
        file.create_dataset('idx2char', data=np.array(idx2char, dtype=np.int32), chunks=(1000,))

        char2idx = [[] for _ in range(len(idx2char))]
        for idx, value in enumerate(idx2char):
            char2idx[value].append(idx)

        for idx, value in enumerate(char2idx):
            file.create_dataset(str(idx), data=np.array(value, dtype=np.int32))

        default = file.create_dataset(
            'default',
            shape=(len(characters), 64, 64),
            dtype=np.uint8
        )

        for idx, char in enumerate(characters.keys()):
            with Image.open(os.path.join(default_dir, f'{char}.png')) as image:
                default[idx] = np.array(image.convert('L'), dtype=np.uint8)

    print(f'HDF5 dataset created at {file_path} with {total_images} images.')
    print(f'{len(characters)} characters found in total.')


# create_font_dataset(ROOT_DIR, DEFAULT_ROOT_DIR, FILE_PATH)
# exit()


class FontDataset(Dataset):
    def __init__(self, file_path, device='cuda'):
        self.file_path = file_path
        self.data = h5py.File(self.file_path, 'r')
        self.fonts = list(self.data['fonts'])
        self.indices = list(self.data['indices'])
        self.transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=(0.5,), std=(0.5,))
        ])
        self.transform_adjustments = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.RandomRotation(5, fill=1.0),  # Random Rotation
            v2.RandomAdjustSharpness(sharpness_factor=0.3, p=0.5),  # Random Blur
            v2.GaussianNoise(mean=0, sigma=0.05, clip=False),  # Random Noise
            v2.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),  # Random Color Adjustment
            v2.Normalize(mean=(0.5,), std=(0.5,))  # Normalize
        ])
        self.device = device

    def __len__(self):
        return len(self.data['images'])

    def __getitem__(self, idx):
        # Pick a random font
        # start, end = random.choice(self.indices)

        # Pick a random character from the font as ground truth
        # gt_image_idx = random.randrange(start, end)
        # gt_image = self.transform(self.data['images'][gt_image_idx]).to(self.device)

        gt_image = self.transform(self.data['images'][idx]).to(self.device)

        font_idx = np.searchsorted(self.data['indices'], idx, side='right')
        start, end = self.data['indices'][font_idx - 1], self.data['indices'][font_idx]

        # Sample k random characters from the font as style images
        style_image_indices = random.sample(range(start, end), k=6)
        style_images = [self.transform_adjustments(self.data['images'][i]) for i in style_image_indices]
        style_images = torch.cat(style_images).to(self.device)

        char = self.data['idx2char'][idx]
        content_image_idx = random.choice(self.data[str(char)])
        content_image = self.transform_adjustments(self.data['images'][content_image_idx]).to(self.device)

        # img = torch.cat([gt_image, content_image, style_images]).cpu().squeeze().numpy()
        #
        # fig, axes = plt.subplots(2, 4, figsize=(12, 6))  # 2 rows, 4 columns
        # for i, ax in enumerate(axes.flat):
        #     ax.imshow(img[i], cmap='gray')
        #     ax.axis('off')  # Turn off axis
        #
        # plt.tight_layout()
        # plt.show()

        # Return as a PyTorch tensors
        return {
            'gt_images': gt_image,
            'content_images': content_image,
            'style_images': style_images
            # 'style_image_paths': style_paths,
            # 'image_paths': gt_path
        }


# dataset = FontDataset(FILE_PATH)
# sample = dataset[1000]
#
# img = torch.cat([sample['gt_images'], sample['content_images'], sample['style_images']]).cpu().squeeze().numpy()
#
# fig, axes = plt.subplots(2, 4, figsize=(12, 6))  # 2 rows, 4 columns
# for i, ax in enumerate(axes.flat):
#     ax.imshow(img[i], cmap='gray')
#     ax.axis('off')  # Turn off axis
#
# plt.tight_layout()
# plt.show()


class FontDatasetLoader:
    def __init__(self, file_path, batch_size=16, device='cuda'):
        self.dataset = FontDataset(file_path, device)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )

    def __len__(self):
        return len(self.dataset)


# dataset = FontDatasetLoader(FILE_PATH)
#
# print(len(dataset.dataloader))
#
# batch = next(iter(dataset.dataloader))
#
# for k, v in batch.items():
#     print(f'Key: {k}')
#     print(v.shape)

# # Create HDF5 file
# with h5py.File(hdf5_file, "w") as f:
#     # Traverse each font folder
#     with os.scandir(root_folder) as root_entries:
#         for font_entry in root_entries:
#             if font_entry.is_dir():  # Process directories only
#                 font_name = font_entry.name
#                 images = []  # Collect images for the font
#
#                 with os.scandir(font_entry.path) as image_entries:
#                     for image_entry in image_entries:
#                         if image_entry.is_file() and image_entry.name.endswith(".png"):
#                             with Image.open(image_entry.path) as img:
#                                 img = img.convert("L")  # Grayscale
#                                 images.append(np.array(img, dtype=np.uint8))  # Convert to NumPy
#
#                 if images:
#                     # Save the font group with its characters dataset
#                     images = np.stack(images)  # Shape: (num_images, height, width)
#                     f.create_dataset(f"{font_name}/characters", data=images, compression="gzip")
#
# print(f"Saved grouped font data to {hdf5_file}")


"""
# Create a random grayscale image tensor (size: 1x28x28)
image = torch.rand(1, 28, 28)

# Squeeze the batch dimension (if present)
image = image.squeeze(0)

# Plotting the image
plt.imshow(image, cmap='gray')
plt.axis('off')  # Hide axis
plt.show()
"""
