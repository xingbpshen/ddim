import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class CelebAHQDataset(Dataset):
    def __init__(self, image_dir, attr_file, attrs=None, transform=None):
        self.image_dir = image_dir
        self.attr_df = pd.read_csv(attr_file, delim_whitespace=True, skiprows=1)
        self.transform = transform

        if attrs is None:
            self.attrs = self.attr_df.columns.tolist()  # uses all columns
        else:
            self.attrs = attrs

        for attr in self.attrs:
            if attr not in self.attr_df.columns:
                raise ValueError(f"Attribute '{attr}' not found in the attribute file.")

    def __len__(self):
        return len(self.attr_df)

    def __getitem__(self, idx):

        img_name = os.path.join(self.image_dir, self.attr_df.index[idx])
        image = Image.open(img_name).convert("RGB")

        if self.transform:
            image = self.transform(image)
        else:
            transform = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            image = transform(image)

        # Convert attributes from [-1, 1] to [0, 1]
        attr_values = {attr: (torch.tensor((float(self.attr_df.iloc[idx][attr]) + 1) / 2, dtype=torch.long)) for attr in
                       self.attrs}

        return image, attr_values
