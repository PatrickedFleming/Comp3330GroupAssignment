import torch
TYPESDICT = {'Asparagus': 0, 'Carrotts': 1, 'Oysters': 2, 'Pork': 3, 'Salmon': 4, 'Zuccini': 5, 'Strawberries': 6, 'Sausages': 7, 'Garlic': 8, 'Ginger': 9, 'Cauliflower': 10, 'Capsicum': 11, 'Pumpkin': 12, 'Rockmelon': 13, 'Watermelon': 14, 'Avocado': 15, 'Tomato': 16, 'Pineapple': 17, 'Pears': 18, 'Apples': 19, 'Peach': 20, 'Trout': 21, 'Snapper': 22, 'Barra': 23, 'Prawns': 24, 'TropicalFish': 25, 'Steak': 26, 'Chicken': 27, 'Lamb': 28, 'Mushrooms': 29, 'RedOnion': 30, 'Tortellini': 31, 'Blueberries': 32, 'Lettuce': 33, 'Milk': 34, 'Eggs': 35, 'Juice': 36, 'Kiwi': 37, 'Butter': 38, 'Cheese': 39}

# Check which device is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {}'.format(device))

import os
import glob
import pandas as pd
from PIL import Image

# LOAD DATASET
class Dataset(torch.utils.data.Dataset):
    """FRIDGE INHABINTANTS dataset"""
    def __init__(self, data_dir, transform=None):
        self.image_paths = glob.glob(os.path.join(data_dir, '*.JPG'))
        self.image_paths.sort()
        labels_path = os.path.join(data_dir, "labels.csv")
        self.labels = None#pd.read_csv(labels_path, header=None).to_numpy()[:,1] if os.path.isfile(labels_path) else None
        self.transform = transform
        self.n = len(self.image_paths)
    def __len__(self):
        return self.n
    def __getitem__(self, idx):
        """Get one example"""
        img_path = self.image_paths[idx]
        img = Image.open(img_path)
        img_transformed = self.transform(img)
        if self.labels is None:
            label = TYPESDICT[os.path.basename(img_path).split('_')[1]]
        else:
            label = self.labels[idx]
        return img_transformed, label
    
# Transform / resize dataset:
from torchvision import transforms

# Transforms to prepare data and perform data augmentation
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(scale=(0.6, 1.0), size=(112,112)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])
eval_transforms = transforms.Compose([
    transforms.Resize(size=(112,112)),
    transforms.ToTensor()
])

train_data = Dataset("FoodDatasetTrain", transform=train_transforms)
valid_data = Dataset("FoodDatasetValidate", transform=eval_transforms)
test_data = Dataset("FoodDatasetTest", transform=eval_transforms)

print("Train: {} examples".format(len(train_data)))
print("Valid: {} examples".format(len(valid_data)))
print("Test: {} examples".format(len(test_data)))