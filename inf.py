import torch
import os
import glob
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from Model1 import Model
from Common import *
IMAGESPATH = "FoodDatasetTest"



# Check which device is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {}'.format(device))


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

print("Loaded dataset from {}".format(IMAGESPATH))
test_data = Dataset(IMAGESPATH, transform=eval_transforms)
print("{} examples".format(len(test_data)))
test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=len(test_data), shuffle=False)
print("{} batches".format(len(test_loader))) # Should be 1



    
# Create an object from the model
model = torch.load("CNN2.pth").to(device)
model.eval()
for data, label in test_loader:
    data = data.to(device)
    label = label.to(device)

    modelResult = model(data)
    
    maxx = torch.max(modelResult, dim=1)

    acc = ((modelResult.argmax(dim=1) == label).float().mean())
    

print("Accuracy: " + str(round(float(acc)*100,2)) + "%")
    