import torch
import os
import glob
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

IMAGESPATH = "FoodDatasetTest"

TYPESDICT = {'Asparagus': 0, 'Carrotts': 1, 'Oysters': 2, 'Pork': 3, 'Salmon': 4, 'Zuccini': 5, 'Strawberries': 6, 'Sausages': 7, 'Garlic': 8, 'Ginger': 9, 'Cauliflower': 10, 'Capsicum': 11, 'Pumpkin': 12, 'Rockmelon': 13, 'Watermelon': 14, 'Avocado': 15, 'Tomato': 16, 'Pineapple': 17, 'Pears': 18, 'Apples': 19, 'Peach': 20, 'Trout': 21, 'Snapper': 22, 'Barra': 23, 'Prawns': 24, 'TropicalFish': 25, 'Steak': 26, 'Chicken': 27, 'Lamb': 28, 'Mushrooms': 29, 'RedOnion': 30, 'Tortellini': 31, 'Blueberries': 32, 'Lettuce': 33, 'Milk': 34, 'Eggs': 35, 'Juice': 36, 'Kiwi': 37, 'Butter': 38, 'Cheese': 39}
REVERSED_TYPESDICT = {v: k for k, v in TYPESDICT.items()}

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


# CREATE THE MODEL
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv_layer_1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=3, padding='same'),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.conv_layer_2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, kernel_size=3, padding='same'),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.conv_layer_3 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, padding='same'),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.conv_layer_4 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, padding='same'),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.conv_layer_5 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, kernel_size=3, padding='same'),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.conv_layer_6 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 512, kernel_size=3, padding='same'),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )


        self.fc_layer_1 = torch.nn.Sequential(
            torch.nn.Linear(3*3*256, 128), #torch.nn.Linear(7*7*128, 128),
            torch.nn.Dropout(),
            torch.nn.ReLU()
        )
        self.fc_layer_2 = torch.nn.Linear(128, len(TYPESDICT))
    def forward(self, x):
        x = self.conv_layer_1(x)
        x = self.conv_layer_2(x)
        x = self.conv_layer_3(x)
        x = self.conv_layer_4(x)
        x = self.conv_layer_5(x)
        #x = self.conv_layer_6(x)
        
        x = x.view(x.size(0), -1) # Flatten
        x = self.fc_layer_1(x)
        x = self.fc_layer_2(x)
        return x
    
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
    