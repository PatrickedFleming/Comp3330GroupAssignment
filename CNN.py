#This is heavily based of week 5 lab, notebook 4.
import torch
from Common import *
# Check which device is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {}'.format(device))

import os
import glob
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from Model1 import Model

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

# Wrap in DataLoader objects with batch size and shuffling preference
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=64, shuffle=True)
valid_loader = torch.utils.data.DataLoader(dataset=valid_data, batch_size=len(valid_data), shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=len(test_data), shuffle=False)

# Check number of batches
print("Train: {} batches".format(len(train_loader)))
print("Valid: {} batches".format(len(valid_loader))) # Should be 1
print("Test: {} batches".format(len(test_loader))) # Should be 1

print(valid_data[0][0].shape)

# Show some examples:
if False:
    fig = plt.figure()
    fig.set_figheight(12)
    fig.set_figwidth(12)
    for idx in range(16):
        ax = fig.add_subplot(4,4,idx+1)
        ax.axis('off')
        ax.set_title(REVERSED_TYPESDICT[train_data[idx][1]])
        plt.imshow(train_data[idx][0].permute(1,2,0))
    plt.axis('off')
    plt.show()


    
# Create an object from the model
model = Model().to(device)

# Set loss function and optimizer
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
loss_fn = torch.nn.CrossEntropyLoss()

train_losses, val_losses = [], []
train_accs, val_accs = [], []
# Train for 10 epochs
for epoch in range(75):
    ### Training
    # Track epoch loss and accuracy
    epoch_loss, epoch_accuracy = 0, 0
    # Switch model to training (affects batch norm and dropout)
    model.train()
    # Iterate through batches
    for i, (data, label) in enumerate(train_loader):
        # Reset gradients
        optimizer.zero_grad()
        # Move data to the used device
        data = data.to(device)
        label = label.to(device)
        # Forward pass
        output = model(data)
        loss = loss_fn(output, label)
        # Backward pass
        loss.backward()
        # Adjust weights
        optimizer.step()
        # Compute metrics
        acc = ((output.argmax(dim=1) == label).float().mean())
        epoch_accuracy += acc/len(train_loader)
        epoch_loss += loss/len(train_loader)
    print('Epoch: {}, train accuracy: {:.2f}%, train loss: {:.4f}'.format(epoch+1, epoch_accuracy*100, epoch_loss))
    train_losses.append(epoch_loss.item())
    train_accs.append(epoch_accuracy.item())
    ### Evaluation
    # Track epoch loss and accuracy
    epoch_valid_accuracy, epoch_valid_loss = 0, 0
    # Switch model to evaluation (affects batch norm and dropout)
    model.eval()
    # Disable gradients
    with torch.no_grad():
        # Iterate through batches
        for data, label in valid_loader:
            # Move data to the used device
            data = data.to(device)
            label = label.to(device)
            # Forward pass
            valid_output = model(data)
            valid_loss = loss_fn(valid_output, label)
            # Compute metrics
            acc = ((valid_output.argmax(dim=1) == label).float().mean())
            epoch_valid_accuracy += acc/len(valid_loader)
            epoch_valid_loss += valid_loss/len(valid_loader) 
    print('Epoch: {}, val accuracy: {:.2f}%, val loss: {:.4f}'.format(epoch+1, epoch_valid_accuracy*100, epoch_valid_loss))
    val_losses.append(epoch_valid_loss.item())
    val_accs.append(epoch_valid_accuracy.item())

fig, (ax1, ax2) = plt.subplots(2, figsize=(12, 8), sharex=True)
ax1.plot(train_losses, color='b', label='train')
ax1.plot(val_losses, color='g', label='valid')
ax1.set_ylabel("Loss")
ax1.legend()
ax2.plot(train_accs, color='b', label='train')
ax2.plot(val_accs, color='g', label='valid')
ax2.set_ylabel("Accuracy")
ax2.set_xlabel("Epoch")
ax2.legend()

test_accuracy, test_loss = 0, 0
with torch.no_grad():
    # Iterate through batches
    for data, label in test_loader:
        # Move data to the used device
        data = data.to(device)
        label = label.to(device)
        # Forward pass
        test_output_i = model(data)
        test_loss_i = loss_fn(test_output_i, label)
        # Compute metrics
        acc = ((test_output_i.argmax(dim=1) == label).float().mean())
        test_accuracy += acc/len(test_loader)
        test_loss += test_loss_i/len(test_loader)
        # Print Results
        if True:
            print("RESULTS")
            print(label)
            print(test_output_i.argmax(dim=1))
            print("")

print("Test loss: {:.4f}".format(test_loss))
print("Test accuracy: {:.2f}%".format(test_accuracy*100))
plt.show()

# SAVE MODEL
if True:
    torch.save(model, 'CNN.pth')
