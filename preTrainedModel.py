import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, Subset
from torchvision import models, datasets, transforms
from torch import nn, optim
from sklearn.model_selection import train_test_split
import sys
if __name__ == '__main__':
    modelToRun = sys.argv[1]
    SHOW_PLOTS = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose([
        transforms.Resize((224,224)), #resize the image to 224x224
        transforms.ToTensor(),  #convert the image to a pytorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), #normalize the image
    ])

    # LOAD IN ENTIRE TRAINING DATASET
    dataset = datasets.ImageFolder(root='./dataset', transform=data_transform)

    class_names = dataset.classes

    train_size = 0.80
    val_size = 0.20


    #dataset given by teacher
    # LOAD IN THE TEACHER DATA
    teacherDataset = datasets.ImageFolder(root='./FoodDatasetComplete', transform=data_transform)


    # Split indices into training and combined validation/test indices
    train_indices, val_indices = train_test_split(range(len(dataset)), test_size=(val_size), random_state=42)


    train_set = Subset(dataset, train_indices)  # Subset for training datata
    test_set = teacherDataset               # Subset for validation data given by teacher
    val_set = Subset(dataset, val_indices)    # Subset for test data

    batch_size = 32
    num_workers = 4 #default 0
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    valid_loader = DataLoader(val_set, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=num_workers, shuffle=True)


    from torch.optim.lr_scheduler import ReduceLROnPlateau  # Import ReduceLROnPlateau scheduler from PyTorch

    class CustomCallback:
        def __init__(self, early_stop_patience=5, reduce_lr_factor=0.2, reduce_lr_patience=3, reduce_lr_min_lr=0.0000001, checkpoint_path='checkpoint.pth', log_dir='logs'):
            # Initialize callback parameters
            self.early_stop_patience = early_stop_patience  # Patience for early stopping
            self.reduce_lr_factor = reduce_lr_factor  # Factor by which to reduce learning rate
            self.reduce_lr_patience = reduce_lr_patience  # Patience for reducing learning rate
            self.reduce_lr_min_lr = reduce_lr_min_lr  # Minimum learning rate
            self.checkpoint_path = checkpoint_path  # Path to save model checkpoints
            # self.log_dir = log_dir  # Directory for logging

            # Initialize variables for early stopping
            self.early_stop_counter = 0  # Counter for early stopping
            self.best_val_loss = float('inf')  # Best validation loss

            self.optimizer = None  # Optimizer for training
            self.scheduler = None  # Learning rate scheduler

        def set_optimizer(self, optimizer):
            # Set optimizer for training
            self.optimizer = optimizer

        def on_epoch_end(self, epoch, val_loss):
            # Early Stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.early_stop_counter = 0  # Reset counter if validation loss improves
            else:
                self.early_stop_counter += 1  # Increment counter if validation loss does not improve

            if self.early_stop_counter >= self.early_stop_patience:
                print("Early stopping triggered!")
                return True  # Stop training if early stopping criterion is met

            # Reduce LR on Plateau
            if self.scheduler is not None:
                self.scheduler.step(val_loss)  # Adjust learning rate based on validation loss

            return False  # Continue training

        def on_train_begin(self):
            # Initialize Reduce LR on Plateau scheduler
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=self.reduce_lr_factor,
                                                patience=self.reduce_lr_patience, min_lr=self.reduce_lr_min_lr)

        def on_train_end(self):
            pass

        def set_model(self, model):
            self.model = model  # Set model for the callback

    def trainModel(model, setepochs):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        custom_callback = CustomCallback()
        custom_callback.set_optimizer(optimizer)
        model.to(device)
        custom_callback.set_model(model)
        lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, min_lr=1e-6)
        train_losses = []  # List to store training losses
        train_accuracies = []  # List to store training accuracies
        val_losses = []  # List to store validation losses
        val_accuracies = []  # List to store validation accuracies

        # Training loop
        num_epochs = setepochs  # Number of epochs for training
        for epoch in range(num_epochs):
            # Training
            model.train()  # Set the model to training mode
            running_train_loss = 0.0  # Initialize running training loss
            correct_train = 0  # Initialize number of correctly predicted training samples
            total_train = 0  # Initialize total number of training samples
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU if available
                optimizer.zero_grad()  # Zero the parameter gradients
                outputs = model(inputs)  # Forward pass
                if isinstance(outputs, tuple):
                    logits = outputs[0]  # Unpack logits if model returns tuple
                else:
                    logits = outputs
                loss = criterion(logits, labels)  # Calculate loss
                loss.backward()  # Backward pass
                optimizer.step()  # Optimize parameters

                running_train_loss += loss.item() * inputs.size(0)  # Accumulate training loss
                _, predicted = torch.max(logits, 1)  # Get predicted labels
                total_train += labels.size(0)  # Increment total training samples
                correct_train += (predicted == labels).sum().item()  # Increment correctly predicted samples

            # Calculate epoch-wise training loss and accuracy
            epoch_train_loss = running_train_loss / len(train_loader.dataset)  # Average training loss
            train_accuracy = correct_train / total_train  # Training accuracy

            # Validation
            model.eval()  # Set the model to evaluation mode
            running_val_loss = 0.0  # Initialize running validation loss
            correct_val = 0  # Initialize number of correctly predicted validation samples
            total_val = 0  # Initialize total number of validation samples
            with torch.no_grad():
                for inputs, labels in valid_loader:
                    inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU if available
                    outputs = model(inputs)  # Forward pass
                    if isinstance(outputs, tuple):
                        logits = outputs[0]  # Unpack logits if model returns tuple
                    else:
                        logits = outputs
                    loss = criterion(logits, labels)  # Calculate loss

                    running_val_loss += loss.item() * inputs.size(0)  # Accumulate validation loss
                    _, predicted = torch.max(logits, 1)  # Get predicted labels
                    total_val += labels.size(0)  # Increment total validation samples
                    correct_val += (predicted == labels).sum().item()  # Increment correctly predicted validation samples

            # Calculate epoch-wise validation loss and accuracy
            epoch_val_loss = running_val_loss / len(valid_loader.dataset)  # Average validation loss
            val_accuracy = correct_val / total_val  # Validation accuracy

            # Append values to lists
            train_losses.append(epoch_train_loss)  # Append training loss
            train_accuracies.append(train_accuracy)  # Append training accuracy
            val_losses.append(epoch_val_loss)  # Append validation loss
            val_accuracies.append(val_accuracy)  # Append validation accuracy

            # Step LR scheduler
            lr_scheduler.step(epoch_val_loss)  # Adjust learning rate based on validation loss

            # Check early stopping
            if custom_callback.on_epoch_end(epoch, epoch_val_loss):
                break  # Stop training if early stopping criterion is met

            # Print epoch results
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {epoch_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

        # Plot training and validation losses starting from index 1
        epochs = range(1, len(train_losses) + 1)  # Generate the range of epochs starting from 1

        # Plot training and validation accuracies
        plt.plot(epochs, train_accuracies, label='Training Accuracy')  # Plot training accuracies over epochs
        plt.plot(epochs, val_accuracies, label='Validation Accuracy')  # Plot validation accuracies over epochs
        plt.xlabel('Epoch')  # Set label for the x-axis
        plt.ylabel('Accuracy')  # Set label for the y-axis
        plt.title('Training and Validation Accuracies')  # Set title for the plot
        plt.legend()  # Display legend
        if SHOW_PLOTS:
            plt.show()  # Show the plot
        # Plot training and validation losses starting from index 1
        epochs = range(1, len(train_losses) + 1)  # Generate the range of epochs starting from 1

        # Plot training and validation losses
        plt.plot(epochs, train_losses, label='Training Loss')  # Plot training losses over epochs
        plt.plot(epochs, val_losses, label='Validation Loss')  # Plot validation losses over epochs
        plt.xlabel('Epoch')  # Set label for the x-axis
        plt.ylabel('Loss')  # Set label for the y-axis
        plt.title('Training and Validation Losses')  # Set title for the plot
        plt.legend()  # Display legend
        plt.grid(True)  # Display grid
        if SHOW_PLOTS:
            plt.show()  # Show the plot

        output= {'lastTrainAccuracy': train_losses[-1], 'lastValAccuracy': val_accuracies[-1], 'lastTrainLoss': train_losses[-1], 'lastValLoss': val_losses[-1]}
        print("$" + str(output))
    
    def evalModel(model, criterion):
        model.eval()  # Set model to evaluation mode
        test_correct = 0  # Initialize number of correctly predicted samples
        correct_top5 = 0  # Initialize number of correctly top-5 predicted samples
        test_total = 0  # Initialize total number of samples
        test_running_loss = 0.0  # Initialize running test loss

        with torch.no_grad():  # Turn off gradients during evaluation
            for inputs, labels in test_loader:  # Iterate through test data
                inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU
                outputs = model(inputs)  # Get model predictions
                if isinstance(outputs, tuple):
                    logits = outputs[0]  # Unpack logits from model outputs if necessary
                else:
                    logits = outputs
                loss = criterion(logits, labels)  # Calculate loss

                test_running_loss += loss.item() * inputs.size(0)  # Update running test loss
                _, predicted = torch.max(logits, 1)  # Get predicted labels
                _, predicted_top5 = torch.topk(logits, 5, 1)  # Get top-5 predicted labels

                test_total += labels.size(0)  # Update total number of samples
                correct_top5 += (predicted_top5 == labels.view(-1, 1)).sum().item()  # Update number of correctly top-5 predicted samples
                test_correct += (predicted == labels).sum().item()  # Update number of correctly predicted samples

        # Calculate test loss and accuracy
        test_loss = test_running_loss / len(test_loader.dataset)  # Average test loss
        test_accuracy = test_correct / test_total  # Test accuracy
        top5_accuracy = (correct_top5 / test_total) 

        # Print test results
        print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

        print("$" + str({"testLoss": test_loss, "testAccuracy": test_accuracy, "testTop5Accuracy": top5_accuracy}))

    # ==== GoogleNetwork
    def trainAndEvaluateGoogle():
        default_weight_googlenet = models.GoogLeNet_Weights.DEFAULT
        googlenet_model = models.googlenet(weights=default_weight_googlenet)
        for param in googlenet_model.parameters():
            param.requires_grad = False
        num_ftrs = googlenet_model.fc.in_features
        googlenet_model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),  # Fully connected layer with 512 units
            nn.ReLU(inplace=True),  # ReLU activation
            nn.Linear(512, 256),  # Fully connected layer with 256 units
            nn.ReLU(inplace=True),  # ReLU activation
            nn.Linear(256, 40),  # Output layer with 40 units for 40 classes
            nn.Softmax(dim=1)  # Softmax activation for classification
        )
        tuned_googlenet_model = googlenet_model

        data_transform = transforms.Compose([
            transforms.Resize((224,224)), #resize the image to 224x224
            transforms.ToTensor(),  #convert the image to a pytorch tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), #normalize the image
        ])

        trainModel(tuned_googlenet_model, 10)

        evalModel(tuned_googlenet_model, nn.CrossEntropyLoss())

    #=== MobileNet3
    def trainAndEvaluateM3():
        default_weight_mobilenet = models.MobileNet_V3_Small_Weights.DEFAULT
        mobilenet_model = models.mobilenet_v3_small(weights=default_weight_mobilenet)

        # Freeze all layers
        for param in mobilenet_model.parameters():
            param.requires_grad = False
        
        num_features = mobilenet_model.classifier[0].in_features
        new_head = nn.Sequential(
            nn.Linear(num_features, 512),  # Fully connected layer with 512 units
            nn.ReLU(inplace=True),  # ReLU activation
            nn.Linear(512, 256),  # Fully connected layer with 256 units
            nn.ReLU(inplace=True),  # ReLU activation
            nn.Linear(256, 40),  # Output layer with 40 units for 40 classes
            nn.Softmax(dim=1)  # Softmax activation for classification
        )
        tuned_mobilenet_model = mobilenet_model

        tuned_mobilenet_model.classifier = new_head
        trainModel(tuned_mobilenet_model, 10)
        evalModel(tuned_mobilenet_model, nn.CrossEntropyLoss())


    #== SWIN_V2_b

    def trainAndEvaluateSwinV2B():
        default_weight_swin_v2_b = models.Swin_V2_B_Weights.DEFAULT
        swin_v2_b_model = models.swin_v2_b(weights=default_weight_swin_v2_b)
        for param in swin_v2_b_model.parameters():
            param.requires_grad = False
        num_features = swin_v2_b_model.head.in_features
        new_head = nn.Sequential(
            nn.Linear(num_features, 512),  # Fully connected layer with 512 units
            nn.ReLU(inplace=True),  # ReLU activation
            nn.Linear(512, 256),  # Fully connected layer with 256 units
            nn.ReLU(inplace=True),  # ReLU activation
            nn.Linear(256, 40),  # Output layer with 40 units for 40 classes
            nn.Softmax(dim=1)  # Softmax activation for classification
        )

        tuned_swin_v2_b_model = swin_v2_b_model

        tuned_swin_v2_b_model.head = new_head
        print(tuned_swin_v2_b_model)

        # TRAIN
        trainModel(tuned_swin_v2_b_model, 10)

        # EVALUATE
        evalModel(tuned_swin_v2_b_model, nn.CrossEntropyLoss())

    #=== DENSE NET
    def trainAndEvaluateDenseNet201():
        default_weight_denseNet201 = models.DenseNet201_Weights.DEFAULT
        densenet201_model = models.densenet201(weights=default_weight_denseNet201)
        for param in densenet201_model.parameters():
            param.requires_grad = False
        num_features = densenet201_model.classifier.in_features
        new_head = nn.Sequential(
            nn.Linear(num_features, 512),  # Fully connected layer with 512 units
            nn.ReLU(inplace=True),  # ReLU activation
            nn.Linear(512, 256),  # Fully connected layer with 256 units
            nn.ReLU(inplace=True),  # ReLU activation
            nn.Linear(256, 40),  # Output layer with 40 units for 40 classes
            nn.Softmax(dim=1)  # Softmax activation for classification
        )
        tuned_densenet201__model = densenet201_model

        tuned_densenet201__model.classifier = new_head

        trainModel(tuned_densenet201__model, 10)
        evalModel(tuned_densenet201__model, nn.CrossEntropyLoss())
#========================    
    listofmodels = {'google':trainAndEvaluateGoogle, 'mobilenet3':trainAndEvaluateM3, 'swin_v2_b':trainAndEvaluateSwinV2B, 'denseNet201':trainAndEvaluateDenseNet201}
    if (modelToRun in listofmodels):
        print("$RUN:" + str(modelToRun))
        listofmodels[modelToRun]()