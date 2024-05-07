import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

def plot_images(original_images, augmented_images, labels, num_images=6):
    plt.figure(figsize=(12, 5))
    for i in range(num_images):
        # Original image
        ax = plt.subplot(2, num_images, i + 1)
        plt.imshow(original_images[i].squeeze(), cmap='gray')
        plt.title("Original: {}".format(int(labels[i])))
        plt.axis("off")
        
        # Augmented image
        ax = plt.subplot(2, num_images, num_images + i + 1)
        plt.imshow(augmented_images[i].squeeze(), cmap='gray')
        plt.title("Augmented: {}".format(int(labels[i])))
        plt.axis("off")
    # Save the figure
    plt.savefig('original_vs_augmented.png')
    plt.show()


def plot_predictions_comparison(best_model, last_model, images, labels, num_images=3):
    plt.figure(figsize=(12, 8))
    num_total_images = images.shape[0]
    random_indices = np.random.choice(num_total_images, size=num_images, replace=False)
    
    best_model.eval()
    last_model.eval()
    
    with torch.no_grad():
        for i, idx in enumerate(random_indices):
            image = images[idx].unsqueeze(0)  # Add batch dimension
            
            # Prediction for the best model
            best_outputs = best_model(image)
            best_probs = torch.softmax(best_outputs, dim=1)[0]
            best_top_probs, best_top_classes = best_probs.topk(3)
            best_top_probs = best_top_probs.numpy()
            best_top_classes = best_top_classes.numpy()
            
            # Prediction for the last model
            last_outputs = last_model(image)
            last_probs = torch.softmax(last_outputs, dim=1)[0]
            last_top_probs, last_top_classes = last_probs.topk(3)
            last_top_probs = last_top_probs.numpy()
            last_top_classes = last_top_classes.numpy()
            
            # Plot for the best model
            ax = plt.subplot(2, num_images, i + 1)
            plt.imshow(images[idx].reshape(28, 28), cmap='gray_r')
            
            best_title_text = '\n'.join([f'Class: {cls}, Probability: {prob:.2%}' for cls, prob in zip(best_top_classes, best_top_probs)])
            plt.title(f'Best Model\n{best_title_text}', color='#017653')
            plt.axis("off")
            
            # Plot for the last model
            ax = plt.subplot(2, num_images, num_images + i + 1)
            plt.imshow(images[idx].reshape(28, 28), cmap='gray_r')
            
            last_title_text = '\n'.join([f'Class: {cls}, Probability: {prob:.2%}' for cls, prob in zip(last_top_classes, last_top_probs)])
            plt.title(f'Last Model\n{last_title_text}', color='#017653')
            plt.axis("off")
    
    plt.savefig('last_and_best_comparison_images.png')  
    plt.show()

########## PREPARATiON

# Model / data parameters
num_classes = 10
input_shape = (1, 28, 28)

# Define data augmentation transformations
train_transform = transforms.Compose([
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(contrast=0.05),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Define normalization transformation for test data
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


# Load the data and apply transformations
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=test_transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=test_transform)

# DataLoaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# Load a batch of data
data_iter = iter(train_loader)
images, labels = next(data_iter)


# Define a list to store original images
original_images = []
for i in range(len(images)):
    original_images.append(images[i].numpy())  # Convert torch tensor to numpy array

# Apply transformations to create augmented images
augmented_images = []
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=train_transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=False)

data_iter = iter(train_loader)
images, labels = next(data_iter)

for i in range(len(images)):
    augmented_images.append(images[i].numpy())


# Call plot_images() function
plot_images(original_images, augmented_images, labels)


#########################  MODEL

# Define the model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Instantiate the model
model = Net()

print(model)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Training loop
best_loss = float('inf')  # Initialize with a very large value
best_model_path = 'best_model.pth'
last_model_path = 'last_model.pth'

# Training loop
num_epochs = 15
history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        if i % 100 == 99:
            print('[%d, %5d] loss: %.5f' % (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0
    
    # Evaluate on validation set and save the best model
    with torch.no_grad():
        model.eval()
        val_loss = 0.0
        for inputs_val, labels_val in train_loader:
            outputs_val = model(inputs_val)
            val_loss += criterion(outputs_val, labels_val).item()
        
        avg_val_loss = val_loss / len(train_loader)

        history['val_loss'].append(avg_val_loss)

        if avg_val_loss < best_loss:
            print(f"Validation loss decreased ({best_loss:.6f} --> {avg_val_loss:.6f}). Saving model...")
            torch.save(model.state_dict(), best_model_path)
            best_loss = avg_val_loss
        else:
            print("Validation loss did not improve. Moving on...")

    avg_train_loss = running_loss / len(train_loader)
    history['train_loss'].append(avg_train_loss)
    # Save the last model after each epoch
    torch.save(model.state_dict(), last_model_path)

print('Finished Training')

################## RESULTS

def calculate_test_metrics(model, test_loader, criterion):
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_loss /= len(test_loader)
    test_accuracy = 100 * correct / total

    return test_loss, test_accuracy

# Load the best model
best_model = Net()  # Instantiate your model class
best_model.load_state_dict(torch.load(best_model_path))

# Load the last model
last_model = Net()  # Instantiate your model class
last_model.load_state_dict(torch.load(last_model_path))

# Calculate test metrics for best model
best_test_loss, best_test_accuracy = calculate_test_metrics(best_model, test_loader, criterion)

# Calculate test metrics for last model
last_test_loss, last_test_accuracy = calculate_test_metrics(last_model, test_loader, criterion)

# Display the results
print("")
print("Best Model Test Loss:", best_test_loss)
print("Best Model Test Accuracy:", best_test_accuracy)
print("")
print("Last Model Test Loss:", last_test_loss)
print("Last Model Test Accuracy:", last_test_accuracy)
print("")

plot_predictions_comparison(best_model, last_model, images, labels)
