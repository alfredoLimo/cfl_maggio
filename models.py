"""
LeNet-5 (around 60k parameters)
ResNet-9 (around 6M parameters)

Check gpu function
Training functions to test the models

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from math import prod


#############################################################################################################
# Models 
#############################################################################################################

# LeNet-5 model
class LeNet5(nn.Module):
    def __init__(self, in_channels=1, num_classes=10, input_size=(28, 28)):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5, stride=1, padding=2)  # Convolutional layer with 6 feature maps of size 5x5
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)  # Subsampling layer with 6 feature maps of size 2x2
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)  # Convolutional layer with 16 feature maps of size 5x5
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)  # Subsampling layer with 16 feature maps of size 2x2
        
        # Dinamically calculate the size of the features after convolutional layers
        dummy_input = torch.zeros(1, in_channels, *input_size)
        dummy_output = self.pool2(self.conv2(self.pool1(self.conv1(dummy_input))))
        self.feature_size = prod(dummy_output.size()[1:])

        self.fc1 = nn.Linear(self.feature_size, 120)  # Fully connected layer, output size 120
        self.fc2 = nn.Linear(120, 84)  # Fully connected layer, output size 84
        self.fc3 = nn.Linear(84, num_classes)  # Fully connected layer, output size num_classes

    def forward(self, x):
        x = F.relu(self.conv1(x))  # Apply ReLU after conv1
        x = self.pool1(x)  # Apply subsampling pool1
        x = F.relu(self.conv2(x))  # Apply ReLU after conv2
        x = self.pool2(x)  # Apply subsampling pool2
        x = x.view(x.size(0), -1)  # Flatten for fully connected layers
        x = F.relu(self.fc1(x))  # Apply ReLU after fc1
        x = F.relu(self.fc2(x))  # Apply ReLU after fc2
        x = self.fc3(x)  # Output layer
        return x
    

# Resnet-9 layer
def residual_block(in_channels, out_channels, pool=False):
    layers = [
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

# ResNet-9 model
class ResNet9(nn.Module):
    def __init__(self, in_channels, num_classes, input_size=(28, 28)):
        super().__init__()
        self.prep = residual_block(in_channels, 64)
        self.layer1_head = residual_block(64, 128, pool=True)
        self.layer1_residual = nn.Sequential(residual_block(128, 128), residual_block(128, 128))
        self.layer2 = residual_block(128, 256, pool=True)
        self.layer3_head = residual_block(256, 512, pool=True)
        self.layer3_residual = nn.Sequential(residual_block(512, 512), residual_block(512, 512))
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Changed to adaptive average pooling:         self.MaxPool2d = nn.Sequential(nn.MaxPool2d(4))
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Calculate the size of the features after the convolutional layers
        dummy_input = torch.zeros(1, in_channels, *input_size)
        dummy_output = self.pool(self.layer3_head(self.layer2(self.layer1_head(self.prep(dummy_input)))))
        self.feature_size = dummy_output.size(1) * dummy_output.size(2) * dummy_output.size(3)

        # Output layer
        self.linear = nn.Linear(self.feature_size, num_classes)

    def forward(self, x):
        x = self.prep(x)
        x = self.layer1_head(x)
        x = self.layer1_residual(x) + x
        x = self.layer2(x)
        x = self.layer3_head(x)
        x = self.layer3_residual(x) + x
        x = self.pool(x)  # Changed to adaptive average pooling
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x
    

# dictionary with the models
models = {
    'LeNet5': LeNet5,
    'ResNet9': ResNet9,
}    


#############################################################################################################
# Helper functions 
#############################################################################################################

# define device
def check_gpu(manual_seed=True, print_info=True):
    if manual_seed:
        torch.manual_seed(0)
    if torch.cuda.is_available():
        if print_info:
            print("CUDA is available")
        device = 'cuda'
        torch.cuda.manual_seed_all(0) 
    elif torch.backends.mps.is_available():
        if print_info:
            print("MPS is available")
        device = torch.device("mps")
        torch.mps.manual_seed(0)
    else:
        if print_info:
            print("CUDA is not available")
        device = 'cpu'
    return device


# simple train function
def simple_train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')


# simple test function
def simple_test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({100. * correct / len(test_loader.dataset):.0f}%)\n')
    

# Dataset class
class CombinedDataset(Dataset):
    def __init__(self, features, labels, transform=None):
        self.features = features
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.labels[idx]

        if self.transform:
            x = self.transform(x)

        return x, y




#############################################################################################################
# test the models
#############################################################################################################
def main():
    # Libraries
    import non_iiddata_generator_no_drifting as noniidgen
    from non_iiddata_generator_no_drifting import merge_data
    import torch
    import torch.optim as optim
    from torch.utils.data import DataLoader

    # Training settings
    model_name = "ResNet9"   # Options: "LeNet5", "ResNet9"
    batch_size = 64
    test_batch_size = 1000
    epochs = 10
    lr = 0.01
    momentum = 0.9
    seed = 1
    transform = None
    # dataset settings
    dataset_name = "CIFAR10"
    client_number = 10
    set_rotation = True
    rotations = 4
    scaling_rotation_low = 0.1
    scaling_rotation_high = 0.2
    set_color = True
    colors = 3
    scaling_color_low = 0.1
    scaling_color_high = 0.2
    random_order = True

    print(f"\n\033[94mTraining {model_name} on {dataset_name} with {client_number} clients\033[0m\n")

    device = check_gpu(manual_seed=True, print_info=True)
    torch.manual_seed(seed)

    # load data 
    train_images, train_labels, test_images, test_labels = noniidgen.load_full_datasets(dataset_name)

    # create data: split_feature_skew
    clients_data = noniidgen.split_feature_skew(
        train_features = train_images,
        train_labels = train_labels,
        test_features = test_images,
        test_labels = test_labels,
        client_number = client_number,
        set_rotation = set_rotation,
        rotations = rotations,
        scaling_rotation_low = scaling_rotation_low,
        scaling_rotation_high = scaling_rotation_high,
        set_color = set_color,
        colors = colors,
        scaling_color_low = scaling_color_low,
        scaling_color_high = scaling_color_high,
        random_order = random_order
    )

    # merge the data (for Centralized Learning Simulation)
    train_features, train_labels, test_features, test_labels = merge_data(clients_data)

    # Create the datasets
    train_dataset = CombinedDataset(train_features, train_labels, transform=transform)
    test_dataset = CombinedDataset(test_features, test_labels, transform=transform)

    # Create the data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    # model = LeNet5(in_channels=3, num_classes=10, input_size=(32,32)).to(device)
    model = models[model_name](in_channels=3, num_classes=10, input_size=(32,32)).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)


    for epoch in range(1, epochs + 1):
        simple_train(model, device, train_loader, optimizer, epoch)
        simple_test(model, device, test_loader)

if __name__ == '__main__':
    main()
