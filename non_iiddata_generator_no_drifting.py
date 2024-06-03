import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from scipy.stats import truncnorm

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

# For reproducibility only
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def draw_split_statistic(
    data_list: torch.Tensor,
    plot_indices = None
) -> None:
    '''
    Print label counts and plot images.
    
    Args:
        data_list (list): A list of dictionaries where each dictionary contains the features and labels for each client.
                        * Output of split_ fns
        plot_indices (list): A list of indices to plot the first 100 images for each client.

    Warning:
        Working for only 10 classes dataset. (EMNIST e CIFAR100 NOT SUPPORTED)
    '''
    # Print label counts for each dictionary
    for i, data in enumerate(data_list):
        train_labels = data['train_labels']
        test_labels = data['test_labels']
        
        train_label_counts = torch.tensor([train_labels.tolist().count(x) for x in range(10)])
        test_label_counts = torch.tensor([test_labels.tolist().count(x) for x in range(10)])
        
        print(f"Client {i}:")
        print("Training label counts:", train_label_counts)
        print("Test label counts:", test_label_counts)
        print("\n")
    
    # If plot_indices is provided, plot the first 100 images with labels for the specified dictionaries
    if plot_indices:
        for idx in plot_indices:
            if idx < len(data_list):
                data = data_list[idx]
                train_features = data['train_features']
                train_labels = data['train_labels']
                
                num_images = min(100, train_features.shape[0])
                fig, axes = plt.subplots(10, 10, figsize=(15, 15))
                fig.suptitle(f'Dictionary {idx} - First {num_images} Training Images', fontsize=16)
                
                for i in range(num_images):
                    ax = axes[i // 10, i % 10]
                    image = train_features[i]
                    
                    if image.shape[0] == 3:
                        # For CIFAR (3, H, W) -> (H, W, 3)
                        image = image.permute(1, 2, 0).numpy()
                    else:
                        # For MNIST (1, H, W) -> (H, W)
                        image = image.squeeze().numpy()
                    
                    ax.imshow(image, cmap='gray' if image.ndim == 2 else None)
                    ax.set_title(train_labels[i].item())
                    ax.axis('off')
                
                plt.tight_layout(rect=[0, 0, 1, 0.96])
                plt.show()

def load_full_datasets(
    dataset_name: str = "MNIST",
) -> list:
    '''
    Load datasets into four separate parts: train labels, train images, test labels, test images.

    Args:
        dataset_name (str): Name of the dataset to load. Options are "MNIST", "FMNIST", "EMNIST", "CIFAR10", "CIFAR100".

    TODO: EMNIST IS NOT WELL.

    Returns:
        list: [4] of torch.Tensor. [train_images, train_labels, test_images, test_labels]
    '''
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    if dataset_name == "MNIST":
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    elif dataset_name == "FMNIST":
        train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    elif dataset_name == "EMNIST": # not auto-downloaded successfully
        train_dataset = datasets.EMNIST(root='./data', split='letters', train=True, download=True, 
                                            transform = transforms.Compose([ 
                                            lambda img: transforms.functional.rotate(img, -90), 
                                            lambda img: transforms.functional.hflip(img), 
                                            transforms.ToTensor()
                                            ])
                                        )               
        test_dataset = datasets.EMNIST(root='./data', split='letters', train=False, download=True,
                                            transform = transforms.Compose([ 
                                            lambda img: transforms.functional.rotate(img, -90), 
                                            lambda img: transforms.functional.hflip(img), 
                                            transforms.ToTensor()
                                            ])
                                        )         
    elif dataset_name == "CIFAR10":
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    elif dataset_name == "CIFAR100":
        train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Dataset {dataset_name} is not supported.")

    # Extracting train and test images and labels
    train_images = torch.stack([data[0] for data in train_dataset]).squeeze(1)
    test_images = torch.stack([data[0] for data in test_dataset]).squeeze(1)
    
    if dataset_name in ["CIFAR10", "CIFAR100"]:
        train_labels = torch.tensor(train_dataset.targets).clone().detach()
        test_labels = torch.tensor(test_dataset.targets).clone().detach()
    else:
        train_labels = train_dataset.targets.clone().detach()
        test_labels = test_dataset.targets.clone().detach()

    return [train_images, train_labels, test_images, test_labels]

def rotate_dataset(
    dataset: torch.Tensor,
    degrees: list
) -> torch.Tensor:
    '''
    Rotates all images in the dataset by a specified degree.

    Args:
        dataset (torch.Tensor): Input dataset, a tensor of shape (N, ) where N is the number of images.
        degrees (list) : List of degrees to rotate each image.

    Returns:
        torch.Tensor: The rotated dataset, a tensor of the same shape (N, ) as the input.
    '''

    if len(dataset) != len(degrees):
        raise ValueError("The length of degrees list must be equal to the number of images in the dataset.")
    
    rotated_images = []
    
    for img_tensor, degree in zip(dataset, degrees):
        # Convert the tensor to a PIL image
        img = transforms.ToPILImage()(img_tensor)
        
        # Rotate the image
        rotated_img = img.rotate(degree)
        
        # Convert the PIL image back to a tensor
        rotated_img_tensor = transforms.ToTensor()(rotated_img).squeeze(0)
        
        rotated_images.append(rotated_img_tensor)
    
    # Stack all tensors into a single tensor
    rotated_dataset = torch.stack(rotated_images)
    
    return rotated_dataset

def color_dataset(
    dataset: torch.Tensor,
    colors: list
) -> torch.Tensor:
    '''
    Colors all images in the dataset by a specified color.

    Args:
        dataset (torch.Tensor): Input dataset, a tensor of shape (N, H, W) or (N, 3, H, W)
                                where N is the number of images.
        colors (list) : List of 'red', 'green', 'blue'.

    Warning:
        MNIST, FMNIST, EMNIST are 1-channel. CIFAR10, CIFAR100 are 3-channel.

    Returns:
        torch.Tensor: The colored dataset, a tensor of the shape (N, 3, H, W) with 3 channels.
    '''

    if len(dataset) != len(colors):
        raise ValueError("The length of colors list must be equal to the number of images in the dataset.")

    if dataset.dim() == 3:
        # Handle 1-channel dataset
        colored_dataset = dataset.unsqueeze(1).repeat(1, 3, 1, 1) # Shape becomes (N, 3, H, W)
    elif dataset.dim() == 4 and dataset.size(1) == 3:
        colored_dataset = dataset.clone()
    else:
        raise ValueError("This function only supports 1-channel (N, H, W) or 3-channel (N, 3, H, W) datasets.")

    for i, color in enumerate(colors):
        # Map the grayscale values to the specified color
        if color == 'red':
            colored_dataset[i, 0, :, :] = 1  # Set the red channel for the image
        elif color == 'green':
            colored_dataset[i, 1, :, :] = 1  # Set the green channel for the image
        elif color == 'blue':
            colored_dataset[i, 2, :, :] = 1  # Set the blue channel for the image
        else:
            raise ValueError("Color must be 'red', 'green', or 'blue'")

    return colored_dataset

def split_basic(
    features: torch.Tensor,
    labels: torch.Tensor,
    client_number: int = 10,
    permute: bool = True
) -> list:
    """
    Splits a dataset into a specified number of clusters (clients).
    
    Args:
        features (torch.Tensor): The dataset features.
        labels (torch.Tensor): The dataset labels.
        client_number (int): The number of clients to split the data into.
        permute (bool): Whether to shuffle the data before splitting.
        
    Returns:
        list: A list of dictionaries where each dictionary contains the features and labels for each client.
    """

    # Ensure the features and labels have the same number of samples
    assert len(features) == len(labels), "The number of samples in features and labels must be the same."

    # Randomly shuffle the dataset while maintaining correspondence between features and labels
    if permute:
        indices = torch.randperm(len(features))
        features = features[indices]
        labels = labels[indices]
    
    # Calculate the number of samples per client
    samples_per_client = len(features) // client_number
    
    # List to hold the data for each client
    client_data = []
    
    for i in range(client_number):
        start_idx = i * samples_per_client
        end_idx = start_idx + samples_per_client
        
        # Handle the last client which may take the remaining samples
        if i == client_number - 1:
            end_idx = len(features)
        
        client_features = features[start_idx:end_idx]
        client_labels = labels[start_idx:end_idx]
        
        client_data.append({
            'features': client_features,
            'labels': client_labels
        })
    
    return client_data

def split_unbalanced(
    features: torch.Tensor,
    labels: torch.Tensor,
    client_number: int = 10,
    std_dev: float = 0.1,
    permute: bool = True
) -> list:
    """
    Splits a dataset into a specified number of clusters (clients).
    
    Args:
        features (torch.Tensor): The dataset features.
        labels (torch.Tensor): The dataset labels.
        client_number (int): The number of clients to split the data into.
        std_dev (float): standard deviation of the normal distribution for the number of samples per client.
        permute (bool): Whether to shuffle the data before splitting.
        
    Returns:
        list: A list of dictionaries where each dictionary contains the features and labels for each client.
    """

    # Ensure the features and labels have the same number of samples
    assert len(features) == len(labels), "The number of samples in features and labels must be the same."
    assert std_dev > 0, "Standard deviation must be larger than 0."

    # Generate random percentage from a truncated normal distribution
    percentage = truncnorm.rvs(-0.5/std_dev, 0.5/std_dev, loc=0.5, scale=std_dev, size=client_number)
    normalized_percentage = percentage / np.sum(percentage)

    # Randomly shuffle the dataset while maintaining correspondence between features and labels
    if permute:
        indices = torch.randperm(len(features))
        features = features[indices]
        labels = labels[indices]

    # Calculate the number of samples per client based on the normalized samples
    total_samples = len(features)
    samples_per_client = (normalized_percentage * total_samples).astype(int)

    # Adjust to ensure the sum of samples_per_client equals the total_samples
    difference = total_samples - samples_per_client.sum()
    for i in range(abs(difference)):
        samples_per_client[i % client_number] += np.sign(difference)
    
    # List to hold the data for each client
    client_data = []
    start_idx = 0
    
    for i in range(client_number):
        end_idx = start_idx + samples_per_client[i]
        
        client_features = features[start_idx:end_idx]
        client_labels = labels[start_idx:end_idx]
        
        client_data.append({
            'features': client_features,
            'labels': client_labels
        })
        
        start_idx = end_idx
    
    return client_data

def assigning_rotation_features(
    datapoint_number: int,
    rotations: int = 4,
    scaling: float = 0.1,
    random_order: bool = True
) -> list:
    '''
    Assigns a rotation to each datapoint based on a softmax distribution.

    Args:
        datapoint_number (int): The number of datapoints to assign rotations to.
        rotations (int): The number of possible rotations. Recommended to be [2,4].
        scaling (float): The scaling factor for the softmax distribution. 0: Uniform distribution.
        random_order (bool): Whether to shuffle the order of the rotations.
    
    Returns:
        list: A list of rotations assigned to the datapoints.
    '''
    assert 0 <= scaling <= 1, "k must be between 0 and 1."
    assert rotations > 1, "Must have at least 2 rotations."

    # Scale the values based on k
    values = np.arange(rotations, 0, -1)  # From N to 1
    scaled_values = values * scaling
    
    # Apply softmax to get the probabilities
    exp_values = np.exp(scaled_values)
    probabilities = exp_values / np.sum(exp_values)

    angles = [i * 360 / rotations for i in range(rotations)]
    if random_order:
        np.random.shuffle(angles)

    angles_assigned = np.random.choice(angles, size=datapoint_number, p=probabilities)

    return angles_assigned

def assigning_color_features(
    datapoint_number: int,
    colors: int = 3,
    scaling: float = 0.1,
    random_order: bool = True
) -> list:
    '''
    Assigns colors to the datapoints based on the softmax probabilities.

    Args:
        datapoint_number (int): Number of datapoints to assign colors to.
        colors (int): Number of colors to assign. Must be 2 or 3.
        scaling (float): Scaling factor for the softmax probabilities. 0: Uniform distribution.
        random_order (bool): Whether to shuffle the order of the colors.

    Returns:
        list: A list of colors assigned to the datapoints.
    '''

    assert 0 <= scaling <= 1, "k must be between 0 and 1."
    assert colors == 2 or colors == 3, "Color must be 2 or 3."
    
    # Scale the values based on k
    values = np.arange(colors, 0, -1)  # From N to 1
    scaled_values = values * scaling
    
    # Apply softmax to get the probabilities
    exp_values = np.exp(scaled_values)
    probabilities = exp_values / np.sum(exp_values)

    if colors == 2:
        letters = ['red', 'blue']
    else:
        letters = ['red', 'blue', 'green']

    if random_order:
        np.random.shuffle(letters)

    colors_assigned = np.random.choice(letters, size=datapoint_number, p=probabilities)

    # unique, counts = np.unique(colors_assigned, return_counts=True)
    # for letter, count in zip(unique, counts):
    #     print(f'{letter}: {count}')

    return colors_assigned

def split_feature_skew(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    client_number: int = 10,
    set_rotation: bool = False,
    rotations: int = None,
    scaling_rotation_low: float = 0.1,
    scaling_rotation_high: float = 0.1,
    set_color: bool = False,
    colors: int = None,
    scaling_color_low: float = 0.1,
    scaling_color_high: float = 0.1,
    random_order: bool = True,
    show_distribution: bool = False
) -> list:
    '''
    Splits an overall dataset into a specified number of clusters (clients) with ONLY feature skew.
    
    Args:
        train_features (torch.Tensor): The training dataset features.
        train_labels (torch.Tensor): The training dataset labels.
        test_features (torch.Tensor): The testing dataset features.
        test_labels (torch.Tensor): The testing dataset labels.
        client_number (int): The number of clients to split the data into.
        set_rotation (bool): Whether to assign rotations to the features.
        rotations (int): The number of possible rotations. Recommended to be [2,4].
        scaling_rotation_low (float): The low bound scaling factor of rotation for the softmax distribution.
        scaling_rotation_high (float): The high bound scaling factor of rotation for the softmax distribution.
        set_color (bool): Whether to assign colors to the features.
        colors (int): The number of colors to assign. Must be [2,3].
        scaling_color_low (float): The low bound scaling factor of color for the softmax distribution.
        scaling_color_high (float): The high bound scaling factor of color for the softmax distribution.
        random_order (bool): Whether to shuffle the order of the rotations and colors.
        show_distribution (bool): Whether to print the distribution of the assigned features.

    Warning:
        random_order should be identical for both training and testing if not DRIFTING.

    Returns:
        list: A list of dictionaries where each dictionary contains the features and labels for each client.
                Both train and test.
    '''
    # Ensure the features and labels have the same number of samples
    assert len(train_features) == len(train_labels), "The number of samples in features and labels must be the same."
    assert len(test_features) == len(test_labels), "The number of samples in features and labels must be the same."
    assert scaling_color_high >= scaling_color_low, "High scaling must be larger than low scaling."
    assert scaling_rotation_high >= scaling_rotation_low, "High scaling must be larger than low scaling."

    # generate basic split
    basic_split_data_train = split_basic(train_features, train_labels, client_number)
    basic_split_data_test = split_basic(test_features, test_labels, client_number)

    # Process train and test splits with rotations if required
    if set_rotation:
        for client_data_train, client_data_test in zip(basic_split_data_train, basic_split_data_test):

            len_train = len(client_data_train['labels'])
            len_test = len(client_data_test['labels'])
            total_rotations = assigning_rotation_features(
                len_train + len_test, rotations, 
                np.random.uniform(scaling_rotation_low,scaling_rotation_high), random_order
                )
            
            print(dict(Counter(total_rotations))) if show_distribution else None

            # Split the total_rotations list into train and test
            train_rotations = total_rotations[:len_train]
            test_rotations = total_rotations[len_train:]

            client_data_train['features'] = rotate_dataset(client_data_train['features'], train_rotations)
            client_data_test['features'] = rotate_dataset(client_data_test['features'], test_rotations)

    if set_color:
        for client_data_train, client_data_test in zip(basic_split_data_train, basic_split_data_test):

            len_train = len(client_data_train['labels'])
            len_test = len(client_data_test['labels'])
            total_colors = assigning_color_features(
                len_train + len_test, colors, 
                np.random.uniform(scaling_color_low,scaling_color_high), random_order
                )
            
            print(dict(Counter(total_colors))) if show_distribution else None

            # Split the total_colors list into train and test
            train_colors = total_colors[:len_train]
            test_colors = total_colors[len_train:]

            client_data_train['features'] = color_dataset(client_data_train['features'], train_colors)
            client_data_test['features'] = color_dataset(client_data_test['features'], test_colors)

    rearranged_data = []

    # Iterate through the indices of the lists
    for i in range(client_number):
        # Create a new dictionary for each client
        client_data = {
            'train_features': basic_split_data_train[i]['features'],
            'train_labels': basic_split_data_train[i]['labels'],
            'test_features': basic_split_data_test[i]['features'],
            'test_labels': basic_split_data_test[i]['labels']
        }
        # Append the new dictionary to the list
        rearranged_data.append(client_data)
            
    return rearranged_data
    
def split_label_skew(
    train_features: torch.Tensor,
    train_labels: torch.Tensor, 
    test_features: torch.Tensor,
    test_labels: torch.Tensor, 
    client_number: int = 10,
    scaling_label_low: float = 0.4,
    scaling_label_high: float = 0.6,
) -> list:
    '''
    Splits an overall dataset into a specified number of clusters (clients) with ONLY label skew.

    Args:
        train_features (torch.Tensor): The training dataset features.
        train_labels (torch.Tensor): The training dataset labels.
        test_features (torch.Tensor): The testing dataset features.
        test_labels (torch.Tensor): The testing dataset labels.
        client_number (int): The number of clients to split the data into.
        scaling_label_low (float): The low bound scaling factor of label for the softmax distribution.
        scaling_label_high (float): The high bound scaling factor of label for the softmax distribution.

    Warning:
        Datasets vary in sensitivity to scaling. Fine-tune the scaling factors for each dataset for optimal results.    

    Returns:
        list: A list of dictionaries where each dictionary contains the features and labels for each client.
                Both train and test.
    '''

    def calculate_probabilities(labels, scaling):
        # Count the occurrences of each label
        label_counts = torch.bincount(labels, minlength=10).float()
        scaled_counts = label_counts ** scaling
        
        # Apply softmax to get probabilities
        probabilities = F.softmax(scaled_counts, dim=0)
        
        return probabilities

    def create_sub_dataset(features, labels, probabilities, num_points):
        selected_indices = []
        while len(selected_indices) < num_points:
            for i in range(len(labels)):
                if torch.rand(1).item() < probabilities[labels[i]].item():
                    selected_indices.append(i)
                if len(selected_indices) >= num_points:
                    break
        
        selected_indices = torch.tensor(selected_indices)
        sub_features = features[selected_indices]
        sub_labels = labels[selected_indices]
        remaining_indices = torch.ones(len(labels), dtype=torch.bool)
        remaining_indices[selected_indices] = 0
        remaining_features = features[remaining_indices]
        remaining_labels = labels[remaining_indices]

        return sub_features, sub_labels, remaining_features, remaining_labels

    avg_points_per_client_train = len(train_labels) // client_number
    avg_points_per_client_test = len(test_labels) // client_number

    rearranged_data = []

    remaining_train_features = train_features
    remaining_train_labels = train_labels
    remaining_test_features = test_features
    remaining_test_labels = test_labels

    for i in range(client_number):
        
        # For the last client, take all remaining data
        if i == client_number - 1:

            client_data = {
                'train_features': remaining_train_features,
                'train_labels': remaining_train_labels,
                'test_features': remaining_test_features,
                'test_labels': remaining_test_labels
            } 
            rearranged_data.append(client_data)
            break

        probabilities = calculate_probabilities(remaining_train_labels, np.random.uniform(scaling_label_low,scaling_label_high))

        sub_train_features, sub_train_labels, remaining_train_features, remaining_train_labels = create_sub_dataset(
            remaining_train_features, remaining_train_labels, probabilities, avg_points_per_client_train)
        sub_test_features, sub_test_labels, remaining_test_features, remaining_test_labels = create_sub_dataset(
            remaining_test_features, remaining_test_labels, probabilities, avg_points_per_client_test)
        
        client_data = {
            'train_features': sub_train_features,
            'train_labels': sub_train_labels,
            'test_features': sub_test_features,
            'test_labels': sub_test_labels
        }        
        rearranged_data.append(client_data)

    return rearranged_data

def merge_data(
    data: list
) -> list:
    '''
    Merges the data from multiple clients into a single dataset.

    Args:
        data (list): A list of dictionaries where each dictionary contains the features and labels for each client (output of previous functions).
    
    Returns:
        list: A list of four torch.Tensors containing the training features, training labels, testing features, and testing labels.
    
    '''
    train_features = []
    train_labels = []
    test_features = []
    test_labels = []
    for client_data in data:
        train_features.append(client_data['train_features'])
        train_labels.append(client_data['train_labels'])
        test_features.append(client_data['test_features'])
        test_labels.append(client_data['test_labels'])

    # Concatenate all the data
    train_features = torch.cat(train_features, dim=0)
    train_labels = torch.cat(train_labels, dim=0)
    test_features = torch.cat(test_features, dim=0)
    test_labels = torch.cat(test_labels, dim=0)

    return [train_features, train_labels, test_features, test_labels]



