# Config


# Training settings
model_name = "ResNet9"   # Options: "LeNet5", "ResNet9"
batch_size = 64
test_batch_size = 1000
n_rounds = 50
lr = 0.01
momentum = 0.9
seed = 1
transform = None

# dataset settings
dataset_name = "CIFAR10" # Options: "CIFAR10", "CIFAR100" "MNIST", "FMNIST"
client_number = 10
set_rotation = True
rotations = 4
scaling_rotation_low = 1
scaling_rotation_high = 1
set_color = True
colors = 3
scaling_color_low = 1
scaling_color_high = 1
random_order = True
show_distribution = True

n_classes_dict = {
    "CIFAR10": 10,
    "CIFAR100": 100,
    "MNIST": 10,
    "FMNIST": 10
}
n_classes = n_classes_dict[dataset_name]

input_size_dict = {
    "CIFAR10": (32, 32),
    "CIFAR100": (32, 32),
    "MNIST": (28, 28),
    "FMNIST": (28, 28)
}
input_size = input_size_dict[dataset_name]

