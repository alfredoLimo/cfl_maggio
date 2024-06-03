import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import non_iiddata_generator_no_drifting as noniidgen
import config as cfg
import torch
import os



# data generation
def generate_dataset():

    train_images, train_labels, test_images, test_labels = noniidgen.load_full_datasets(cfg.dataset_name)

    # create data: split_feature_skew
    clients_data = noniidgen.split_feature_skew(
        train_features = train_images,
        train_labels = train_labels,
        test_features = test_images,
        test_labels = test_labels,
        client_number = cfg.client_number,
        set_rotation = cfg.set_rotation,
        rotations = cfg.rotations,
        scaling_rotation_low = cfg.scaling_rotation_low,
        scaling_rotation_high = cfg.scaling_rotation_high,
        set_color = cfg.set_color,
        colors = cfg.colors,
        scaling_color_low = cfg.scaling_color_low,
        scaling_color_high = cfg.scaling_color_high,
        random_order = cfg.random_order,
        show_distribution = cfg.show_distribution,
    )

    # save dictionary for each client
    for i in range(cfg.client_number):
        np.save(f'./data/client_{i+1}', clients_data[i])
        print(f"Data for client {i+1} saved")


# create folder if not exists
def create_folders():
    # Create directories for results
    if not os.path.exists(f"results/{cfg.model_name}/{cfg.dataset_name}"):
        os.makedirs(f"results/{cfg.model_name}/{cfg.dataset_name}")
    else:
        # remove the directory and create a new one
        os.system(f"rm -r results/{cfg.model_name}/{cfg.dataset_name}")
        os.makedirs(f"results/{cfg.model_name}/{cfg.dataset_name}")
    if not os.path.exists(f"histories/{cfg.model_name}/{cfg.dataset_name}"):
        os.makedirs(f"histories/{cfg.model_name}/{cfg.dataset_name}")
    if not os.path.exists(f"checkpoints/{cfg.model_name}/{cfg.dataset_name}"):
        os.makedirs(f"checkpoints/{cfg.model_name}/{cfg.dataset_name}")
    if not os.path.exists(f"images/{cfg.model_name}/{cfg.dataset_name}"):
        os.makedirs(f"images/{cfg.model_name}/{cfg.dataset_name}")


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


# plot and save plot on server side
def plot_loss_and_accuracy(loss, accuracy,  show=True):
    # read args
    rounds = cfg.n_rounds
    
    # Plot loss and accuracy
    plt.figure(figsize=(12, 6))

    plt.plot(loss, label='Loss')
    plt.plot(accuracy, label='Accuracy')
    min_loss_index = loss.index(min(loss))
    max_accuracy_index = accuracy.index(max(accuracy))
    print(f"\n\033[1;34mServer Side\033[0m \nMinimum Loss occurred at round {min_loss_index + 1} with a loss value of {loss[min_loss_index]:.3f} \nMaximum Accuracy occurred at round {max_accuracy_index + 1} with an accuracy value of {accuracy[max_accuracy_index]*100:.2f}\n")
    plt.scatter(min_loss_index, loss[min_loss_index], color='blue', marker='*', s=100, label='Min Loss')
    plt.scatter(max_accuracy_index, accuracy[max_accuracy_index], color='orange', marker='*', s=100, label='Max Accuracy')
    
    # Labels and title
    plt.xlabel('Rounds')
    plt.ylabel('Metrics')
    plt.title('Distributed Metrics (Weighted Average on Test-Set)')
    plt.legend()
    plt.savefig(f"images/{cfg.model_name}/{cfg.dataset_name}/training_{rounds}_rounds.png")
    if show:
        plt.show()
    return min_loss_index+1, max_accuracy_index+1
