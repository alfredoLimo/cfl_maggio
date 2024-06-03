"""
This code creates a Flower client that can be used to train a model locally and share the updated 
model with the server. When it is started, it connects to the Flower server and waits for instructions.
If the server sends a model, the client trains the model locally and sends back the updated model.
If abilitated, at the end of the training the client evaluates the last model, and plots the 
metrics during the training.

This is code is set to be used locally, but it can be used in a distributed environment by changing the server_address.
In a distributed environment, the server_address should be the IP address of the server, and each client machine should 
have this code running.
"""

# Libraies
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader
import torch
import utils
import flwr as fl
import argparse
import models
import config as cfg
import numpy as np
from sklearn.model_selection import train_test_split



# Define Flower client )
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, val_loader, optimizer, num_examples, 
                 client_id, train_fn, evaluate_fn, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.num_examples = num_examples
        self.client_id = client_id
        self.train_fn = train_fn
        self.evaluate_fn = evaluate_fn
        self.device = device

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        try: 
            self.set_parameters(parameters)
            for epoch in range(config["local_epochs"]):
                self.train_fn(self.model, self.device, self.train_loader, self.optimizer, epoch, self.client_id)
        except Exception as e:
            print(f"An error occurred during training of Honest client {self.client_id}: {e}, returning model with error") 
        
        return self.get_parameters(config), self.num_examples["train"], {}
    

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        try:
            loss, accuracy = self.evaluate_fn(self.model, self.device, self.val_loader)
            return float(loss), self.num_examples["val"], {"accuracy": float(accuracy)}
        except Exception as e:
            print(f"An error occurred during inference of client {self.client_id}: {e}, returning same zero metrics") 
            return float(10000), self.num_examples["val"], {"accuracy": float(0)}




# main
def main()->None:
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--id",
        type=int,
        choices=range(1, 40),
        required=True,
        help="Specifies the artificial data partition",
    )
    args = parser.parse_args()

    # check gpu and set manual seed
    device = utils.check_gpu(manual_seed=True)

    # model and history folder
    model = models.models[cfg.model_name](in_channels=3, num_classes=cfg.n_classes, input_size=cfg.input_size).to(device)
    # train_fn = utils.trainings[args.model]
    # evaluate_fn = utils.evaluations[args.model]
    # plot_fn = utils.plot_functions[args.model]
    # config = utils.config_tests[args.dataset][args.model]

    # check if metrics.csv exists otherwise delete it
    # utils.check_and_delete_metrics_file(config['history_folder'] + f"client_{args.data_type}_{args.id}", question=False)

    # Load data
    data = np.load(f'./data/client_{args.id}.npy', allow_pickle=True).item()

    # Split the data into training and testing subsets
    train_features, val_features, train_labels, val_labels = train_test_split(
        data['train_features'], data['train_labels'], test_size=0.2, random_state=42
    )

    num_examples = {
        "train": train_features.shape[0],
        "val": val_features.shape[0]
    }

    # Create the datasets
    train_dataset = models.CombinedDataset(train_features, train_labels, transform=None)
    val_dataset = models.CombinedDataset(val_features, val_labels, transform=None)

    # Create the data loaders
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.test_batch_size, shuffle=False)

    # Optimizer and Loss function
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum)

    # Start Flower client
    client = FlowerClient(model, train_loader, val_loader, optimizer, num_examples, args.id, 
                           models.simple_train, models.simple_test, device).to_client()
    fl.client.start_client(server_address="[::]:8098", client=client) # local host

    # read saved data and plot
    # plot_fn(args.id, show=False)





if __name__ == "__main__":
    main()
