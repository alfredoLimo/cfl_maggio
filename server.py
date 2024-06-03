"""
This code implements the FedAvg, empowered with our Federated Behavioural Planes for visualizing the client's behaviour.
When it starts, the server waits for the clients to connect. When the established number of clients is reached, the 
learning process starts. The server sends the model to the clients, and the clients train the model locally. After training,
the clients send the updated model back to the server. The server evaluate the client models on the clean validation set
to create the Federated Behavioural Planes (Error and Counterfactual). The planes are saved in in the folder
"images/{dataset}/{model}/gifs/{data_type}/". Error Behavioural Plane is in the folder "error_traj", and Counterfactual 
Behavioural Plane in folder "cf_traj". Then client models are aggregated with FedAvg. The aggregated model is then sent
to the clients for the next round of training. The server saves the model and metrics after each round.

This is code is set to be used locally, but it can be used in a distributed environment by changing the server_address.
In a distributed environment, the server_address should be the IP address of the server, and each client machine should 
run the appopriate client code (client.py).
"""

# Libraries
import flwr as fl
import numpy as np
from typing import List, Tuple, Union, Optional, Dict
from flwr.common import Parameters, Scalar, Metrics
from flwr.server.client_proxy import ClientProxy
from flwr.common import FitRes
import argparse
import torch
from torch.utils.data import DataLoader
import os
from collections import OrderedDict
import json
import time
import pandas as pd
import config as cfg
import utils
import models
from sklearn.model_selection import train_test_split



# Config_client
def fit_config(server_round: int):
    """Return training configuration dict for each round."""
    config = {
        "current_round": server_round,
        "local_epochs": 2,
        "tot_rounds": 20,
    }
    return config


# Custom weighted average function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    # validities = [num_examples * m["validity"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


# Custom strategy to save model after each round
class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self, model, dataset, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.dataset = dataset

        # # read data for testing
        # self.X_test, self.y_test = utils.load_data_test(data_type=self.data_type, dataset=self.dataset)

        # if self.dataset == 'diabetes':
        #     # randomly pick N samples <= 10605
        #     idx = np.random.choice(len(self.X_test), 300, replace=False)
        #     self.X_test = self.X_test[idx]
        #     self.y_test = self.y_test[idx]
        # elif self.dataset == 'mnist':
        #     # randomly pick N samples <= 938
        #     idx = np.random.choice(len(self.X_test), 300, replace=False)
        #     self.X_test = self.X_test[idx]
        #     self.y_test = self.y_test[idx]      
        
        # print(f"Used Size Server-Test Set: {self.X_test.shape}")

        # # create folder if not exists
        # if not os.path.exists(self.checkpoint_folder + f"{self.data_type}"):
        #     os.makedirs(self.checkpoint_folder + f"{self.data_type}")

    # Override aggregate_fit method to add saving functionality
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate model weights using weighted average and store checkpoint"""
        
        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures) # aggregated_metrics from aggregate_fit is empty except if i pass fit_metrics_aggregation_fn

        # Save model
        if aggregated_parameters is not None:

            print(f"Saving round {server_round} aggregated_parameters...")
            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)
            # Convert `List[np.ndarray]` to PyTorch`state_dict`
            params_dict = zip(self.model.state_dict().keys(), aggregated_ndarrays)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.model.load_state_dict(state_dict, strict=True)
            # Save the model
            torch.save(self.model.state_dict(), f"checkpoints/{cfg.model_name}/{cfg.dataset_name}/model_{server_round}.pth")
        
        return aggregated_parameters, aggregated_metrics





# Main
def main() -> None:
    # parser = argparse.ArgumentParser(description="Flower")
    # parser.add_argument(
    #     "--rounds",
    #     type=int,
    #     default=20,
    #     help="Specifies the number of FL rounds",
    # )
    # args = parser.parse_args()

    # Start time
    start_time = time.time()

    # Create directories 
    utils.create_folders()

    # Create all data for clients
    utils.generate_dataset()

    # Pick the indipendent test set from each client
    test_x, test_y = [], []
    for client_id in range(cfg.client_number):
        data = np.load(f'./data/client_{client_id+1}.npy', allow_pickle=True).item()
        test_x.append(data['test_features'])
        test_y.append(data['test_labels'])
    test_x = np.concatenate(test_x, axis=0)
    test_y = np.concatenate(test_y, axis=0)
    # Split the data into test subsets (final evaluation & server validation)
    test_x_server, test_x, test_y_server, test_y = train_test_split(test_x, test_y, test_size=0.5, random_state=42)

    # Create the datasets
    test_dataset = models.CombinedDataset(test_x, test_y, transform=None)

    # Create the data loaders
    test_loader = DataLoader(test_dataset, batch_size=cfg.test_batch_size, shuffle=False)

    # model and history folder
    device = utils.check_gpu(manual_seed=True, print_info=True)
    model = models.models[cfg.model_name](in_channels=3, num_classes=cfg.n_classes, input_size=cfg.input_size).to(device)

    # Define strategy
    strategy = SaveModelStrategy(
        model=model, # model to be trained
        min_fit_clients=cfg.client_number, #+cfg.n_attackers, # Never sample less than 10 clients for training
        min_evaluate_clients=cfg.client_number, #+cfg.n_attackers,  # Never sample less than 5 clients for evaluation
        min_available_clients=cfg.client_number, #+cfg.n_attackers, # Wait until all 10 clients are available
        fraction_fit=1.0, # Sample 100 % of available clients for training
        fraction_evaluate=1.0, # Sample 100 % of available clients for evaluation
        evaluate_metrics_aggregation_fn=weighted_average,
        on_evaluate_config_fn=fit_config,
        on_fit_config_fn=fit_config,
        dataset=cfg.dataset_name,
    )

    print(f"\n\033[94mTraining {cfg.model_name} on {cfg.dataset_name} with {cfg.client_number} clients\033[0m\n")

    # Start Flower server for three rounds of federated learning
    history = fl.server.start_server(
        server_address="0.0.0.0:8098",   # 0.0.0.0 listens to all available interfaces
        config=fl.server.ServerConfig(num_rounds=cfg.n_rounds),
        strategy=strategy,
    )
    # convert history to list
    loss = [k[1] for k in history.losses_distributed]
    accuracy = [k[1] for k in history.metrics_distributed['accuracy']]

    # Save loss and accuracy to a file
    print(f"Saving metrics to as .json in histories folder...")
    with open(f'histories/{cfg.model_name}/{cfg.dataset_name}/distributed_metrics.json', 'w') as f:
        json.dump({'loss': loss, 'accuracy': accuracy}, f)

    # Single Plot
    best_loss_round, best_acc_round = utils.plot_loss_and_accuracy(loss, accuracy, show=False)

    # Load the best model
    model.load_state_dict(torch.load(f"checkpoints/{cfg.model_name}/{cfg.dataset_name}/model_{best_loss_round}.pth"))

    # Evaluate the model on the test set
    loss_test, accuracy_test = models.simple_test(model, device, test_loader)
    print(f"\n\033[93mTest Loss: {loss_test:.3f}, Test Accuracy: {accuracy_test*100:.2f}\033[0m\n")

    # Print training time in minutes (grey color)
    print(f"\033[90mTraining time: {round((time.time() - start_time)/60, 2)} minutes\033[0m")
    time.sleep(1)
    
if __name__ == "__main__":
    main()
