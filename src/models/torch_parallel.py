import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import yaml
import argparse
import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys
import time
import trimesh as tr

# Add project root to sys.path to allow relative imports
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.models.torch_mlp import MLP, generate_torus_data

# --- Worker Function for Parallel Training ---
def train_single_model_process(model_id, config_dict, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, device_for_process_str):
    """
    Trains a single MLP model. This function is designed to be run in a separate process.
    """
    try:
        model_config = config_dict['model']
        training_config = config_dict['training']
        
        # Set seed for this process for reproducibility and variation if desired
        process_seed = training_config.get('seed', 42) + model_id
        torch.manual_seed(process_seed)
        np.random.seed(process_seed)

        device = torch.device(device_for_process_str)

        # Initialize model
        # Updated to use num_hidden_layers and hidden_dim
        model = MLP(
            input_dim=model_config['input_dim'],
            num_hidden_layers=model_config['num_hidden_layers'], # Updated
            hidden_dim=model_config['hidden_dim'],             # Updated
            output_dim=model_config['output_dim'],
            activation_fn_name=model_config.get('activation_fn_name', 'relu'),
            dropout_rate=model_config.get('dropout_rate', 0.0),
            use_batch_norm=model_config.get('use_batch_norm', False)
        ).to(device)

        # Optimizer
        opt_config = training_config.get('optimizer', {'type': 'adam'})
        lr = training_config['learning_rate']
        if opt_config['type'].lower() == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=opt_config.get('weight_decay', 0))
        elif opt_config['type'].lower() == 'adamw':
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=opt_config.get('weight_decay', 0.01))
        elif opt_config['type'].lower() == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=opt_config.get('weight_decay', 0))
        else:
            raise ValueError(f"Unsupported optimizer: {opt_config['type']}")

        # Loss function
        loss_fn_name = training_config.get('loss_function', 'bce').lower()
        if loss_fn_name == 'bce':
            criterion = nn.BCELoss()
        elif loss_fn_name == 'mse':
            criterion = nn.MSELoss()
        else:
            raise ValueError(f"Unsupported loss function: {loss_fn_name}")

        # DataLoaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=training_config['batch_size'], shuffle=True)
        
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        test_loader = DataLoader(test_dataset, batch_size=training_config['batch_size'], shuffle=False)

        # Training Loop (simplified, add schedulers, early stopping, grad clipping from torch_mlp.py if full features needed)
        for epoch in range(training_config['epochs']):
            model.train()
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
            # Basic logging, can be expanded
            # print(f"Model {model_id}, Epoch {epoch+1}, Loss: {loss.item():.4f}")


        # Evaluation
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item() * data.size(0) # Sum batch loss
                predicted = (output > 0.5).float() # Assuming binary classification
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        avg_test_loss = test_loss / total if total > 0 else float('inf')
        test_accuracy = correct / total if total > 0 else 0.0
        
        metrics = {'test_loss': avg_test_loss, 'test_accuracy': test_accuracy}
        print(f"Model {model_id} trained. Test Acc: {test_accuracy:.4f}, Test Loss: {avg_test_loss:.4f}")

        # Extract layer outputs
        # The MLP.extract_layer_outputs method returns (1, num_layers, num_datapoints, hidden_dimension)
        layer_outputs_tensor = model.extract_layer_outputs(test_loader, device) # test_loader for full test set

        return model_id, model.state_dict(), metrics, layer_outputs_tensor

    except Exception as e:
        print(f"Error in model {model_id} process: {e}")
        import traceback
        traceback.print_exc()
        return model_id, None, {'test_loss': float('inf'), 'test_accuracy': 0.0}, None


class ParallelTrainer:
    def __init__(self, config_path):
        # Workaround for potential inconsistent file reading for YAML
        file_content = ""
        try:
            with open(config_path, 'r') as f:
                file_content = f.read()
            lines = file_content.splitlines()
            while lines and lines[-1].strip() == "```":
                lines.pop()
            cleaned_content = "\n".join(lines)
            self.config = yaml.safe_load(cleaned_content)
        except Exception as e:
            print(f"Error loading or cleaning YAML content from {config_path} in ParallelTrainer: {e}")
            print("Original content that caused error (first 500 chars):")
            print(file_content[:500])
            raise
        
        self.num_networks = self.config['training'].get('num_networks', 1)
        
        # Data loading and preparation (once in main process)
        data_config = self.config['data']
        
        # Set input_dim in config if null, based on data
        if self.config['model'].get('input_dim') is None:
            self.config['model']['input_dim'] = 3
            print(f"INFO: model.input_dim was null, set to 3 for torus data for ParallelTrainer.")

        if data_config['type'] == 'synthetic':
            # Use data_config['generation']['n'] for synthetic data sample size
            num_samples = data_config.get('generation', {}).get('n', 1000)
            big_radius = data_config.get('generation', {}).get('big_radius', 3)
            small_radius = data_config.get('generation', {}).get('small_radius', 1)
            X, y = generate_torus_data(num_samples, big_radius, small_radius)

            # Split data
            split_ratio = data_config.get('split_ratio', 0.8)
            shuffle = data_config.get('shuffle_data', True)
            random_seed = data_config.get('random_seed_data', None)
            
            if shuffle:
                if random_seed is not None:
                    np.random.seed(random_seed)
                indices = np.arange(X.shape[0])
                np.random.shuffle(indices)
                X = X[indices]
                y = y[indices]

            train_size = int(split_ratio * len(X))
            self.X_train, self.X_test = X[:train_size], X[train_size:]
            self.y_train, self.y_test = y[:train_size], y[train_size:]
            
            # Convert to tensors to pass to processes
            self.X_train_tensor = torch.tensor(self.X_train, dtype=torch.float32)
            self.y_train_tensor = torch.tensor(self.y_train, dtype=torch.float32)
            self.X_test_tensor = torch.tensor(self.X_test, dtype=torch.float32)
            self.y_test_tensor = torch.tensor(self.y_test, dtype=torch.float32)

            self.collected_results = []
        else:
            raise NotImplementedError(f"Data type {data_config['type']} not supported yet for parallel training.")

    def train_models_parallel(self):
        # For CPU-bound tasks in parallel, ProcessPoolExecutor is suitable.
        # Each process will run on CPU.
        device_for_processes = 'cpu' 
        
        # Determine number of workers
        max_workers_config = self.config['training'].get('max_parallel_workers', os.cpu_count())
        num_workers = min(os.cpu_count(), self.num_networks, max_workers_config if max_workers_config else os.cpu_count())
        
        print(f"Starting parallel training for {self.num_networks} models using {num_workers} worker processes on CPU.")

        futures = []
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            for i in range(self.num_networks):
                # Pass the full config_dict, and raw data tensors
                future = executor.submit(
                    train_single_model_process,
                    i, # model_id
                    self.config, # full config dict
                    self.X_train_tensor, 
                    self.y_train_tensor,
                    self.X_test_tensor,
                    self.y_test_tensor,
                    device_for_processes
                )
                futures.append(future)

            for future in as_completed(futures):
                try:
                    result = future.result() # (model_id, state_dict, metrics, layer_outputs_tensor)
                    if result[1] is not None: # Check if state_dict is not None (i.e., no error in process)
                        self.collected_results.append(result)
                    else:
                        print(f"Model {result[0]} failed training, result not collected.")
                except Exception as e:
                    print(f"Exception collecting result from process: {e}")
        
        # Sort results by model_id for consistency, if needed
        self.collected_results.sort(key=lambda x: x[0])
        print(f"Parallel training finished. Collected results for {len(self.collected_results)} models.")
        return self.collected_results

    def process_and_stack_results(self):
        if not self.collected_results:
            print("No results collected to process.")
            # Define expected shape for empty tensor
            num_datapoints = len(self.X_test_tensor)
            # Infer num_layers and max_feature_dim from a dummy model if possible, or use placeholders
            # This is tricky if no models succeeded. For now, returning empty or raising error.
            # For simplicity, assume if no results, we can't determine the shape.
            # A robust way needs a model instance to call extract_layer_outputs to get shape.
            # Let's return an empty tensor with a warning.
            print("Warning: Cannot determine exact shape for empty activation tensor without a model template.")
            return torch.empty(self.num_networks, 0, num_datapoints, 0), {}


        # Stack layer outputs
        # Each layer_outputs_tensor from train_single_model_process is (1, num_layers, num_datapoints, hidden_dim)
        # We need to cat them along the first dimension (model index)
        
        valid_activation_tensors = [res[3] for res in self.collected_results if res[3] is not None]
        
        if not valid_activation_tensors:
            print("No valid activation tensors collected.")
            num_datapoints = len(self.X_test_tensor)
            print("Warning: Cannot determine exact shape for empty activation tensor.")
            return torch.empty(self.num_networks, 0, num_datapoints, 0), {}

        # Assuming all tensors have the same num_layers, num_datapoints, and hidden_dim structure from MLP.extract_layer_outputs
        # (padding to max_hidden_dim is done inside MLP.extract_layer_outputs)
        stacked_activations = torch.cat(valid_activation_tensors, dim=0)
        # Expected shape: (num_successful_models, num_layers, num_datapoints, max_hidden_dimension)
        # If num_successful_models != self.num_networks, this might not be (self.num_networks, ...)

        # Process metrics
        all_metrics = [res[2] for res in self.collected_results if res[2] is not None]
        if all_metrics:
            avg_test_accuracy = np.mean([m['test_accuracy'] for m in all_metrics])
            avg_test_loss = np.mean([m['test_loss'] for m in all_metrics])
            summary_metrics = {'average_test_accuracy': avg_test_accuracy, 'average_test_loss': avg_test_loss}
            print(f"Summary Metrics: Avg Test Accuracy: {avg_test_accuracy:.4f}, Avg Test Loss: {avg_test_loss:.4f}")
        else:
            summary_metrics = {}
            print("No metrics collected.")
            
        return stacked_activations, summary_metrics


# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train multiple MLP models in parallel on CPU using PyTorch.")
    parser.add_argument("config_path", type=str, help="Path to the YAML configuration file.")
    args = parser.parse_args()

    # Basic check for model.input_dim (as in other scripts)
    # Applying YAML loading workaround here as well
    temp_file_content = ""
    try:
        with open(args.config_path, 'r') as f_check:
            temp_file_content = f_check.read()
        temp_lines = temp_file_content.splitlines()
        while temp_lines and temp_lines[-1].strip() == "```":
            temp_lines.pop()
        temp_cleaned_content = "\n".join(temp_lines)
        temp_config_check = yaml.safe_load(temp_cleaned_content)
    except Exception as e:
        print(f"Error loading or cleaning YAML content from {args.config_path} in main pre-check: {e}")
        print("Original content that caused error (first 500 chars):")
        print(temp_file_content[:500])
        raise

    if temp_config_check['model'].get('input_dim') is None:
        if temp_config_check['data'].get('synthetic_type') == 'moons':
            print("WARNING: model.input_dim in YAML is null. Should be 2 for moons data. Will be set internally.")
        elif temp_config_check['data'].get('synthetic_type') == 'torus':
             print("WARNING: model.input_dim in YAML is null. Should be 3 for torus data. Will be set internally.")
        else:
            print("WARNING: model.input_dim in YAML is null and synthetic_type is not moons/torus. Please set it manually in YAML.")


    start_time = time.time()
    
    trainer = ParallelTrainer(args.config_path)
    collected_run_results = trainer.train_models_parallel()
    
    if collected_run_results:
        final_stacked_activations, summary_metrics_results = trainer.process_and_stack_results()
        print(f"\nShape of final stacked layer outputs: {final_stacked_activations.shape}")
        # Expected: (num_networks (or num_successful), num_layers_collected, num_datapoints_in_test_set, max_feature_dim)
        # num_layers_collected and max_feature_dim are determined by MLP.extract_layer_outputs's padding logic.
    else:
        print("No models were successfully trained or no results collected.")

    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds.")
