import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import argparse

# Try-except import strategy for make_functional_with_buffers and vmap
try:
    from torch.func import make_functional_with_buffers, vmap
    print("INFO: Imported make_functional_with_buffers and vmap from torch.func")
except ImportError:
    try:
        from functorch import make_functional_with_buffers, vmap
        print("INFO: Imported make_functional_with_buffers and vmap from functorch")
    except ImportError:
        print("ERROR: Could not import make_functional_with_buffers and vmap from torch.func or functorch.")
        raise ImportError("make_functional_with_buffers and vmap are not available.")
import numpy as np
from pathlib import Path
import sys

# Add project root to sys.path to allow relative imports
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.models.torch_mlp import MLP, generate_moons_data # Assuming generate_moons_data is also in torch_mlp or a utils file

# --- Helper for Functional Forward Pass ---
# This function will be vmapped. It takes functional model parameters, buffers, and one input.
def functional_forward_pass_template(functional_model_callable, params, buffers, x, extract_activations=False):
    # functional_model_callable is the result of make_functional_with_buffers, it's callable.
    if extract_activations:
        # Assuming the functional_model_callable, when given correctly structured MLP class,
        # will return (output, activations_list) if its original forward method does.
        # The MLP.forward method is designed to return (output, activations_list)
        y_pred, activations = functional_model_callable(params, buffers, x, extract_activations=True)
        return y_pred, activations
    else:
        y_pred = functional_model_callable(params, buffers, x)
        return y_pred

class VectorizedTrainer:
    def __init__(self, config_path):
        # Workaround for potential inconsistent file reading
        file_content = ""
        try:
            with open(config_path, 'r') as f:
                file_content = f.read()
            # Manually strip trailing backticks and whitespace that might cause YAML errors
            lines = file_content.splitlines()
            while lines and lines[-1].strip() == "```":
                lines.pop()
            cleaned_content = "\n".join(lines)
            self.config = yaml.safe_load(cleaned_content)
        except Exception as e:
            print(f"Error loading or cleaning YAML content from {config_path}: {e}")
            print("Original content that caused error (first 500 chars):")
            print(file_content[:500])
            raise

        model_config = self.config['model']
        training_config = self.config['training']
        self.num_networks = training_config.get('num_networks', 1) # Default to 1 if not specified

        # Device setup
        self.device = torch.device(training_config.get('device', 'cpu'))
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU.")
            self.device = torch.device('cpu')
        elif self.device.type == 'mps' and not torch.backends.mps.is_available():
            print("MPS not available, falling back to CPU.")
            self.device = torch.device('cpu')
        
        print(f"Using device: {self.device}")

        # Create num_networks MLP instances (on CPU first)
        # Updated to use num_hidden_layers and hidden_dim
        mlps = [MLP(input_dim=model_config['input_dim'],
                    num_hidden_layers=model_config['num_hidden_layers'], # Updated
                    hidden_dim=model_config['hidden_dim'],             # Updated
                    output_dim=model_config['output_dim'],
                    activation_fn_name=model_config.get('activation_fn_name', 'relu'),
                    dropout_rate=model_config.get('dropout_rate', 0.0),
                    use_batch_norm=model_config.get('use_batch_norm', False))
                for _ in range(self.num_networks)]

        # Convert all MLPs to functional form and collect their params and buffers
        all_params = []
        all_buffers = []
        
        # Use the first MLP to get the functional model template
        # make_functional_with_buffers returns func_model, params, buffers
        self.functional_model_callable, params_template, buffers_template = make_functional_with_buffers(mlps[0], disable_autograd_tracking=True)

        for i, mlp in enumerate(mlps):
            if i == 0: # Use the already converted first model
                current_params = params_template
                current_buffers = buffers_template
            else:
                _, current_params, current_buffers = make_functional_with_buffers(mlp, disable_autograd_tracking=True)
            all_params.append(list(current_params)) # Ensure it's a list of tensors
            all_buffers.append(list(current_buffers)) # Ensure it's a list of tensors

        # Stack parameters and buffers
        # params_template is a list of tensors (weights and biases for each layer)
        # stacked_params will be a list of tensors, where each tensor is stacked along dim 0 (num_networks)
        
        stacked_params_no_grad = [torch.stack([model_params[i] for model_params in all_params]) for i in range(len(params_template))]
        self.stacked_params = [p.to(self.device).requires_grad_(True) for p in stacked_params_no_grad] # Set requires_grad=True
        
        # Handle cases with no buffers (e.g. MLP without batchnorm)
        if len(buffers_template) > 0:
            self.stacked_buffers = [torch.stack([model_buffers[i] for model_buffers in all_buffers]).to(self.device) for i in range(len(buffers_template))]
        else:
            self.stacked_buffers = [] # No buffers to stack or use

        # Optimizer: Operates on the stacked parameters
        opt_config = training_config.get('optimizer', {'type': 'adam', 'weight_decay': 0.0})
        lr = training_config['learning_rate']
        
        if opt_config['type'].lower() == 'adam':
            self.optimizer = optim.Adam(self.stacked_params, lr=lr, weight_decay=opt_config.get('weight_decay', 0))
        elif opt_config['type'].lower() == 'adamw':
            self.optimizer = optim.AdamW(self.stacked_params, lr=lr, weight_decay=opt_config.get('weight_decay', 0.01))
        elif opt_config['type'].lower() == 'sgd':
            self.optimizer = optim.SGD(self.stacked_params, lr=lr, weight_decay=opt_config.get('weight_decay', 0))
        else:
            raise ValueError(f"Unsupported optimizer: {opt_config['type']}")

        # Loss function
        loss_fn_name = training_config.get('loss_function', 'bce').lower()
        if loss_fn_name == 'bce':
            # For vmap, loss needs to be computed per model, then aggregated.
            # Using reduction='none' allows this.
            self.criterion = nn.BCELoss(reduction='none') 
        elif loss_fn_name == 'mse':
            self.criterion = nn.MSELoss(reduction='none')
        else:
            raise ValueError(f"Unsupported loss function: {loss_fn_name}")
            
        # Vmapped forward function
        # Pass self.functional_model as a static argument
        # Args to functional_forward_pass_template: functional_model_callable, params, buffers, x, extract_activations
        # vmap over params (dim 0), buffers (dim 0), x (None - same x for all models), extract_activations (None)
        self.vmapped_forward = vmap(
            lambda p, b, x, ext_act: functional_forward_pass_template(self.functional_model_callable, p, b, x, ext_act),
            in_dims=(0, 0, None, None), # Stacked params, stacked buffers, common input x, common flag
            out_dims=0 # Output will be stacked along dim 0
        )


    def train(self):
        data_config = self.config['data']
        training_config = self.config['training']

        # Data Loading (using generate_moons_data as an example)
        if data_config['type'] == 'synthetic' and data_config.get('synthetic_type') == 'moons':
            # Set input_dim in config if null, based on data
            if self.config['model']['input_dim'] is None: # This fallback logic might be redundant if config is always correct
                 if data_config.get('synthetic_type') == 'moons':
                     self.config['model']['input_dim'] = 2 
                     print(f"INFO: model.input_dim was null, set to 2 for moons data in VectorizedTrainer.")
                 elif data_config.get('synthetic_type') == 'torus': # Example for another type
                     self.config['model']['input_dim'] = 3
                     print(f"INFO: model.input_dim was null, set to 3 for torus data in VectorizedTrainer.")
            
            # Use data_config['generation']['n'] for synthetic data sample size
            num_samples = data_config.get('generation', {}).get('n', 1000) # Default if generation or n not found
            noise = data_config.get('noise', 0.1)

            X, y = generate_moons_data(num_samples, noise)
            
            split_ratio = data_config.get('split_ratio', 0.8)
            train_size = int(split_ratio * len(X))
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
        else:
            raise NotImplementedError(f"Data type {data_config['type']} or synthetic_type {data_config.get('synthetic_type')} not supported yet.")

        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=training_config['batch_size'], shuffle=True)
        
        test_dataset = torch.utils.data.TensorDataset(X_test, y_test) # For potential evaluation
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=training_config['batch_size'], shuffle=False)


        print(f"Starting training for {self.num_networks} networks.")
        for epoch in range(training_config['epochs']):
            epoch_loss_sum = 0.0
            num_batches = 0
            for batch_x, batch_y in self.train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                # Vectorized forward pass
                # output shape: (num_networks, batch_size, output_dim)
                predictions = self.vmapped_forward(self.stacked_params, self.stacked_buffers, batch_x, False)

                # Loss computation
                # predictions: (num_networks, batch_size, output_dim)
                # batch_y: (batch_size, output_dim)
                # Expand batch_y to match predictions: (1, batch_size, output_dim) then let it broadcast
                # or (num_networks, batch_size, output_dim) by repeating
                batch_y_expanded = batch_y.unsqueeze(0).expand_as(predictions)
                
                loss_all_models = self.criterion(predictions, batch_y_expanded) # (num_networks, batch_size, output_dim)
                
                # Aggregate loss: mean over batch_size, output_dim, then mean over num_networks
                mean_loss_per_model = loss_all_models.mean(dim=[1, 2]) # Shape: (num_networks,)
                total_mean_loss = mean_loss_per_model.mean() # Scalar

                self.optimizer.zero_grad()
                total_mean_loss.backward() # Computes gradients for self.stacked_params
                self.optimizer.step()

                epoch_loss_sum += total_mean_loss.item()
                num_batches += 1
            
            avg_epoch_loss = epoch_loss_sum / num_batches
            print(f"Epoch {epoch+1}/{training_config['epochs']}, Average Loss: {avg_epoch_loss:.4f}")
            
            # TODO: Add evaluation and other metrics if needed

    def extract_layer_outputs_vectorized(self, data_loader):
        # Ensure params and buffers are on the correct device for inference
        # (already done in __init__ if they were moved there, but good practice)
        current_params = [p.to(self.device) for p in self.stacked_params]
        current_buffers = [b.to(self.device) for b in self.stacked_buffers] if self.stacked_buffers else []

        # This list will store, for each type of activation, a list of batched activations
        # Example: collected_activations_batched[0] = [act_type0_batch1, act_type0_batch2, ...]
        # where act_type0_batch1 is (num_networks, batch_size_1, dim_type0)
        collected_activations_batched = []

        print(f"Extracting layer outputs for {self.num_networks} models...")
        for batch_x, _ in data_loader:
            batch_x = batch_x.to(self.device)
            
            # _, list_of_activation_tensors = self.vmapped_forward(...)
            # Each tensor in list_of_activation_tensors is (num_networks, batch_size, feature_dim_k)
            _, batch_activations_list = self.vmapped_forward(current_params, current_buffers, batch_x, True)

            if not collected_activations_batched:
                # Initialize based on the number of activation types collected
                collected_activations_batched = [[] for _ in range(len(batch_activations_list))]

            for i, activation_tensor_for_type_i in enumerate(batch_activations_list):
                # activation_tensor_for_type_i shape: (num_networks, current_batch_size, dim_for_type_i)
                collected_activations_batched[i].append(activation_tensor_for_type_i.cpu())
        
        if not collected_activations_batched or not collected_activations_batched[0]:
            print("No activations collected.")
            # Determine expected shape for empty tensor
            num_datapoints = sum(len(b[1]) for b in data_loader) # Total samples
            return torch.empty(self.num_networks, 0, num_datapoints, 0)


        # Concatenate along batch dimension (dim=1)
        # Result: list of tensors, each (num_networks, num_total_datapoints, feature_dim_k)
        concatenated_activations = [torch.cat(act_type_batches, dim=1) for act_type_batches in collected_activations_batched]

        # Pad and Stack to target shape: (num_networks, num_layers, num_datapoints, max_feature_dimension)
        # num_layers is len(concatenated_activations)
        # num_datapoints is concatenated_activations[0].shape[1]
        
        num_layers_collected = len(concatenated_activations)
        num_datapoints = concatenated_activations[0].shape[1]

        max_feature_dim = 0
        for act_tensor in concatenated_activations: # act_tensor is (num_networks, num_datapoints, dim_k)
            dim_k = act_tensor.shape[2]
            if dim_k > max_feature_dim:
                max_feature_dim = dim_k
        
        padded_and_stacked_activations_list = []
        for act_tensor in concatenated_activations:
            # act_tensor is (num_networks, num_datapoints, dim_k)
            padding_size = max_feature_dim - act_tensor.shape[2]
            if padding_size > 0:
                # Pad on the feature dimension (last dim, index 2)
                # torch.pad expects (pad_left, pad_right, pad_top, pad_bottom, ...)
                # We only want to pad the last dimension: (0, padding_size)
                padded_tensor = torch.nn.functional.pad(act_tensor, (0, padding_size))
            else:
                padded_tensor = act_tensor
            padded_and_stacked_activations_list.append(padded_tensor)
            
        # Stack along the 'num_layers' dimension (new dim at index 1)
        # Each tensor in list is (num_networks, num_datapoints, max_feature_dim)
        # Resulting tensor: (num_layers_collected, num_networks, num_datapoints, max_feature_dim)
        final_stacked_tensor_temp = torch.stack(padded_and_stacked_activations_list, dim=0)

        # Permute to get (num_networks, num_layers_collected, num_datapoints, max_feature_dim)
        final_stacked_tensor = final_stacked_tensor_temp.permute(1, 0, 2, 3)
        
        return final_stacked_tensor

# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train vectorized MLP models using PyTorch.")
    parser.add_argument("config_path", type=str, help="Path to the YAML configuration file.")
    args = parser.parse_args()

    # The input_dim check is removed.
    # The VectorizedTrainer.__init__ method has its own fallback for input_dim if needed,
    # but the test config explicitly sets model.input_dim = 2.

    trainer = VectorizedTrainer(args.config_path)
    trainer.train()
    
    print("\nExtracting layer outputs from the test set (vectorized)...")
    # Using test_loader for extraction
    if hasattr(trainer, 'test_loader'):
        all_models_activations = trainer.extract_layer_outputs_vectorized(trainer.test_loader)
        print(f"Shape of extracted layer outputs tensor for all models: {all_models_activations.shape}")
        # Expected: (num_networks, num_layers_collected, num_datapoints_in_test_set, max_feature_dim)
    else:
        print("Test loader not available for activation extraction.")
