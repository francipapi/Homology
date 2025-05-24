import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import yaml
import argparse
from sklearn.datasets import make_moons
# import trimesh # Add if using torus data generation

# --- MLP Class ---
class MLP(nn.Module):
    def __init__(self, input_dim, num_hidden_layers, hidden_dim, output_dim, activation_fn_name='relu', dropout_rate=0.2, use_batch_norm=True):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        # self.hidden_dims = hidden_dims # Replaced by num_hidden_layers and hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.activation_fn_name = activation_fn_name.lower()
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm

        self.layers = nn.ModuleList()
        
        current_dim = input_dim
        
        # Hidden layers
        for _ in range(num_hidden_layers):
            self.layers.append(nn.Linear(current_dim, hidden_dim))
            self._add_activation_norm_dropout(hidden_dim) # Pass hidden_dim for BatchNorm size
            current_dim = hidden_dim # Update current_dim for the next layer
            
        # Output layer
        self.layers.append(nn.Linear(current_dim, output_dim))
        self.layers.append(nn.Sigmoid()) # Assuming binary classification

        # Initialize weights
        self._initialize_weights()

    def _add_activation_norm_dropout(self, h_dim):
        if self.activation_fn_name == 'relu':
            self.layers.append(nn.ReLU())
        elif self.activation_fn_name == 'tanh':
            self.layers.append(nn.Tanh())
        else:
            raise ValueError(f"Unsupported activation function: {self.activation_fn_name}")

        if self.use_batch_norm:
            self.layers.append(nn.BatchNorm1d(h_dim))
        
        if self.dropout_rate > 0:
            self.layers.append(nn.Dropout(self.dropout_rate))

    def _initialize_weights(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity=self.activation_fn_name if self.activation_fn_name in ['relu', 'tanh'] else 'leaky_relu') # kaiming for relu/tanh
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    def forward(self, x, extract_activations=False):
        activations = []
        # Ensure x is 2D (batch_size, features)
        if x.ndim == 1:
            x = x.unsqueeze(0) # Add batch dimension if single sample
        
        for layer in self.layers:
            x = layer(x) # Apply current layer
            if extract_activations:
                if isinstance(layer, nn.Linear):
                    # This is "output of Linear layer (before activation)"
                    activations.append(x.detach().clone()) 
                elif isinstance(layer, (nn.ReLU, nn.Tanh, nn.Sigmoid)):
                    # This is "output of each activation function"
                    activations.append(x.detach().clone())
        
        if extract_activations:
            return x, activations
        return x

    def extract_layer_outputs(self, data_loader, device):
        self.eval()
        all_layer_outputs = [] # To store outputs from all layers across all batches

        for batch_idx, (data, _) in enumerate(data_loader):
            data = data.to(device)
            _, batch_activations = self.forward(data, extract_activations=True)
            
            if not all_layer_outputs: # Initialize list of lists for each layer
                all_layer_outputs = [[] for _ in range(len(batch_activations))]

            for i, activation in enumerate(batch_activations):
                all_layer_outputs[i].append(activation.cpu())
        
        # Concatenate activations for each layer
        # Resulting shape for each element in final_activations: (num_datapoints, layer_output_dim)
        final_activations = [torch.cat(layer_outputs, dim=0) for layer_outputs in all_layer_outputs if layer_outputs]

        # Reshape to target: (1, num_layers, num_datapoints, hidden_dimension)
        # num_layers here refers to the number of collected activation tensors.
        # hidden_dimension varies per layer.
        # This will result in a list of tensors, each shaped (1, 1, num_datapoints, layer_output_dim)
        # and then stacked.
        
        reshaped_activations = []
        for act_tensor in final_activations:
            num_datapoints = act_tensor.shape[0]
            layer_output_dim = act_tensor.shape[1] if act_tensor.ndim > 1 else 1
            if act_tensor.ndim == 1: # for scalar outputs like from some activation functions or final output
                act_tensor = act_tensor.unsqueeze(1)
            reshaped_activations.append(act_tensor.unsqueeze(0).unsqueeze(0)) # (1, 1, num_datapoints, layer_output_dim)

        # The request is (1, num_layers, num_datapoints, hidden_dimension)
        # This implies that all hidden_dimensions must be the same, or we need padding.
        # For now, let's return a list of tensors, as dimensions might vary.
        # Or, if the goal is to stack them, they must have compatible shapes.
        # The prompt: "The hidden_dimension would be the dimension of that specific layer's output."
        # This suggests a list of tensors is more appropriate if dimensions differ.
        # However, "(1, num_layers, num_datapoints, hidden_dimension)" implies a single tensor.

        # Let's clarify: "num_layers here refers to the number of hidden layers + output layer."
        # The way activations are collected (input to Linear, output of Act/BN) means more tensors than "layers" in the MLP definition.
        # For simplicity, let's define "num_layers" as the number of tensors in `final_activations`.

        # If a single tensor is strictly required and dimensions vary, padding would be needed.
        # Given the ambiguity, returning a list of tensors (each [num_datapoints, layer_dim]) is safer.
        # The prompt "The final output structure for activations should be a list of tensors, 
        # where each tensor corresponds to a layer's output for all data points." supports this.
        # Then it says "Reshape/stack these to match the target shape: (1, num_layers, num_datapoints, hidden_dimension)"
        # This is contradictory if hidden_dimension varies.

        # Let's assume for now the user wants them stacked if possible, or a list if not.
        # For the requested shape (1, num_layers, num_datapoints, hidden_dimension),
        # it implies all layers must have the same hidden_dimension for stacking into a single tensor.
        # This is generally not true (e.g. input layer, output layer).

        # Let's return the list of (num_datapoints, layer_dim) tensors.
        # The user can then decide how to process this list further if a single tensor is needed.
        # Or, let's attempt to create the specified shape if we consider "hidden_dimension" to be max_hidden_dim and pad.
        # This seems overly complex for now.
        # Let's stick to the "list of tensors" which is explicitly mentioned.

        # Re-reading: "The final output structure for activations should be a list of tensors,
        # where each tensor corresponds to a layer's output for all data points."
        # This is `final_activations`.
        # "Reshape/stack these to match the target shape: (1, num_layers, num_datapoints, hidden_dimension)."
        # This part is tricky. If hidden_dimension is specific to *that* layer, it cannot be a single tensor.
        # It probably means a list of tensors, where each tensor is (1, 1, num_datapoints, layer_specific_dim).

        # Let's provide a list of tensors, where each tensor is shaped (num_datapoints, layer_dim)
        # The reshaping to (1, num_layers, ...) seems to imply a different structure.
        # Let's re-evaluate the forward pass activation collection.
        # "store the output of each Linear layer (before activation) and each activation function"

        # If we store only post-activation outputs of hidden layers and the final output layer:
        # MLP: Input -> L1 -> A1 -> L2 -> A2 -> L3 -> Sigmoid
        # Activations: A1_out, A2_out, Sigmoid_out
        # These would have dimensions: h_dim1, h_dim2, output_dim. Still variable.

        # Let's try to follow the shape (1, num_layers, num_datapoints, hidden_dimension) by returning a list of tensors,
        # where each tensor in the list is (1, num_datapoints, hidden_dimension_for_that_layer).
        # And `num_layers` would be the length of this list.

        processed_activations_for_stacking = []
        max_dim = 0
        for act_tensor in final_activations: # Each is (num_datapoints, layer_dim)
            if act_tensor.shape[1] > max_dim:
                max_dim = act_tensor.shape[1]
        
        for act_tensor in final_activations:
            num_datapoints = act_tensor.shape[0]
            layer_dim = act_tensor.shape[1]
            # Pad to max_dim if necessary for stacking later into the (..., hidden_dimension) part
            # For now, let's not pad, and assume "hidden_dimension" means "dimension_of_this_layer"
            # and the "stack" means creating a list of tensors of shape (1, num_datapoints, layer_dim)
            # and then concatenating them along a new "num_layers" dimension.

            # Target for each item: (1, num_datapoints, layer_dim)
            processed_activations_for_stacking.append(act_tensor.unsqueeze(0))

        # Stack along the new 'num_layers' dimension (dim=1)
        # This requires all tensors to have the same dimensions for features, which is not guaranteed.
        # Example: input layer (dim 2), hidden (dim 32), output (dim 1)

        # Let's go with the explicit "list of tensors" for now, as that's robust.
        # final_activations is already a list of [ (num_datapoints, layer_dim_i) ]
        # The request "Reshape/stack these to match the target shape: (1, num_layers, num_datapoints, hidden_dimension)"
        # is only possible if hidden_dimension is constant or padded.
        # Given the context, it's more likely they want a list of tensors, where each is (num_datapoints, layer_dim)
        # and "num_layers" is the len of this list. The (1, ...) part is confusing.

        # Let's assume the target shape (1, num_layers, num_datapoints, hidden_dimension)
        # means that IF all hidden_dimensions were the same, we could stack.
        # Since they are not, we will return a list of tensors.
        # Each tensor in the list will be (num_datapoints, layer_output_dim).
        # The "1" at the beginning and "num_layers" as a dimension seem to imply a single tensor.

        # Re-interpreting: "The final output structure for activations should be a list of tensors,
        # where each tensor corresponds to a layer's output for all data points." -> This is `final_activations`.
        # "Reshape/stack these to match the target shape: (1, num_layers, num_datapoints, hidden_dimension)."
        # This could mean: create a single tensor by padding if necessary.
        # Let's try to make it a single tensor with padding.
        # `num_layers` = number of activation outputs collected.
        # `hidden_dimension` = max dimension among these.

        num_collected_layers = len(final_activations)
        num_datapoints = final_activations[0].shape[0] if num_collected_layers > 0 else 0
        
        max_feature_dim = 0
        for act_tensor in final_activations: # act_tensor is (num_datapoints, layer_dim)
            if act_tensor.shape[1] > max_feature_dim:
                max_feature_dim = act_tensor.shape[1]

        if num_collected_layers == 0:
            return torch.empty(1, 0, num_datapoints, max_feature_dim)

        # Pad each activation tensor to (num_datapoints, max_feature_dim)
        padded_activations = []
        for act_tensor in final_activations:
            padding_size = max_feature_dim - act_tensor.shape[1]
            if padding_size > 0:
                # Pad on the feature dimension (dim 1)
                padded_tensor = torch.nn.functional.pad(act_tensor, (0, padding_size))
            else:
                padded_tensor = act_tensor
            padded_activations.append(padded_tensor) # List of (num_datapoints, max_feature_dim)

        # Stack them to create (num_layers, num_datapoints, max_feature_dim)
        stacked_activations = torch.stack(padded_activations, dim=0)

        # Add the leading '1' dimension: (1, num_layers, num_datapoints, max_feature_dim)
        output_tensor = stacked_activations.unsqueeze(0)
        
        return output_tensor


# --- Data Generation ---
def generate_moons_data(n_samples, noise):
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    return X, y

# In torch_simple_mlp.py, there was a generate_torus_data.
# For now, only moons is directly used by default config.
# We can add generate_torus_data if trimesh is confirmed available/desirable.

# --- Training Function ---
def train_model(config_path):
    # Workaround for potential inconsistent file reading for YAML
    file_content = ""
    try:
        with open(config_path, 'r') as f:
            file_content = f.read()
        # Manually strip trailing backticks and whitespace that might cause YAML errors
        lines = file_content.splitlines()
        while lines and lines[-1].strip() == "```":
            lines.pop()
        cleaned_content = "\n".join(lines)
        config = yaml.safe_load(cleaned_content)
    except Exception as e:
        print(f"Error loading or cleaning YAML content from {config_path}: {e}")
        print("Original content that caused error (first 500 chars):")
        print(file_content[:500])
        raise

    model_config = config['model']
    training_config = config['training']
    data_config = config['data']

    # Device
    device = torch.device(training_config['device'])

    # Model
    model = MLP(
        input_dim=model_config['input_dim'],
        num_hidden_layers=model_config['num_hidden_layers'], # Updated
        hidden_dim=model_config['hidden_dim'],             # Updated
        output_dim=model_config['output_dim'],
        activation_fn_name=model_config.get('activation_fn_name', 'relu'), # Corrected key from 'activation'
        dropout_rate=model_config.get('dropout_rate', 0.0),
        use_batch_norm=model_config.get('use_batch_norm', False)
    ).to(device)

    # Optimizer
    lr = training_config['learning_rate']
    opt_config = training_config.get('optimizer', {'type': 'adam'}) # Get the optimizer config dict
    optimizer_type = opt_config.get('type', 'adam').lower() # Get the type string
    
    if optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=opt_config.get('weight_decay', 0.0))
    elif optimizer_type == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=opt_config.get('weight_decay', 0.01))
    elif optimizer_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=opt_config.get('weight_decay', 0.0))
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_type}")

    # Loss function
    loss_fn_name = training_config.get('loss_function', 'bce').lower() # This is fine, loss_function is a direct key
    if loss_fn_name == 'bce':
        criterion = nn.BCELoss()
    elif loss_fn_name == 'mse':
        criterion = nn.MSELoss()
    else:
        raise ValueError(f"Unsupported loss function: {loss_fn_name}")

    # Data
    if data_config['type'] == 'synthetic':
        # Use data_config['generation']['n'] for synthetic data sample size
        num_samples = data_config.get('generation', {}).get('n', 1000) # Default if generation or n not found
        noise = data_config.get('noise', 0.1) # noise can be a top-level key in data for synthetic types

        if data_config.get('synthetic_type', 'moons') == 'moons':
            X, y = generate_moons_data(num_samples, noise)
        # Add other synthetic types like torus here if needed
        # elif data_config.get('synthetic_type') == 'torus':
        #     # Example for torus if it used different specific params from 'generation' block
        #     big_radius = data_config.get('generation', {}).get('big_radius', 3)
        #     small_radius = data_config.get('generation', {}).get('small_radius', 1)
        #     X, y = generate_torus_data(num_samples, big_radius, small_radius) # Assuming generate_torus_data exists
        #     X, y = generate_torus_data(...) # Requires trimesh and more params
        else:
            raise ValueError(f"Unsupported synthetic data type: {data_config.get('synthetic_type')}")
        
        split_ratio = data_config.get('split_ratio', 0.8)
        train_size = int(split_ratio * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
    # Add other data types like 'csv' here
    # elif data_config['type'] == 'csv':
    #     # Load from CSV, preprocess, split
    #     pass
    else:
        raise ValueError(f"Unsupported data type: {data_config['type']}")

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=training_config['batch_size'], shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=training_config['batch_size'], shuffle=False)

    # Scheduler (optional)
    scheduler = None
    scheduler_config = training_config.get('scheduler', {}) # Get scheduler config dict
    scheduler_type = scheduler_config.get('type', 'none').lower()

    if scheduler_type == 'reduce_on_plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            factor=scheduler_config.get('factor', 0.1), 
            patience=scheduler_config.get('patience', 10)
        )
    elif scheduler_type == 'step_lr':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_config.get('step_size', 30),
            gamma=scheduler_config.get('gamma', 0.1)
        )
    # Add more schedulers as needed

    # Early Stopping (optional)
    early_stopping_config = training_config.get('early_stopping', {})
    use_early_stopping = early_stopping_config.get('use', False)
    best_val_loss = float('inf')
    epochs_no_improve = 0

    # Training Loop
    for epoch in range(training_config['epochs']):
        model.train()
        train_loss_sum = 0
        correct_train = 0
        total_train = 0

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()

            grad_clip_config = training_config.get('gradient_clipping', {})
            if grad_clip_config.get('use', False) and grad_clip_config.get('max_norm') is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_config['max_norm'])
            
            optimizer.step()
            train_loss_sum += loss.item()

            predicted = (output > 0.5).float()
            total_train += target.size(0)
            correct_train += (predicted == target).sum().item()
        
        avg_train_loss = train_loss_sum / len(train_loader)
        train_accuracy = correct_train / total_train if total_train > 0 else 0

        # Evaluation Loop
        model.eval()
        test_loss_sum = 0
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                test_loss_sum += loss.item()
                
                predicted = (output > 0.5).float()
                total_test += target.size(0)
                correct_test += (predicted == target).sum().item()

        avg_test_loss = test_loss_sum / len(test_loader)
        test_accuracy = correct_test / total_test if total_test > 0 else 0

        print(f"Epoch {epoch+1}/{training_config['epochs']} - "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f} - "
              f"Test Loss: {avg_test_loss:.4f}, Test Acc: {test_accuracy:.4f}")

        if scheduler:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_test_loss)
            else:
                scheduler.step()
        
        # Early Stopping
        if use_early_stopping:
            min_delta = early_stopping_config.get('min_delta', 0.0001)
            patience = early_stopping_config.get('patience', 10)
            if avg_test_loss < best_val_loss - min_delta:
                best_val_loss = avg_test_loss
                epochs_no_improve = 0
                # Optionally save best model
                # torch.save(model.state_dict(), "best_model.pth") # Add path config
            else:
                epochs_no_improve += 1
            
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break
    
    print("Training finished.")

    # Extract layer outputs (demonstration)
    # This uses the test_loader by default.
    # The 'extract_final_activations' key in model_config is not standard per current YAML structure,
    # so let's make this unconditional for the test run, or rely on a new top-level config key.
    # For testing, we always want to see this output.
    print("\nExtracting layer outputs from the test set for torch_mlp.py...")
    model.to(device) 
    layer_outputs_tensor = model.extract_layer_outputs(test_loader, device)
    print(f"torch_mlp.py: Shape of extracted layer outputs tensor: {layer_outputs_tensor.shape}")
    # Example: (1, num_activation_layers, num_test_samples, max_feature_dim_padded)
    # Individual layers can be accessed via layer_outputs_tensor[0, layer_idx, :, :feature_dim_for_layer_idx]


# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an MLP model using a YAML configuration file.")
    parser.add_argument("config_path", type=str, help="Path to the YAML configuration file.")
    args = parser.parse_args()
    
    train_model(args.config_path)

    # Example of how to create a dummy config and run if no config is provided (for testing)
    # This should ideally be run by providing a config file.
    # For quick testing, one could create a dummy config here:
    # if not os.path.exists(args.config_path):
    #     print(f"Config file {args.config_path} not found. Creating a dummy config for testing.")
    #     dummy_config = { ... } # Define a dummy config dict
    #     with open("dummy_config.yaml", 'w') as f:
    #         yaml.dump(dummy_config, f)
    #     args.config_path = "dummy_config.yaml"
    #     train_model(args.config_path)
