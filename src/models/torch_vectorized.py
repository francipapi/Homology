import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import argparse
from torch.utils.data import ConcatDataset, DataLoader

# Import strategy for make_functional_with_buffers and vmap
try:
    from torch.func import make_functional_with_buffers, vmap
except ImportError:
    try:
        from functorch import make_functional_with_buffers, vmap
    except ImportError:
        print("ERROR: Could not import make_functional_with_buffers and vmap.")
        print("Please install either torch>=2.0.0 or functorch")
        raise ImportError("Required vectorization utilities not available.")

import numpy as np
from pathlib import Path
import sys

# --- Dynamic Project Root Addition ---
try:
    current_file_path = Path(__file__).resolve()
    project_root_found = False
    _temp_path = current_file_path
    for _ in range(5): 
        if (_temp_path.parent / 'src').is_dir():
            project_root = _temp_path.parent
            if str(project_root) not in sys.path:
                 sys.path.append(str(project_root))
            print(f"Project root identified and added to sys.path: {project_root}")
            project_root_found = True
            break
        _temp_path = _temp_path.parent
    if not project_root_found:
        project_root = Path(__file__).resolve().parent.parent.parent # Original fallback
        if str(project_root) not in sys.path:
            sys.path.append(str(project_root))
        print(f"Falling back to project root (original logic): {project_root}")

    from src.models.torch_mlp import MLP, generate_torus_data, load_data_from_file
    print("Successfully imported MLP, generate_torus_data, and load_data_from_file.")
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print(f"Searched for 'src' directory upwards from {current_file_path if '__file__' in locals() else 'current script directory'}.")
    print("Please ensure your project structure is correct or adjust PYTHONPATH.")
    if 'src.models.torch_mlp' not in sys.modules:
         raise ImportError(
             "Could not import MLP and generate_torus_data from src.models.torch_mlp. "
             "Check 'src' directory in Python path and 'models/torch_mlp.py'."
         )
except NameError: 
    print("Warning: __file__ not defined. Relative imports might fail. Ensure 'src' is in PYTHONPATH.")
    from src.models.torch_mlp import MLP, generate_torus_data, load_data_from_file


# --- Helper for Functional Forward Pass ---
def functional_forward_pass_template(
        functional_model_callable,
        params,
        buffers,
        x,
        extract_activations: bool = False):
    """
    Wrapper used by vmap.  When `extract_activations` is True we call the
    underlying stateless model with the keyword that MLP.forward expects
    (`extract_hidden_activations`).
    """
    if extract_activations:
        y_pred, activations = functional_model_callable(
            params, buffers, x,                     # positional args
            extract_hidden_activations=True         # <-- fixed name
        )
        return y_pred, activations
    else:
        y_pred = functional_model_callable(params, buffers, x)
        return y_pred

class VectorizedTrainer:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        model_config = self.config['model']
        training_config = self.config['training']
        self.num_networks = training_config.get('num_networks', 1)

        device_name = training_config.get('device', 'cpu')
        if device_name == 'mps':
            if not torch.backends.mps.is_available():
                print("MPS not available, falling back to CPU.")
                self.device = torch.device('cpu')
            else:
                self.device = torch.device('mps')
                print("MPS device selected. For unsupported op fallback, set env var PYTORCH_ENABLE_MPS_FALLBACK=1.")
        elif device_name == 'cuda':
            if not torch.cuda.is_available():
                print("CUDA not available, falling back to CPU.")
                self.device = torch.device('cpu')
            else:
                cuda_device_id = training_config.get('cuda_device_id', 0)
                self.device = torch.device(f'cuda:{cuda_device_id}')
        else:
            self.device = torch.device('cpu')
        
        print(f"Using device: {self.device}")

        mlps = [MLP(
            input_dim=model_config['input_dim'],
            num_hidden_layers=model_config['num_hidden_layers'],
            hidden_dim=model_config['hidden_dim'],
            output_dim=model_config['output_dim'],
            activation_fn_name=model_config.get('activation_fn_name', 'relu'),
            dropout_rate=model_config.get('dropout_rate', 0.0),
            use_batch_norm=model_config.get('use_batch_norm', False)
        ) for _ in range(self.num_networks)]

        all_params_cpu = []
        all_buffers_cpu = []
        
        self.functional_model_callable, params_template_cpu, buffers_template_cpu = make_functional_with_buffers(
            mlps[0], 
            disable_autograd_tracking=True
        )

        for i, mlp in enumerate(mlps):
            if i == 0:
                current_params_cpu = params_template_cpu
                current_buffers_cpu = buffers_template_cpu
            else:
                _, current_params_cpu, current_buffers_cpu = make_functional_with_buffers(
                    mlp, 
                    disable_autograd_tracking=True
                )
            all_params_cpu.append(list(current_params_cpu))
            all_buffers_cpu.append(list(current_buffers_cpu))

        stacked_params_no_grad_cpu = [
            torch.stack([model_params_set[i] for model_params_set in all_params_cpu])
            for i in range(len(params_template_cpu))
        ]
        self.stacked_params = [p.to(self.device).requires_grad_(True) for p in stacked_params_no_grad_cpu]
        
        if len(buffers_template_cpu) > 0:
            stacked_buffers_cpu = [
                torch.stack([model_buffers_set[i] for model_buffers_set in all_buffers_cpu])
                for i in range(len(buffers_template_cpu))
            ]
            self.stacked_buffers = [b.to(self.device) for b in stacked_buffers_cpu]
        else:
            self.stacked_buffers = []

        opt_config = training_config.get('optimizer', {'name': 'adam', 'weight_decay': 0.0})
        lr = training_config['learning_rate']
        optimizer_name = opt_config['name'].lower()
        weight_decay = opt_config.get('weight_decay', 0.0)
        
        if optimizer_name == 'adam':
            self.optimizer = optim.Adam(self.stacked_params, lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'adamw':
            self.optimizer = optim.AdamW(self.stacked_params, lr=lr, weight_decay=weight_decay if weight_decay > 0 else 0.01)
        elif optimizer_name == 'sgd':
            self.optimizer = optim.SGD(self.stacked_params, lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        loss_fn_name = training_config.get('loss_function', 'bce').lower()
        if loss_fn_name == 'bce':
            self.criterion = nn.BCELoss(reduction='none')
        elif loss_fn_name == 'mse':
            self.criterion = nn.MSELoss(reduction='none')
        else:
            raise ValueError(f"Unsupported loss function: {loss_fn_name}")
            
        self.vmapped_forward = vmap(
            lambda p_tuple, b_tuple, x_batch, extract_act_flag: functional_forward_pass_template(
                self.functional_model_callable, p_tuple, b_tuple, x_batch, extract_act_flag
            ),
            in_dims=(0, 0, None, None),
            out_dims=0,
            randomness='different'
        )

    def train(self):
        data_config = self.config['data']
        training_config = self.config['training']

        print(f"\nTraining {self.num_networks} networks on {self.device}")
        print(f"Batch size: {training_config['batch_size']}, Epochs: {training_config['epochs']}")

        # Data generation or loading
        data_source = data_config.get('data_source')
        if data_source is not None:
            print(f"Loading data from: {data_source}")
            X_cpu, y_cpu = load_data_from_file(data_source)
        elif data_config['type'] == 'synthetic' and data_config.get('synthetic_type') == 'torus':
            num_samples = data_config.get('generation', {}).get('n', 1000)
            big_radius = data_config.get('generation', {}).get('big_radius', 3)
            small_radius = data_config.get('generation', {}).get('small_radius', 1)
            X_cpu, y_cpu = generate_torus_data(num_samples, big_radius, small_radius)
        else:
            raise ValueError(f"Unsupported data configuration. Either set data_source or use synthetic data.")
            
        split_ratio = data_config.get('split_ratio', 0.8)
        train_size = int(split_ratio * len(X_cpu))
        X_train_cpu, X_test_cpu = X_cpu[:train_size], X_cpu[train_size:]
        y_train_cpu, y_test_cpu = y_cpu[:train_size], y_cpu[train_size:]

        train_dataset = torch.utils.data.TensorDataset(X_train_cpu, y_train_cpu)
        self.train_loader = torch.utils.data.DataLoader(
                train_dataset, 
                batch_size=training_config['batch_size'],
                shuffle=True,
                pin_memory=(self.device.type != 'cpu'),
                num_workers=training_config.get('num_workers', 2 if self.device.type != 'cpu' else 0),
                persistent_workers=(training_config.get('num_workers', 2) > 0 and self.device.type != 'cpu')
            )
            
        test_dataset = torch.utils.data.TensorDataset(X_test_cpu, y_test_cpu)
        self.test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=training_config['batch_size'],
            shuffle=False,
            pin_memory=(self.device.type != 'cpu'),
            num_workers=training_config.get('num_workers', 2 if self.device.type != 'cpu' else 0),
            persistent_workers=(training_config.get('num_workers', 2) > 0 and self.device.type != 'cpu')
        )

        for epoch in range(training_config['epochs']):
            epoch_loss_sum_device = torch.tensor(0.0, device=self.device)
            epoch_correct_device = torch.tensor(0.0, device=self.device)
            epoch_total_elements_device = torch.tensor(0.0, device=self.device)
            num_batches = 0
            
            for batch_x_cpu, batch_y_cpu in self.train_loader:
                batch_x = batch_x_cpu.to(self.device)
                batch_y = batch_y_cpu.to(self.device)

                # predictions shape: (num_networks, batch_size, output_dim)
                predictions = self.vmapped_forward(
                    self.stacked_params, 
                    self.stacked_buffers, 
                    batch_x, 
                    False 
                )
                
                output_dim = predictions.shape[2]

                # --- Standardize batch_y and Expand ---
                # Ensure batch_y is (batch_size, output_dim) before expanding
                if batch_y.ndim == 1 and output_dim == 1: # batch_y is (B,), output_dim is 1
                    # This happens if y_data is 1D and output_dim is 1.
                    batch_y = batch_y.unsqueeze(-1) # Convert batch_y to (B, 1)
                
                # Now batch_y should be (B, D) or already (N, B, D)
                if batch_y.ndim == 2: 
                    # This implies batch_y is (B, D). Check consistency.
                    if not (batch_y.shape[0] == predictions.shape[1] and batch_y.shape[1] == output_dim):
                         raise ValueError(
                            f"Loaded batch_y shape {batch_y.shape} (after potential unsqueeze) is not "
                            f"({predictions.shape[1]}, {output_dim}) [batch_size, output_dim] "
                            f"as expected from predictions shape {predictions.shape} (N, B, D)."
                        )
                    # Unsqueeze for num_networks and expand
                    batch_y_expanded = batch_y.unsqueeze(0).expand_as(predictions) # (1, B, D) -> (N, B, D)
                elif batch_y.shape == predictions.shape: # Already (N, B, D) from a custom collate perhaps
                     batch_y_expanded = batch_y
                else:
                    raise ValueError(
                        f"Post-standardization, batch_y shape {batch_y.shape} "
                        f"is not 2D (batch_size, output_dim) and does not match predictions shape {predictions.shape} "
                        f"for expansion."
                    )
                
                # --- Prepare Tensors for Operations to avoid MPS broadcast issues ---
                op_predictions = predictions
                op_target = batch_y_expanded
                is_output_dim_one = (output_dim == 1)

                if is_output_dim_one:
                    op_predictions = predictions.squeeze(-1) # Shape: (N, B)
                    op_target = batch_y_expanded.squeeze(-1)   # Shape: (N, B)
                
                # --- Loss Calculation ---
                # loss_all_elements shape: (N, B) if D=1, or (N, B, D) if D>1
                loss_all_elements = self.criterion(op_predictions, op_target) 
                
                if is_output_dim_one:
                    mean_loss_per_model = loss_all_elements.mean(dim=1) # Mean over batch_size. Shape (N,)
                else:
                    mean_loss_per_model = loss_all_elements.mean(dim=[1, 2]) # Mean over BS & Dims. Shape (N,)
                
                total_mean_loss = mean_loss_per_model.mean() # Scalar loss

                # --- Accuracy Calculation ---
                with torch.no_grad():
                    # Use original `predictions` for thresholding, then squeeze if needed for comparison.
                    if self.config['training'].get('loss_function', 'bce').lower() == 'bce':
                        temp_preds_for_binary = predictions # (N, B, D)
                        if is_output_dim_one:
                             temp_preds_for_binary = predictions.squeeze(-1) # (N, B)
                        predictions_binary = (temp_preds_for_binary > 0.5).float()
                    else: # For MSE etc., accuracy logic might differ.
                          # This assumes predictions_binary should match op_target's shape.
                        predictions_binary = op_predictions # Already (N,B) or (N,B,D)
                    
                    # op_target is already (N,B) or (N,B,D) and matches predictions_binary
                    correct_predictions_tensor = (predictions_binary == op_target).float()
                    epoch_correct_device += correct_predictions_tensor.sum() 
                    epoch_total_elements_device += op_target.numel()

                self.optimizer.zero_grad(set_to_none=True)
                total_mean_loss.backward()
                self.optimizer.step()

                epoch_loss_sum_device += total_mean_loss
                num_batches += 1
            
            avg_epoch_loss = (epoch_loss_sum_device / num_batches) if num_batches > 0 else torch.tensor(0.0, device=self.device)
            avg_accuracy = (epoch_correct_device / epoch_total_elements_device) if epoch_total_elements_device > 0 else torch.tensor(0.0, device=self.device)
            
            print(f"Epoch {epoch+1}/{training_config['epochs']} - Loss: {avg_epoch_loss.item():.4f} - Accuracy: {avg_accuracy.item():.4f}")
        
        # Extract layer outputs if enabled
        layer_extraction_config = self.config.get('layer_extraction', {})
        if layer_extraction_config.get('enabled', False):
            print("\nExtracting layer outputs...")
            # Combine train and test datasets into one
            full_dataset = ConcatDataset([self.train_loader.dataset, self.test_loader.dataset])
            full_loader = DataLoader(
                full_dataset,
                batch_size=training_config['batch_size'],
                shuffle=False,
                pin_memory=(self.device.type != 'cpu'),
                num_workers=training_config.get('num_workers', 2 if self.device.type != 'cpu' else 0),
                persistent_workers=(training_config.get('num_workers', 2) > 0 and self.device.type != 'cpu')
            )
            all_models_activations = self.extract_layer_outputs_vectorized(full_loader)
            print(f"torch_vectorized.py: Shape of extracted layer outputs tensor: {all_models_activations.shape}")
            
            # Save layer outputs
            output_dir = Path(layer_extraction_config.get('output_dir', 'results/layer_outputs'))
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / 'torch_vectorized_layer_outputs.pt'
            torch.save({
                'layer_outputs': all_models_activations.cpu(),
                'config': self.config
            }, output_file)
            print(f"Layer outputs saved to: {output_file}")
        else:
            print("Layer extraction disabled. Skipping layer output extraction.")

    def extract_layer_outputs_vectorized(self, data_loader):
        current_params_infer = [p.detach() for p in self.stacked_params]
        current_buffers_infer = [b.detach() for b in self.stacked_buffers] if self.stacked_buffers else []
        
        collected_activations_batched = [] 

        with torch.no_grad():
            for batch_x_cpu, _ in data_loader:
                batch_x = batch_x_cpu.to(self.device)
                
                _, batch_activations_list_device = self.vmapped_forward(
                    current_params_infer, 
                    current_buffers_infer,
                    batch_x, 
                    True
                )

                if not collected_activations_batched: 
                    collected_activations_batched = [[] for _ in range(len(batch_activations_list_device))]

                for i, activation_tensor_for_layer_i in enumerate(batch_activations_list_device):
                    collected_activations_batched[i].append(activation_tensor_for_layer_i)
        
        if not collected_activations_batched or not any(act_list for act_list in collected_activations_batched if act_list): # Check if any sublist has content
            print("Warning: No activations collected.")
            return torch.empty((self.num_networks, 0, 0, 0), device=self.device, dtype=torch.float)

        concatenated_activations_device = []
        for act_type_batches in collected_activations_batched:
            if act_type_batches: 
                concatenated_activations_device.append(torch.cat(act_type_batches, dim=1))
        
        if not concatenated_activations_device:
             print("Warning: Concatenation resulted in no activation tensors.")
             return torch.empty((self.num_networks, 0, 0, 0), device=self.device, dtype=torch.float)

        max_feature_dim = 0
        if any(act_tensor.nelement() > 0 for act_tensor in concatenated_activations_device):
            max_feature_dim = max(
                act_tensor.shape[2] for act_tensor in concatenated_activations_device if act_tensor.nelement() > 0 and act_tensor.ndim == 3
            ) # Ensure tensor is 3D for shape[2]
        
        padded_activations_list_device = []
        for act_tensor in concatenated_activations_device: 
            if act_tensor.nelement() == 0 or act_tensor.ndim != 3: # Basic sanity checks
                # Handle potentially non-3D or empty tensors if logic allows them
                padded_tensor = torch.empty(
                    (act_tensor.shape[0] if act_tensor.ndim > 0 else self.num_networks, # num_networks
                     act_tensor.shape[1] if act_tensor.ndim > 1 else 0,                  # num_datapoints
                     max_feature_dim),                                                  # max_features
                    dtype=act_tensor.dtype if act_tensor.nelement() > 0 else torch.float, 
                    device=act_tensor.device if act_tensor.nelement() > 0 else self.device
                )
                padded_activations_list_device.append(padded_tensor)
                continue

            padding_size = max_feature_dim - act_tensor.shape[2]
            if padding_size > 0:
                padded_tensor = torch.nn.functional.pad(act_tensor, (0, padding_size)) 
            elif padding_size < 0:
                raise ValueError(
                    f"Internal error: Padding size is negative ({padding_size}). "
                    f"act_tensor shape: {act_tensor.shape}, max_feature_dim: {max_feature_dim}"
                )
            else:
                padded_tensor = act_tensor
            padded_activations_list_device.append(padded_tensor)
            
        if not padded_activations_list_device:
            print("Warning: No valid activations to stack after padding.")
            return torch.empty((self.num_networks, 0, 0, 0), device=self.device, dtype=torch.float)

        final_stacked_tensor_temp = torch.stack(padded_activations_list_device, dim=0)
        final_stacked_tensor_device = final_stacked_tensor_temp.permute(1, 0, 2, 3)
        
        return final_stacked_tensor_device

# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train vectorized MLP models using PyTorch.")
    parser.add_argument("config_path", type=str, help="Path to the YAML configuration file.")
    args = parser.parse_args()

    trainer = VectorizedTrainer(args.config_path)
    trainer.train()


