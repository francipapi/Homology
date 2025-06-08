import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import argparse
import multiprocessing as mp
import numpy as np
from pathlib import Path
import sys
import os
import time
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle
import tempfile

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
        project_root = Path(__file__).resolve().parent.parent.parent
        if str(project_root) not in sys.path:
            sys.path.append(str(project_root))
            print(f"Falling back to project root (original logic): {project_root}")
    from src.models.torch_mlp import MLP, generate_torus_data, load_data_from_file
    print("Successfully imported MLP, generate_torus_data, and load_data_from_file.")
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print(f"Searched for 'src' directory upwards from {current_file_path if '__file__' in locals() else 'current script directory'}." )
    print("Please ensure your project structure is correct or adjust PYTHONPATH.")
    if 'src.models.torch_mlp' not in sys.modules:
        raise ImportError(
            "Could not import MLP and generate_torus_data from src.models.torch_mlp. "
            "Check 'src' directory in Python path and 'models/torch_mlp.py'."
        )
except NameError:
    print("Warning: __file__ not defined. Relative imports might fail. Ensure 'src' is in PYTHONPATH.")
    from src.models.torch_mlp import MLP, generate_torus_data, load_data_from_file


def train_single_network(args):
    """
    Train a single network in a separate process.
    """
    network_id, config, data_lists, random_seed = args

    # Reconstruct tensors directly from Python lists to avoid numpy pickling issues
    device = torch.device('cpu')  # ensure device is defined before tensor construction
    X_train = torch.tensor(data_lists['X_train'], dtype=torch.float32, device=device)
    y_train = torch.tensor(data_lists['y_train'], dtype=torch.float32, device=device)
    X_test  = torch.tensor(data_lists['X_test'],  dtype=torch.float32, device=device)
    y_test  = torch.tensor(data_lists['y_test'],  dtype=torch.float32, device=device)

    # Set random seed for reproducibility
    torch.manual_seed(random_seed + network_id)
    np.random.seed(random_seed + network_id)

    # Extract configuration
    model_config    = config['model']
    training_config = config['training']

    # Create model
    model = MLP(
        input_dim          = model_config['input_dim'],
        num_hidden_layers  = model_config['num_hidden_layers'],
        hidden_dim         = model_config['hidden_dim'],
        output_dim         = model_config['output_dim'],
        activation_fn_name = model_config.get('activation_fn_name', 'relu'),
        dropout_rate       = model_config.get('dropout_rate', 0.0),
        use_batch_norm     = model_config.get('use_batch_norm', False)
    ).to(device)

    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    train_loader  = DataLoader(train_dataset,
                               batch_size=training_config['batch_size'],
                               shuffle=True,
                               num_workers=0)

    test_dataset  = TensorDataset(X_test, y_test)
    test_loader   = DataLoader(test_dataset,
                               batch_size=training_config['batch_size'],
                               shuffle=False,
                               num_workers=0)

    # Setup optimizer
    lr = training_config['learning_rate']
    opt_cfg = training_config.get('optimizer', {'name': 'adam'})
    name   = opt_cfg.get('name', 'adam').lower()
    wd     = opt_cfg.get('weight_decay', 0.0)
    if name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif name == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr,
                                weight_decay=wd if wd>0 else 0.01)
    elif name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=wd)
    else:
        raise ValueError(f"Unsupported optimizer: {name}")

    # Setup scheduler
    scheduler_cfg = training_config.get('lr_scheduler', {})
    scheduler = None
    if scheduler_cfg.get('type') == 'reduce_on_plateau':
        kwargs = {k: scheduler_cfg[k] for k in ['factor','patience','min_lr'] if k in scheduler_cfg}
        try:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                            verbose=scheduler_cfg.get('verbose', False),
                                                            **kwargs)
        except TypeError:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)
    elif scheduler_cfg.get('type') == 'step_lr':
        scheduler = optim.lr_scheduler.StepLR(optimizer,
                                              step_size=scheduler_cfg.get('step_size', 30),
                                              gamma=scheduler_cfg.get('gamma', 0.1))

    # Setup loss function
    loss_name = training_config.get('loss_fn', 'bce').lower()
    if loss_name == 'bce':
        criterion = nn.BCELoss()
    elif loss_name == 'mse':
        criterion = nn.MSELoss()
    elif loss_name == 'cross_entropy':
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported loss function: {loss_name}")

    # Gradient clipping and early stopping configs
    grad_clipping_cfg = training_config.get('gradient_clipping', {})
    use_gc   = grad_clipping_cfg.get('enabled', False)
    max_norm = grad_clipping_cfg.get('max_norm', 1.0)

    early_stop_cfg = training_config.get('early_stopping', {})
    use_es    = early_stop_cfg.get('enabled', False)
    es_pat    = early_stop_cfg.get('patience', 10)
    min_delta = early_stop_cfg.get('min_delta', 0.0)
    best_loss = float('inf')
    patience  = 0

    # Training loop
    start_time = time.time()
    final_loss = 0.0
    final_acc  = 0.0
    for epoch in range(training_config['epochs']):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            if use_gc:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        # Evaluation
        model.eval()
        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for bx, by in test_loader:
                bx, by = bx.to(device), by.to(device)
                out = model(bx)
                l = criterion(out, by)
                total_loss += l.item()
                if loss_name == 'bce':
                    preds = (out > 0.5).float()
                    total   += by.size(0)
                    correct += (preds == by).sum().item()
        avg_loss = total_loss / len(test_loader) if test_loader else 0
        acc      = correct / total if total else 0

        if scheduler:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_loss)
            else:
                scheduler.step()

        if use_es:
            if avg_loss < best_loss - min_delta:
                best_loss = avg_loss
                patience  = 0
            else:
                patience += 1
                if patience >= es_pat:
                    break

        final_loss = avg_loss
        final_acc  = acc

    training_time = time.time() - start_time

    # Save model if threshold met
    model_saved = False
    save_thresh = training_config.get('save_model_threshold', 0.0)
    if final_acc >= save_thresh:
        temp_dir = Path(tempfile.gettempdir()) / 'torch_parallel_models'
        temp_dir.mkdir(exist_ok=True)
        model_path = temp_dir / f'network_{network_id}_acc_{final_acc:.4f}.pth'
        torch.save(model.state_dict(), model_path)
        model_saved = True

    # Extract layer outputs
    layer_outputs = None
    le_cfg = config.get('layer_extraction', {})
    if le_cfg.get('enabled', False):
        full_dataset = ConcatDataset([train_dataset, test_dataset])
        full_loader  = DataLoader(full_dataset,
                                   batch_size=training_config['batch_size'],
                                   shuffle=False,
                                   num_workers=0)
        lo_tensor = model.extract_layer_outputs(full_loader, device)
        layer_outputs = lo_tensor.cpu().numpy()

    return network_id, final_loss, final_acc, layer_outputs, training_time, model_saved


class ParallelTrainer:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config         = yaml.safe_load(f)
        self.model_config    = self.config['model']
        self.training_config = self.config['training']
        self.data_config     = self.config['data']
        self.num_networks    = self.training_config.get('num_networks', 1)
        self.max_workers     = self.training_config.get('max_parallel_workers')
        if self.max_workers is None:
            self.max_workers = min(mp.cpu_count(), self.num_networks)
        print(f"Initializing ParallelTrainer with {self.num_networks} networks")
        print(f"Using {self.max_workers} parallel workers")

    def prepare_data(self):
        print("Preparing data...")
        ds = self.data_config
        if ds.get('data_source'):
            X, y = load_data_from_file(ds['data_source'])
        elif ds['type'] == 'synthetic' and ds.get('synthetic_type') == 'torus':
            gen = ds.get('generation', {})
            X, y = generate_torus_data(gen.get('n',1000),
                                       gen.get('big_radius',3),
                                       gen.get('small_radius',1))
        else:
            raise ValueError("Unsupported data configuration. Either set data_source or use synthetic data.")
        if ds.get('shuffle_data', True):
            seed = ds.get('random_seed_data', 42)
            torch.manual_seed(seed)
            perm = torch.randperm(len(X))
            X, y = X[perm], y[perm]
        split_ratio = ds.get('split_ratio', 0.8)
        train_size   = int(split_ratio * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Convert to Python lists for safe cross-process transfer
        self.data_lists = {
            'X_train': X_train.numpy().tolist(),
            'y_train': y_train.numpy().tolist(),
            'X_test':  X_test.numpy().tolist(),
            'y_test':  y_test.numpy().tolist()
        }
        print(f"Data prepared: {len(X_train)} train, {len(X_test)} test samples")

    def train(self):
        self.prepare_data()
        print(f"\nStarting parallel training of {self.num_networks} networks...")
        seed = self.training_config.get('seed', 42)
        args = [(i, self.config, self.data_lists, seed) for i in range(self.num_networks)]

        start_time = time.time()
        results = []
        # Ensure 'spawn' for macOS compatibility
        if mp.get_start_method(allow_none=True) != 'spawn':
            mp.set_start_method('spawn', force=True)
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_id = {executor.submit(train_single_network, arg): arg[0] for arg in args}
            for future in as_completed(future_to_id):
                nid = future_to_id[future]
                try:
                    res = future.result()
                    results.append(res)
                    _, loss, acc, _, ttime, saved = res
                    saved_str = ' [SAVED]' if saved else ''
                    print(f"Network {nid:3d} completed - Loss: {loss:.4f}, Acc: {acc:.4f}, Time: {ttime:.2f}s{saved_str}")
                except Exception as exc:
                    print(f"Network {nid} generated an exception: {exc}")

        total_time = time.time() - start_time
        results.sort(key=lambda x: x[0])
        losses = [r[1] for r in results]
        accs   = [r[2] for r in results]
        times  = [r[4] for r in results]
        saved_count = sum(1 for r in results if r[5])

        print(f"\nTraining completed in {total_time:.2f}s")
        print(f"Average loss: {np.mean(losses):.4f} ± {np.std(losses):.4f}")
        print(f"Average accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
        print(f"Average training time per network: {np.mean(times):.2f}s")
        print(f"Parallel efficiency: {sum(times)/total_time:.1f}x")
        print(f"Models saved: {saved_count}/{self.num_networks}")

        if self.config.get('layer_extraction', {}).get('enabled', False):
            self.save_layer_outputs(results)
        return results

    def save_layer_outputs(self, results):
        print("\nSaving layer outputs...")
        outs = [r[3] for r in results if r[3] is not None]
        if not outs:
            print("No layer outputs to save.")
            return
        all_layer_outputs = np.stack(outs, axis=0)
        output_dir = Path(self.config['layer_extraction'].get('output_dir', 'results/layer_outputs'))
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / 'torch_parallel_cpu_layer_outputs.pt'
        torch.save({
            'layer_outputs': torch.from_numpy(all_layer_outputs),
            'config': self.config,
            'num_networks': self.num_networks
        }, output_file)
        print(f"Layer outputs saved to: {output_file}")
        print(f"Shape: {all_layer_outputs.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train multiple MLP models in parallel using multiprocessing.")
    parser.add_argument("config_path", type=str, help="Path to the YAML configuration file.")
    args = parser.parse_args()
    trainer = ParallelTrainer(args.config_path)
    trainer.train()
