import argparse
import time
from functools import partial

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.data import dataset
import mnist

class MLP(nn.Module):
    """A simple MLP."""

    def __init__(
        self, num_layers: int, input_dim: int, hidden_dim: int, output_dim: int
    ):
        super().__init__()
        layer_sizes = [input_dim] + [hidden_dim] * num_layers + [output_dim]
        self.layers = [
            nn.Linear(idim, odim)
            for idim, odim in zip(layer_sizes[:-1], layer_sizes[1:])
        ]

    def __call__(self, x):
        for l in self.layers[:-1]:
            x = nn.relu(l(x))
        return nn.sigmoid(self.layers[-1](x))
    
    def predict(self, sample):
        sample = mx.array(sample)
        logits = self.__call__(sample)
        return nn.softmax(logits, axis=1)
    
    def partials(self, x):
        x = mx.array(x)
        partial = []
        for l in self.layers[:-1]:
            x = nn.relu(l(x))
            partial.append(x)
        return np.array(partial)



def loss_fn(model, X, y):
    return nn.losses.binary_cross_entropy(model(X), y, reduction="mean", with_logits=False)


def batch_iterate(batch_size, X, y):
    perm = mx.array(np.random.permutation(y.size))
    for s in range(0, y.size, batch_size):
        ids = perm[s : s + batch_size]
        yield X[ids], y[ids]

def decay_learning_rate(initial_value, decay_coeff, epoch):
    return initial_value/(decay_coeff*(epoch+1)) 

def train(param): 
    verbose = True
    X_gen = param[0]
    y_gen = param[1]
    seed = 0
    num_layers = 10
    hidden_dim = 20
    num_classes = 1
    batch_size = 32
    num_epochs = 3
    learning_rate = 0.0001

    mx.set_default_device(mx.cpu)

    np.random.seed(seed)

    train_images, test_images, train_labels, test_labels = train_test_split(
        X_gen, y_gen, test_size=0.2, random_state=51, shuffle=True
    )

    scaler = StandardScaler()
    train_images = scaler.fit_transform(train_images)
    test_images = scaler.transform(test_images)

    # Convert NumPy arrays to MLX arrays
    train_images = mx.array(train_images)
    train_labels = mx.array(train_labels.astype(np.int32))
    test_images = mx.array(test_images)
    test_labels = mx.array(test_labels.astype(np.int32))

    # Load the model
    model = MLP(num_layers, train_images.shape[-1], hidden_dim, num_classes)
    mx.eval(model.parameters())

    optimizer = optim.Adam(learning_rate=learning_rate)
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)


    @partial(mx.compile, inputs=model.state)
    def eval_fn(X, y):
        return mx.mean(mx.argmax(model(X), axis=1) == y)

    for e in range(num_epochs):
        tic = time.perf_counter()
        for X, y in batch_iterate(batch_size, train_images, train_labels):
            loss, dloss_dw = loss_and_grad_fn(model, X, y)
            optimizer.update(model, dloss_dw)
            print(model.parameters())
            mx.eval(model.parameters(), optimizer.state)
        accuracy = eval_fn(test_images, test_labels)
        toc = time.perf_counter()
        if e % 1 ==0 and verbose:
            print(
                f"Epoch {e}: Test accuracy {accuracy.item():.3f},"
                f" Time {toc - tic:.3f} (s)"
            )

    # Example new sample (replace with your actual data)
    new_sample = np.array([[0.5, -1.2, 3.3]])  # Shape: (1, 3)

    # Convert to MLX array
    new_sample_mlx = mx.array(new_sample)

    if eval_fn(test_images, test_labels) < 0.999:
        print("Target accuracy not reached")
        return

    return model.partials(X_gen)

def main():

    # Load Gen_easy 
    X_gen, y_gen = data.generate(4000, 3, 1)
    start = time.time()
    print(train([X_gen, y_gen]))
    end = time.time()
    print(end-start)

if __name__ == "__main__":
    main()