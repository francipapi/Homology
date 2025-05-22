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
from multiprocessing import Pool



def eval_fn(model, X, y):
        return mx.mean(mx.argmax(model(X), axis=1) == y)

def create_layers(depth, width, input, output):
    layers = []
    size = input
    for _ in range(depth):
        layers.append(nn.Linear(size, width))
        layers.append(nn.ReLU())
        size = width
    layers.append(nn.Linear(width, 1))
    #layers.append(nn.Sigmoid())
    return layers

def loss_fn(model, X, y):
    return nn.losses.binary_cross_entropy(model(X), y, reduction="mean", with_logits=True)

def batch_iterate(batch_size, X, y):
    perm = mx.array(np.random.permutation(y.size))
    for s in range(0, y.size, batch_size):
        ids = perm[s : s + batch_size]
        yield X[ids], y[ids]

def train(param):
    mx.set_default_device(mx.cpu)
    X, y = param
    epochs=150
    batch_size = 32
    verbose = True
    learning_rate = 0.001

    X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=51, shuffle=True
        )

    scaler = StandardScaler()
    train_images = scaler.fit_transform(X_train)
    test_images = scaler.transform(X_test)

    # Convert NumPy arrays to MLX arrays
    X_train = mx.array(X_train)
    y_train = mx.array(y_train.astype(np.int32))
    X_test = mx.array(X_test)
    y_test = mx.array(y_test.astype(np.int32))

    model = nn.Sequential(*create_layers(10, 40, 3, 1))
    mx.eval(model.parameters())

    optimizer = optim.Adam(learning_rate=learning_rate, betas=[0.9, 0.999], bias_correction=True)
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    # Training Loop

    for e in range(epochs): 
        tic = time.perf_counter()
        for X_batch, y_batch in batch_iterate(batch_size, X_train, y_train):
            loss, grad = loss_and_grad_fn(model, X_batch, y_batch)
            optimizer.update(model, grad)
            mx.eval(model.state)
        toc = time.perf_counter()
        if e%10 == 0:
            accuracy = np.sum((np.array(model(X_test).flatten()) >= 0.5).astype(int) == np.array(y_test.flatten()))/np.array(y_test).shape[0]
            print(
                f"Epoch {e}: Test Loss {loss_fn(model, X_test, y_test).item():.5f},"
                f" Time {toc - tic:.3f} (s),"
                f" Val accuracy {accuracy:.5f}"
            )
        if accuracy == 1:
            break

    if accuracy <= 0.999:
        return None
    
    partial = []
    x = mx.array(X)
    for l in model.layers[:-1]:
        x = l(x)
        if isinstance(l, nn.ReLU):
            partial.append(x)
    return np.array(partial)

def main():
    X, y = data.generate(4000,3,1)
    train_data = []
    for _ in range(10):
        train_data.append([X,y])


    with Pool(processes=12) as pool:  # Outer multiprocessing pool
        rain_results = pool.map(train, train_data)
    
if __name__ == "__main__":
    main()
