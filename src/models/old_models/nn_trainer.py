import trimesh as tr
import numpy as np
import gudhi as gd
import plotly.graph_objects as go
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


n = 4000 #number of points sampled from each torus

torus1 = tr.creation.torus(3,1)
torus2 = tr.creation.torus(3,1)
rotation_matrix = tr.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
torus2.apply_transform(rotation_matrix)
translation_matrix = tr.transformations.translation_matrix([3, 0, 0])
torus2.apply_transform(translation_matrix)

points1 = np.array(torus1.sample(n))
points2 = np.array(torus2.sample(n))
X = np.concatenate((points1, points2))
y = np.concatenate((np.zeros((n,1)), np.ones((n,1))))

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=52, shuffle=True)

y_train = y_train.astype(np.float32)
y_test = y_test.astype(np.float32)

model = Sequential([
    Input(shape=(3,)),
    Dense(5, activation='relu'),
    Dense(5, activation='relu'),
    Dense(5, activation='relu'),
    Dense(5, activation='relu'),
    Dense(5, activation='relu'),
    Dense(5, activation='relu'),
    Dense(5, activation='relu'),
    Dense(1, activation='sigmoid')])

model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

def getLayerOut(m,i):
  output = []
  x = i
  for layer in model.layers:
    x = layer(x)
    output.append(x)
  return output

layer_output = getLayerOut(model,X)
for i, o in enumerate(layer_output):
  print(f"Layer {i}: {o.shape}")

# Define the number of points you want to keep (e.g., 500)
n_points = 2000
kmeans = KMeans(n_clusters=n_points)
kmeans.fit(layer_output[0])
downsampled_points = kmeans.cluster_centers_

rips_complex = gd.RipsComplex(points=downsampled_points, max_edge_length=0.75)
simplex_tree = rips_complex.create_simplex_tree(max_dimension=3)
persistence = simplex_tree.persistence()
betti_numbers = simplex_tree.betti_numbers()

# Print Betti numbers
for i, betti in enumerate(betti_numbers):
    print(f"Betti number B{i}: {betti}")