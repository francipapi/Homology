import trimesh as tr
import numpy as np
import gudhi as gd
import plotly.graph_objects as go
from sklearn.cluster import KMeans
n = 1000 #number of points sampled from each torus

torus1 = tr.creation.torus(1,0.3)
torus2 = tr.creation.torus(1,0.3)
rotation_matrix = tr.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
torus2.apply_transform(rotation_matrix)
translation_matrix = tr.transformations.translation_matrix([1, 0, 0])
torus2.apply_transform(translation_matrix)
torus1.apply_transform(tr.transformations.translation_matrix([-3, 3, 3]))
torus2.apply_transform(tr.transformations.translation_matrix([-3, 3, 3]))

torus3 = tr.creation.torus(1,0.3)
torus4 = tr.creation.torus(1,0.3)
rotation_matrix1 = tr.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
torus4.apply_transform(rotation_matrix)
translation_matrix1 = tr.transformations.translation_matrix([1, 0, 0])
torus4.apply_transform(translation_matrix)
torus4.apply_transform(tr.transformations.translation_matrix([-3, -3, 3]))
torus3.apply_transform(tr.transformations.translation_matrix([-3, -3, 3]))

torus5 = tr.creation.torus(1,0.3)
torus6 = tr.creation.torus(1,0.3)
rotation_matrix1 = tr.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
torus6.apply_transform(rotation_matrix)
translation_matrix1 = tr.transformations.translation_matrix([1, 0, 0])
torus6.apply_transform(translation_matrix)
torus5.apply_transform(tr.transformations.translation_matrix([-3, 3, -3]))
torus6.apply_transform(tr.transformations.translation_matrix([-3, 3, -3]))

torus7 = tr.creation.torus(1,0.3)
torus8 = tr.creation.torus(1,0.3)
rotation_matrix1 = tr.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
torus8.apply_transform(rotation_matrix)
translation_matrix1 = tr.transformations.translation_matrix([1, 0, 0])
torus8.apply_transform(translation_matrix)
torus8.apply_transform(tr.transformations.translation_matrix([-3, -3, -3]))
torus7.apply_transform(tr.transformations.translation_matrix([-3, -3, -3]))

torus9 = tr.creation.torus(1,0.3)
torus10 = tr.creation.torus(1,0.3)
rotation_matrix = tr.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
torus10.apply_transform(rotation_matrix)
translation_matrix = tr.transformations.translation_matrix([1, 0, 0])
torus10.apply_transform(translation_matrix)
torus9.apply_transform(tr.transformations.translation_matrix([3, 3, 3]))
torus10.apply_transform(tr.transformations.translation_matrix([3, 3, 3]))

torus11 = tr.creation.torus(1,0.3)
torus12 = tr.creation.torus(1,0.3)
rotation_matrix1 = tr.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
torus12.apply_transform(rotation_matrix)
translation_matrix1 = tr.transformations.translation_matrix([1, 0, 0])
torus12.apply_transform(translation_matrix)
torus12.apply_transform(tr.transformations.translation_matrix([3, -3, 3]))
torus11.apply_transform(tr.transformations.translation_matrix([3, -3, 3]))

torus13 = tr.creation.torus(1,0.3)
torus14 = tr.creation.torus(1,0.3)
rotation_matrix1 = tr.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
torus14.apply_transform(rotation_matrix)
translation_matrix1 = tr.transformations.translation_matrix([1, 0, 0])
torus14.apply_transform(translation_matrix)
torus13.apply_transform(tr.transformations.translation_matrix([3, 3, -3]))
torus14.apply_transform(tr.transformations.translation_matrix([3, 3, -3]))

torus15 = tr.creation.torus(1,0.3)
torus16 = tr.creation.torus(1,0.3)
rotation_matrix1 = tr.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
torus16.apply_transform(rotation_matrix)
translation_matrix1 = tr.transformations.translation_matrix([1, 0, 0])
torus16.apply_transform(translation_matrix)
torus16.apply_transform(tr.transformations.translation_matrix([3, -3, -3]))
torus15.apply_transform(tr.transformations.translation_matrix([3, -3, -3]))

points1 = np.array(torus1.sample(n))
points2 = np.array(torus2.sample(n))
points3 = np.array(torus3.sample(n))
points4 = np.array(torus4.sample(n))
points5 = np.array(torus5.sample(n))
points6 = np.array(torus6.sample(n))
points7 = np.array(torus7.sample(n))
points8 = np.array(torus8.sample(n))
points9 = np.array(torus9.sample(n))
points10 = np.array(torus10.sample(n))
points11 = np.array(torus11.sample(n))
points12 = np.array(torus12.sample(n))
points13 = np.array(torus13.sample(n))
points14 = np.array(torus14.sample(n))
points15 = np.array(torus15.sample(n))
points16 = np.array(torus16.sample(n))
X = np.concatenate((points1, points2, points3, points4, points5, points6, points7, points8, points9, points10, points11, points12, points13, points14, points15, points16))
y = np.concatenate((np.zeros((n,1)), np.ones((n,1)), np.zeros((n,1)), np.ones((n,1)), np.zeros((n,1)), np.ones((n,1)), np.zeros((n,1)), np.ones((n,1)), np.zeros((n,1)), np.ones((n,1)), np.zeros((n,1)), np.ones((n,1)), np.zeros((n,1)), np.ones((n,1)), np.zeros((n,1)), np.ones((n,1))))

import numpy as np 
import networkx as nx
from sklearn.neighbors import kneighbors_graph
def distance_matrix(points, k):
    knn_graph = kneighbors_graph(points, n_neighbors=k, mode='connectivity', include_self=False)

    # Create a NetworkX graph from the sparse matrix
    graph = nx.Graph(knn_graph)


    for u, v in graph.edges:
        graph[u][v]['weight'] = 1
    
    num_points = len(points)
    distance_matrix = np.full((num_points, num_points), float('inf'))
    for i, paths in nx.all_pairs_shortest_path_length(graph):
        for j, dist in paths.items():
            distance_matrix[i, j] = dist

    return distance_matrix

for k in range(5,45,5):
  for d in np.linspace(2,8,7):
    rips_complex = gd.RipsComplex(distance_matrix=distance_matrix(X,k), max_edge_length=d)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=1)
    simplex_tree.collapse_edges()
    simplex_tree.expansion(3)
    persistence = simplex_tree.persistence()
    betti_numbers = simplex_tree.betti_numbers()

    # Print Betti numbers
    print(k,",",d)
    for i, betti in enumerate(betti_numbers):
        print(f"Betti number B{i}: {betti}")