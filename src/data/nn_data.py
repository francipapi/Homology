import trimesh as tr
import numpy as np
import gudhi as gd
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from gudhi.dtm_rips_complex import DTMRipsComplex
import trimesh as tr
import numpy as np
import gudhi as gd
import plotly.graph_objects as go

n = 2000 #number of points sampled from each torus

torus1 = tr.creation.torus(1,0.3)
torus2 = tr.creation.torus(1,0.3)
rotation_matrix = tr.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
torus2.apply_transform(rotation_matrix)
translation_matrix = tr.transformations.translation_matrix([1, 0, 0])
torus2.apply_transform(translation_matrix)

torus3 = tr.creation.torus(1,0.3)
torus4 = tr.creation.torus(1,0.3)
rotation_matrix1 = tr.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
torus4.apply_transform(rotation_matrix)
translation_matrix1 = tr.transformations.translation_matrix([1, 0, 0])
torus4.apply_transform(translation_matrix)
torus4.apply_transform(tr.transformations.translation_matrix([0, 3, 0]))
torus3.apply_transform(tr.transformations.translation_matrix([0, 3, 0]))

points1 = np.array(torus1.sample(n))
points2 = np.array(torus2.sample(n))
points3 = np.array(torus3.sample(n))
points4 = np.array(torus4.sample(n))
X = np.concatenate((points1, points2, points3, points4))
y = np.concatenate((np.zeros((n,1)), np.ones((n,1)), np.zeros((n,1)), np.ones((n,1))))

points = gd.sparsify_point_set(X)
rips_complex = DTMRipsComplex(points=points, k=1)
simplex_tree = rips_complex.create_simplex_tree(max_dimension=1)
simplex_tree.collapse_edges()
simplex_tree.expansion(3)
persistence = simplex_tree.persistence()
betti_numbers = simplex_tree.betti_numbers()

# Print Betti numbers
for i, betti in enumerate(betti_numbers):
    print(f"Betti number B{i}: {betti}")