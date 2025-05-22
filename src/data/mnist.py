import numpy as np
import struct
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from skimage.transform import resize  # Ensure scikit-image is installed

def load_idx(filename):
    """
    Loads an IDX file and returns it as a NumPy array.
    
    Parameters:
    - filename: Path to the IDX file.
    
    Returns:
    - Numpy array containing the data.
    """
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for _ in range(dims))
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)
    return data

# Paths to the IDX files
train_images_path = 'train-images.idx3-ubyte'
train_labels_path = 'train-labels.idx1-ubyte'
test_images_path = 't10k-images.idx3-ubyte'
test_labels_path = 't10k-labels.idx1-ubyte'

# Load data
print("Loading MNIST data...")
X_train = load_idx(train_images_path)
y_train = load_idx(train_labels_path)
X_test = load_idx(test_images_path)
y_test = load_idx(test_labels_path)
print("Data loaded successfully.")

# Preprocess data
print("Preprocessing data...")
X_train_flat = X_train.reshape(X_train.shape[0], -1).astype('float32') / 255.0  # Shape: (60000, 784)
X_test_flat = X_test.reshape(X_test.shape[0], -1).astype('float32') / 255.0    # Shape: (10000, 784)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_flat)  # Shape: (60000, 784)
X_test_scaled = scaler.transform(X_test_flat)        # Shape: (10000, 784)
print("Preprocessing completed.")

# Apply PCA
print("Applying PCA...")
pca = PCA(n_components=50)
X_train_pca = pca.fit_transform(X_train_scaled)  # Shape: (60000, 50)
X_test_pca = pca.transform(X_test_scaled)        # Shape: (10000, 50)
print(f"PCA completed. Explained variance by 50 components: {np.sum(pca.explained_variance_ratio_):.2%}")

# Function to reconstruct and display an image
def reconstruct_and_display(pca, scaler, pca_data, original_scaled_data, low_res=True, target_resolution=(14, 14), sample_index=0):
    """
    Reconstructs an image from its PCA representation and displays it alongside the original image.
    
    Parameters:
    - pca: Trained PCA model.
    - scaler: Trained scaler.
    - pca_data: PCA-transformed data (e.g., X_test_pca).
    - original_scaled_data: Original scaled data before PCA (e.g., X_test_scaled).
    - low_res: If True, downsample the image to the target resolution.
    - target_resolution: Tuple indicating the desired resolution (height, width).
    - sample_index: Index of the image to reconstruct and display.
    """
    # Select the PCA-transformed vector and reshape to 2D
    pca_vector = pca_data[sample_index].reshape(1, -1)  # Shape: (1, 50)
    
    # Reconstruct the image from PCA vector (returns scaled data)
    reconstructed_scaled = pca.inverse_transform(pca_vector)  # Shape: (1, 784)
    
    # Inverse scale the data to [0, 1]
    reconstructed = scaler.inverse_transform(reconstructed_scaled)  # Shape: (1, 784)
    
    # Reshape to 28x28
    reconstructed_image_28 = reconstructed.reshape(28, 28)
    
    if low_res:
        # Downsample the image to the target resolution (14x14)
        low_res_image = resize(
            reconstructed_image_28,
            target_resolution,
            anti_aliasing=True
        )
    else:
        low_res_image = reconstructed_image_28
    
    # Retrieve the original image from the scaled data
    original_reconstructed_scaled = original_scaled_data[sample_index].reshape(1, -1)  # Shape: (1, 784)
    original_reconstructed = scaler.inverse_transform(original_reconstructed_scaled).reshape(28, 28)
    
    # Plotting
    plt.figure(figsize=(15, 5))
    
    # Original Image
    plt.subplot(1, 3, 1)
    plt.imshow(original_reconstructed, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    # Reconstructed Image (28x28)
    plt.subplot(1, 3, 2)
    plt.imshow(reconstructed_image_28, cmap='gray')
    plt.title('Reconstructed Image (28x28)')
    plt.axis('off')
    
    # Low-Resolution Reconstructed Image (14x14)
    plt.subplot(1, 3, 3)
    plt.imshow(low_res_image, cmap='gray')
    plt.title(f'Low-Resolution (14x14)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Example: Reconstruct and display the first test image
print("Reconstructing and displaying a low-resolution image...")
reconstruct_and_display(
    pca=pca,
    scaler=scaler,
    pca_data=X_test_pca,
    original_scaled_data=X_test_scaled,
    low_res=True,
    target_resolution=(14, 14),
    sample_index=0
)