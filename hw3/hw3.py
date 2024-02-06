from scipy.linalg import eigh
import matplotlib.pyplot as plt
import numpy as np


def load_and_center_dataset(filename):
    data = np.load(filename, allow_pickle=True)
    mean = np.mean(data, axis=0)
    data = data - mean
    data = data.astype(float)
    return data


def get_covariance(data):
    X = data
    X_T = np.transpose(X)
    numerator = np.dot(X_T, X)
    denominator = X.shape[0] - 1
    S = numerator / denominator
    return S


def get_eig(S, m):
    start_index = S.shape[0] - m
    end_index = S.shape[0] - 1

    eigen_values, eigen_vectors = eigh(S, subset_by_index=[start_index, end_index])
    descending_eigen_values = eigen_values[::-1]
    descending_eigen_vectors = eigen_vectors[:, ::-1]

    eigen_values_matrix = np.diag(descending_eigen_values)
    eigen_vectors_matrix = descending_eigen_vectors
    return eigen_values_matrix, eigen_vectors_matrix


def get_eig_prop(S, prop):
    eigen_values, eigen_vectors = np.linalg.eig(S)

    descending_eigen_values = eigen_values[::-1]
    descending_eigen_vectors = eigen_vectors[:, ::-1]

    total_eigen_values = sum(descending_eigen_values)
    variance_matrix = descending_eigen_values / total_eigen_values

    valid_indices = []
    for idx, variance in enumerate(variance_matrix):
        if variance >= prop:
            valid_indices.append(idx)

    eigen_values_matrix = np.diag(descending_eigen_values[valid_indices][::-1])
    eigen_vectors_matrix = descending_eigen_vectors[:, valid_indices][:, ::-1]
    return eigen_values_matrix, eigen_vectors_matrix


def project_image(image, U):
    return U @ U.T @ image


def display_image(orig, proj):
    orig = orig.reshape((32, 32)).T
    proj = proj.reshape((32, 32)).T

    # Performing transpose of the images because NumPy and Matplotlib
    # handle image data differently.
    # NumPy handles image as (height, width).
    # Matplotlib handles image as (width, height).

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3))

    image_1 = ax1.imshow(orig, cmap="viridis", aspect="equal")
    ax1.set_title("Original")
    cbar1 = fig.colorbar(image_1, ax=ax1)

    image_2 = ax2.imshow(proj, cmap="viridis", aspect="equal")
    ax2.set_title("Projection")
    cbar2 = fig.colorbar(image_2, ax=ax2)

    return fig, ax1, ax2
