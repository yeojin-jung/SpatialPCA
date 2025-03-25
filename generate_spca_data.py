import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

from scipy.sparse import csr_matrix

from numpy.linalg import norm
from sklearn.metrics.pairwise import rbf_kernel

def align_order(k, K):
    """
    Create an order for reordering vectors with noise.

    Parameters:
    k (int): Index to be set to zero in the order.
    K (int): Total number of elements.

    Returns:
    np.array: An array representing the order for reordering vectors with noise.
    """
    order = np.zeros(K, dtype=int)  # Initialize an array of zeros with length K.
    # Set all indices except k to a random permutation of 1 to K-1.
    order[np.where(np.arange(K) != k)[0]] = np.random.choice(np.arange(1, K), K - 1, replace=False)
    order[k] = 0  # Set the k-th index to 0.
    return order  # Return the order array.


def reorder_with_noise(v, order, K, r):
    """
    Reorder a vector with noise based on a given order.

    Parameters:
    v (np.array): The input vector to be reordered.
    order (np.array): The order in which to reorder the vector.
    K (int): Total number of elements.
    r (float): Probability of applying noise.

    Returns:
    np.array: The reordered vector, possibly with noise.
    """
    u = np.random.rand()  # Generate a random number between 0 and 1.
    if u < r:
        # If u is less than r, apply noise by randomly permuting the order array and using it to reorder v.
        return v[order[np.random.choice(range(K), K, replace=False)]]
    else:
        # Otherwise, sort v in descending order and reorder it according to the order array.
        sorted_row = np.sort(v)[::-1]
        return sorted_row[order]
    

def generate_W_strong(coords_df, n, p, K, r):
    """
    Generate a strong weight matrix W.

    Parameters:
    coords_df (pd.DataFrame): DataFrame containing coordinates and group information for each node.
    N (int): Length of documents.
    n (int): Number of nodes.
    p (int): Number of words.
    K (int): Number of topics.
    r (float): Noise parameter.

    Returns:
    np.array: A strong weight matrix W.
    """
    W = np.zeros((K, n)) 
    
    for k in coords_df["grp"].unique():  # Loop over each unique group.
        for b in coords_df[coords_df["grp"] == k]["grp_blob"].unique():  # Loop over each unique blob within the group.
            alpha = np.random.normal(0.3, 0.5, K) 
            subset_df = coords_df[
                (coords_df["grp"] == k) & (coords_df["grp_blob"] == b)
            ]  # Subset the DataFrame to include only rows with the current group and blob.

            c = subset_df.shape[0]  # Get the number of rows in the subset.
            order = align_order(k, K)  # Generate an order array for reordering with noise.
            weight = reorder_with_noise(alpha, order, K, r)  # Reorder the weights with noise.
            inds = (coords_df["grp"] == k) & (coords_df["grp_blob"] == b)  # Get the indices of nodes in the current subset.
            # Assign the weights to the appropriate columns in W and add some noise.
            W[:, inds] = np.column_stack([weight] * c) + np.abs(
                np.random.normal(scale=0.05, size=c * K).reshape((K, c))
            )

    WTW = W.T.dot(W)
    eigenvalues, eigenvectors = np.linalg.eigh(WTW)
    eigenvalues_clipped = np.clip(eigenvalues, a_min=1e-8, a_max=None)
    WTW_stabilized = eigenvectors.dot(np.diag(np.sqrt(eigenvalues_clipped))).dot(eigenvectors.T)
    normalized_W = W.dot(np.linalg.inv(WTW_stabilized))

    return normalized_W # Return the normalized weight matrix W.


def generate_graph(n, p, K, r):
    """
    Generate a graph with specified properties.

    Parameters:
    N (int): Total number of documents.
    n (int): Number of nodes.
    p (int): Number of words.
    K (int): Number of topics.
    r (float): Noise parameter.

    Returns:
    pd.DataFrame: A DataFrame containing coordinates and group information for each node.
    """
    coords = np.zeros((n, 2)) 
    coords[:, 0] = np.random.uniform(0, 1, n)  # Assign random x-coordinates between 0 and 1.
    coords[:, 1] = np.random.uniform(0, 1, n)  # Assign random y-coordinates between 0 and 1.

    # Assign groups based on equally dividing the x-axis into K segments
    group_edges = np.linspace(0, 1, K+1)  # Create K+1 edges to define K segments
    grps = np.digitize(coords[:, 0], group_edges) - 1  # Assign groups based on x-coordinates

    # Create a DataFrame with coordinates and group assignments
    coords_df = pd.DataFrame(coords, columns=["x", "y"])
    coords_df["grp"] = grps  # Assign each node to a group based on its x-coordinate
    coords_df["grp_blob"] = grps  # Store the original group assignments
    
    return coords_df  # Return the DataFrame


def get_initial_centers(val, centers):
    quantiles = []
    for i in range(centers):
        quantiles.append(i * int(val.shape[0] / centers))
    return quantiles


def generate_graph_kmeans(n, p, K, r, n_clusters=30):
    coords = np.zeros((n, 2))
    coords[:, 0] = np.random.uniform(0, 1, n)
    coords[:, 1] = np.random.uniform(0, 1, n)

    cluster_obj = KMeans(
        n_clusters=n_clusters, init=coords[get_initial_centers(coords, n_clusters), :], n_init=1
    )
    grps = cluster_obj.fit_predict(coords)

    coords_df = pd.DataFrame(coords, columns=["x", "y"])
    coords_df["grp"] = grps % K
    coords_df["grp_blob"] = grps
    return coords_df


def generate_weights_edge(coords_df, nearest_n, phi):
    """
    Generate weights and edges for a graph based on coordinates.

    Parameters:
    coords_df (pd.DataFrame): DataFrame containing coordinates and group information for each node.
    nearest_n (int): Number of nearest neighbors to consider.
    phi (float): Parameter for the RBF kernel.

    Returns:
    tuple: A tuple containing the sparse weight matrix and a DataFrame of edges with weights.
    """
    K = rbf_kernel(coords_df[["x", "y"]], gamma=phi)  # Calculate RBF kernel matrix for coordinates.
    np.fill_diagonal(K, 0)  # Set the diagonal to zero to remove self-loops.
    weights = np.zeros_like(K)  # Initialize an array of zeros with the same shape as K.

    for i in range(K.shape[0]):
        # Get indices of the nearest_n largest values in row i of K.
        top_indices = np.argpartition(K[i], -nearest_n)[-nearest_n:]
        weights[i, top_indices] = K[i, top_indices]  # Assign the corresponding weights.

    weights = (weights + weights.T) / 2  # Symmetrize the weight matrix.
    weights_csr = csr_matrix(weights)  # Convert the weight matrix to a sparse matrix format.

    rows, cols = weights_csr.nonzero()  # Get the non-zero indices of the sparse matrix.
    w = weights[weights.nonzero()]  # Get the corresponding weights.
    # Create a DataFrame containing source, target, and weight columns for the edges.
    edge_df = pd.DataFrame({"src": rows, "tgt": cols, "weight": w})
    return weights_csr, edge_df  # Return the sparse matrix and edge DataFrame.


def generate_data(n, m, p, k, r, n_clusters, alpha):
    coords_df = generate_graph_kmeans(n, p, k, r, n_clusters)
    weights, edge_df = generate_weights_edge(coords_df, k, 0.05)

    # High-dimensional case: p >> n
    U0 = generate_W_strong(coords_df, n, p, k, r) # k x n (normalized)
    U0 = U0.T # n x k
    random_matrix = np.random.randn(n, n-k) # n x n-k
    U1_proj = random_matrix -  U0 @ U0.T @ random_matrix # n x n-k
    U1, _ = np.linalg.qr(U1_proj) # n x n
    U_true = np.hstack((U0, U1)) # n x n
    print(np.allclose(U_true.T @ U_true, np.eye(n), atol=1e-8))

    V0= np.zeros((p, k))
    V00 = np.random.uniform(0, 1, (m, k)) # m x k
    V00_norm, _ = np.linalg.qr(V00) # m x k
    V0[0:m, :] = V00_norm # p x k
    V1 = np.random.uniform(0, 1, (p, p-k)) # p x p-k
    V1_proj = V1 - (V0 @ V0.T @ V1)
    V1_norm, _ = np.linalg.qr(V1_proj) 
    V1_norm.shape
    V_true = np.hstack((V0, V1_norm)) # p x p
    print(np.allclose(V_true.T @ V_true, np.eye(p), atol=1e-8))

    delta = 0.9
    Lambda1 = np.diag([alpha] * k)
    Lambda2 = np.diag([alpha - delta] * (n - k))
    Lambda = np.block([[Lambda1, np.zeros((k, p - k))],
                        [np.zeros((n - k, k)), Lambda2, np.zeros((n - k, p - n))]])

    X = U_true @ Lambda @ V_true.T
    X = X + np.random.normal(0, 0.01, X.shape)

    return X, U_true, Lambda, V_true, coords_df, edge_df, weights


def plot_scatter(coords_df):
    unique_groups = coords_df["grp"].unique()
    cmap = plt.get_cmap("Set3", len(unique_groups))
    colors = [cmap(i) for i in range(len(unique_groups))]

    for group, color in zip(unique_groups, colors):
        grp_data = coords_df[coords_df["grp"] == group]
        plt.scatter(grp_data["x"], grp_data["y"], label=group, color=color)
