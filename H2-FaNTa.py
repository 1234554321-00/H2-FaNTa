

import os
import pandas as pd
import networkx as nx
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from torch import nn
from scipy import sparse
import warnings
import torch
import torch.nn as nn
import numpy as np
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import traceback
import pandas as pd
from collections import defaultdict
from sklearn.metrics import classification_report
import torch.optim as optim
import torch
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import time
from scipy import stats



def process_data(folder_path):
    # Read the Excel file with all three columns
    df_genres = pd.read_excel(
        os.path.join(folder_path, 'movie_genres.xlsx'),
        usecols=['movieID', 'genreID', 'Labels']
    )

    # Create mappings
    genre_mapping = defaultdict(list)
    genre_id_mapping = {}

    # Create a mapping of unique genres to integer labels
    unique_genres = df_genres['genreID'].unique()
    genre_id_mapping = {genre: idx for idx, genre in enumerate(sorted(unique_genres))}

    # Create a mapping of movieID to list of genreIDs
    for _, row in df_genres.iterrows():
        movie_id = row['movieID']
        genre = row['genreID']
        genre_mapping[movie_id].append(genre_id_mapping[genre])

    # Convert the genre lists to a format suitable for training
    processed_genres = [(movie_id, genres[0]) for movie_id, genres in genre_mapping.items()]

    # Create the final DataFrame with sorted movieIDs
    ground_truth_ratings = pd.DataFrame(processed_genres, columns=['movieID', 'genreID'])
    ground_truth_ratings = ground_truth_ratings.sort_values('movieID').reset_index(drop=True)
    
    return ground_truth_ratings, genre_id_mapping

def create_heterogeneous_graph(folder_path):
    # Create an empty graph
    G = nx.Graph()
    # Create dictionaries to store the number of nodes for each node type
    node_counts = {'userID': 0, 'movieID': 0, 'directorID': 0, 'actorID': 0, 'genreID': 0}

    # Create a dictionary to store mapping between nodes and their attributes
    node_attributes = {}
    # Create a dictionary to store mapping between edges and their weights
    edge_weights = {}

    # Create dictionaries to store the number of nodes and edges for each type of relationship
    relationship_counts = {}

    # Create a dictionary to map each file to its corresponding columns
    file_columns = {
        'user_movies.xlsx': ['userID', 'movieID', 'rating'],
        'movie_directors.xlsx': ['movieID', 'directorID'],
        'movie_actors.xlsx': ['movieID', 'actorID'],
        'movie_genres.xlsx': ['movieID', 'genreID']
    }

    # Iterate through the files and read them to populate the graph
    for file_name, columns in file_columns.items():
        file_path = os.path.join(folder_path, file_name)
        if os.path.exists(file_path):
            # Read the Excel file into a pandas DataFrame
            df = pd.read_excel(file_path, usecols=columns)

            # Add nodes and edges to the graph based on the file's content
            if 'userID' in columns:
                for _, row in df.iterrows():
                    user_node = f"userID:{row['userID']}"
                    movie_node = f"movieID:{row['movieID']}"
                    rating = row['rating']

                    # Add nodes only if they don't exist
                    if user_node not in G:
                        G.add_node(user_node, type='userID')
                        node_counts['userID'] += 1

                    if movie_node not in G:
                        G.add_node(movie_node, type='movieID')
                        node_counts['movieID'] += 1

                    G.add_edge(user_node, movie_node, weight=rating)

            if 'directorID' in columns:
                for _, row in df.iterrows():
                    movie_node = f"movieID:{row['movieID']}"
                    director_node = f"directorID:{row['directorID']}"

                    # Add nodes only if they don't exist
                    if movie_node not in G:
                        G.add_node(movie_node, type='movieID')
                        node_counts['movieID'] += 1

                    if director_node not in G:
                        G.add_node(director_node, type='directorID')
                        node_counts['directorID'] += 1

                    G.add_edge(movie_node, director_node)

            if 'actorID' in columns:
                for _, row in df.iterrows():
                    movie_node = f"movieID:{row['movieID']}"
                    actor_node = f"actorID:{row['actorID']}"

                    # Add nodes only if they don't exist
                    if movie_node not in G:
                        G.add_node(movie_node, type='movieID')
                        node_counts['movieID'] += 1

                    if actor_node not in G:
                        G.add_node(actor_node, type='actorID')
                        node_counts['actorID'] += 1

                    G.add_edge(movie_node, actor_node)

            if 'genreID' in columns:
                for _, row in df.iterrows():
                    movie_node = f"movieID:{row['movieID']}"
                    genre_node = f"genreID:{row['genreID']}"

                    # Add nodes only if they don't exist
                    if movie_node not in G:
                        G.add_node(movie_node, type='movieID')
                        node_counts['movieID'] += 1

                    if genre_node not in G:
                        G.add_node(genre_node, type='genreID')
                        node_counts['genreID'] += 1

                    G.add_edge(movie_node, genre_node)

    # Print the number of nodes and edges for the graph and the node counts
    print("Graph information:")
    print("Nodes:", len(G.nodes()))
    print("Edges:", len(G.edges()))
    for node_type, count in node_counts.items():
        print(f"Number of {node_type} nodes: {count}")

    return G

#****************************************************************************************
#----------------------------------- Hypergraph and Incidence Matrices Movie-User--------------------------------------------
#****************************************************************************************
def hypergraph_MU(folder_path):

    # Create an empty hypergraph
    hyper_MU = {}
    relationship_counts = {}

    # Create a dictionary to store mapping between nodes and their attributes
    att_MU = {}
    # Create a dictionary to store mapping between edges and their weights
    edge_weights = {}

    # Create a dictionary to map the 'user_movies.xlsx' file to its corresponding columns
    file_columns = {
        'user_movies.xlsx': ['userID', 'movieID', 'rating'],
    }

    # Iterate through the files and read them to populate the hypergraph
    for file_name, columns in file_columns.items():
        file_path = os.path.join(folder_path, file_name)
        if os.path.exists(file_path):
            # Read the Excel file into a pandas DataFrame
            df = pd.read_excel(file_path, usecols=columns)

            # Update the hypergraph and relationship counts based on the file's content
            for _, row in df.iterrows():
                movie_node = f"movieID:{row['movieID']}"
                user_node = f"userID:{str(row['userID'])}"
                rating = row['rating']

                # Add the movie node to the hypergraph if it doesn't exist
                if movie_node not in hyper_MU:
                    hyper_MU[movie_node] = []

                # Add the user node to the hypergraph if it doesn't exist
                if user_node not in hyper_MU:
                    hyper_MU[user_node] = []

                # Add the user node to the movie hyperedge
                hyper_MU[movie_node].append(user_node)

                # Set the type attribute in att_MU
                att_MU[user_node] = {'type': 'userID'}
                att_MU[movie_node] = {'type': 'movieID'}

                edge_weights[(movie_node, user_node)] = rating

                # Count nodes and edges for the userID-movieID relationship
                relationship = 'userID-movieID'
                relationship_counts[relationship] = relationship_counts.get(relationship, {'nodes': 0, 'edges': 0})
                relationship_counts[relationship]['nodes'] += 2  # Two nodes (movie and user)
                relationship_counts[relationship]['edges'] += 1

    # Filter out hyperedges with empty relationships
    hyper_MU = {k: v for k, v in hyper_MU.items() if v}
    
    # Count the number of edges
    num_edges = sum(len(nodes) for nodes in hyper_MU.values())

    print("Hypergraph information of MU:")
    print("Number of hyperedges of MU (nodes):", len(hyper_MU))
    print("Number of edges of MU:", num_edges)

    return hyper_MU, att_MU


#****************************************************************************************
#----------------------------------- Hypergraph and Incidence Matrices Movie-Director--------------------------------------------
#****************************************************************************************
def hypergraph_MD(folder_path):
 
    # Create an empty hyper_MD
    hyper_MD = {}
    relationship_counts_MD = {}

    # Create a dictionary to store mapping between nodes and their attributes
    att_MD = {}
    
    # Create a dictionary to map the 'director_movies.xlsx' file to its corresponding columns
    file_columns = {
        'movie_directors.xlsx': ['movieID', 'directorID'],
    }

    # Iterate through the files and read them to populate the hyper_MD
    for file_name, columns in file_columns.items():
        file_path = os.path.join(folder_path, file_name)
        if os.path.exists(file_path):
            # Read the Excel file into a pandas DataFrame
            df = pd.read_excel(file_path, usecols=columns)

            # Update the hyper_MD and relationship_counts based on the file's content
            for _, row in df.iterrows():
                movie_node = f"movieID:{row['movieID']}"
                director_node = f"directorID:{str(row['directorID'])}"

                # Add the movie node to the hypergraph if it doesn't exist
                if movie_node not in hyper_MD:
                    hyper_MD[movie_node] = []

                # Add the director node to the hyper_MD if it doesn't exist
                if director_node not in hyper_MD:
                    hyper_MD[director_node] = []

                # Add the director node to the movie hyperedge
                hyper_MD[movie_node].append(director_node)

                # Set the type attribute in att_MD
                att_MD[director_node] = {'type': 'directorID'}
                att_MD[movie_node] = {'type': 'movieID'}

                # Count nodes and edges for the directorID-movieID relationship
                relationship = 'directorID-movieID'
                relationship_counts_MD[relationship] = relationship_counts_MD.get(relationship, {'nodes': 0, 'edges': 0})
                relationship_counts_MD[relationship]['nodes'] += 2  # Two nodes (movie and director)
                relationship_counts_MD[relationship]['edges'] += 1

    # Filter out hyperedges with empty relationships
    hyper_MD = {k: v for k, v in hyper_MD.items() if v}

    # Count the number of edges
    num_edges = sum(len(nodes) for nodes in hyper_MD.values())

    print("Hypergraph information of MD:")
    print("Number of hyperedges of MD (nodes):", len(hyper_MD))
    print("Number of edges of MD:", num_edges)

    return hyper_MD, att_MD

def generate_incidence_matrices_MD(hyper_MD, att_MD):
    """
    Generates incidence matrices for movies and directors.

    Args:
        hyper_MD (dict): Hypergraph representing connections between movies and directors.
        att_MD (dict): Dictionary containing attributes for nodes.

    Returns:
        tuple: A tuple containing the movie-director incidence matrix and its transpose.
    """
    movie_nodes = [node for node in att_MD if att_MD[node]['type'] == 'movieID']
    director_nodes = [node for node in att_MD if att_MD[node]['type'] == 'directorID']

    num_movies = len(movie_nodes)
    num_directors = len(director_nodes)
    incidence_matrix_MD = np.zeros((num_directors, num_movies), dtype=float)  # Swap dimensions

    for movie_index, movie_node in enumerate(movie_nodes):
        directors_connected = hyper_MD.get(movie_node, [])
        for director_node in directors_connected:
            if director_node in director_nodes:
                director_index = director_nodes.index(director_node)
                incidence_matrix_MD[director_index, movie_index] = 1  # Swap indices
    
    print("incidence_matrix_MD Shape", incidence_matrix_MD.shape)
    
    return incidence_matrix_MD

#****************************************************************************************
#----------------------------------- Hypergraph and Incidence Matrices Movie-Actor--------------------------------------------
#****************************************************************************************
def hypergraph_MA(folder_path):
    """
    Generate a hypergraph based on the files found in the specified folder path.

    Args:
    - folder_path (str): Path to the folder containing the files.

    Returns: 
    - hyper_MA (dict): Dictionary representing the hypergraph.
    - att_MA (dict): Dictionary containing attributes of nodes in the hypergraph.
    """
    # Create an empty hyper_MA
    hyper_MA = {}
    relationship_counts_MA = {}

    # Create a dictionary to store mapping between nodes and their attributes
    att_MA = {}
    
    # Create a dictionary to map the 'actor_movies.xlsx' file to its corresponding columns
    file_columns = {
        'movie_actors.xlsx': ['movieID', 'actorID'],
    }

    # Iterate through the files and read them to populate the hyper_MA
    for file_name, columns in file_columns.items():
        file_path = os.path.join(folder_path, file_name)
        if os.path.exists(file_path):
            # Read the Excel file into a pandas DataFrame
            df = pd.read_excel(file_path, usecols=columns)

            # Update the hyper_MA and relationship_counts based on the file's content
            for _, row in df.iterrows():
                movie_node = f"movieID:{row['movieID']}"
                actor_node = f"actorID:{str(row['actorID'])}"

                # Add the movie node to the hypergraph if it doesn't exist
                if movie_node not in hyper_MA:
                    hyper_MA[movie_node] = []

                # Add the actor node to the hyper_MA if it doesn't exist
                if actor_node not in hyper_MA:
                    hyper_MA[actor_node] = []

                # Add the actor node to the movie hyperedge
                hyper_MA[movie_node].append(actor_node)

                # Set the type attribute in att_MA
                att_MA[actor_node] = {'type': 'actorID'}
                att_MA[movie_node] = {'type': 'movieID'}

                # Count nodes and edges for the actorID-movieID relationship
                relationship = 'actorID-movieID'
                relationship_counts_MA[relationship] = relationship_counts_MA.get(relationship, {'nodes': 0, 'edges': 0})
                relationship_counts_MA[relationship]['nodes'] += 2  # Two nodes (movie and actor)
                relationship_counts_MA[relationship]['edges'] += 1

    # Filter out hyperedges with empty relationships
    hyper_MA = {k: v for k, v in hyper_MA.items() if v}

    # Count the number of edges
    num_edges = sum(len(nodes) for nodes in hyper_MA.values())

    print("Hypergraph information of MA:")
    print("Number of hyperedges of MA (nodes):", len(hyper_MA))
    print("Number of edges of MA:", num_edges)

    return hyper_MA, att_MA

def generate_incidence_matrices_MA(hyper_MA, att_MA):

    # Extract movie and actor nodes
    movie_nodes = [node for node in att_MA if att_MA[node]['type'] == 'movieID']
    actor_nodes = [node for node in att_MA if att_MA[node]['type'] == 'actorID']

    # Initialize incidence matrix
    num_movies = len(movie_nodes)
    num_actors = len(actor_nodes)
    incidence_matrix_MA = np.zeros((num_actors, num_movies), dtype=float)

    # Populate the incidence matrix
    for movie_index, movie_node in enumerate(movie_nodes):
        actors_connected = hyper_MA.get(movie_node, [])
        for actor_node in actors_connected:
            if actor_node in actor_nodes:
                actor_index = actor_nodes.index(actor_node)
                incidence_matrix_MA[actor_index, movie_index] = 1  # Adjust based on your hypergraph structure

    print("incidence_matrix_MA Shape", incidence_matrix_MA.shape)
    
    return incidence_matrix_MA

#****************************************************************************************
#----------------------------------- Hypergraph and Incidence Matrices Movie-Genre--------------------------------------------
#****************************************************************************************

def hypergraph_MG(folder_path):

    # Create an empty hyper_MG
    hyper_MG = {}
    relationship_counts_MG = {}

    # Create a dictionary to store mapping between nodes and their attributes
    att_MG = {}
    
    # Create a dictionary to map the 'genre_movies.xlsx' file to its corresponding columns
    file_columns = {
        'movie_genres.xlsx': ['movieID', 'genreID'],
    }

    # Iterate through the files and read them to populate the hyper_MG
    for file_name, columns in file_columns.items():
        file_path = os.path.join(folder_path, file_name)
        if os.path.exists(file_path):
            # Read the Excel file into a pandas DataFrame
            df = pd.read_excel(file_path, usecols=columns)

            # Update the hyper_MG and relationship_counts based on the file's content
            for _, row in df.iterrows():
                movie_node = f"movieID:{row['movieID']}"
                genre_node = f"genreID:{str(row['genreID'])}"

                # Add the movie node to the hypergraph if it doesn't exist
                if movie_node not in hyper_MG:
                    hyper_MG[movie_node] = []

                # Add the genre node to the hyper_MG if it doesn't exist
                if genre_node not in hyper_MG:
                    hyper_MG[genre_node] = []

                # Add the genre node to the movie hyperedge
                hyper_MG[movie_node].append(genre_node)

                # Set the type attribute in att_MG
                att_MG[genre_node] = {'type': 'genreID'}
                att_MG[movie_node] = {'type': 'movieID'}

                # Count nodes and edges for the genreID-movieID relationship
                relationship = 'genreID-movieID'
                relationship_counts_MG[relationship] = relationship_counts_MG.get(relationship, {'nodes': 0, 'edges': 0})
                relationship_counts_MG[relationship]['nodes'] += 2  # Two nodes (movie and genre)
                relationship_counts_MG[relationship]['edges'] += 1

    # Filter out hyperedges with empty relationships
    hyper_MG = {k: v for k, v in hyper_MG.items() if v}

    # Count the number of edges
    num_edges = sum(len(nodes) for nodes in hyper_MG.values())

    print("Hypergraph information of MG:")
    print("Number of hyperedges of MG (nodes):", len(hyper_MG))
    print("Number of edges of MG:", num_edges)

    return hyper_MG, att_MG

def generate_incidence_matrices_MG(hyper_MG, att_MG):
 
    # Extract movie and genre nodes
    movie_nodes = [node for node in att_MG if att_MG[node]['type'] == 'movieID']
    genre_nodes = [node for node in att_MG if att_MG[node]['type'] == 'genreID']

    # Initialize incidence matrix
    num_movies = len(movie_nodes)
    num_genres = len(genre_nodes)
    incidence_matrix_MG = np.zeros((num_genres, num_movies), dtype=float)

    # Populate the incidence matrix
    for movie_index, movie_node in enumerate(movie_nodes):
        genres_connected = hyper_MG.get(movie_node, [])
        for genre_node in genres_connected:
            if genre_node in genre_nodes:
                genre_index = genre_nodes.index(genre_node)
                incidence_matrix_MG[genre_index, movie_index] = 1  # Adjust based on your hypergraph structure

    print("incidence_matrix_MG Shape", incidence_matrix_MG.shape)
  
    return incidence_matrix_MG


#****************************************************************************************
#----------------------------------- Hypergraph Convolutional Embedding --------------------------------------------
#****************************************************************************************

def generate_incidence_matrix(hypergraph, attributes, primary_type='movieID', secondary_type=None, max_primary_nodes=None):
    
    # Extract nodes by type
    primary_nodes = [node for node in attributes if attributes[node]['type'] == primary_type]
    secondary_nodes = [node for node in attributes if attributes[node]['type'] == secondary_type]
    
    # If max_primary_nodes is specified, truncate primary nodes
    if max_primary_nodes is not None:
        primary_nodes = primary_nodes[:max_primary_nodes]
    
    # Initialize matrix
    num_primary = len(primary_nodes)
    num_secondary = len(secondary_nodes)
    incidence_matrix = np.zeros((num_secondary, num_primary), dtype=float)
    
    # Populate matrix
    connection_count = 0
    for primary_idx, primary_node in enumerate(primary_nodes):
        connected_nodes = hypergraph.get(primary_node, [])
        for secondary_node in connected_nodes:
            if secondary_node in secondary_nodes:
                secondary_idx = secondary_nodes.index(secondary_node)
                incidence_matrix[secondary_idx, primary_idx] = 1
                connection_count += 1
    
    return incidence_matrix
#-----------------------------------------------------

def print_statistics(behavior, H, results):
    """
    Print statistical information about the hypergraph processing.
    
    Args:
        behavior (str): Name of the behavior/relation
        H (sparse.csr_matrix): Incidence matrix
        results (dict): Dictionary containing the computed matrices and scores
    """
    print(f"Number of vertices: {H.shape[0]}")
    print(f"Number of hyperedges: {H.shape[1]}")
    print(f"Laplacian matrix shape: {results['laplacian'].shape}")
    print(f"Behavior weight (alpha): {results['behavior_weight']}")
    
    if 'attention_scores' in results:
        attention_scores = results['attention_scores']
        print(f"Attention score statistics:")
        print(f"- Mean: {np.mean(attention_scores):.4f}")
        print(f"- Max: {np.max(attention_scores):.4f}")
        print(f"- Min: {np.min(attention_scores):.4f}")
    
    print(f"\nFirst 5x5 elements of the behavior-specific Laplacian matrix:")
    print(results['laplacian'].toarray()[:5, :5])


def calculate_hypergraph_laplacian(H, use_attention=False, feature_dim=64):
    """
    Calculate the hypergraph Laplacian matrix with optional attention mechanism.
    
    Args:
        H (np.ndarray): Incidence matrix
        use_attention (bool): Whether to use attention mechanism
        feature_dim (int): Dimension for attention features if use_attention is True
    """
    H = sparse.csr_matrix(H)
    
    # Calculate degree matrices
    vertex_degrees = np.array(H.sum(axis=1)).flatten()
    edge_degrees = np.array(H.sum(axis=0)).flatten()
    Dv = sparse.diags(vertex_degrees)
    De = sparse.diags(edge_degrees)
    
    # Calculate weights
    if use_attention:
        W = sparse.diags(calculate_attention_weights(H, feature_dim))
    else:
        W = sparse.eye(H.shape[1])
    
    # Calculate inverses
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Dv_sqrt_inv = sparse.diags(1.0 / np.sqrt(vertex_degrees))
        De_inv = sparse.diags(1.0 / edge_degrees)
    
    # Calculate Laplacian
    I = sparse.eye(H.shape[0])
    L = I - Dv_sqrt_inv @ H @ W @ De_inv @ H.T @ Dv_sqrt_inv
    
    return L, Dv, De, W

def hypergraph_GNN(incidence_matrices, use_attention=False, alpha=None):
    """
    Process incidence matrices with optional attention and behavior weights.
    
    Args:
        incidence_matrices (dict): Dictionary of incidence matrices
        use_attention (bool): Whether to use attention mechanism
        alpha (dict): Optional behavior weights
    """
    if alpha is None:
        alpha = {k: 1.0/len(incidence_matrices) for k in incidence_matrices.keys()}
        
    results = {}
    for behavior, H in incidence_matrices.items():
        print(f"\nProcessing {behavior} behavior...")
        
        try:
            L, Dv, De, W = calculate_hypergraph_laplacian(H, use_attention)
            
            results[behavior] = {
                'laplacian': L,
                'vertex_degree_matrix': Dv,
                'edge_degree_matrix': De,
                'weight_matrix': W,
                'behavior_weight': alpha[behavior]
            }
            
            if use_attention:
                results[behavior]['attention_scores'] = W.diagonal()
                
            print_statistics(behavior, H, results[behavior])
            
        except Exception as e:
            print(f"Error processing {behavior}: {str(e)}")
            continue
    
    return results

def calculate_attention_weights(H, feature_dim=64):
    """
    Calculate attention-based weights for hyperedges using a simple attention mechanism.
    
    Args:
        H (np.ndarray): Incidence matrix
        feature_dim (int): Dimension of the node features
        
    Returns:
        np.ndarray: Attention weights for hyperedges
    """
    # Initialize random node features for demonstration
    # In practice, these should be actual node features
    n_vertices = H.shape[0]
    n_edges = H.shape[1]
    
    # Convert to torch tensors for easier computation
    H_tensor = torch.FloatTensor(H.toarray() if sparse.issparse(H) else H)
    
    # Initialize random node features
    node_features = torch.randn(n_vertices, feature_dim)
    
    # Calculate edge features by aggregating connected node features
    edge_features = torch.mm(H_tensor.t(), node_features)  # [n_edges, feature_dim]
    
    # Calculate attention scores
    attention_weights = F.softmax(torch.sum(edge_features ** 2, dim=1), dim=0)
    
    return attention_weights.detach().numpy()

def calculate_behavior_specific_laplacian(H_k, X_m=None, X_s=None, feature_dim=64):
    """
    Calculate behavior-specific Laplacian with similarity-based attention weights.
    
    Args:
        H_k (np.ndarray): Behavior-specific incidence matrix
        X_m (np.ndarray, optional): Master node features
        X_s (np.ndarray, optional): Slave node features
        feature_dim (int): Feature dimension
    """
    # Convert to sparse matrix
    H_k = sparse.csr_matrix(H_k)
    
    # Calculate degree matrices
    vertex_degrees = np.array(H_k.sum(axis=1)).flatten()
    edge_degrees = np.array(H_k.sum(axis=0)).flatten()
    Dv_k = sparse.diags(vertex_degrees)
    De_k = sparse.diags(edge_degrees)
    
    # Calculate attention-based weights using similarity attention
    attention_weights = calculate_attention_weights(H_k, X_m, X_s, feature_dim)
    W_k = sparse.diags(attention_weights)
    
    # Calculate matrix inverses
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Dv_sqrt_inv = sparse.diags(1.0 / np.sqrt(vertex_degrees))
        De_inv = sparse.diags(1.0 / edge_degrees)
    
    # Calculate the behavior-specific Laplacian
    I = sparse.eye(H_k.shape[0])
    L_k = I - Dv_sqrt_inv @ H_k @ W_k @ De_inv @ H_k.T @ Dv_sqrt_inv
    
    return L_k, Dv_k, De_k, W_k

def multi_behavior_hypergraphGNN(incidence_matrices, feature_dim=64, alpha=None):
    """
    Process multiple behavior-specific incidence matrices with similarity-based attention.
    """
    if alpha is None:
        alpha = {k: 1.0/len(incidence_matrices) for k in incidence_matrices.keys()}
    
    results = {}
    
    # Initialize master and slave features (could be replaced with actual features)
    X_m = torch.randn(max(H.shape[0] for H in incidence_matrices.values()), feature_dim)
    X_s_dict = {
        behavior: torch.randn(H.shape[1], feature_dim) 
        for behavior, H in incidence_matrices.items()
    }
    
    for behavior, H_k in incidence_matrices.items():
        print(f"\nProcessing {behavior} behavior...")
        print(f"Incidence matrix shape: {H_k.shape}")
        
        # Calculate behavior-specific components with similarity attention
        L_k, Dv_k, De_k, W_k = calculate_behavior_specific_laplacian(
            H_k, 
            X_m=X_m, 
            X_s=X_s_dict[behavior],
            feature_dim=feature_dim
        )
        
        results[behavior] = {
            'laplacian': L_k,
            'vertex_degree_matrix': Dv_k,
            'edge_degree_matrix': De_k,
            'weight_matrix': W_k,
            'attention_scores': W_k.diagonal(),
            'behavior_weight': alpha[behavior]
        }
        
        # Print statistics
        print(f"Number of vertices: {H_k.shape[0]}")
        print(f"Number of hyperedges: {H_k.shape[1]}")
        print(f"Behavior weight (alpha): {alpha[behavior]}")
        print(f"Average attention score: {np.mean(W_k.diagonal())}")
    
    return results

class HyperAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.Wq = nn.Linear(dim, dim)
        self.Wk = nn.Linear(dim, dim)
        self.Wv = nn.Linear(dim, dim)
        self.scale = dim ** -0.5
        
    def forward(self, X, H):
        """
        Modified attention mechanism to handle varying dimensions
        Args:
            X: Node features
            H: Incidence matrix
        """
        # Transform features
        Q = self.Wq(X)  # [num_vertices x dim]
        K = self.Wk(X)  # [num_vertices x dim]
        V = self.Wv(X)  # [num_vertices x dim]
        
        # Convert H to tensor if it's sparse
        H_tensor = torch.FloatTensor(H.toarray() if sparse.issparse(H) else H)
        
        # Create edge features by aggregating node features
        edge_features = torch.mm(H_tensor.t(), V)  # [num_edges x dim]
        
        # Calculate attention scores
        attention = torch.mm(Q, edge_features.t()) * self.scale  # [num_vertices x num_edges]
        attention = attention * H_tensor  # Apply hypergraph structure mask
        
        # Normalize attention weights
        attention = F.softmax(attention, dim=1)
        
        return attention

def calculate_semantic_score(edge_embeddings):
    """Calculate semantic coherence score for behavior importance"""
    if edge_embeddings.shape[0] == 0:
        return 0.0
    
    centroid = edge_embeddings.mean(dim=0)
    similarity = F.cosine_similarity(edge_embeddings, centroid.unsqueeze(0))
    return similarity.mean().item()

def calculate_structural_score(H, edge_embeddings, sigma=1.0):
    """Calculate structural score using HyperSim metric"""
    if edge_embeddings.shape[0] == 0:
        return 0.0
    
    # Convert H to dense tensor
    H_tensor = torch.FloatTensor(H.toarray() if sparse.issparse(H) else H)
    
    # Calculate structural similarity
    degrees = H_tensor.sum(dim=0)  # Edge degrees
    intersection = torch.mm(H_tensor.t(), H_tensor)  # Edge-edge intersection
    structural_sim = intersection / (degrees.unsqueeze(0) * degrees.unsqueeze(1)).sqrt()
    
    # Calculate embedding similarity
    dist = torch.cdist(edge_embeddings, edge_embeddings)
    embedding_sim = torch.exp(-dist**2 / (2*sigma**2))
    
    # Combine similarities
    combined_sim = structural_sim * embedding_sim
    return combined_sim.mean().item()

def calculate_hypergraph_laplacian(H, feature_dim=64, hidden_dim=32):
    """
    Calculate behavior-specific Laplacian with attention mechanism.
    """
    H = sparse.csr_matrix(H)
    num_vertices = H.shape[0]
    
    # Initialize vertex features
    X = torch.empty(num_vertices, feature_dim)
    nn.init.xavier_uniform_(X)
    
    # Calculate degree matrices
    vertex_degrees = np.array(H.sum(axis=1)).flatten()
    edge_degrees = np.array(H.sum(axis=0)).flatten()
    Dv = sparse.diags(vertex_degrees)
    De = sparse.diags(edge_degrees)
    
    # Calculate attention-based weights
    attention_module = HyperAttention(feature_dim)
    attention_weights = attention_module(X, H)
    
    # Create edge embeddings by aggregating vertex features
    H_tensor = torch.FloatTensor(H.toarray() if sparse.issparse(H) else H)
    edge_embeddings = torch.mm(H_tensor.t(), X)  # [num_edges x feature_dim]
    
    # Calculate behavior importance
    struct_score = calculate_structural_score(H, edge_embeddings)
    sem_score = calculate_semantic_score(edge_embeddings)
    behavior_weight = 0.5 * struct_score + 0.5 * sem_score
    
    # Convert attention weights to sparse diagonal matrix
    W = sparse.diags(attention_weights.mean(dim=0).detach().numpy())
    
    # Calculate inverses
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Dv_sqrt_inv = sparse.diags(1.0 / np.sqrt(vertex_degrees))
        De_inv = sparse.diags(1.0 / edge_degrees)
    
    # Calculate Laplacian
    I = sparse.eye(H.shape[0])
    L = I - Dv_sqrt_inv @ H @ W @ De_inv @ H.T @ Dv_sqrt_inv
    
    return L, Dv, De, W, X, behavior_weight

def save_embeddings(results, save_path='embeddings'):
    """
    Save embeddings for each behavior in CSV format
    """
    for behavior, data in results.items():
        # Get node features (embeddings)
        embeddings = data['node_features']
        
        # Convert to numpy array
        if torch.is_tensor(embeddings):
            embeddings = embeddings.detach().numpy()
            
        # Create DataFrame
        df = pd.DataFrame(embeddings)
        df.columns = [f'dim_{i}' for i in range(df.shape[1])]
        df.insert(0, 'node_id', range(len(df)))
        df.insert(1, 'behavior', behavior)
        
        # Save to CSV
        filename = f'{save_path}_{behavior}_embeddings.csv'
        df.to_csv(filename, index=False)
        print(f"\nSaved {behavior} embeddings to {filename}")
        print(f"Embedding shape: {embeddings.shape}")
        print("\nFirst few rows of embeddings:")
        print(df.head())
        
        # Print some statistics
        print(f"\nEmbedding statistics for {behavior}:")
        print("Mean:", embeddings.mean(axis=0)[:5], "...")
        print("Std:", embeddings.std(axis=0)[:5], "...")
        print("Min:", embeddings.min(axis=0)[:5], "...")
        print("Max:", embeddings.max(axis=0)[:5], "...")

def hypergraph_GNN(incidence_matrices, feature_dim=64, hidden_dim=32):
    """
    Process multi-behavior hypergraph.
    """
    results = {}
    
    for behavior, H in incidence_matrices.items():
        print(f"\nProcessing {behavior} behavior...")
        
        try:
            # Calculate Laplacian and related matrices
            L, Dv, De, W, X, behavior_weight = calculate_hypergraph_laplacian(
                H, 
                feature_dim=feature_dim,
                hidden_dim=hidden_dim
            )
            
            results[behavior] = {
                'laplacian': L,
                'vertex_degree_matrix': Dv,
                'edge_degree_matrix': De,
                'weight_matrix': W,
                'node_features': X,
                'behavior_weight': behavior_weight
            }
            
            # Print statistics
            print(f"Behavior: {behavior}")
            print(f"Number of vertices: {H.shape[0]}")
            print(f"Number of hyperedges: {H.shape[1]}")
            print(f"Behavior importance score: {behavior_weight:.4f}")
            
        except Exception as e:
            print(f"Error processing {behavior}: {str(e)}")
            continue
    
    # Save and print embeddings
    save_embeddings(results)
    
    return results

# ----------------- Dynamic Pruning-------------------------------------

class DynamicPruningGNN(nn.Module):
    def __init__(self, theta_0=0.1, theta_max=0.9, lambda_param=0.1, tau=1.0, epsilon=1e-6):
        super(DynamicPruningGNN, self).__init__()
        self.theta_0 = theta_0
        self.theta_max = theta_max
        self.lambda_param = lambda_param
        self.tau = tau
        self.epsilon = epsilon
        self.iteration = 0

    def calculate_threshold(self, level='comp'):
        """Calculate iteration-based threshold for different pruning levels"""
        exp_term = 1 - np.exp(-self.lambda_param * self.iteration)
        theta_base = self.theta_0 + exp_term * (self.theta_max - self.theta_0)
        
        # Adjust threshold based on pruning level
        if level == 'edge':
            return min(theta_base, 0.8)  # More conservative for edge pruning
        elif level == 'node':
            return min(theta_base, 0.7)  # Most conservative for node pruning
        return theta_base  # Default for component pruning

    # Component-Level Pruning Methods
    def calculate_component_importance(self, H, X, W):
        """Calculate structural importance of components"""
        # Structural importance
        structural_score = torch.norm(torch.mm(X, W), dim=1)
        
        # Attention-based significance
        attention_module = HyperAttention(X.shape[1])
        attention_scores = attention_module(X, H).mean(dim=0)
        
        # Combine scores
        beta = 0.5  # Trade-off parameter
        importance = beta * structural_score + (1 - beta) * attention_scores
        return F.softmax(importance / self.tau, dim=0)

    def prune_components(self, H, X, W):
        """Perform component-level pruning"""
        component_scores = self.calculate_component_importance(H, X, W)
        theta_comp = self.calculate_threshold('comp')
        
        # Create pruning mask
        keep_mask = component_scores >= theta_comp
        
        # Update features
        X_pruned = X * keep_mask.unsqueeze(1)
        return X_pruned, keep_mask

    # Edge-Level Pruning Methods
    def calculate_edge_importance(self, H, X):
        """Calculate edge importance using attention-weighted similarity"""
        H_dense = H.toarray() if sparse.issparse(H) else H
        H_tensor = torch.FloatTensor(H_dense)
        
        # Calculate cosine similarity between connected nodes
        cos_sim = F.cosine_similarity(X.unsqueeze(1), X.unsqueeze(0), dim=2)
        edge_scores = cos_sim * H_tensor
        
        return F.softmax(edge_scores / self.tau, dim=1)

    def prune_edges(self, H, X, component_mask):
        """Perform edge-level pruning"""
        edge_scores = self.calculate_edge_importance(H, X)
        theta_edge = self.calculate_threshold('edge')
        
        # Only prune edges in retained components
        H_pruned = H.copy()
        mask = (edge_scores >= theta_edge) & component_mask.unsqueeze(1)
        H_pruned[~mask] = 0
        
        return sparse.csr_matrix(H_pruned), mask

    # Node-Level Pruning Methods
    def calculate_node_importance(self, H, X):
        """Calculate node importance based on connectivity and feature magnitude"""
        H_dense = H.toarray() if sparse.issparse(H) else H
        
        # Calculate connectivity strength
        connectivity = H_dense.sum(axis=1)
        
        # Calculate feature magnitude
        feature_magnitude = torch.norm(X, dim=1)
        
        # Combine scores
        importance = connectivity * feature_magnitude
        return F.softmax(importance / self.tau, dim=0)

    def prune_nodes(self, H, X, edge_mask):
        """Perform node-level pruning"""
        node_scores = self.calculate_node_importance(H, X)
        theta_node = self.calculate_threshold('node')
        
        # Only keep nodes connected to retained edges
        connected_nodes = (H * edge_mask).sum(axis=1) > 0
        keep_mask = (node_scores >= theta_node) & connected_nodes
        
        # Update features
        X_pruned = X * keep_mask.unsqueeze(1)
        return X_pruned, keep_mask

    def update_embeddings_with_pruning(self, X, pruned_matrices, behavior_weights, W, Theta, max_nodes):
        """Update node embeddings using pruned matrices"""
        updated_X = torch.zeros_like(X)
        
        for behavior, H_pruned in pruned_matrices.items():
            # Calculate degree matrices for pruned incidence matrix
            vertex_degrees = np.array(H_pruned.sum(axis=1)).flatten()
            edge_degrees = np.array(H_pruned.sum(axis=0)).flatten()
            
            # Handle zero degrees
            vertex_degrees = np.maximum(vertex_degrees, self.epsilon)
            edge_degrees = np.maximum(edge_degrees, self.epsilon)
            
            # Ensure degrees match max_nodes
            vertex_degrees = np.pad(vertex_degrees, 
                                  (0, max_nodes - len(vertex_degrees)), 
                                  'constant', 
                                  constant_values=1.0)
            
            # Create degree matrices
            Dv_sqrt_inv = sparse.diags(1.0 / np.sqrt(vertex_degrees))
            De_inv = sparse.diags(1.0 / edge_degrees)
            
            # Convert to dense tensors for computation
            H_tensor = torch.FloatTensor(H_pruned.toarray())
            Dv_sqrt_inv_tensor = torch.FloatTensor(Dv_sqrt_inv.toarray())
            De_inv_tensor = torch.FloatTensor(De_inv.toarray())
            
            # Message passing with pruned matrices
            transformed_X = torch.mm(X, W)
            msg = torch.mm(Dv_sqrt_inv_tensor, H_tensor)
            msg = torch.mm(msg, De_inv_tensor)
            msg = torch.mm(msg, H_tensor.t())
            msg = torch.mm(msg, Dv_sqrt_inv_tensor)
            msg = torch.mm(msg, transformed_X)
            msg = torch.mm(msg, Theta[behavior])
            
            # Accumulate weighted messages
            updated_X += behavior_weights[behavior] * msg
        
        return F.relu(updated_X)

    def forward(self, incidence_matrices, feature_dim=64, hidden_dim=32, num_layers=2):
        """Forward pass with three-level pruning"""
        results = {}
        max_nodes = max(H.shape[0] for H in incidence_matrices.values())
        
        # Initialize parameters
        X = torch.empty(max_nodes, feature_dim)
        nn.init.xavier_uniform_(X)
        W = torch.empty(feature_dim, feature_dim)
        nn.init.xavier_uniform_(W)
        Theta = {behavior: torch.empty(feature_dim, feature_dim) for behavior in incidence_matrices.keys()}
        
        for iteration in range(num_layers):
            self.iteration = iteration
            pruned_matrices = {}
            
            for behavior, H in incidence_matrices.items():
                # 1. Component-level pruning
                X_comp, comp_mask = self.prune_components(H, X, W)
                
                # 2. Edge-level pruning
                H_edge, edge_mask = self.prune_edges(H, X_comp, comp_mask)
                
                # 3. Node-level pruning
                X_final, node_mask = self.prune_nodes(H_edge, X_comp, edge_mask)
                
                pruned_matrices[behavior] = H_edge
                
                # Store pruning statistics
                results[f'layer_{iteration}_{behavior}'] = {
                    'component_mask': comp_mask,
                    'edge_mask': edge_mask,
                    'node_mask': node_mask,
                    'pruned_matrix': H_edge
                }
            
            # Update embeddings
            X = self.update_embeddings_with_pruning(X_final, pruned_matrices, 
                                                  self.calculate_behavior_weights({}), 
                                                  W, Theta, max_nodes)
        
        return results
    
#-------------------------Contrative Learning ---------------------------------

class FaNTaLoss(nn.Module):
    def __init__(self, tau=0.1, gamma=0.7, delta=0.5, 
                 theta_low=0.1, theta_high=0.8, eta=0.3,
                 lambda1=1.0, lambda2=0.1, lambda3=0.1, lambda4=0.01,
                 lambda5=0.01):  # Added lambda5 for pruning loss
        super(FaNTaLoss, self).__init__()
        self.tau = tau
        self.gamma = gamma
        self.delta = delta
        self.theta_low = theta_low
        self.theta_high = theta_high
        self.eta = eta
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.lambda4 = lambda4
        self.lambda5 = lambda5
        
        # Initialize pruning-specific parameters
        self.alpha_comp = 0.4
        self.alpha_edge = 0.4
        self.alpha_node = 0.2
        self.beta_gamma = 0.1
        self.beta_delta = 0.1
        self.gamma0 = 0.8
        self.delta0 = 0.6

    def calculate_pruning_ratio(self, H_original, H_pruned, X_original, X_pruned):
        """Calculate pruning ratios for different levels"""
        # Component-level ratio (using attention head preservation)
        PrR_comp = 1 - (H_pruned.sum() / H_original.sum())
        
        # Edge-level ratio
        edge_mask_orig = (H_original != 0).float()
        edge_mask_pruned = (H_pruned != 0).float()
        PrR_edge = 1 - (edge_mask_pruned.sum() / edge_mask_orig.sum())
        
        # Node-level ratio
        node_mask_orig = (X_original.sum(dim=1) != 0).float()
        node_mask_pruned = (X_pruned.sum(dim=1) != 0).float()
        PrR_node = 1 - (node_mask_pruned.sum() / node_mask_orig.sum())
        
        # Combined ratio
        PrR = (self.alpha_comp * PrR_comp + 
               self.alpha_edge * PrR_edge + 
               self.alpha_node * PrR_node)
        
        return PrR, PrR_comp, PrR_edge, PrR_node

    def calculate_thresholds(self, PrR):
        """Calculate dynamic thresholds based on pruning ratio"""
        gamma = self.gamma0 * (1 - self.beta_gamma * PrR)
        delta = self.delta0 * (1 - self.beta_delta * PrR)
        return gamma, delta

class ContrastiveHGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_behaviors, num_layers=2):
        super(ContrastiveHGNN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_behaviors = num_behaviors
        self.pruning = DynamicPruningGNN()
        
        # Initial transformation layer for node features
        self.initial_transform = nn.Linear(input_dim, hidden_dim)
        
        # Node embedding layers
        self.node_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)  
            for i in range(num_layers)
        ])
        
        # Edge embedding layers
        self.edge_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])
        
        # Behavior-specific transformations
        self.behavior_transforms = nn.ParameterDict({
            f'behavior_{i}': nn.Parameter(torch.randn(hidden_dim, hidden_dim))
            for i in range(num_behaviors)
        })
        
        # Final projection layers - Modified dimensions
        self.node_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Edge features to output dimension projection
        self.edge_projection = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),  # Changed input dimension to match node_embeddings
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Attention mechanism
        self.attention = nn.Parameter(torch.randn(output_dim, 1))
        
        # Behavior importance weights
        self.behavior_importance = nn.Parameter(torch.ones(num_behaviors))
        
        # Loss function
        self.loss_fn = FaNTaLoss()
        
        # Initialize parameters
        self._initialize_parameters()












    def compute_attention_similarity(self, edge_i, edge_j, attention_weights):
        """Compute attention-based similarity between edges"""
        att_ij = attention_weights[edge_i, edge_j]
        att_ji = attention_weights[edge_j, edge_i]
        max_att = torch.max(attention_weights)
        return (att_ij * att_ji) / (max_att + 1e-8)

    def identify_false_negatives(self, node_emb, attention_weights, pruning_ratio, 
                               pruned_H, behavior_weight):
        """Identify false negative pairs using dual-threshold mechanism"""
        # Calculate dynamic thresholds
        gamma, delta = self.loss_fn.calculate_thresholds(pruning_ratio)
        
        # Node similarity matrix
        sim_matrix = F.cosine_similarity(node_emb.unsqueeze(1), 
                                       node_emb.unsqueeze(0), dim=2)
        
        # Attention similarity for edges
        edge_sim = torch.zeros_like(sim_matrix)
        for i in range(pruned_H.size(0)):
            for j in range(pruned_H.size(0)):
                edge_sim[i,j] = self.compute_attention_similarity(i, j, attention_weights)
        
        # Apply dual thresholds
        node_mask = sim_matrix > (gamma * behavior_weight)
        att_mask = edge_sim > (delta * behavior_weight)
        false_neg_mask = node_mask & att_mask
        
        return false_neg_mask
        
    def identify_hard_negatives(self, node_emb, attention_weights, mu=0.7, nu=0.3):
        """Identify hard negative pairs"""
        sim_matrix = F.cosine_similarity(node_emb.unsqueeze(1), 
                                       node_emb.unsqueeze(0), dim=2)
        
        edge_sim = torch.zeros_like(sim_matrix)
        for i in range(sim_matrix.size(0)):
            for j in range(sim_matrix.size(0)):
                edge_sim[i,j] = self.compute_attention_similarity(i, j, attention_weights)
        
        hard_neg_mask = (sim_matrix > mu) & (edge_sim < nu)
        return hard_neg_mask

    def compute_structural_loss(self, H_original, H_pruned, X_prev, X_curr):
        """Compute structural preservation loss"""
        # Adjacency difference
        A_orig = torch.mm(H_original, H_original.t())
        A_pruned = torch.mm(H_pruned, H_pruned.t())
        adj_loss = torch.norm(A_orig - A_pruned, p='fro')**2
        
        # Feature preservation
        feat_loss = torch.norm(torch.mm(H_pruned, X_curr) - 
                             torch.mm(H_original, X_prev), p='fro')**2
        
        return adj_loss + feat_loss

    def compute_pruning_loss(self, pruning_masks, thresholds_curr, thresholds_prev):
        """Compute pruning regularization loss"""
        loss = 0
        for level in ['comp', 'edge', 'node']:
            # L1 sparsity
            mask = pruning_masks[level]
            loss += 0.1 * torch.norm(mask, p=1)
            
            # L2 smoothness of thresholds
            thresh_diff = thresholds_curr[level] - thresholds_prev[level]
            loss += 0.1 * torch.norm(thresh_diff, p=2)**2
        
        return loss

    def compute_loss(self, behavior_outputs, behavior_weights):
        """Compute the total loss incorporating pruning-aware components"""
        total_loss = 0
        device = next(iter(behavior_outputs.values()))['node_embeddings'].device
        eps = 1e-8
        
        for i, (behavior, outputs) in enumerate(behavior_outputs.items()):
            node_emb = F.normalize(outputs['node_embeddings'], p=2, dim=1)
            edge_emb = F.normalize(outputs['edge_embeddings'], p=2, dim=1)
            attention = outputs['attention_weights']
            H_tensor = outputs['H_tensor']
            H_pruned = outputs['H_pruned']
            X_prev = outputs['X_prev']
            pi_k = behavior_weights[i]
            
            # Calculate pruning ratios
            PrR, _, _, _ = self.loss_fn.calculate_pruning_ratio(
                H_tensor, H_pruned, X_prev, node_emb
            )
            
            # 1. Pruning-aware Contrastive Loss
            valid_pairs = H_pruned > 0
            similarity = torch.mm(node_emb, edge_emb.t())
            masked_sim = similarity * valid_pairs.float()
            
            cl_loss = -torch.mean(
                F.log_softmax(masked_sim[valid_pairs] / self.loss_fn.tau, dim=0)
            )
            
            # 2. False Negative Correction Loss
            fn_mask = self.identify_false_negatives(
                node_emb, attention, PrR, H_pruned, pi_k
            )
            if fn_mask.sum() > 0:
                fn_loss = -torch.mean(
                    F.log_softmax(similarity[fn_mask] / self.loss_fn.tau, dim=0)
                )
            else:
                fn_loss = torch.tensor(0.0, device=device)
            
            # 3. Hard Negative Mining Loss
            hard_neg_mask = self.identify_hard_negatives(node_emb, attention)
            if hard_neg_mask.sum() > 0:
                hard_loss = -torch.mean(
                    F.log_softmax(-similarity[hard_neg_mask] / self.loss_fn.tau, dim=0)
                )
            else:
                hard_loss = torch.tensor(0.0, device=device)
            
            # 4. Structural Preservation Loss
            struct_loss = self.compute_structural_loss(
                H_tensor, H_pruned, X_prev, node_emb
            )
            
            # 5. Pruning Regularization Loss
            prune_loss = self.compute_pruning_loss(
                outputs['pruning_masks'],
                outputs['thresholds_curr'],
                outputs['thresholds_prev']
            )
            
            # Combine all losses
            behavior_loss = (
                self.loss_fn.lambda1 * pi_k * cl_loss +
                self.loss_fn.lambda2 * fn_loss +
                self.loss_fn.lambda3 * hard_loss +
                self.loss_fn.lambda4 * struct_loss +
                self.loss_fn.lambda5 * prune_loss
            )
            
            if torch.isfinite(behavior_loss):
                total_loss += behavior_loss
        
        return total_loss

    def forward(self, incidence_matrices, node_features):
        device = node_features.device
        behavior_outputs = {}
        
        # First transform input features
        x = self.initial_transform(node_features)
        
        for i, (behavior, H) in enumerate(incidence_matrices.items()):
            # Store original matrices for loss computation
            H_original = self._to_tensor(H, device)
            X_prev = x.clone()
            
            # Apply pruning
            pruning_results = self.pruning(
                {'behavior': H_original}, 
                x.clone()
            )
            
            H_pruned = pruning_results[f'layer_0_behavior']['pruned_matrix']
            H_norm = self._normalize_matrix(H_pruned)
            
            # Process through layers
            current_features = x
            for node_layer, edge_layer in zip(self.node_layers, self.edge_layers):
                edge_msg = torch.mm(H_norm.t(), current_features)
                edge_msg = edge_layer(edge_msg)
                edge_msg = torch.mm(edge_msg, 
                                  self.behavior_transforms[f'behavior_{i}'])
                node_msg = torch.mm(H_norm, edge_msg)
                current_features = F.relu(node_layer(node_msg) + current_features)
            
            # Project features
            node_embeddings = self.node_projection(current_features)
            edge_features = torch.mm(H_norm.t(), node_embeddings)
            edge_embeddings = self.edge_projection(edge_features)
            
            # Compute attention
            attention_scores = torch.mm(node_embeddings, self.attention)
            attention_weights = F.softmax(attention_scores, dim=0)
            
            # Store outputs
            behavior_outputs[behavior] = {
                'node_embeddings': node_embeddings,
                'edge_embeddings': edge_embeddings,
                'attention_weights': attention_weights,
                'H_tensor': H_original,
                'H_pruned': H_pruned,
                'X_prev': X_prev,
                'pruning_masks': pruning_results[f'layer_0_behavior'],
                'thresholds_curr': self.pruning.calculate_threshold(),
                'thresholds_prev': getattr(self.pruning, '_prev_threshold', 0.0)
            }
        
        # Normalize behavior importance weights
        behavior_weights = F.softmax(self.behavior_importance, dim=0)
        
        if self.training:
            loss = self.compute_loss(behavior_outputs, behavior_weights)
            return loss, behavior_outputs
        
        return behavior_outputs
    
    def compute_loss(self, behavior_outputs, behavior_weights):
        """Compute the total loss across all behaviors with improved numerical stability"""
        total_loss = 0
        device = next(iter(behavior_outputs.values()))['node_embeddings'].device
        eps = 1e-8
        
        for i, (behavior, outputs) in enumerate(behavior_outputs.items()):
            node_emb = F.normalize(outputs['node_embeddings'], p=2, dim=1)
            edge_emb = F.normalize(outputs['edge_embeddings'], p=2, dim=1)
            attention = outputs['attention_weights']
            H_tensor = outputs['H_tensor']
            pi_k = behavior_weights[i]
            
            # 1. Contrastive Loss with absolute values
            valid_pairs = H_tensor > 0
            similarity = torch.clamp(torch.mm(node_emb, edge_emb.t()), min=-1.0, max=1.0)
            masked_similarity = similarity * valid_pairs.float()
            
            pos_score = masked_similarity[valid_pairs]
            pos_score = torch.abs(pos_score) / self.loss_fn.tau  # Add abs()
            
            neg_score = torch.abs(similarity) / self.loss_fn.tau  # Add abs()
            valid_nodes = valid_pairs.any(dim=1)
            
            if valid_nodes.any():
                log_softmax = F.log_softmax(neg_score[valid_nodes], dim=1)
                cl_loss = -torch.mean(log_softmax[valid_pairs[valid_nodes]])  # Use mean instead of raw sum
            else:
                cl_loss = torch.tensor(0.0, device=device)
            
            # 2. False Negative Detection Loss with absolute values
            sim_matrix = torch.abs(torch.clamp(torch.mm(node_emb, node_emb.t()), min=-1.0, max=1.0))
            degrees = torch.sum(H_tensor, dim=1) + eps
            intersection = torch.mm(H_tensor, H_tensor.t())
            norm_term = torch.sqrt(torch.outer(degrees, degrees))
            struct_sim = torch.clamp(intersection / norm_term, min=0.0, max=1.0)
            
            fn_mask = ((sim_matrix > self.loss_fn.gamma * pi_k) & 
                    (struct_sim > self.loss_fn.delta)).float()
            fn_loss = torch.mean(fn_mask * sim_matrix) if fn_mask.sum() > 0 else torch.tensor(0.0, device=device)
            
            # 3. Hard Negative Mining Loss with absolute values
            lower_bound = pi_k * self.loss_fn.theta_low
            upper_bound = pi_k * self.loss_fn.theta_high
            hard_mask = ((sim_matrix > lower_bound) & 
                        (sim_matrix < upper_bound)).float()
            hard_neg_loss = torch.mean(hard_mask * sim_matrix) if hard_mask.sum() > 0 else torch.tensor(0.0, device=device)
            
            # 4. Hypergraph Regularization with positive definite guarantee
            D_v = torch.diag(1.0 / torch.sqrt(torch.sum(H_tensor, dim=1) + eps))
            L = torch.eye(H_tensor.size(0), device=device) - torch.mm(
                torch.mm(D_v, H_tensor),
                torch.mm(H_tensor.t(), D_v)
            )
            reg_loss = torch.abs(torch.trace(torch.mm(torch.mm(node_emb.t(), L), node_emb)))
            
            # Combine losses with positive weights
            behavior_loss = (
                self.loss_fn.lambda1 * torch.abs(pi_k * cl_loss) +  # Ensure positive
                self.loss_fn.lambda2 * fn_loss +
                self.loss_fn.lambda3 * torch.abs(pi_k * hard_neg_loss) +
                self.loss_fn.lambda4 * reg_loss
            )
            
            if torch.isfinite(behavior_loss):
                total_loss += behavior_loss
        
        return total_loss

def train_model(model, incidence_matrices, num_epochs=200, lr=0.001):  # Increased epochs
    """Train the contrastive HGNN model with improved numerical stability"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)  # Increased weight decay
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Print matrix information
    print("\nIncidence Matrix Info:")
    for behavior, H in incidence_matrices.items():
        print(f"{behavior} matrix shape: {H.shape}")
        sparsity = np.sum(H == 0) / (H.shape[0] * H.shape[1])
        print(f"{behavior} matrix sparsity: {sparsity:.2%}")
    
    # Get maximum dimensions
    max_nodes = max(H.shape[0] for H in incidence_matrices.values())
    max_edges = max(H.shape[1] for H in incidence_matrices.values())
    
    # Initialize node features with Xavier initialization
    node_features = torch.empty(max_nodes, model.input_dim).to(device)
    nn.init.xavier_uniform_(node_features)
    
    # Normalize matrices
    padded_matrices = {}
    for behavior, H in incidence_matrices.items():
        padded_H = np.zeros((max_nodes, max_edges))
        padded_H[:H.shape[0], :H.shape[1]] = H
        # Add small epsilon and normalize
        padded_H = (padded_H + 1e-8) / np.sum(padded_H + 1e-8)
        padded_matrices[behavior] = padded_H
    
    print(f"\nTraining on device: {device}")
    print(f"Number of model parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Node features shape: {node_features.shape}")
    
    # Improved learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',
        factor=0.7,  # Less aggressive reduction
        patience=7,   # Increased patience
        verbose=True,
        min_lr=1e-6  # Add minimum learning rate
    )
    
    best_loss = float('inf')
    patience_counter = 0
    max_patience = 15  # Increased early stopping patience
    losses = []  # Track losses
    
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        
        try:
            # Forward pass with gradient scaling
            with torch.cuda.amp.autocast(enabled=device.type=='cuda'):
                loss, outputs = model(padded_matrices, node_features)
            
            # Check for invalid loss
            if not torch.isfinite(loss):
                print(f"Warning: Non-finite loss detected: {loss.item()}")
                continue
            
            # Backward pass with gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)  # Increased max norm
            optimizer.step()
            
            losses.append(loss.item())
            
            # Learning rate scheduling
            scheduler.step(loss)
            
            # Early stopping check with moving average
            window_size = 5
            if len(losses) >= window_size:
                avg_loss = sum(losses[-window_size:]) / window_size
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
            
            if patience_counter >= max_patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
            
            # More detailed logging
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, '
                      f'Avg Loss: {sum(losses[-10:])/10:.4f}, '
                      f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        except Exception as e:
            print(f"\nError in epoch {epoch+1}:")
            print(f"Error message: {str(e)}")
            raise e
   
    # Get final embeddings
    model.eval()
    with torch.no_grad():
        outputs = model(padded_matrices, node_features)
    
    return outputs, losses

# Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        # Add batch normalization and dropout
        self.bn_input = nn.BatchNorm1d(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        # Initialize weights properly
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        x = self.bn_input(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        return x

def process_data(folder_path):
    df_genres = pd.read_excel(
        os.path.join(folder_path, 'movie_genres.xlsx'),
        usecols=['movieID', 'genreID', 'Labels']
    )
    
    # Create continuous genre IDs starting from 0
    unique_genres = sorted(df_genres['genreID'].unique())
    genre_id_mapping = {genre: idx for idx, genre in enumerate(unique_genres)}
    
    # Map movie IDs to their genres using the new mapping
    genre_mapping = defaultdict(list)
    for _, row in df_genres.iterrows():
        movie_id = row['movieID']
        genre = row['genreID']
        genre_mapping[movie_id].append(genre_id_mapping[genre])
    
    # Take the first genre for each movie
    processed_genres = [(movie_id, genres[0]) for movie_id, genres in genre_mapping.items()]
    ground_truth_ratings = pd.DataFrame(processed_genres, columns=['movieID', 'genreID'])
    ground_truth_ratings = ground_truth_ratings.sort_values('movieID').reset_index(drop=True)
    
    num_genres = len(unique_genres)
    return ground_truth_ratings, genre_id_mapping, num_genres

def generate_movie_embeddings(num_movies, embedding_dim):
    return torch.randn(num_movies, embedding_dim)

def train_mlp(movie_embeddings, labels, config):
    input_dim = config['embedding_dim']
    output_dim = config['num_genres']  # Use the actual number of genres
    
    # Print shape information for debugging
    print(f"Input dimension: {input_dim}")
    print(f"Output dimension (num_genres): {output_dim}")
    print(f"Number of unique labels: {len(torch.unique(labels))}")
    print(f"Max label value: {torch.max(labels).item()}")
    
    # Define the MLP model
    mlp_model = MLP(input_dim=input_dim, hidden_dim=config['hidden_dim'], output_dim=output_dim)
    optimizer = optim.Adam(mlp_model.parameters(), lr=config['learning_rate'])
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(config['num_epochs']):
        mlp_model.train()
        optimizer.zero_grad()
        
        # Forward pass
        outputs = mlp_model(movie_embeddings)
        
        # Compute loss
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # Print loss
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{config['num_epochs']}, Loss: {loss.item():.4f}")

    print("Training complete!")
    return mlp_model

def create_data_loaders(movie_embeddings, labels, batch_size=32):
    """Create train, validation, and test data loaders."""
    # First split: 80% train+val, 20% test
    X_temp, X_test, y_temp, y_test = train_test_split(
        movie_embeddings, labels, test_size=0.2, random_state=42)
    
    # Second split: 70% train, 10% validation (87.5% of remaining data is train)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.125, random_state=42)
    
    # Create TensorDatasets
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader

def calculate_metrics(y_true, y_pred, y_pred_proba):
    """Calculate all evaluation metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Convert to one-hot for AUC calculation
    n_classes = y_pred_proba.shape[1]
    y_true_one_hot = np.eye(n_classes)[y_true]
    
    # Calculate AUC for each class and take mean
    auc_scores = []
    for i in range(n_classes):
        try:
            auc = roc_auc_score(y_true_one_hot[:, i], y_pred_proba[:, i])
            auc_scores.append(auc)
        except ValueError:
            continue
    auc = np.mean(auc_scores) if auc_scores else 0
    
    accuracy = accuracy_score(y_true, y_pred)
    
    return {
        'mae': mae,
        'rmse': rmse,
        'auc': auc,
        'accuracy': accuracy
    }

def evaluate_model(model, data_loader, device):
    """Evaluate model on given data loader."""
    model.eval()
    all_preds = []
    all_preds_proba = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            # Get predicted probabilities
            probs = F.softmax(outputs, dim=1)
            
            # Get predicted classes
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_preds_proba.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return calculate_metrics(
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_preds_proba)
    )

def train_and_evaluate(movie_embeddings, labels, config, run_id):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        movie_embeddings, labels, batch_size=config['batch_size'])
    
    # Initialize model
    model = MLP(
        input_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        output_dim=config['num_genres']
    ).to(device)
    
    # Initialize optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    best_val_loss = float('inf')
    best_model = None
    patience = 10
    patience_counter = 0
    
    # Training loop
    for epoch in range(config['num_epochs']):
        model.train()
        train_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()
        
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break
    
    # Load best model for evaluation
    model.load_state_dict(best_model)
    
    # Evaluate on all sets
    train_metrics = evaluate_model(model, train_loader, device)
    val_metrics = evaluate_model(model, val_loader, device)
    test_metrics = evaluate_model(model, test_loader, device)
    
    return {
        'train': train_metrics,
        'val': val_metrics,
        'test': test_metrics
    }

def run_evaluation(movie_embeddings, labels, config, num_runs=10):
    """Run multiple evaluations and compute statistics."""
    all_metrics = {
        'train': {'mae': [], 'rmse': [], 'auc': [], 'accuracy': []},
        'val': {'mae': [], 'rmse': [], 'auc': [], 'accuracy': []},
        'test': {'mae': [], 'rmse': [], 'auc': [], 'accuracy': []}
    }
    
    start_time = time.time()
    
    for run in range(num_runs):
        print(f"\nStarting run {run + 1}/{num_runs}")
        run_metrics = train_and_evaluate(movie_embeddings, labels, config, run)
        
        # Store metrics
        for split in ['train', 'val', 'test']:
            for metric in ['mae', 'rmse', 'auc', 'accuracy']:
                all_metrics[split][metric].append(run_metrics[split][metric])
    
    total_time = time.time() - start_time
    
    # Calculate statistics
    results = {}
    for split in ['train', 'val', 'test']:
        results[split] = {}
        for metric in ['mae', 'rmse', 'auc', 'accuracy']:
            values = all_metrics[split][metric]
            mean = np.mean(values)
            std = np.std(values)
            conf_interval = stats.t.interval(
                0.95, len(values)-1, loc=mean, scale=stats.sem(values))
            
            results[split][metric] = {
                'mean': mean,
                'std': std,
                'conf_interval': conf_interval
            }
    
    # Print results
    print("\nEvaluation Results (10 runs):")
    print("=" * 50)
    for split in ['train', 'val', 'test']:
        print(f"\n{split.upper()} Set Results:")
        print("-" * 30)
        for metric in ['mae', 'rmse', 'auc', 'accuracy']:
            print(f"{metric.upper()}:")
            print(f"  Mean: {results[split][metric]['mean']:.4f}")
            print(f"  Std:  {results[split][metric]['std']:.4f}")
            print(f"  95% CI: ({results[split][metric]['conf_interval'][0]:.4f}, "
                  f"{results[split][metric]['conf_interval'][1]:.4f})")
    
    print(f"\nTotal time for {num_runs} runs: {total_time:.2f} seconds")
    print(f"Average time per run: {total_time/num_runs:.2f} seconds")
    
    return results


# Example usage in main function
def main():
    config = {
        'folder_path': 'C:\\IMDB',
        'embedding_dim': 64,
        'hidden_dim': 256,  # Increased hidden dimension
        'learning_rate': 0.001,
        'num_epochs': 100,  # Increased epochs (with early stopping)
        'batch_size': 64,   # Increased batch size
        'num_genres': None
    }

    # Process data and get the actual number of genres
    ground_truth_ratings, genre_id_mapping, num_genres = process_data(config['folder_path'])
    config['num_genres'] = num_genres
    
    # Print data statistics
    print(f"\nDataset Statistics:")
    print(f"Number of genres: {num_genres}")
    print("\nGenre distribution:")
    for genre, count in ground_truth_ratings['genreID'].value_counts().items():
        print(f"Genre {genre}: {count} samples")
    
    num_movies = ground_truth_ratings['movieID'].nunique()
    movie_embeddings = generate_movie_embeddings(num_movies, config['embedding_dim'])
    labels = torch.tensor(ground_truth_ratings['genreID'].values, dtype=torch.long)
    
    # Run evaluation
    results = run_evaluation(movie_embeddings, labels, config, num_runs=10)
    return results

if __name__ == "__main__":
    main()