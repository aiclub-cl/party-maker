"""
Module for the functions that cluster the members using K-means.
"""
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def visualize_clusters(members, interest_vectors, kmeans):
    """
    Creates a visualization of the k-means clusters

    Inputs:
    members (list): List of members names
    interest_vectors (numpy array): The interest vectors of all the members. (shape: (n_members, n_unique_interests))
    kmeans (KMeans): The K-means model

    Output:
    None
    """
    # Reduce the dimensionality of the interest vectors to 2D
    pca = PCA(n_components=2)
    reduced_vectors = pca.fit_transform(interest_vectors)

    # Create a scatter plot of the reduced vectors, colored by cluster
    colors = plt.cm.get_cmap("viridis", kmeans.n_clusters)
    for idx, member in enumerate(members):
        cluster = kmeans.labels_[idx]
        plt.scatter(reduced_vectors[idx, 0], reduced_vectors[idx, 1], c=[colors(cluster)], label=member["name"], alpha=0.7)

    # Add labels for each member
    for idx, member in enumerate(members):
        plt.annotate(member["name"], (reduced_vectors[idx, 0], reduced_vectors[idx, 1]), textcoords="offset points", xytext=(0, 3), ha="center")

    # Add a legend outside the plot area
    plt.title("Community Member Clusters")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def create_groups(members, group_size, interest_vectors, visualize=False):
    """
    Clusters the members using K-means and returns the groups.

    Input:
    members (list): List of members names
    group_size (int): Number of members per group
    interest_vectors (numpy array): The interest vectors of all the members. (shape: (n_members, n_unique_interests))
    visualize (bool): If True, the clusters will be visualized. (Default: False)

    Output:
    None
    """
    # Create the K-means model and fit it to the data
    n_clusters = len(members) // group_size
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(interest_vectors)

    # Assign members to clusters
    for idx, member in enumerate(members):
        cluster = kmeans.labels_[idx]
        if "cluster" not in member:
            member["cluster"] = []
        member["cluster"].append(cluster)

    # Print the groups
    for i in range(n_clusters):
        group = [member["name"] for member in members if i in member["cluster"]]
        print(f"Group {i + 1}: {', '.join(group)}")

    # Call the function to visualize the clusters
    if visualize:
        visualize_clusters(members, interest_vectors, kmeans)
