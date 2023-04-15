"""
Module for the functions that cluster the members using K-means.
"""
from sklearn.cluster import KMeans
from collections import defaultdict

def create_groups(members, group_size, interest_vectors):
    """
    Clusters the members using K-means and returns the groups.

    Input:
    data (numpy array): The scaled features.
    group_size (int): The number of members in each group.

    Output:
    groups (dict): The groups of members.
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
