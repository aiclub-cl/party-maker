"""
Module for the functions that handle the CSV file with the members data.
"""
import csv
from sklearn.preprocessing import MultiLabelBinarizer

def extract_unique_interests(file_name):
    """
    Reads the CSV file and returns the unique interests found.

    Input:
    file_name (str): CSV file name.

    Output:
    unique_interests (list): The unique interests found.
    """
    unique_interests = set()

    with open(file_name, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)

        for row in reader:
            unique_interests.add(row['Hobby'])
            unique_interests.add(row['Fav_Genre'])
            unique_interests.add(row['Fav_Weather'])
            unique_interests.add(row['AI_Exp'])
            unique_interests.add(row['Ext_or_Int'])

    return sorted(unique_interests)


def generate_interest_vectors(file_name, unique_interests):
    """
    Reads the CSV file and returns the interest vectors of all the members.

    Input:
    file_name (str): CSV file name.
    unique_interests (list): The unique interests.

    Output:
    members (list): The list of members.
    interest_vectors (numpy array): The interest vectors of all the members. (shape: (n_members, n_unique_interests))
    """
    members = []

    with open(file_name, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)

        for row in reader:
            interests = [row['Hobby'], row['Fav_Genre'], row['Fav_Weather'], row['AI_Exp'], row['Ext_or_Int']]
            members.append({"name": row['Username'], "interests": interests})

    # Binarize the interests to create feature vectors
    binarizer = MultiLabelBinarizer(classes=unique_interests)
    interest_vectors = binarizer.fit_transform([member["interests"] for member in members])

    return members, interest_vectors
