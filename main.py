"""
Runs the project.
"""
from csv_handler import extract_unique_interests, generate_interest_vectors
from clustering import create_groups

file_name = "members.csv"
group_size = 4

unique_interests = extract_unique_interests(file_name)
members, interest_vectors = generate_interest_vectors(file_name, unique_interests)

# Create groups of members based on their interests.
groups = create_groups(members, group_size, interest_vectors)