"""
Runs the project.
"""
from csv_handler import extract_unique_interests, generate_interest_vectors
from clustering import create_groups

# Name of the file containing the data.
file_name = "members.csv"

# Number of members in each group.
group_size = 4

# Extract the unique interests from the file.
unique_interests = extract_unique_interests(file_name)

# Generate vectors of interests for each member.
members, interest_vectors = generate_interest_vectors(file_name, unique_interests)

# Create groups of members based on their interests.
create_groups(members, group_size, interest_vectors, visualize=True)
