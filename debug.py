import pickle

pickle_file_path = 'assets/debug/out_188_gt.pkl'

# Load the contents of the pickle file
with open(pickle_file_path, 'rb') as file:
    data = pickle.load(file)

# Print the contents of the loaded data
print(data)