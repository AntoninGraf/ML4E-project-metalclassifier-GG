import os

def count_files_in_folder(folder_path):
    try:
        # List all items in the folder
        items = os.listdir(folder_path)
        
        # Filter out the directories
        files = [item for item in items if os.path.isfile(os.path.join(folder_path, item))]
        
        # Return the number of files
        return len(files)
    except Exception as e:
        print(f"An error occurred: {e}")
        return 0

# Example usage

# Path to the data file
folder_path = "./data/Live_files" 

number_of_files = count_files_in_folder(folder_path)
print(f"Number of files in '{folder_path}': {number_of_files}")
