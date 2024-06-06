import time
import os
import numpy as np
from datafilereader import DataFileReader
import pickle
import glob

# Set up features
featureListR = [17,20,21,22,26,28,31,32,39,42,44,71]
featureListL = [4,5,6,7,8,9,10,12,61]
labels = ["unknown", "5_CTS", "10_CTS", "20_CTS", "50_CTS", "1_CHF", "2_CHF", "5_CHF"]

# Path to the data file
folder_path = "./data/Live_files" 

# Load the pre-trained SVM model
with open("data/Tests/model1.pkl", "rb+") as f:
    model = pickle.load(f)

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


# Function to process each test file
def process_file(file_path):
    
    reader = DataFileReader(file_path)
    #get all measured Z 
    _,Z = reader.get_all_mesurements()
    N = len(Z) #number of measurements for this coin
    R = np.real(Z)
    L = np.imag(Z)/(2*np.pi*f)
    # Initialize calibration values
    C_idx = reader.get_reference_impedance_index()
    print("Calibration index: ", C_idx)
    # substract all the data by the calibration
    R = R[1:,:]-R[C_idx,:]
    L = L[1:,:]-L[C_idx,:]
    #extract the needed features
    R = R[:,featureListR]
    L = L[:,featureListL]
    p = R[:,0:9]/L
    #concatenate the features
    for n in range(N-1):
        X.append(np.concatenate((R[n,:],L[n,:],p[n,:]),axis=0))
        #X.append(d) ,p[n,:]
        Y.append(i)
    # reshape the array to 2D
    X = X.reshape(1, -1)
    Y = model.predict(X)
    print("Coin is of type: ", labels[Y[0]])

    return 

# Main function to process all test files
def main():
    # Define the directory containing the test files
    test_files_dir = "./data/tests/"
    test_files_pattern = os.path.join(test_files_dir, "test_set_*.h5")
    
    # Find all test files matching the pattern
    test_files = glob.glob(test_files_pattern)
    
    for test_file in test_files:
        print(f"Processing file: {test_file}")
        predictions = process_file(test_file)
        
        print(f"Results for {test_file}:")
        for coin, count in predictions.items():
            if count > 0:
                print(f"  {coin}: {count}")

if __name__ == "__main__":
    main()
