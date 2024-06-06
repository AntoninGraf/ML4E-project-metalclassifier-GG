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

# Find the number of files in the test folder
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
def process_files(file_path):
    #SETUP
    NmbFiles = count_files_in_folder(folder_path)
    for i in range(1, NmbFiles + 1):
        dataset = DataFileReader(file_path+"test_set_"[i]+".h5")
        #get all measured Z for this coin
        _,Z = dataset.get_all_mesurements()
        N = len(Z) #number of measurements for this coin
        R = np.real(Z)
        L = np.imag(Z)/(2*np.pi*f)
        # substract all the data by the calibration
        C_idx = dataset.get_reference_impedance_index()
        print("Calibration index: ", C_idx)
        R = R[1:,:]-R[C_idx,:]
        L = L[1:,:]-L[C_idx,:]
        #extract the needed features
        R = R[:,featureListR]
        L = L[:,featureListL]
        p = R[:,0:9]/L
        #concatenate the features
        for n in range(N-1):
            X.append(np.concatenate((R[n,:],L[n,:],p[n,:]),axis=0))
            Y.append(i)
    return 

# Main function to process all test files
def main():
  
if __name__ == "__main__":
    main()
