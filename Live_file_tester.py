import os
import numpy as np
from datafilereader import DataFileReader
import pickle
import glob

# Path to the data folder containing the live files
folder_path = "./data/Live_files/" 

# Set up features
featureListR = [17, 20, 21, 22, 26, 28, 31, 32, 39, 42, 44, 71]
featureListL = [4, 5, 6, 7, 8, 9, 10, 12, 61]
labels = ["unknown", "5_CTS", "10_CTS", "20_CTS", "50_CTS", "1_CHF", "2_CHF", "5_CHF"]

# Load the pre-trained models
with open("data/Tests/modelAn1.pkl", "rb") as f:
    ANO = pickle.load(f)

with open("data/Tests/model1.pkl", "rb") as f:
    SVM = pickle.load(f)

# Count the number of files in the folder
def count_files_in_folder(folder_path):
    try:
        files = [item for item in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, item))]
        return len(files)
    except Exception as e:
        print(f"An error occurred: {e}")
        return 0

# Process each test file
def process_files(folder_path):
    NmbFiles = count_files_in_folder(folder_path)
    if NmbFiles == 0:
        print("No files found in the folder.")
        return
    else :
        print(f"Found {NmbFiles} files in the folder.")
    
    file_paths = glob.glob(os.path.join(folder_path, "test_set_*.h5"))
    
    for file_path in file_paths:
        #get the number of the file
        file = file_path.split("_")[-1].split(".")[0]
        
        dataset = DataFileReader(file_path)
        
        # Get all measured Z for this coin
        f, Z = dataset.get_all_mesurements()
        R = np.real(Z)
        L = np.imag(Z) / (2 * np.pi * f)
        
        # Get the calibration index
        C_idx = dataset.get_reference_impedance_index()
        if C_idx is None or len(C_idx) == 0:
            print(f"No reference measurement found in {file_path}")
            continue
        
        R_cal = R[C_idx,:]
        L_cal = L[C_idx,:]
        
        # Remove the calibration index from the list of measurements
        R = np.delete(R, C_idx, axis=0)
        L = np.delete(L, C_idx, axis=0)
        
        # Calibrate the coin measurements
        R = R - R_cal
        L = L - L_cal
        # Extract the needed features
        R = R[:, featureListR]
        L = L[:, featureListL]
        
    
        # Ensure no zero or near-zero values in L to avoid division by zero
        L[L == 0] = np.nan  # Replace zero with NaN to handle safely
        p = R[:,0:9]/L
        
        # Initialize prediction counter
        predictions = {label: 0 for label in labels}
        
        # Get the number of measurements after removing calibration
        N = R.shape[0]

        for j in range(N):
            # Concatenate the features
            X = np.concatenate((R[j, :], L[j, :], p[j, :]), axis=0)
            # Check for NaN values before prediction
            if np.isnan(X).any():
                print(f"Skipping measurement {j} due to NaN values in features.")
                continue
            
            # Reshape the array to 2D
            X = X.reshape(1, -1)
            
            # Anomaly detection
            try:
                result = ANO.predict(X)
            except ValueError as ve:
                print(f"Error in anomaly detection for measurement {j}: {ve}")
                continue
            
            if result == -1:
                print(f"Measurement {j} is unknown")
                predictions["unknown"] += 1
            else:
                # Classification
                try:
                    Y = SVM.predict(X)
                    print(f"Measurement {j} is of type: {labels[Y[0]]}")
                    predictions[labels[Y[0]]] += 1
                except ValueError as ve:
                    print(f"Error in classification for measurement {j}: {ve}")
                    continue

        print(f"Results for test set N°{file}:")
        for coin, count in predictions.items():
            if count > 0:
                print(f"  {coin}: {count}")

# Main function to process all test files
def main(folder_path):
    count_files_in_folder(folder_path)
    process_files(folder_path)

# Run the main function
if __name__ == "__main__":
    main(folder_path)
