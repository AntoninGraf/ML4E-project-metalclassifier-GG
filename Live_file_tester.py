import time
import os
import numpy as np
from datafilereader import DataFileReader
import pickle
import glob

# Set up features
featureListR = [17, 20, 21, 22, 26, 28, 31, 32, 39, 42, 44, 71]
featureListL = [4, 5, 6, 7, 8, 9, 12, 61]
labels = ["unknown", "5_CTS", "10_CTS", "20_CTS", "50_CTS", "1_CHF", "2_CHF", "5_CHF"]

# Load the pre-trained SVM model
with open("data/Tests/model1.pkl", "rb+") as f:
    model = pickle.load(f)

# Function to process each test file
def process_file(file_path):
    reader = DataFileReader(file_path)
    f, Z = reader.get_all_mesurements()
    
    # Separate resistance and reactance
    R = np.real(Z)
    L = np.imag(Z) / (2 * np.pi * f)
    
    # Initialize calibration values
    R_cal = R[0]
    L_cal = L[0]
    
    # Initialize prediction counter
    predictions = {label: 0 for label in labels}
    
    # Loop through measurements (skip the first one if it's a reference)
    start_idx = 1 if reader.__is_reference(0) else 0
    for idx in range(start_idx, len(f)):
        R_measure = R[idx] - R_cal
        L_measure = L[idx] - L_cal
        # Extract the needed features
        R_features = R_measure[featureListR]
        L_features = L_measure[featureListL]
        # Concatenate the features
        X = np.concatenate((R_features, L_features), axis=0)
        # Reshape the array to 2D
        X = X.reshape(1, -1)
        # Predict the coin type
        Y = model.predict(X)[0]
        predictions[labels[Y]] += 1
    
    return predictions

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
