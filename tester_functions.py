import time
import os
import numpy as np
from dev.datafilereader import DataFileReader
import pickle
import glob

# Count the number of files in the folder
def count_files_in_folder(folder_path):
    try:
        files = [item for item in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, item))]
        return len(files)
    except Exception as e:
        print(f"An error occurred: {e}")
        return 0

#live files function (live_files.ipynb)
# Process each test file
def process_files(folder_path):

    # Set up features
    featureListR = [17, 20, 21, 22, 26, 28, 31, 32, 39, 42, 44, 71]
    featureListL = [4, 5, 6, 7, 8, 9, 10, 12, 61]
    labels = ["unknown", "5_CTS", "10_CTS", "20_CTS", "50_CTS", "1_CHF", "2_CHF", "5_CHF"]

    # Load the pre-trained models
    with open("models/model1.pkl","rb+") as f:
        SVM = pickle.load(f)

    with open("models/modelOneVsAll.pkl", "rb") as f:
        SVMO = pickle.load(f)

    NmbFiles = count_files_in_folder(folder_path)

    if NmbFiles == 0:
        print("No files found in the folder.")
        return None, None
    else:
        print(f"Found {NmbFiles} files in the folder.")
    
    file_paths = glob.glob(os.path.join(folder_path, "test_set_*.h5"))
    
    summary_results = []
    detailed_results = {}

    for file_path in file_paths:
        # Get the number of the file
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
        p = R[:,0:9] / L
        
        # Collect results for each measurement
        results = []
        
        # Initialize prediction counter for the summary
        predictions = {label: 0 for label in labels}

        # Get the number of measurements after removing calibration
        N = R.shape[0]

        for j in range(N):
            # Concatenate the features
            X = np.concatenate((R[j, :], L[j, :], p[j, :]), axis=0)
            # Check for NaN values before prediction
            if np.isnan(X).any():
                results.append(("Measurement {}".format(j), "Skipped due to NaN"))
                continue
            
            # Reshape the array to 2D
            X = X.reshape(1, -1)
            
            # Anomaly detection
            try:
                result = SVMO.predict(X)
            except ValueError as ve:
                results.append(("Measurement {}".format(j), "Anomaly detection error"))
                continue
            
            if result == -1:
                results.append(("Measurement {}".format(j), "Unknown"))
                predictions["unknown"] += 1
            else:
                # Classification
                try:
                    Y = SVM.predict(X)
                    results.append(("Measurement {}".format(j), labels[Y[0]]))
                    predictions[labels[Y[0]]] += 1
                except ValueError as ve:
                    results.append(("Measurement {}".format(j), "Classification error"))
                    continue
        
        # Convert results to NumPy array
        results_array = np.array(results, dtype=object)
        detailed_results[file] = results_array
        
        # Add summary for this file
        file_summary = {"file": file, "predictions": predictions}
        summary_results.append(file_summary)

    return summary_results, detailed_results

#live test function (live_test.ipynb)
def liveTest(file_path):

    #Set up features
    featureListR = [17,20,21,22,26,28,31,32,39,42,44,71]
    featureListL = [4,5,6,7,8,9,10,12,61]

    labels = ["unknown", "5_CTS", "10_CTS", "20_CTS", "50_CTS", "1_CHF", "2_CHF", "5_CHF"]

    #SETUP
    Calibrated = False
    reader = DataFileReader(file_path)
    with open("models/model1.pkl","rb+") as f:
        SVM = pickle.load(f)

    with open("models/modelOneVsAll.pkl", "rb") as f:
        SVMO = pickle.load(f)

    # CALIBRATION
    print("\n\n")
    print("=====================================================")
    print("Please Start Calibration (one short press in the air)")
    while not Calibrated:
    
        if reader.has_file_changed():
            f, Z, is_reference = reader.get_last_mesurement()
            R_cal = np.real(Z)
            L_cal = np.imag(Z)/(2*np.pi*f)
            Calibrated = True
            print("Calibration done")
            

        #time.sleep(2)  # Give user time to calibrate
        
    # TESTING
    print("Please take the measurement with the coin to be tested (one short press)")
    while  Calibrated:
        if reader.has_file_changed():
            f, Z, is_reference = reader.get_last_mesurement()
            R = np.real(Z)
            L = np.imag(Z)/(2*np.pi*f)
            # calibrate the coin
            R = R-R_cal
            L = L-L_cal
            
            #extract the needed features
            R = R[featureListR]
            L = L[featureListL]
            p = R[0:9]/L
            #concatenate the features
            X = np.concatenate((R,L,p),axis=0)
            # reshape the array to 2D
            X = X.reshape(1, -1)
            Y_pred = SVMO.predict(X)
            #result = ANO.predict(X)
            if Y_pred == -1:
                print("Coin is unknown")
            else:
                Y = SVM.predict(X)
                print("Coin is of type: ", labels[Y[0]])

        #time.sleep(2)  # Give user time to mesure the coin to be tested