import time
import os
import numpy as np
from datafilereader import DataFileReader
#from sklearn.externals import joblib


# Path to the data file
file_path = "./data/Tests/coin_data.h5" 

#Set up features
featureListR = [17,20,21,22,26,28,31,32,39,42,44,71]
featureListL = [4,5,6,7,8,9,12,61]


# Load the pre-trained SVM model
# model = joblib.load('svm_model.pkl')
# scaler = joblib.load('scaler.pkl')

#Main loop to continuously check the file for changes
def main(file_path):

    Calibrated = False
    reader = DataFileReader(file_path)

    # CALIBRATION
    print("Please Start Calibration")
    while not Calibrated:
        
        if reader.has_file_changed():
            f, Z, is_reference = reader.get_last_mesurement()
            R_cal = np.real(Z)
            L_cal = np.imag(Z)/(2*np.pi*f)
            Calibrated = True
            print("Calibration done")
            return R_cal, L_cal

        time.sleep(5)  # Give user time to calibrate
        
    # Test
    print("Please take the measurement for the coin to be tested")
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
            return R, L

        time.sleep(5)  # Give user time to mesure the coin to be tested
        
    

main(file_path)

# if __name__ == "__main__":
#     file_path = "./dat a/Tests/coin_data.h5" 
