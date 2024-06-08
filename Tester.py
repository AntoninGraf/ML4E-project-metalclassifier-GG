import time
import os
import numpy as np
from datafilereader import DataFileReader
import pickle

# Path to the data file
file_path = "./data/Tests/coin_data.h5" 

#Set up features
featureListR = [17,20,21,22,26,28,31,32,39,42,44,71]
featureListL = [4,5,6,7,8,9,10,12,61]


labels = ["unknown", "5_CTS", "10_CTS", "20_CTS", "50_CTS", "1_CHF", "2_CHF", "5_CHF"]

# Load the pre-trained SVM model

with open("data/Tests/model1.pkl","rb+") as f:
    SVM = pickle.load(f)

with open("data/Tests/modelAn1.pkl","rb+") as f:
    ANO = pickle.load(f)


# Main function
def main(file_path):

    #SETUP
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
            

        time.sleep(2)  # Give user time to calibrate
        
    # TESTING
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
            p = R[0:9]/L
            #concatenate the features
            X = np.concatenate((R,L,p),axis=0)
            # reshape the array to 2D
            X = X.reshape(1, -1)
            result = ANO.predict(X)
            if result == -1:
                print("Coin is unknown")
            else:
                Y = SVM.predict(X)
                print("Coin is of type: ", labels[Y[0]])

        time.sleep(2)  # Give user time to mesure the coin to be tested
        
    
main(file_path)

# if __name__ == "__main__":
#     file_path = "./dat a/Tests/coin_data.h5" 
