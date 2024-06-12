"""
Cours :       MachLearn - Metal classifier project
Created : 2024

Description : This file is to demonstrate how to plot the data file from ISS data file
"""
import os
import os.path
import argparse
import matplotlib.pyplot as plt
from dev.datafileviewer_template import DataFileViewer
import matplotlib

if __name__ == "__main__":
    
    DEFAULT_FILE_PATH = "./data/dataSetAGF/5CHF-2pce.h5"
    
    # create the parser
    parser = argparse.ArgumentParser(description='Display the impedance data from a file')
    parser.add_argument('--file_path',nargs='?', type=str, help='path to the file to display')
    args = parser.parse_args()
    file_path = args.file_path

    print("file_path : ", file_path)

    # check if the file exists
    if file_path is None:
        file_path = DEFAULT_FILE_PATH

    print("file_path : ", file_path)
    
   # check if the file exists
    if not os.path.isfile(file_path):
        raise ValueError("File does not exist")
    
    # Start the viewer to display the content of the file
    imdisp = DataFileViewer(file_path)
    plt.show() # Display the figure and keep it open

    
    