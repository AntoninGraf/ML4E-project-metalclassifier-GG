"""
Cours :       MachLearn - Inductive Smart Sensors - Projet
Created : 2024

Description : Read the content of a data file containing impedance measurements
"""

import time
import h5py
from util.settings import Dtype, Settings
import numpy as np
import os
import numpy as np
import matplotlib.pyplot as plt

class DataFileReader:
    """
    Class to read and write HDF5 files containing impedance measurements

    """
    
    DEFAULT_N_SAMPLES = 500
    FREQ_MIN = 0
    FREQ_MAX = 200e3
    LIMIT_FREQ = True
    
    def __init__(self, filename, mode='r', max_str_len=50):   # Mode : 'r' to read, 'a' to edit, 'w' to write
        """
        Constructor
        
        Arguments:
        filename -- path and filename : str
        mode -- 'r' to read, 'a' to edit, 'w' to write : str
        """
        self.__settings = Settings.settings
        self.__filename = filename
        
        # Max metadata length (if string)
        self.__str_len = max_str_len
        
        f = h5py.File(self.__filename,mode)
        
        # Create liste of measurement metadata
        self.__mes_metadata_keys = {           # List of measurement metadata that are not received from serial port
            'Measurement description' : Dtype.string,
            'Measurement UID' :         Dtype.uint32,
            'Date and time' :           Dtype.string,
            'Software version' :        Dtype.string
        }
        # Add metadata that are received from the serial port
        for key in list(self.__settings['Metadata labels']['com'].keys()):
            self.__mes_metadata_keys[key] = self.__settings['Metadata labels']['com'][key]
            
        # Create compound type
        self.__metadata_dtypes = {}
        self.__metadata_dtypes['names'] = []
        self.__metadata_dtypes['formats'] = []

        metadata_com_labels = self.__settings['Metadata labels']['com']
        
        for key in list(self.__mes_metadata_keys.keys()):
            self.__metadata_dtypes['names'].append(key)
            if key in list(metadata_com_labels.keys()):
                if metadata_com_labels[key] == Dtype.uint16:
                    self.__metadata_dtypes['formats'].append(('<i2', (1,)))
                elif metadata_com_labels[key] == Dtype.uint96:
                    self.__metadata_dtypes['formats'].append(('<i4', (3,)))
                elif metadata_com_labels[key] == Dtype.float32:
                    self.__metadata_dtypes['formats'].append(('<f4', (1,)))
                elif metadata_com_labels[key] == Dtype.string:
                    self.__metadata_dtypes['formats'].append(('i1', (self.__str_len,)))
                elif metadata_com_labels[key] == Dtype.bool:
                    self.__metadata_dtypes['formats'].append(('i1', (1,)))
            else:
                if(list(f.keys()) != []):
                    if(len(f['metadata'][0][key]) > self.__str_len):
                        self.__str_len = len(f['metadata'][0][key])
                if self.__mes_metadata_keys[key] == Dtype.string:
                    self.__metadata_dtypes['formats'].append(('i1', (self.__str_len,)))
                elif self.__mes_metadata_keys[key] == Dtype.uint32:
                    self.__metadata_dtypes['formats'].append(('i4', (1,)))
        
        f.close()
        
        # get the current time stamp of the file
        self.current_modification_timestamp = self.__get_current_modification_timestamp()
        
    # --- public mehtods
    def get_reference_impedance_index(self):
        """
        Get the indexes of the reference measurements

        Returns:
        Reference indexes : 1D numpy array
        """

        ref_index = np.asarray([n for n in range(self.__get_N_mes()) if self.__is_reference(n)])
        if ref_index.size == 0:
            return None
        return ref_index
    
    def get_all_mesurements(self):
        """
        Get all measurements in file

        Returns:
        frequency indices of computed impedance : 1D numpy array
        computed impedance : 2D numpy array of complex numbers. First dimension represents the measurement index,
        and second dimension represents the frequency index

        """

        frequency,Z = self.__get_impedance_matrix(self.LIMIT_FREQ)
        return frequency,Z
    
    def get_last_mesurement(self):
        """
        Get the last measurement

        Returns:
        frequency indices of computed impedance : 1D numpy array
        computed impedance : 1D numpy array of complex numbers
        is_reference : bool
        
        """
        metadata,vd,vs,frequency,Z = self.__get_mes(-1)
        if self.LIMIT_FREQ:
            idx_f = np.logical_and(frequency > self.FREQ_MIN, frequency < self.FREQ_MAX)
        
        is_reference = self.__is_reference(self.__get_N_mes()-1)
        
        return frequency[idx_f],Z[idx_f],is_reference
    
    def has_file_changed(self):
        # determine if the file has changed
        current_timestamp = os.path.getmtime(self.__filename)
        haschanged = current_timestamp != self.current_modification_timestamp
        self.current_modification_timestamp = current_timestamp
        return haschanged
    
    # --- private mehtods
    def __get_impedance_matrix(self, limit_f=True):
        """
        Get all impedances in file, in a single matrix
        
        Returns:
        Frequency indices of computed impedance : 1D numpy array
        Computed impedances : 2D numpy array of complex numbers. First dimension represents the measurement index,
        and second dimension represents the frequency index
        """
        
        f = h5py.File(self.__filename, 'r')
        if list(f.keys()) == []:
            return [],[]
        
        mes_existing = f['existing'][:].astype(bool)
        
        frequency = f['impedance'][0,0,:]
        Z_array = f['impedance'][mes_existing,1:,:]
        
        Z = Z_array[:,0,:]+1j*Z_array[:,1,:]
        
        f.close()
        
        if limit_f:
            idx_f = np.logical_and(frequency > self.FREQ_MIN, frequency < self.FREQ_MAX)
            frequency = frequency[idx_f]
            Z = Z[:,idx_f]
        
        return frequency,Z
       
    def __is_reference(self, n):
        """
        Returns the given measurement's 'Reference' attribute
        
        Argument:
        n -- measurement index : int
        
        Returns:
        Reference : bool
        """
        return self.__get_metadata(n)['Reference'] != 0
    
        
    def __get_mes(self, n):
        """
        Get measurement data, by index
        
        Argument:
        n -- measurement index : int
        
        Returns:
        metadata : dict
        vd : 1D-numpy array
        vs : 1D-numpy array
        frequency indices of computed impedance : 1D numpy array
        computed impedance : 1D numpy array of complex numbers
        """
        
        f = h5py.File(self.__filename, 'r')
        
        mes_existing = f['existing'][:].astype(bool)
        n_ = np.where(mes_existing)[0][n]
        
        rawdata = f['rawdata'][n_]
        vd = rawdata[0,:]
        vs = rawdata[1,:]
        
        Z_array = f['impedance'][n_]
        frequency = Z_array[0,:]
        Z = Z_array[1,:]+1j*Z_array[2,:]

        metadata_raw = f['metadata'][n_]
        metadata = {}

        for meta_key in list(self.__mes_metadata_keys.keys()):
            num_key = self.__metadata_dtypes['names'].index(meta_key)
            if(self.__mes_metadata_keys[meta_key] == Dtype.string):
                # Convert stored metzadata to string
                meta_str = metadata_raw[num_key][:np.where(metadata_raw[num_key] == 0)[0][0]]
                metadata[meta_key] = meta_str.tobytes().decode('utf-8')
            else:
                if(meta_key == 'Device UID'):
                    metadata[meta_key] = (int(metadata_raw[num_key][0]) << 64) + (int(metadata_raw[num_key][1]) << 32) + int(metadata_raw[num_key][2])
                else:
                    metadata[meta_key] = metadata_raw[num_key][0]
                    
        f.close()

        return metadata,vd,vs,frequency,Z
    
    def __get_metadata(self,n):
        """
        Get measurement metadata, by index
        
        Argument:
        n -- measurement index : int
        
        Returns:
        metadata : dict
        """
        f = h5py.File(self.__filename, 'r')
        
        mes_existing = f['existing'][:].astype(bool)
        n_ = np.where(mes_existing)[0][n]
        
        metadata_raw = f['metadata'][n_]
        
        metadata = {}

        for meta_key in list(self.__mes_metadata_keys.keys()):
            if(self.__mes_metadata_keys[meta_key] == Dtype.string):
                # Convert stored metzadata to string
                meta_str = metadata_raw[meta_key][:np.where(metadata_raw[meta_key] == 0)[0][0]]
                metadata[meta_key] = meta_str.tobytes().decode('utf-8')
            else:
                if(meta_key == 'Device UID'):
                    metadata[meta_key] = (int(metadata_raw[meta_key][0]) << 64) + (int(metadata_raw[meta_key][1]) << 32) + int(metadata_raw[meta_key][2])
                else:
                    metadata[meta_key] = metadata_raw[meta_key][0]
                    
        f.close()
        return metadata
    
    def __get_N_mes(self):
        """
        Get number of measurments in file
        
        Returns:
        Number of measurments : int
        """
        f = h5py.File(self.__filename, 'r')
        if(list(f.keys()) == []):
            N = 0
        else:
            N = np.sum(f['existing'][:])
        f.close()
        return N
    
    def __get_current_modification_timestamp(self):
        return os.path.getmtime(self.__filename)