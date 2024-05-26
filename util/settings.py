
"""
Created : 2022

Description : Project settings
"""

from enum import Enum

# Metadata types
class Dtype(Enum):
    uint16 = 0
    uint96 = 1
    float32 = 2
    string = 3
    bool = 4
    uint32 = 5

class Settings:
    __Nbits = 12        # Hardware configuration
    __Baudrate = 115200 #    

    __sw_version = '1.0'

    __metadata_file = {     # File metadata
        # key              : editable
        'User name'        : True,
        'File description' : True
    }

    __metadata_mes_com = {          # Metadata known from serial com
        # key                     : data type
        'Reference'               : Dtype.bool,
        'Number of points'        : Dtype.uint16,
        'Number of sequences'     : Dtype.uint16,
        'Shunt resistor [Ohm]'    : Dtype.uint16,
        'Reference voltage [V]'   : Dtype.float32,
        'Sampling frequency [Hz]' : Dtype.float32,
        'Device UID'              : Dtype.uint96,
        'Hardware version'        : Dtype.string,
        'Firmware version'        : Dtype.string
    }

    __metadata_labels = {
        'file' : __metadata_file,
        'com'  : __metadata_mes_com
    }

    settings = {
        'Nbits' :            __Nbits,
        'Baudrate' :         __Baudrate,
        'Metadata labels' :  __metadata_labels,
        'Software version' : __sw_version
    }
