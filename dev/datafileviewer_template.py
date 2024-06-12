"""
Cours :       MachLearn - Metal classifier project
Created : 2024

Description : Displaying the content of a data file containing impedance measurements
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from dev.datafilereader import DataFileReader


class DataFileViewer():
    """
        Class to display the data file from the ISS
        
    """
    
    def __init__(self, file_path):
        """
            Constructor
        
            Arguments:
            file_path -- path and filename of the data file : str
        """
           
        # determine if the file is available
        check_file = os.path.isfile(file_path)
        if not check_file:
            raise ValueError("File does not exist")
        
        # set the attributes
        self.file_path = file_path
        self.dfr = DataFileReader(file_path)

        # create the figure and the interval of update
        self.fig, self.plots_axes = self.__create_figure()
        self.update_interval = 1000  # Update interval in milliseconds (1000 ms = 1 second)
        self.animation = FuncAnimation(self.fig, self.update_plot, interval=self.update_interval,cache_frame_data=False)
        self.plot()

       
        
    def plot(self):
        """
            Display the data file in a figure
        
        """
        # get data
        frequency,Z = self.dfr.get_all_mesurements()
        ref_idx = self.dfr.get_reference_impedance_index()
        self.__plot_figure(frequency,Z,ref_idx)

        
    def update_plot(self,frame):
        """
            Update the figure if the file has changed

            Arguments:
            frame -- frame number : int
        
        """
        # update the plot if the file has changed
        haschanged = self.dfr.has_file_changed()
        #print("Checking for changes : ,", haschanged, " at ", time.time())
        if haschanged:
            print("File has changed, updating plot")
            frequency,Z,is_reference = self.dfr.get_last_mesurement()
            ref_idx = 0 if is_reference else None # if the last measurement is a reference set the corresponding index 0
            
            self.__plot_figure(frequency,Z,ref_idx)
            
        return haschanged
    
    # ----------------- private methods -------------------------
    def __plot_figure(self, frequency,Z,ref_idx=None):
        """
            plot the figure

            Arguments:
            frequency -- frequency array : np.array
            Z -- impedance array : np.array
            ref_idx -- index of the reference impedance : int
        
        """
        # plot the data
        if len(Z) > 0:
            self.__plot_impedance(frequency,Z,ref_idx)
            self.__higlight_last()
        else:
            self.plots_axes['R'].plot(0,0) # plot something empty otherwise the animation doesn't call the update function
        plt.pause(0.01) # add small pause otherwise the plot is not refreshed
        
    def __higlight_last(self):
        """
            Higlight the last lines in red
        
        """
        # highlight the last line
        for ax in self.plots_axes.values():
            lines = ax.get_lines()
            line = lines[-1]
            line.set_color('red')
            line.set_linewidth(2)
            # remove the previous one
            if(len(lines)>1):
                line = lines[-2]
                line.set_color('blue')
                line.set_linewidth(1)
      
    def __plot_impedance(self,frequency,Z,ref_idx=None):
        """
            Plot the impedance on the figure

            Arguments:
            frequency -- frequency array : np.array
            Z -- impedance array : np.array
            ref_idx -- index of the reference impedance : int
        
        """
        # check if Z is a 1D array (1 impedance measurement) and add a dimension
        if len(Z.shape) == 1:
            Z = Z [None,:] # add a dimension

        # extract the reference impedance if available and plot it
        if ref_idx is not None:
            mask = np.ones(Z.shape[0], dtype=bool)
            mask[ref_idx] = False
            Zdata = Z[mask,:]
            Z0 = Z[ref_idx,:]

            self.plots_axes['R'].plot(frequency,np.real(Z0).T,'b--')
            self.plots_axes['L'].plot(frequency,(np.imag(Z0)/(2*np.pi*frequency)*1e6).T,'b--')
        else:
            Zdata = Z

        # plot the impedance
        self.plots_axes['R'].plot(frequency,np.real(Zdata).T,'b')
        self.plots_axes['L'].plot(frequency,(np.imag(Zdata)/(2*np.pi*frequency)*1e6).T,'b')
        

    def __create_figure(self):
        """
            Create the figure
        
        """
        # create the figure
        plt.show(block=False) # Display the plot window in non-blocking mode
        fig, ax = plt.subplots(2, 1, figsize=(10, 8))
        plots_axes = {}
        plots_axes['R'] = ax[0]
        plots_axes['L'] = ax[1]

        plots_axes['L'].set_xlabel('Frequency (Hz)')
        plots_axes['R'].set_ylabel('R [Ohm]')
        plots_axes['L'].set_ylabel('L [uH]')
        plots_axes['R'].set_title('Impedance vs Frequency')
        plt.pause(0.01)

        return fig, plots_axes