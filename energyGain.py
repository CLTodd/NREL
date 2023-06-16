# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 13:03:40 2023

@author: ctodd
"""

#import pdb
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import re # can probably figure out a way to not use this
from flasc.dataframe_operations import dataframe_manipulations as dfm
 

class energyGain():
    
    def __init__(self, df, dfUpstream, testTurbines=[], refTurbines=[],
                 wdCol=None, wsCol=None):
        """
        testTurbines: list, turbine numbers to be considered test turbines
        refTurbines: list, turbine numbers to be considered reference turbines
        wdCol: string, name of the column in df to use for reference wind direction
            Calculates a column named "wd" if None
        wsCol: string, name of the column in df to use for reference wind speed
            Calculates a column named "ws" if None
        """
        
        self.df = df
        self.dfUpstream = dfUpstream
        self.testTurbines = testTurbines
        self.referenceTurbines = refTurbines
        self.allTurbines = [int(re.sub("\D+","",colname)) for colname in list(df) if re.match('^pow_\d+', colname)]
        self.wdCol = wdCol
        self.wsCol = wsCol
        
        # Set the columns to be referenced for wind speed and direction if not given   
        if self.wdCol == None:
            self.setWD() 
        if self.wsCol==None:
            self.setWS()

    def setWS(self, colname=None):
        """
        Setting the column to be referenced for wind speed in none was provided
        
        """
        if colname != None:
            self.wsCol = colname
            return None
        
        # Set-difference to find the list of turbines that should be excluded from this wind speed calculation
        exc = list(set(self.allTurbines) - set(self.referenceTurbines)) # should this be changed to allow something other than reference turbines 
        # Set reference wind speed and direction for the data frame
        self.df = dfm.set_ws_by_upstream_turbines(self.df, self.dfUpstream, exclude_turbs=exc)
        self.wsCol = "ws"
        return None
    
    def setWD(self, colname=None):
        """
        Setting the column to be referenced for wind direction if none was provided
        """
        
        if colname != None:
            self.wdCol = colname
            return None
        
        self.df = dfm.set_wd_by_all_turbines(self.df)
        self.wdCol = "wd"
        return None
    
    def powerRatio(self, windDirectionBin, windSpeedBin,controlMode,verbose=False):
        """
        Power ratio for a specific wind direction bin and wind speed bin. 
        
        windDirectionBin: list of length 2
        windSpeedBin: list of length 2
        controlMode: string, "baseline" or "controlled"
        wdToUse: string, name of the column with the reference wind direction.
            Calculates a column named "wd" if None
        wsToUse: string, name of the column with the reference wind speed.
            Calculates a column named "ws" if None
        """
        # Assuming valid inputs for now
        
        # Set wind speed if necessary
        if self.wsCol==None:
            self.setWS()
        
        # Set wind direction if necessary
        if self.wdCol==None:
            self.setWD()
        
        # Calculate Ratio
        numerator = self.averagePower(windDirectionBin,
                                      windSpeedBin,
                                      self.testTurbines,
                                      controlMode=controlMode)
        
        denominator = self.averagePower(windDirectionBin,
                                        windSpeedBin,
                                        self.referenceTurbines,
                                        controlMode=controlMode)
        # If either of these are strings, 
        # there are no observations in this bin to calculate a ratio from
        if type(numerator) is str:
            sadMessage = numerator + "Can't calculate power ratio numerator (average power)."
            if verbose:
                print(sadMessage)
            return sadMessage
        
        if type(denominator) is str:
            sadMessage = denominator + "Can't calculate power ratio denominator (average power)."
            if verbose:
                print(sadMessage)
            return sadMessage
        

        return numerator/denominator

    def averagePower(self, windDirectionBin,windSpeedBin, 
                     turbineList, controlMode, verbose=False):
        """
        Average Power for a specific wind direction bin and wind speed bin.
        
        windDirectionBin: list of length 2
        windSpeedBin: list of length 2
        controlMode: string, "baseline", "controlled", or "both"
        wdToUse: string, name of the column with the reference wind direction.
            Calculates a column named "wd" if None
        wsToUse: string, name of the column with the reference wind speed.
            Calculates a column named "ws" if None
        """
        
        # Set wind direction if necessary
        if self.wdCol==None:
            self.setWD()
            
        # Set wind speed if necessary
        if self.wsCol==None:
            self.setWS()
        
        # Select relevant rows
        dfTemp = self.df[ (self.df[self.wdCol]>= windDirectionBin[0]) &
                          (self.df[self.wdCol]< windDirectionBin[1]) &
                          (self.df[self.wsCol]>= windSpeedBin[0]) &
                          (self.df[self.wsCol]< windSpeedBin[1])
                        ]
        
        # Filter for control mode if necessary
        if controlMode != "both":
            dfTemp = dfTemp[(dfTemp['control_mode']==controlMode)]
                            
        # Select only the columns that are for the desired turbines
        # This only works for up to 1000 turbines, otherwise formatting gets messed up
        powerColumns = ["pow_{:03.0f}".format(number) for number in turbineList]
        dfPower = dfTemp[powerColumns]
        
        # If the data frame is empty then this returns NaN. 
        # This is an imperfect work around imo
        if dfPower.empty:
            sadMessage = f"No observations for turbines {turbineList} in {controlMode} mode for wind directions {windDirectionBin} and wind speeds {windSpeedBin}."
            if verbose:
                print(sadMessage)
            return sadMessage
        
        avg = dfPower.mean(axis=None, skipna=True, numeric_only=True)
     
        return avg
    
    def changeInPowerRatio(self, windDirectionBin, windSpeedBin, verbose=False):
        """
        Change in Power Ratio for a specific wind direction bin and wind speed bin.
        
        windDirectionBin: list of length 2
        windSpeedBin: list of length 2
        """
        
        # Set wind speed if necessary
        if self.wsCol ==None:
            self.setWS()
        
        # Set wind direction if necessary
        if self.wdCol ==None:
            self.setWD()
            
        Rpc = self.powerRatio(windDirectionBin, windSpeedBin, "controlled")
        Rpb = self.powerRatio(windDirectionBin, windSpeedBin, "baseline")
        
        # If either of these are strings, 
        # there are no observations in this bin to calculate a ratio from
        if type(Rpc) is str:
            sadMessage = Rpc + "Can't calculate power ratio for controlled mode."
            if verbose:
                print(sadMessage)
            return sadMessage
        
        if type(Rpb) is str:
            sadMessage = Rpb + "Can't calculate power ratio for baseline mode."
            if verbose:
                print(sadMessage)
            return sadMessage
        
        return Rpc - Rpb
        
    def percentPowerGain(self, windDirectionBin, windSpeedBin, verbose=False):
        
        """
        Percent Power Gain for a specific wind direction bin and wind speed bin.
        
        windDirectionBin: list of length 2
        windSpeedBin: list of length 2
        
        """
        
        # Set wind speed if necessary
        if self.wsCol==None:
            self.setWS()
        
        # Set wind direction if necessary
        if self.wdCol==None:
            self.setWD()
        
        Rpc = self.powerRatio(windDirectionBin, windSpeedBin, "controlled")
        Rpb = self.powerRatio(windDirectionBin, windSpeedBin, "baseline")
        
        # If either of these are strings, 
        # there are no observations in this bin to calculate a ratio from
        if type(Rpc) is str:
            sadMessage = Rpc + "Can't calculate power ratio for controlled mode."
            if verbose:
                print(sadMessage)
            return sadMessage
        
        if type(Rpb) is str:
            sadMessage = Rpb + "Can't calculate power ratio for baseline mode."
            if verbose:
                print(sadMessage)
            return sadMessage
        
        return (Rpc - Rpb)/Rpb
    
    def aepGain(self, windDirectionSpecs=[0,360,1], windSpeedSpecs=[0,20,1],
                    hours=8760, AEPmethod=1, absolute=False, useReference=True):
        
        """
        'Annual' energy production gain based on wind condition bin frequencies.
        
        windDirectionSpecs: list of length 3, specifications for wind direction
            bins-- [lower bound (inclusive), upper bound (exclusive), bin width]
        windSpeedSpecs: list of length 3, specifications for wind speed bins--
            [lower bound (inclusive), upper bound (exclusive), bin width]
        hours: numeric, defaults to number of hours in a year
        """
        # For frequency checks
        freqTracker=0
        
        # Which power columns are we interested in
        if useReference:
            powerColumns = [f'pow_{i:03d}' for i in self.testTurbines+self.referenceTurbines]
        else:
            powerColumns = [f'pow_{i:03d}' for i in self.testTurbines]
        
        # dataframe of only the power columns of interest and the corresponding
        # power observations for observations in the wind condition bin
        powerDF = self.df[list((self.df[self.wdCol]>= windDirectionSpecs[0]) &
                               (self.df[self.wdCol]< windDirectionSpecs[1]) &
                               (self.df[self.wsCol]>= windSpeedSpecs[0]) &
                               (self.df[self.wsCol]< windSpeedSpecs[1])),
                          powerColumns]
        # Use this to calculate frequencies later
        N = powerDF.size
        
        # Set up wind condition bins
        windDirectionBins = np.arange(windDirectionSpecs[0],
                                      windDirectionSpecs[1],
                                      windDirectionSpecs[2])
        
        windSpeedBins = np.arange(windSpeedSpecs[0],
                                  windSpeedSpecs[1],
                                  windSpeedSpecs[2])
        
        absoluteSum = 0
        normalizingConstant = 0
        for speed in windSpeedBins:
            
            # Calculate wind condition bin bounds
            upperSpeed = speed + windSpeedSpecs[2]
            
            for direction in windDirectionBins:
                
                # Calculate wind condition bin bounds
                upperDirection = direction + windDirectionSpecs[2]
                
                # Calculate wind condition bin frequency
                freq = powerDF[(self.df[self.wdCol]>= direction) &
                               (self.df[self.wdCol]< upperDirection) &
                               (self.df[self.wsCol]>= speed) &
                               (self.df[self.wsCol]< upperSpeed)
                              ].size/N
                
                freqTracker += freq
                
                # If we aren't using reference turbines, the absolute and 
                # relative AEP formulas are the same for both methods
                if not useReference:
                    # Set average power to one
                    avgPwr = 1
                    
                    # Compute change metric
                    avgPwrTestContr = self.averagePower([direction, upperDirection],
                                                    [speed, upperSpeed],
                                                    self.testTurbines, "controlled")
                    avgPwrTestBase = self.averagePower([direction, upperDirection],
                                                   [speed, upperSpeed],
                                                   self.testTurbines, "baseline")
                    changeMetric = avgPwrTestContr - avgPwrTestBase
                    
                # With Reference turbines, summation formula differs depending on AEP method
                elif AEPmethod==1:
                    changeMetric = self.percentPowerGain([direction, upperDirection],
                                                         [speed, upperSpeed])
                
                    avgPwr = self.averagePower([direction, upperDirection],
                                               [speed, upperSpeed],
                                               self.testTurbines,
                                               "baseline")
                else:# (If AEPmethod==2)
                    changeMetric = self.changeInPowerRatio([direction, upperDirection],
                                                           [speed, upperSpeed])
                    
                    avgPwr = self.averagePower([direction, upperDirection],
                                               [speed, upperSpeed],
                                               self.testTurbines,
                                               "both")
                
                # Think about what's the best way to handle lack of data in either baseline or control
                if (freq==0) or (type(avgPwr) is str) or (type(changeMetric) is str):
                    continue
                
                # Increment the absolute sum
                absoluteSum += ( freq * changeMetric * avgPwr)
                
                 
                if not absolute:
                    # Increment the normalizing constant for % gain AEP
                    if not useReference:
                        # If we aren't using reference turbines, % gain AEP is the same for both AEP methods
                        normalizingConstant += (freq*avgPwrTestBase)
                    elif AEPmethod==1:
                        normalizingConstant += (freq*changeMetric)
                    else: #(AEPmethod==2)
                        pwrRatio = self.powerRatio([direction, upperDirection],
                                                   [speed, upperSpeed],"baseline")
                        normalizingConstant += (freq*pwrRatio*avgPwr)
        
        print(freqTracker) # frequency check
        
        # Percent AEP gain results
        if normalizingConstant == 0:
            # This seems like an imperfect solution
            print("Normalizing constant is 0, returning absolute AEP instead")
        else:
            pctGainAEP = 100*(absoluteSum/normalizingConstant)
            print(f"{pctGainAEP}%")
            return pctGainAEP
        
        # Absolute AEP Gain results
        aepGain = hours*absoluteSum
        print(f"{aepGain} kWh")
        return aepGain
    
    # Change this name later
    # pct power gain
    # I do not think this works right now
    def plot(self, windDirectionSpecs=[0,360,1], windSpeedSpecs=[0,20,1]):
        """
        windDirectionSpecs: list of length 3, specifications for wind direction
            bins-- [lower bound (inclusive), upper bound (exclusive), bin width]
        windSpeedSpecs: list of length 3, specifications for wind speed bins--
            [lower bound (inclusive), upper bound (exclusive), bin width]
        """
        windDirectionBins = np.arange(windDirectionSpecs[0],
                                      windDirectionSpecs[1],
                                      windDirectionSpecs[2])
        windSpeedBins = np.arange(windSpeedSpecs[0],
                                  windSpeedSpecs[1],
                                  windSpeedSpecs[2])
        y=np.empty(shape=0, dtype=float)
        x=np.empty(shape=0, dtype=float)
        for speed in windSpeedBins:
            
            upperSpeed = speed + windSpeedSpecs[2]
            # Not sure this is how I should do the x's
            x_i = np.mean([speed, upperSpeed])
            y_i = np.empty(shape=0, dtype=float)
            
            for direction in windDirectionBins:
                upperDirection = direction + windDirectionSpecs[2]
                
                y_ij = self.percentPowerGain([direction, upperDirection],
                                           [speed, upperSpeed])
                if type(y_ij) is str:
                    continue
                
                y_i = np.append(y_i, y_ij)
            
            y = np.append(y, np.mean(y_i))
            x = np.append(x, x_i)
        
        plt.plot(x,y)
        plt.show()
        return None
    
    # This will need to be updated if the metrics are 
    # changed to return something other than a string when it can't be computed
    # for a particular wind condition bin
    def matrixOfMetrics(self, metricMethod, windDirectionSpecs=[0,360,1], windSpeedSpecs=[0,20,1]):
        """
        For computing wind-condition-bin-specific metrics for many bins at once
        
        metricMethod: Method, the method you want to use to compute the 
            measurements in each wind condition bin
        windDirectionSpecs: list of length 3, specifications for wind direction
            bins-- [lower bound (inclusive), upper bound (exclusive), bin width]
        windSpeedSpecs: list of length 3, specifications for wind speed bins--
            [lower bound (inclusive), upper bound (exclusive), bin width]
        """
        
        # Get the bounds for each wind condition bin
        windDirectionBins = np.arange(windDirectionSpecs[0],
                                      windDirectionSpecs[1],
                                      windDirectionSpecs[2])
        windSpeedBins = np.arange(windSpeedSpecs[0],
                                  windSpeedSpecs[1],
                                  windSpeedSpecs[2])
        
        # Initialize 'empty' matrix (matrix of Nones)        
        nCol = windDirectionBins.size
        nRow = windSpeedBins.size
        dataMatrix = np.full((nRow, nCol), None, dtype=float)
        
        # Going through the matrix
        for j in range(nCol):
            direction = windDirectionBins[j]
            upperDirection = direction + windDirectionSpecs[2]
            
            for i in range(nRow):
                speed = windSpeedBins[i]
                upperSpeed = speed + windSpeedSpecs[2]
                
                y_ij = metricMethod([direction, upperDirection],
                                   [speed, upperSpeed])
                
                # If the metric couldn't be computed for a certain bin, 
                # it would have returned a string
                if type(y_ij) is str:
                    y_ij = None
                
                # Store the measurement
                dataMatrix[i,j] = y_ij
        
        return {"data": dataMatrix, "directions": windDirectionBins, "speeds": windSpeedBins}
         
    # Change this name later, probably
    def heatmap(self, windDirectionSpecs=[0,360,1], windSpeedSpecs=[0,20,1], 
                heatmapMatrix=None, colorMap="seismic", title="Percent Power Gain"):
        """
        For plotting a heatmap of some metric over many wind condition bins
        
        windDirectionSpecs: list of length 3, specifications for wind direction
            bins-- [lower bound (inclusive), upper bound (exclusive), bin width]
        windSpeedSpecs: list of length 3, specifications for wind speed bins--
            [lower bound (inclusive), upper bound (exclusive), bin width]
        heatmapMatrix: Matrix, can pass in a matrix of  data 
            Right now the default is to compute a matrix od percent power gain 
            measurements if no matrix is provided.
        colorMap: any color mapping that works with matplotlib. 
            Diverging color maps and colormaps that exclude white are recommended.
        title: string,
        """

        # Compute matrix of data if needed
        if heatmapMatrix is None:
            print("Computing matrix of Percent Power Gain measurements")
            heatmapMatrix = self.matrixOfMetrics(self.percentPowerGain,
                                                 windDirectionSpecs,
                                                 windSpeedSpecs)["data"]
            
        fig, ax = plt.subplots()
        heatmap = plt.imshow(heatmapMatrix, cmap=colorMap, 
                             interpolation=None, origin='lower')
        
        # Tick marks at multiples of 5 and 1
        ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
        ax.yaxis.set_major_locator(mticker.MultipleLocator(5))
        ax.xaxis.set_minor_locator(mticker.MultipleLocator(1)) 
        ax.yaxis.set_minor_locator(mticker.MultipleLocator(1))
        
        # Labels
        ax.set_title(title)
        ax.set_xlabel(u"Wind Direction (\N{DEGREE SIGN})") #unicode formatted
        ax.set_ylabel("Wind Speed (m/s)")

        fig.colorbar(heatmap) # legend

        plt.show()
        
        return heatmapMatrix
        
    
    
    
    
    
    