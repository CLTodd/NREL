# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 13:03:40 2023

@author: ctodd
"""

import pdb
import matplotlib.pyplot as plt
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
    
    def changeInAEP(self, windDirectionSpecs=[0,360,1], windSpeedSpecs=[0,20,1],
                    hours=8760, AEPmethod=1, absolute=False, useReference=True):
        
        """
        'Annual' energy production based on wind condition bin frequencies.
        
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
                
                # Summation formula differs depending on AEP method
                if AEPmethod==1:
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
                
                # Increment the normalizing constant for % change AEP  
                if AEPmethod==1 and absolute==False:
                    normalizingConstant += (freq*changeMetric)
                elif AEPmethod==2 and absolute==False:
                    pwrRatio = self.powerRatio([direction, upperDirection],
                                               [speed, upperSpeed],"baseline")
                    normalizingConstant += (freq*pwrRatio*avgPwr)
        
        print(freqTracker) # frequency check
        
        # Percent change AEP
        if not absolute:
            if normalizingConstant == 0:
                # This seems like an imperfect solution
                print("Normalizing constant is 0, returning absolute AEP instead")
            else:
                pctAEP = 100*(absoluteSum/normalizingConstant)
                print(f"{pctAEP}%")
                return pctAEP
        
        # Absolute change AEP
        AEP = hours*absoluteSum
        print(f"{AEP} kWh")
        return AEP
    
    # Change this name later
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
    
    # Better comments later
    def matrixOfMetrics(self, metricMethod, windDirectionSpecs=[0,360,1], windSpeedSpecs=[0,20,1]):
        """
        For wind-condition-bin-specific metrics
        """
        
        windDirectionBins = np.arange(windDirectionSpecs[0],
                                      windDirectionSpecs[1],
                                      windDirectionSpecs[2])
        windSpeedBins = np.arange(windSpeedSpecs[0],
                                  windSpeedSpecs[1],
                                  windSpeedSpecs[2])
        I = windDirectionBins.size
        J = windSpeedBins.size
        dataMatrix = np.full((I, J), None, dtype=float)
        
        for i in range(I):
            direction = windDirectionBins[i]
            upperDirection = direction + windDirectionSpecs[2]
            
            for j in range(J):
                speed = windSpeedBins[j]
                upperSpeed = speed + windSpeedSpecs[2]
                
                y_ij = metricMethod([direction, upperDirection],
                                   [speed, upperSpeed])
                if type(y_ij) is str:
                    y_ij = None
                
                dataMatrix[i,j] = y_ij
        
        return {"data": dataMatrix, "directions": windDirectionBins, "speeds": windSpeedBins}
        
    
    # Change this name later
    # this does NOT work right now
    def heatmap(self, windDirectionSpecs=[0,360,1], windSpeedSpecs=[0,20,1], cmap='hot'):
        """
        windDirectionSpecs: list of length 3, specifications for wind direction
            bins-- [lower bound (inclusive), upper bound (exclusive), bin width]
        windSpeedSpecs: list of length 3, specifications for wind speed bins--
            [lower bound (inclusive), upper bound (exclusive), bin width]
        """
        
        resultDict = self.matrixOfMetrics(self.percentPowerGain,
                                          windDirectionSpecs,
                                          windSpeedSpecs)
        
        heatmapMatrix = resultDict["data"] 
        I = heatmapMatrix.shape[0]
        J = heatmapMatrix.shape[1]
        
        windDirectionBins = resultDict["directions"]
        windSpeedBins = resultDict["speeds"]
            
        fig, ax = plt.subplots()
        heatmap = ax.imshow(heatmapMatrix, cmap=cmap, interpolation=None, origin='lower')
        fig.colorbar(heatmap) # legend
        
        xTicks = np.arange(0,I+1,1)-0.5
        plt.xticks(xTicks)
        xLabels = np.append(windDirectionBins, windDirectionSpecs[1])
        ax.set_xticklabels(xLabels)
        
        #breakpoint()
        
        yTicks = np.arange(0,J+1,1)-0.5
        plt.yticks(yTicks)
        yLabels = np.append(windSpeedBins, windSpeedSpecs[1])
        ax.set_yticklabels(yLabels)
        
        plt.show()
        
        
        
        return heatmapMatrix
        
    
    
    
    
    
    