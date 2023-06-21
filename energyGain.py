# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 13:03:40 2023

@author: ctodd
"""

#import pdb
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from timeit import default_timer
import re # can probably figure out a way to not use this
from flasc.dataframe_operations import dataframe_manipulations as dfm
 

class energyGain():
    
    def __init__(self, df, dfUpstream, testTurbines=[], refTurbines=[],
                 wdCol=None, wsCol=None, useReference=True):
        """
        testTurbines: list, turbine numbers to be considered test turbines
        refTurbines: list, turbine numbers to be considered reference turbines
        wdCol: string, name of the column in df to use for reference wind direction
            Calculates a column named "wd" if None
        wsCol: string, name of the column in df to use for reference wind speed
            Calculates a column named "ws" if None
        useReference: Boolean, wheter to compare Test turbines to Reference 
            turbines (True) or Test turbines to themselves in control mode
            versus baseline mode (False).
        """
        #breakpoint()
        self.df = df
        self.dfUpstream = dfUpstream
        self.testTurbines = testTurbines
        self.referenceTurbines = refTurbines
        self.wdCol = wdCol
        self.wsCol = wsCol
        self.useReference = useReference
        # Set the columns to be referenced for wind speed and direction if not given   
        if self.wdCol == None:
            self.setWD() 
        if self.wsCol==None:
            self.setWS()
            
        # quick fix for now
        if df is not None:
            self.allTurbines = [int(re.sub("\D+","",colname)) for colname in list(df) if re.match('^pow_\d+', colname)]
        else:
            self.allTurbines = None
            

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
    
    def powerRatio(self, windDirectionBin, windSpeedBin, controlMode=None, 
                   useReference = True, verbose = False):
        """
        Power ratio for a specific wind direction bin and wind speed bin. 
        
        windDirectionBin: list of length 2
        windSpeedBin: list of length 2
        controlMode: string, "baseline" or "controlled"
        wdToUse: string, name of the column with the reference wind direction.
            Calculates a column named "wd" if None
        wsToUse: string, name of the column with the reference wind speed.
            Calculates a column named "ws" if None
        useReference: Boolean, wheter to compare Test turbines to Reference 
            turbines (True) or Test turbines to themselves in control mode
            versus baseline mode (False). Used for some methods.
        """
        # Assuming valid inputs for now
        
        # Wanted to add this to give flexibility to not use the reference for 
        # one particular method, but part of me feels like this is just confusing 
        # or a bad idea. Might take it away and just always use the object attribute
        if useReference is None:
            useReference = self.useReference
        
        # Set wind speed if necessary
        if self.wsCol==None:
            self.setWS()
        
        # Set wind direction if necessary
        if self.wdCol==None:
            self.setWD()
        
        # Calculate Ratio
        numerator = self.averagePower(windDirectionBin, windSpeedBin,
                                      self.testTurbines, controlMode=controlMode)
        
        if useReference:
            denominator = self.averagePower(windDirectionBin, windSpeedBin,
                                            self.referenceTurbines, controlMode=controlMode)
        else:
            denominator = 1
            print("Reference turbines unused; calculating average power.")
            
        
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
    
    def changeInPowerRatio(self, windDirectionBin, windSpeedBin, useReference=None, verbose=False):
        """
        Change in Power Ratio for a specific wind direction bin and wind speed bin.
        
        windDirectionBin: list of length 2
        windSpeedBin: list of length 2
        """
        if useReference is None:
            useReference = self.useReference
            
        # Set wind speed if necessary
        if self.wsCol ==None:
            self.setWS()
        
        # Set wind direction if necessary
        if self.wdCol ==None:
            self.setWD()

        if useReference:
            # Typical power ratio formula if we are using reference turbines
            control = self.powerRatio(windDirectionBin, windSpeedBin, "controlled", useReference=True)
            baseline = self.powerRatio(windDirectionBin, windSpeedBin, "baseline", useReference=True)
        else:
            control = self.powerRatio(windDirectionBin, windSpeedBin, "controlled", useReference=False)
            baseline = self.powerRatio(windDirectionBin, windSpeedBin, "baseline", useReference=False)
            
            # I think this is important so I'm printing this regardless of verbose
            FYI = "Change in power ratio is simply change in average power without reference turbines.\n"
            FYI += "Returning change in average power. If this isn't what you want, set the useReference argument to True."        
            print(FYI)
 
        # If either of these are strings, 
        # there are no observations in this bin to calculate a ratio from
        if type(control) is str:
            sadMessage = control + "Can't calculate power ratio for controlled mode."
            if verbose:
                print(sadMessage)
            return sadMessage
        
        if type(baseline) is str:
            sadMessage = baseline + "Can't calculate power ratio for baseline mode."
            if verbose:
                print(sadMessage)
            return sadMessage
        
        return control - baseline
        
    def percentPowerGain(self, windDirectionBin, windSpeedBin, useReference=None, verbose=False):
        
        """
        Percent Power Gain for a specific wind direction bin and wind speed bin.
        
        windDirectionBin: list of length 2
        windSpeedBin: list of length 2
        useReference: Boolean, wheter to compare Test turbines to Reference 
            turbines (True) or Test turbines to themselves in control mode
            versus baseline mode (False).
        """
        
        # Wanted to add this to give flexibility to not use the reference for 
        # one particular method, but part of me feels like this is just confusing 
        # or a bad idea. Might take it away and just always use the object attribute
        if useReference is None:
            useReference = self.useReference
        
        # Set wind speed if necessary
        if self.wsCol==None:
            self.setWS()
        
        # Set wind direction if necessary
        if self.wdCol==None:
            self.setWD()
            
        # If useReference==False, this simplifies to average power
        control = self.powerRatio(windDirectionBin, windSpeedBin, "controlled", useReference)
        baseline = self.powerRatio(windDirectionBin, windSpeedBin, "baseline", useReference)
        
        # If either of these are strings, 
        # there are no observations in this bin to calculate a ratio or average power from
        if type(control) is str:
            sadMessage = control + "Can't calculate power ratio for controlled mode."
            if verbose:
                print(sadMessage)
            return sadMessage
        
        if type(baseline) is str:
            sadMessage = baseline + "Can't calculate power ratio for baseline mode."
            if verbose:
                print(sadMessage)
            return sadMessage
        
        return (control - baseline)/baseline
    
    def aepGain(self, windDirectionSpecs=[0,360,1], windSpeedSpecs=[0,20,1],
                    hours=8760, AEPmethod=1, absolute=False, useReference=None):
        
        """
        'Annual' energy production gain based on wind condition bin frequencies.
        
        windDirectionSpecs: list of length 3, specifications for wind direction
            bins-- [lower bound (inclusive), upper bound (exclusive), bin width]
        windSpeedSpecs: list of length 3, specifications for wind speed bins--
            [lower bound (inclusive), upper bound (exclusive), bin width]
        hours: numeric, defaults to number of hours in a year
        AEPmethod: integer, the method used to calculate AEP;
            either 1 or 2 based on the order they appear on the slides
        absolute: Boolean, whether to compute absolute AEP gain (True) or 
            percent AEP gain (False)
        useReference: Boolean, wheter to compare Test turbines to Reference 
            turbines (True) or Test turbines to themselves in control mode
            versus baseline mode (False).
        """
        # For frequency checks
        freqTracker=0
        
        # Wanted to add this to give flexibility to not use the reference for 
        # one particular method, but part of me feels like this is just confusing 
        # or a bad idea. Might take it away and just always use the object attribute
        if useReference is None:
            useReference = self.useReference
        
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
    
    # Works-ish but not done, se not implemented
    # This so far cannot accept aep methods
    def compute1D(self, metricMethod, windDirectionSector=[0,360], 
                  windSpeedRange=[0,20], by=1, dim="direction", verbose=False):
        """
        For computing metrics as a function of either wind or direction 
        (but not both).
        
        metricMethod: Method, the method you want to use to compute the 
            measurements in each wind condition bin
        windDirectionSector: list of length 2, specifications for wind direction
            bins-- [lower bound (inclusive), upper bound (exclusive)]
        windSpeedRange: list of length 2, specifications for wind speed bins--
            [lower bound (inclusive), upper bound (exclusive)]
        dim: string, the varaible you want to compute the metric as a function of
            (dim for dimension as in 1D). Either "direction" or "speed".
        by: step size for the bins for dim
        """
        
        if verbose:
            print("Computing as a function of" + dim)
            
        # My way of dynamically allowing you to choose dim
        # not a huge fan of the way I did this tbh but it works for now
        if dim == "direction":
            windBins = np.arange(windDirectionSector[0],
                                 windDirectionSector[1],
                                 by)
            scope = [windSpeedRange[0], windSpeedRange[1]]
            
        else:
            windBins = np.arange(windSpeedRange[0],
                                 windSpeedRange[1],
                                 by)
            scope = [windDirectionSector[0], windDirectionSector[1]]
            
        y = np.empty(shape=0, dtype=float)
        se = np.empty(shape=0, dtype=float)
        
        for lowerBinBound in windBins:
            upperBinBound = lowerBinBound + by
            Bin = [lowerBinBound, upperBinBound]
            
            if dim == "direction":
                y_i = metricMethod(Bin, scope)
            else:
                y_i = metricMethod(scope, Bin)
            
            if type(y_i) is str:
                np.append(y, None)
                np.append(se, None)
                continue
                    
            y = np.append(y, y_i)
            
        resultDict = {'x': windBins, 
                      'y': y}
        
        return resultDict
 
    # This will need to be updated if the metrics are 
    # changed to return something other than a string when it can't be computed
    # for a particular wind condition bin.
    # This used to be named matrixOfMetrics.
    def compute2D(self, metricMethod, windDirectionSpecs=[0,360,1], windSpeedSpecs=[0,20,1]):
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
         
    # Just sketching out what this method might look like, not implemented at all
    #def se(self, metricMethod, seMethod="bootstrapping", conf=0.95):
        return None
    
    # Mostly 0Untested
    # if nDim is 1, cannot be used with aepGain
    def bootstrapEstimate(self, metricMethod=None, nDim=1, windDirectionSector=[0,360], 
                          windSpeedRange=[0,20], by=1, dim="direction", 
                          windDirectionSpecs=[0,360,1], 
                          windSpeedSpecs=[0,20,1], 
                          seMultiplier=2, lowerPercentile=2.5, upperPercentile=97.5,
                          B=1000, seed=None, retainReps = False,):# figure out how to use kwargs here for hours, aepmethod, and absolute, etc. for metricMethod
        """
        Compute summary statistics of bootsrapped samples based on your metric of choice
        
        nDim: int, 1 or 2; Number of directions along which to compute the 
            metric-- i.e., make bins based on just wind speed or direction or 
            make 2D bins based on both
        metricMethod: Method, the method you want to use to compute the 
             measurements in each wind condition bin (not implemented yet)
         
        windDirectionSector: list of length 2, specifications for wind direction
             bins-- [lower bound (inclusive), upper bound (exclusive)]. 
             Only used if nDim=1
        windSpeedRange: list of length 2, specifications for wind speed bins--
             [lower bound (inclusive), upper bound (exclusive)]
             Only used if nDim=1
        dim: string, the varaible you want to compute the metric as a function of
             (dim for dimension as in 1D). Either "direction" or "speed".
             Only used if nDim=1
        by: step size for the bins for dim. Only used if nDim=1.
        

        windDirectionSpecs: list of length 3, specifications for wind direction
            bins-- [lower bound (inclusive), upper bound (exclusive), bin width]
            Only used if nDim=2
        windSpeedSpecs: list of length 3, specifications for wind speed bins--
            [lower bound (inclusive), upper bound (exclusive), bin width]
            Only used if nDim=2
        """
        # Assuming valid inputs
        start = default_timer()
        Shape = self.df.shape
        resultDict = {"mean": None,
                      "meanMinusSE": None,
                      "meanPlusSE": None,
                      "median": None,
                      "upperPercentile": None,
                      "lowerPercentile": None}
        
        # Set seed for reproducibility
        prng = np.random.default_rng(seed)
        
        # Get all bootstrap samples indices at once
        bootstrapIdx = prng.choice(Shape[0], size=(B, Shape[0]), replace=True)
        
        # Set up wind condition bins
        if nDim==1:

            if dim == "direction":
                windBins = np.arange(windDirectionSector[0],
                                     windDirectionSector[1], by)
            else:
                windBins = np.arange(windSpeedRange[0],
                                     windSpeedRange[1], by)
                
        else: #nDim=2
            windBins = np.array([(direction, speed) 
                             for direction in range(windDirectionSpecs[0],
                                                    windDirectionSpecs[1],
                                                    windDirectionSpecs[2]) 
                             for speed in range(windSpeedSpecs[0],
                                                windSpeedSpecs[1],
                                                windSpeedSpecs[2])])
            #print("Computing as a function of" + dim)
        
        # Matrix to hold all botstrap samples (one row = one bootstrap rep)
        bootstrapMatrix = np.full((B, windBins.shape[0]), None, dtype=float)        
        
        # Create temporary energyGain object
        egTemp = energyGain(df=None,
                            dfUpstream=self.dfUpstream, 
                            testTurbines=self.testTurbines,
                            refTurbines=self.referenceTurbines,
                            wdCol=self.wdCol,
                            wsCol=self.wsCol,
                            useReference=self.useReference)
        
        for rep in range(B):
            indices = bootstrapIdx[rep]
            # Get the sample for this bootstrap rep   
            dfTemp = self.df.iloc[indices]
            
            # reset the data frame indices
            # I'm not entirely sure this matters, but just in case
            dfTemp.reset_index(drop=True, inplace=True)
            
            # Update temporary energyGain object
            egTemp.df=dfTemp
            #breakpoint()
            if nDim==1:
                # compute the desired metric for every wind condition bin
                computeResults = egTemp.compute1D(egTemp.percentPowerGain, # I don't think metric method will work here but come back and check
                                                  windDirectionSector,
                                                  windSpeedRange, by=by, dim=dim)["y"]
            else:#(if nDim==2)
            # Flatten the resulting matrix so that we have an array that goes 
            # through all speeds and then changes directions
                computeResults = egTemp.compute2D(egTemp.percentPowerGain,
                                                windDirectionSpecs,
                                                windSpeedSpecs)["data"].flatten("F")
                #computeData = computeResults["data"].flatten("F")
            
            # Save the array as one row in the bootstrap matrix
            bootstrapMatrix[rep] = computeResults
         
        # Compute sampling distribution statistics across each rep
        # (across rows/by column) one column = all the reps for one bin
        means = np.mean(bootstrapMatrix, axis=0)
        se = np.std(bootstrapMatrix, axis=0)
        meansPlusSE = means + seMultiplier*se
        meansMinusSE = means - seMultiplier*se
        medians = np.percentile(bootstrapMatrix, q=50, axis=0)
        lowerPercentiles = np.percentile(bootstrapMatrix, q=lowerPercentile, axis=0)
        upperPercentiles = np.percentile(bootstrapMatrix, q=upperPercentile, axis=0)
        
        # Add results to dictionary
        resultDict["mean"]= means
        resultDict["se"]= se
        resultDict["meanPlusSE"]= meansPlusSE
        resultDict["meanMinusSE"]= meansMinusSE
        resultDict["median"]= medians
        resultDict["lowerPercentile"]= lowerPercentiles
        resultDict["upperPercentile"]= upperPercentiles
        
        if nDim==1:
            resultDict["binLowerBound"] = windBins
        else:#(if nDim==2)
            resultDict["directionLowerBound"] = windBins[:,0]
            resultDict["speedLowerBound"] = windBins[:,1]
        
        # Return the actual samples if desired
        if retainReps:
            return {"stats":pd.DataFrame(resultDict), 
                    "reps":bootstrapMatrix}
        duration = default_timer() - start
        print(duration)
        return pd.DataFrame(resultDict)
    
    # Change this name later, probably
    # I just realized the tick marks might behave weirdly if either stepsize is specified to be less than on, there was a smart tick mark picker that I could use instead
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
            heatmapMatrix = self.compute2D(self.percentPowerGain,
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
        
    # Change this name later
    # pct power gain
    # This does NOT work, the scatterplot and line plot do not line up
    def plot(self, windDirectionSpecs=[0,360,1], windSpeedSpecs=[0,20,1], x=None, y=None):
        """
        windDirectionSpecs: list of length 3, specifications for wind direction
            bins-- [lower bound (inclusive), upper bound (exclusive), bin width]
        windSpeedSpecs: list of length 3, specifications for wind speed bins--
            [lower bound (inclusive), upper bound (exclusive), bin width]
        """
        # If all data is missing, calculate some default
        if (x is None) and (y is None):
            resultDict = self.compute1D(self.percentPowerGain, windDirectionSpecs[0:2],
                                        windSpeedSpecs[0:2], by=windDirectionSpecs[2],)
            x = resultDict['x']
            y = resultDict['y']
            #title = "Percent Power Gain"
        elif (x is None) or (y is None):
            # If only one is missing, that means someone oopsied
            sillyGoose = "Didn't provide all data."
            return print(sillyGoose)
        
        fig, ax = plt.subplots()
        
        #ax.scatter(x,y)
        
        # # Labels
        # ax.set_title(title)
        # ax.set_xlabel(u"Wind Direction (\N{DEGREE SIGN})") #unicode formatted
        # ax.set_ylabel("Wind Speed (m/s)")
        
        ax.scatter(x=x,y=y, c="blue")
        ax.plot(x,y)
        
        ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
        ax.yaxis.set_major_locator(mticker.MultipleLocator(0.01))
        ax.xaxis.set_minor_locator(mticker.MultipleLocator(1)) 
        ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.005))
        
        ax.grid(visible=True, which="major")
        
        ax.scatter(x=x,y=y, c="red")
        #ax.plot(x,y)
        
        plt.show()
        
        return None
  
# Removing this later
# df_scada = pd.read_feather("C:/Users/ctodd/Documents/GitHub/flasc/examples_smarteole/postprocessed/df_scada_data_60s_filtered_and_northing_calibrated.ftr")
# FI = load_smarteole_floris(wake_model="C:/Users/ctodd/Documents/GitHub/floris/examples/inputs/gch.yaml", wd_std=0.0)
# dfUpstream = ftools.get_upstream_turbs_floris(FI)
# testTurbines = [4,5]
# referenceTurbines = [0,1,2,6]
# df_scadaNoNan = df_scada.dropna(subset=[f'pow_{i:03d}' for i in testTurbines+referenceTurbines])
# thing = energyGain(df_scadaNoNan, dfUpstream, testTurbines, referenceTurbines,wdCol="wd_smarteole", wsCol="ws_smarteole")
# thing.plot([190,250,5],[0,20,1])
    
    
    
    