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
import re
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
        
        self.df = df
        self.dfLong=None
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
            self.__dfLonger__()
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
        self.__dfLonger__()
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
        self.__dfLonger__()
        return None
    
    def __dfLonger__(self):
        df = self.df
        powerColumns = ["pow_{:03.0f}".format(number) for number in self.referenceTurbines + self.testTurbines]
        keep = powerColumns + [self.wdCol, self.wsCol, "time"]
        df[keep].melt(value_vars=powerColumns,
                      value_name="power",
                      var_name="turbine", 
                      id_vars=['time', 'wd_smarteole', 'ws_smarteole'])
        df.set_index(df["time"],inplace=True, drop=True)
        self.dfLong = df
        return None
               
    def setReference(self, lst):
        self.referenceTurbines = lst
        self.__dfLonger__()
    
    def setTest(self, lst):
        self.testTurbines = lst
        self.__dfLonger__()
    
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
    
    def binAdder(self, stepVars = "direction", windDirectionSpecs=[190,250,1],windSpeedSpecs=[0,20,1], copy=True, df=None):
        """
        Add columns for the lower bounds of the wind condition bins to df (or a copy of df)
        
        windDirectionSpecs: list of length 3 or 2, specifications for wind direction
            bins-- [lower bound (inclusive), upper bound (exclusive), bin width].
            If direction is not in stepVars, 3rd element gets ignored if it exists.
        windSpeedSpecs: list of length 3 or 2, specifications for wind speed bins--
            [lower bound (inclusive), upper bound (exclusive), bin width]
            If speed is not in stepVars, 3rd element gets ignored if it exists.
        stepVars: string ("speed" or "direction") or list of the possible strings.
            The variable(s) you want to increment by for the wind condition bins
        copy: boolean, whether to simply return a copy of self.df (True) or to actually update self.df (False)
            Default is true because rows outside of the specs will be deleted
        """
        # Convert to list if needed
        if type(stepVars) is str:
            stepVars = list([stepVars])
        
        if df is None:
            df = self.df.copy()
        
        # Filter for conditions out of the bound of interest
        df = df[(df[self.wdCol]>=windDirectionSpecs[0]) & (df[self.wdCol]<windDirectionSpecs[1]) &
                (df[self.wsCol]>=windSpeedSpecs[0]) & (df[self.wsCol]<windSpeedSpecs[1])]
        
        # Calculating the bins
        if "direction" in stepVars:
            df["directionBinLowerBound"] = (((df[self.wdCol]-windDirectionSpecs[0])//windDirectionSpecs[2])*windDirectionSpecs[2])+windDirectionSpecs[0]
        if "speed" in stepVars:
            df["speedBinLowerBound"] = (((df[self.wsCol]-windSpeedSpecs[0])//windSpeedSpecs[2])*windSpeedSpecs[2])+windSpeedSpecs[0]
        
        # Update self.df if desired
        if not copy:
            self.df = df
            
        # Return the copy with the bin columns
        return df
             
    def binAll(self, stepVars = ["direction", "speed"], windDirectionSpecs=[190,250,1],
               windSpeedSpecs=[0,20,1], retainControlMode=True, 
               retainTurbineLabel=True,  returnWide=True, df=None, group=True):
        """
        windDirectionSpecs: list of length 3 or 2, specifications for wind direction
            bins-- [lower bound (inclusive), upper bound (exclusive), bin width].
            If direction is not in stepVars, 3rd element gets ignored if it exists.
        windSpeedSpecs: list of length 3 or 2, specifications for wind speed bins--
            [lower bound (inclusive), upper bound (exclusive), bin width]
            If speed is not in stepVars, 3rd element gets ignored if it exists.
        stepVars: string ("speed" or "direction") or list of the possible strings
            The variable(s) you want to increment by for the wind condition bins
        retainControlMode: boolean, whether to keep the control mode column (True) or not (False)
        """
        if type(stepVars) is str:    
            stepVars = list([stepVars])
        

        if df is None:
            df = self.binAdder(windDirectionSpecs=windDirectionSpecs,
                               windSpeedSpecs=windSpeedSpecs,
                               stepVars=stepVars)
        
        # Exclude undesirable turbines
        stepVarCols = ["{}BinLowerBound".format(var) for var in stepVars]
        powerColumns = ["pow_{:03.0f}".format(number) for number in self.referenceTurbines + self.testTurbines]     
        colsToKeep = stepVarCols[:]
        if retainControlMode:
           colsToKeep.append("control_mode")
        df = df[colsToKeep + powerColumns]
        
        # Pivot Longer
        dfLong = df.melt(id_vars=colsToKeep, value_name="power", var_name="turbine")
        
        # Convert turbine numbers from strings to integers
        dfLong["turbine"]  = dfLong["turbine"].str.removeprefix("pow_")
        dfLong["turbine"] = dfLong["turbine"].to_numpy(dtype=int)
        
        # Add turbine label column
        if retainTurbineLabel:
            labels = [(num in self.testTurbines) for num in dfLong["turbine"]]
            labels = np.where(labels, "test", "reference")
            dfLong["turbineLabel"] = labels
            colsToKeep.append("turbineLabel")
            
        if not group:
            return dfLong
        
        # Calculating average by group
        dfGrouped = dfLong.groupby(by=colsToKeep).agg(averagePower = pd.NamedAgg(column="power", 
                                                                                 aggfunc=np.mean),
                                                      numObvs = pd.NamedAgg(column="power", 
                                                                            aggfunc='count'))
        
        # Convert grouping index into columns for easier pivoting
        for var in colsToKeep:
            dfGrouped[var] = dfGrouped.index.get_level_values(var)
            
        # Pivot wider     
        if returnWide:
            optionalCols = list( set(colsToKeep) - set(stepVarCols))
            
            dfWide = dfGrouped.pivot(columns=optionalCols, index=stepVarCols, 
                                     values=['averagePower', 'numObvs'])
            return dfWide
        
        # Don't need these columns anymore since they are a part of the multi-index
        dfGrouped.drop(columns=colsToKeep, inplace=True)
        return dfGrouped
        
    # Fix comments later
    def computeAll(self, stepVars = ["direction", "speed"], 
                   windDirectionSpecs=[190,250,1], windSpeedSpecs=[0,20,1],
                   useReference=True, df=None):
        """
        Computes all the things from the slides except AEP gain

        Parameters
        ----------
        stepVars : TYPE, optional
            DESCRIPTION. The default is ["direction", "speed"].
        df : pandas data frame as returned from binAll with returnWide=True, optional
            Calls binAll if None.
        windDirectionSpecs : TYPE, optional
            DESCRIPTION. The default is None.
        windSpeedSpecs : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        df : pandas data frame
            Nicely formatted dataframe that can go directly into aepGain.
        """
        if type(stepVars) is str:    
            stepVars = list([stepVars])
            
        if df is None:
            df = self.binAll(stepVars = stepVars, 
                             windDirectionSpecs=windDirectionSpecs,
                             windSpeedSpecs=windSpeedSpecs,
                             df=df)
        # Sometimes the order of the labels in this tuple seem to change and I haven't figured out why. This should fix the order.
        df = df.reorder_levels([None, "turbineLabel", "control_mode"], axis=1)
                          
        if useReference:
            df["powerRatioBaseline"] = np.divide(df[('averagePower','test','baseline')], 
                                             df[('averagePower', 'reference', 'baseline')])
            df["powerRatioControl"] = np.divide(df[('averagePower', 'test', 'controlled')],
                                            df[('averagePower', 'reference', 'controlled')])
            df["totalNumObvs"] = np.nansum(np.dstack((df[('numObvs', 'test', 'controlled')],
                                                      df[('numObvs', 'reference', 'controlled')],
                                                      df[('numObvs', 'test', 'baseline')],
                                                      df[('numObvs', 'reference', 'baseline')])),
                                           axis=2)[0]
            
            
        else:
            df["powerRatioBaseline"] = df[('averagePower', 'test', 'baseline')]
            df["powerRatioControl"] = df[('averagePower', 'test', 'controlled')]
            
            
            df["totalNumObvs"] = np.nansum(np.dstack((df[('numObvs', 'test', 'controlled')],
                                                      df[('numObvs', 'test', 'baseline')])),
                                           axis=2)[0]
            
            df["totalNumObvsInclRef"] = np.nansum(np.dstack((df["totalNumObvs"],
                                                             df[('numObvs', 'reference', 'controlled')],
                                                             df[('numObvs', 'reference', 'baseline')])),
                                           axis=2)[0]
        
        # Same for both AEP methods
        N = np.nansum(df["totalNumObvs"])
        df["freq"] = df["totalNumObvs"]/N
        df["changeInPowerRatio"] = np.subtract(df['powerRatioControl'],
                                           df['powerRatioBaseline'])
        
        df["percentPowerGain"] = np.divide(df["changeInPowerRatio"],
                                       df['powerRatioControl'])
        
        # Make columns out of the indices just because it's easier to see sometimes
        stepVarCols = ["{}BinLowerBound".format(var) for var in stepVars]
        for var in stepVarCols:
            df[var] = df.index.get_level_values(var)
        
        return df
    
    # Fix comments later
    def aepGain(self, windDirectionSpecs=[190,250,1],windSpeedSpecs=[0,20,1],
                hours=8760, aepMethod=1, absolute=False, useReference=None,df=None):
        """
        Calculates AEP gain  

        Parameters
        ----------
        df : pandas data frame as returned by computeAll, optional
            If None, calls computeAll
        windDirectionSpecs : list, optional
            DESCRIPTION. The default is None.
        windSpeedSpecs : list, optional
            DESCRIPTION. The default is None.
        hours : float, optional
            DESCRIPTION. The default is 8760.
        aepMethod : int, optional
            DESCRIPTION. The default is 1.
        absolute : boolean, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        AEP gain (float)

        """
        if useReference is None:
            useReference = self.useReference
        
        if not useReference:
            # Both methods are equivalent when reference turbines aren't used,
            aepMethod=1
            
        # Calculate nicely formatted df if needed
        if df is None:
            df = self.computeAll(stepVars=["speed","direction"],
                                 windDirectionSpecs=windDirectionSpecs,
                                 windSpeedSpecs=windSpeedSpecs,
                                 df=df,
                                 useReference = useReference)
            
        # Different AEP formulas
        if aepMethod==1:
            if useReference:
                df["aepGainContribution"] = np.multiply(np.multiply(df[('averagePower', 'baseline', 'test')],
                                                                df[('percentPowerGain', '', '')]),
                                                    df[('freq', '', '')])
            else:
                df["aepGainContribution"] = np.multiply(df["changeInPowerRatio"], df[('freq', '', '')])
    
            
            if not absolute:
                denomTerms = np.multiply(df[('averagePower', 'baseline', 'test')], df[('freq', '', '')])
            
        else:
            # Couldn't find an element-wise weighted mean, so I did this
            sumPowerRefBase = np.multiply(df[('averagePower', 'baseline', 'reference')],
                                          df[('numObvs', 'baseline', 'reference')])
            sumPowerRefcontrolled = np.multiply(df[('averagePower', 'controlled', 'reference')],
                                          df[('numObvs', 'controlled', 'reference')])
            
            sumPowerRef = np.nansum(np.dstack((sumPowerRefBase,sumPowerRefcontrolled)),2)[0]
            
            numObvsRef = np.nansum(np.dstack((df[('numObvs', 'controlled', 'reference')],df[('numObvs', 'baseline', 'reference')])),2)[0]
            
            avgPowerRef = np.divide(sumPowerRef, numObvsRef)
            
            df["aepGainContribution"] = np.multiply(np.multiply(avgPowerRef,
                                                              df[('changeInPowerRatio', '', '')]),
                                                  df[('freq', '', '')])
            if not absolute:
                denomTerms = np.multiply(np.multiply(avgPowerRef,
                                              df[('powerRatioBaseline', '', '')]),
                                  df[('freq', '', '')])
                
                
        if not absolute:
            # 'hours' here doesn't really represent hours, 
            # this is just so that our percentages are reported nicely
            hours = 100
            denom = np.nansum(denomTerms)
            df["aepGainContribution"] = df["aepGainContribution"]*(1/denom)
        
        aep = hours*np.nansum(df[('aepGainContribution', '', '')])    
        print(aep)
        return (df, aep)
    
    def bootstrapSamples(self, B=1000, seed=None):
        
        start = default_timer()
        samples = np.ndarray(B, dtype=pd.core.frame.DataFrame)
        
        
        # Sampling rows from different data frames depending on the sampling scheme
        #df = self.dfLong.copy()
        #nrow = self.df.shape[0]
        # Do the actual bootstrapping
        nrow = self.df.shape[0]
        for rep in range(B):
            dfTemp = self.df.sample(n=nrow,
                                    replace=True,
                                    random_state=seed)
            
            dfTemp.reset_index(drop=True,inplace=True)
            samples[rep] = dfTemp
                                                                   
        
        duration = default_timer() - start
        print("Sampling Time:", duration)
        
        return samples
    
  
    # Need to completely rewrite this so that it works with computeAll
    def bootstrapEstimate(self, stepVars=["direction","speed"], 
                          windDirectionSpecs=[200,250,1], windSpeedSpecs=[0,20,1],
                          B=1000, seed=None, metric="percentPowerGain", useReference=True,
                          seMultiplier=2, lowerPercentile=2.5, upperPercentile=97.5,
                          retainReps = False):# figure out how to use kwargs here for hours, aepmethod, and absolute, etc. for metricMethod
        """
        Compute summary statistics of bootsrapped samples based on your metric of choice
        
        metric = string, one of changeInPowerRatio, percentPowerGain

        windDirectionSpecs: list of length 3, specifications for wind direction
            bins-- [lower bound (inclusive), upper bound (exclusive), bin width]
            Only used if nDim=2
        windSpeedSpecs: list of length 3, specifications for wind speed bins--
            [lower bound (inclusive), upper bound (exclusive), bin width]
            Only used if nDim=2
        """
        if type(stepVars) is str:    
            stepVars = list([stepVars])
            
        start = default_timer()
        
        # Get an array of the bootstrap samples
        bootstrapDFs = self.bootstrapSamples(B=B, seed=seed)
        finalCols = list([metric])
        finalColsMultiIdx = list([(metric, '','')])
        
        
        for var in stepVars:
            #metricDict[var] = nones
            name = f'{var}BinLowerBound'
            finalCols.append(name)
            finalColsMultiIdx.append((name,'',''))
        
        
        
        for bootstrap in range(B):
            currentDF = bootstrapDFs[bootstrap]
            # get into correct format
            binadder =self.binAdder(stepVars, windDirectionSpecs, windSpeedSpecs, df=currentDF)
            binall = self.binAll(stepVars, windDirectionSpecs, windSpeedSpecs, df=binadder)
            
            computeall = self.computeAll(stepVars, windDirectionSpecs,
                                         windSpeedSpecs, useReference, df=binall)
            
            
            binStats = computeall[finalColsMultiIdx]
            # Multi-index on the columns makes indexing annoying when all but one level is empty
            dfTemp = binStats.reset_index(drop=True).droplevel(level=[1,2], axis="columns")
            # Make sure columns are in the right order
            dfTemp = dfTemp[finalCols]
            
            if bootstrap==0:
                metricMatrix = np.asarray(dfTemp)
            
            metricMatrix = np.concatenate((metricMatrix, np.asarray(dfTemp)), axis=0)
            
        metricDF = pd.DataFrame(data=metricMatrix, columns=finalCols)

        # Compute Sampling distribution statistics for each wind condition bin
        finalCols.pop(0)
        dfStats = metricDF.groupby(by=finalCols).agg(mean=pd.NamedAgg(column=metric,
                                                                      aggfunc=np.mean),
                                                     se=pd.NamedAgg(column=metric,
                                                                    aggfunc=np.nanstd),
                                                     median=pd.NamedAgg(column=metric,
                                                                        aggfunc=np.nanmedian),
                                                     upperPercentile=pd.NamedAgg(column=metric,
                                                                                 aggfunc=lambda x: np.nanpercentile(x,upperPercentile)),
                                                     lowerPercentile=pd.NamedAgg(column=metric,
                                                                                 aggfunc=lambda x: np.nanpercentile(x, lowerPercentile)),
                                                     nObvs = pd.NamedAgg(column=metric, 
                                                                           aggfunc='count'))
        
        seMultd = seMultiplier*dfStats["se"]
        dfStats["meanMinusSe"] = np.subtract(dfStats["mean"], seMultd)
        dfStats["meanPlusSe"] = np.add(dfStats["mean"], seMultd)
        
        dfStats = dfStats[["mean",'meanMinusSe','meanPlusSe', 
                           'median', 'upperPercentile', 'lowerPercentile',
                           'se', 'nObvs']]
        
        duration = default_timer() - start
        print(duration)
            
        if retainReps:
            return {"summary": dfStats, "reps":bootstrapDFs}
            
        
        return dfStats
    
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
        
    # change this name later
    
    
    # Change this name later
    # pct power gain only right now
    # This does NOT work, the scatterplot and line plot do not line up
    def plot(self, windDirectionSpecs=[0,360,1], windSpeedSpecs=[0,20,1], 
             x=None, y=None, xErr=None, yErr=None):
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
        
        if (xErr is not None) and (yErr is not None):
            ax.errorbar(x=x, y=y,xerr=xErr, yerr=yErr, capsize=5, c="green")
        
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