# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 13:03:40 2023

@author: ctodd

Methods for calculating power ratios, associated metrics, 
and uncertainty on these metrics
"""
from flasc.dataframe_operations import dataframe_manipulations as dfm
from timeit import default_timer
import re
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns
import pandas as pd
pd.options.mode.chained_assignment = None


class energyGain():

    def __init__(self,
                 scada,
                 upstream,
                 directionBins=None,
                 speedBins=None,
                 wdColScada='wd',
                 wsColScada='ws',
                 wind=None,
                 wdColWind=None,
                 wsColWind=None,
                 testTurbines=[],
                 referenceTurbines=[],
                 useReference=True):
        """
        Creates an energyGain object

        Parameters
        ----------
        scada : pandas dataframe
            scada data, formatted as in flasc example

        upstream : pandas dataframe
            output from ftools.get_upstream_turbs_floris, 
            data frame listing which turbines are upstream 
            for each wind speed and direction

        directionBins : numpy array of numerics, optional
            Edges for the wind direction in degrees. Bin membership is 
            left edge inclusive, right edge exclusive. The default None means
            we will ignore wind direction for any calculations that use 
            wind conditions. Must be monotonic INCREASING (bins that share an 
            edge do not need to have their edges be repeated) and on [0,360).
            Rightmost bin edge is expected and will be used as a maximum cutoff for direction.

        speedBins : numpy array, optional
            Edges for the wind speed in m/s. Bin membership is 
            left edge inclusive, right edge exclusive. The default None means
            we will ignore wind speed for any calculations that use 
            wind conditions. Must be monotonic INCREASING (bins that share an 
            edge do not need to have their edges be repeated) and non-negative. 
            Rightmost bin edge is expected and will be used as a maximum cutoff for speed. 

        wdColScada : string
            Name of the column in the scada data that you want to use as the 
            'true' wind direction.
            Defaults to 'wd' (the column output by other FLASC functionality)

        wsColScada : string
            Name of the column in the scada data that you want to use as the 
            'true' wind direction.
            Defaults to 'ws' (the column output by other FLASC functionality)

        wind: pandas dataframe
            data frame with a column for wind direction measurements and/or a 
            column for mind speed measurements, taken at (assumed uniform) 
            time stamps

        wdColWind : string
            Name of the wind direction column in the wind condition time series
            Defaults to None in case you want to ignore direction

        wsColWind : string
            Name of the wind speed column in the wind condition time series
            Defaults to None in case you  want to ignore speed

        testTurbines : list of integers
            Turbine numbers to be considered test turbines, based on 3-digit turbine numbering convention starting at 0 but without the leading zeroes

        referenceTurbines : list of integers
            Turbine numbers to be considered reference turbines, based on 3-digit turbine numbering convention starting at 0 but without the leading zeroes

        useReference : boolean, optional
            Whether to use reference turbines for various energy uplift calculations. 
            This will only be used if a method is called without specifying useReference in the call.
            The default is True.

        Returns
        -------
        energyGain object
        """
        # Object attributes
        self.scada = None
        self.scadaLong = None
        self.pmfJoint = None
        self.pmfSpeed = None
        self.pmfDirection = None
        self.testTurbines = None
        self.referenceTurbines = None
        self.allTurbines = None
        self.directionBins = None
        self.speedBins = None
        self.upstream = None
        self.wind = wind
        self.useReference = useReference

        # Setting attributes
        self.setScada(scada, wdColScada, wsColScada)
        self.setTest(testTurbines)
        self.setReference(referenceTurbines)
        self.setUpstream(upstream)
        # Calculates the pmf based on speed and direction bins
        # and updates the PMF attribute
        self.setBins(directionBins=directionBins,
                     speedBins=speedBins,
                     wdColWind=wdColWind,
                     wsColWind=wsColWind)
                     
        self.setWind(wind,
                     wdColWind=wdColWind,
                     wsColWind=wsColWind)

    def setScada(self, df, wdCol='wd', wsCol='ws'):
        """
        Setter method for scada data object attribute

        Parameters
        ----------
        df : pandas DataFrame
            scada data, as returned by FLORIS.

        wdCol : string
            name of the column that contains the 'reference'
            or 'consensus' wind direction data.

        wsCol : string
            name of the column that contains the 'reference'
            or 'consensus' wind speed data.

        Returns
        -------
        None.

        """
        self.scada = df
        self.setWD(wdCol)
        self.setWS(wsCol)

        if df is not None:

            # Infer the list of all turbines from the power columns in the data frame
            self.allTurbines = [int(re.sub("\D+", "", colname))
                                for colname in list(df) if re.match('^pow_\d+', colname)]

            # If there is no dedicated long term wind data frame,
            # use this new scada data to calculate bin frequencies
            if self.wind is None:
                self.pmf = self.pmfCalculator(df,
                                              speedBins=self.speedBins,
                                              directionBins=self.directionBins)
        else:
            self.allTurbines = None
            if self.wind is None:
                self.pmf = None

        return None

    def setUpstream(self, df):
        """
        Updates the upstream object attribute

        Parameters
        ----------
        df : pandas DataFrame
            output from ftools.get_upstream_turbs_floris, 
            data frame listing which turbines are upstream 
            for each wind speed and direction.

        Returns
        -------
        None.

        """
        self.upstream = df
        return None

    def setWind(self,
                  dfWind,
                  wdColWind='wd',
                  wsColWind='ws'):

        self.wind = dfWind
        # Calculates the pmf based on speed and direction bins
        # and updates the PMF attribute
        self.setBins(directionBins=self.directionBins,
                     speedBins=self.speedBins,
                     wdColWind=wdColWind,
                     wsColWind=wsColWind)

        return None

    def setWD(self, colname):
        """
        Setting the column to be referenced for wind direction 
        ('consensus' wind direction)

        Parameters
        ----------
        colname : string, optional
            Name ofthe column that contains the 'consensus' or 'reference' 
            wind direction.

        Returns
        -------
        None.

        """

        self.wdCol = colname
        return None

    def setWS(self, colname):
        """
        Setting the column to be referenced for wind speed 
        ('consensus' wind speed)

        Parameters
        ----------
        colname : string, optional
            Name ofthe column that contains the 'consensus' or 'reference' wind speeds.

        Returns
        -------
        None.

        """
        self.wsCol = colname

        return None

    def setBins(self,
                directionBins=None,
                speedBins=None,
                wdColWind=None,
                wsColWind=None,
                plot=False):
        """
        Updates the object attributes for speed and direction bins,
        re-calculates the pmf based on these updates, and updates the PMF 
        attribute

        Parameters
        ----------

        directionBins : numeric numpy array, optional
            Edges for the wind direction in degrees. Bin membership is 
            left edge inclusive, right edge exclusive. The default None means
            we will ignore wind direction for any calculations that use 
            wind conditions. Must be monotonic INCREASING (bins that share an 
            edge do not need to have their edges be repeated) and on [0,360). 
            Rightmost bin edge is expected and will be used as a maximum cutoff for direction.  

        speedBins : numeric numpy array, optional
            Edges for the wind speed in m/s. Bin membership is 
            left edge inclusive, right edge exclusive. The default None means
            we will ignore wind speed for any calculations that use 
            wind conditions. Must be monotonic INCREASING (bins that share an 
            edge do not need to have their edges be repeated) and non-negative. 
            Rightmost bin edge is expected and will be used as a maximum cutoff for speed. 

        wdColWind : string, optional
            Name of the wind direction column in the wind condition time series
            Defaults to None in case you want to ignore direction

        wsColWind : string, optional
            Name of the wind speed column in the wind condition time series
            Defaults to None in case you  want to ignore speed

        plot : boolean, optional
            Whether you want to plot the experimental distribution using these wind bins.
            The default is False.

        Returns
        -------
        None or the PMF according to the new wind condition bins, depending on
        returnData

        """
        # Update default object attributes
        self.speedBins = speedBins
        self.directionBins = directionBins

        # If there is no dedicated long term wind condition time series,
        # calculate PMF based on the data
        if self.wind is None:
            df = self.scada
            wdColWind = self.wdCol
            wsColWind = self.wsCol
        else:
            df = self.wind

        # Calculate marginal and/or joint PMFs
        if directionBins is not None:
            # Marginal direction PMF
            self.pmfDirection = self.pmfCalculator(dfWind=df,
                                                   speedBins=None,
                                                   directionBins=directionBins,
                                                   wdColWind=wdColWind,
                                                   wsColWind=None)
            if speedBins is not None:
                # Joint PMF
                self.pmfJoint = self.pmfCalculator(dfWind=df,
                                                   speedBins=speedBins,
                                                   directionBins=directionBins,
                                                   wdColWind=wdColWind,
                                                   wsColWind=wsColWind)
        else:
            # Marginal Speed PMF
            self.pmfSpeed = self.pmfCalculator(dfWind=df,
                                               speedBins=speedBins,
                                               directionBins=None,
                                               wdColWind=None,
                                               wsColWind=wsColWind)

        # Plot the wind rose for the experimental data
        if plot:
            self.windRose(df=self.scada,
                          directionBins=self.directionBins,
                          speedBins=self.speedBins)

        return None

    def setReference(self, lst):
        """
        Updates the list of reference turbines

        Parameters
        ----------
        lst : list of integers
            Turbine numbers to be considered reference turbines, 
            based on 3-digit turbine numbering convention starting at 0 
            but without the leading zeroes.

        Returns
        -------
        None.

        """
        self.referenceTurbines = lst
        self.referenceTurbines.sort()
        return None

    def setTest(self, lst):
        """
        Updates the list of test turbines

        Parameters
        ----------
        lst : list of integers
            Turbine numbers to be considered test turbines, 
            based on 3-digit turbine numbering convention starting at 0 
            but without the leading zeroes.

        Returns
        -------
        None.

        """
        self.testTurbines = lst
        self.testTurbines.sort()
        return None

    def pmfCalculator(self,
                      dfWind,
                      speedBins=None,
                      directionBins=None,
                      wdColWind=None,
                      wsColWind=None):
        """
        Calculates the joint or marginal PMF based on the provided 
        wind condition bins

        Parameters
        ----------
        dfWind : pandas dataframe
            data frame with a column for wind direction measurements and/or a 
            column for wind speed measurements, taken at time stamps

        directionBins : numeric numpy array, optional
            Edges for the wind direction in degrees. Bin membership is 
            left edge inclusive, right edge exclusive. The default None means
            we will ignore wind direction for any calculations that use 
            wind conditions. Must be monotonic INCREASING (bins that share an 
            edge do not need to have their edges be repeated) and on [0,360). 
            Rightmost bin edge is expected and will be used as a maximum cutoff for direction.  

        speedBins : numeric numpy array, optional
            Edges for the wind speed in m/s. Bin membership is 
            left edge inclusive, right edge exclusive. The default None means
            we will ignore wind speed for any calculations that use 
            wind conditions. Must be monotonic INCREASING (bins that share an 
            edge do not need to have their edges be repeated) and non-negative. 
            Rightmost bin edge is expected and will be used as a maximum cutoff for speed. 

        wdColWind : string
            Name of the wind direction column in the wind condition time series
            Defaults to None in case you want to ignore direction

        wsColWind : string
            Name of the wind speed column in the wind condition time series
            Defaults to None in case you  want to ignore speed

        Returns
        -------
        freq : pandas Series 
            Joint or marginal PMF based on the new wind condition bins.
            Will have an index or multiIndex of the wind condition bins.
            The support of the PMF may exclude some wind or speed values that
            were not contained in the bin specifications.
            This gets handled by a helper function.
        """

        # If the correspinding bin variable for one of these is also None,
        # then this update will get ignored
        if wdColWind is None:
            wdColWind = 'wd'
        if wsColWind is None:
            wsColWind = 'ws'

        # This ensures that the frequencies are with respect to all wind data
        # and not just the data that falls in the boundaries of the wind
        # condition bins specified.
        N = dfWind.shape[0]

        # PMF calculations
        if speedBins is None:  # speed is None

            if directionBins is None:  # Both are None
                print("Specify wind direction and/or speed bins")
                freqs = None
            else:  # Just speed is None
                
                # Calculate PMF based on only wind direction
                hist = np.histogram(self.wind[wdColWind],
                                    bins=directionBins)
                freqs = pd.Series(hist[0]/N)
                freqs.index = hist[1][:-1] #omit rightmost edge
                freqs.index.names = ["directionBins"]
        else:  # Speed is not None

            if directionBins is None:  # speed is not none but direction is none

                # Assign bin edges
                # Just the direction bin is None, so
                # Calculate the PMF based on only wind speed
                hist = np.histogram(self.wind[wsColWind],
                                    bins=speedBins)
                freqs = pd.Series(hist[0]/N)
                freqs.index = hist[1][:-1]#omit rightmost edge
                freqs.index.names = ["speedBins"]

            else:  # Neither are None
                # Get bin counts
                hist = np.histogram2d(x=self.wind[wdColWind],
                                      y=self.wind[wsColWind],
                                      bins=(directionBins,
                                            speedBins))

                freqs = np.full(shape=0, fill_value=0)
                speeds = np.full(shape=0, fill_value=np.nan)
                directions = np.full(shape=0, fill_value=np.nan)

                # Calculating joint PMF while keeping track of each variable
                d = 0
                nDirs = hist[1].size-1
                nSpeeds = hist[2].size-1

                while d < nDirs:
                    direction = hist[1][d]
                    s = 0

                    while s < nSpeeds:
                        speed = hist[2][s]
                        # Calculate bin frequency
                        prop = hist[0][d][s]/N
                        freqs = np.concatenate((freqs, [prop]))
                        speeds = np.concatenate((speeds, [speed]))
                        directions = np.concatenate((directions, [direction]))
                        s += 1

                    d += 1

                freqs = pd.Series(freqs)
                freqs.index = pd.MultiIndex.from_arrays([directions, speeds],
                                                       names=('directionBins',
                                                              'speedBins'))

        return freqs

    # ***
    def pmf(self, direction=None, speed=None):
        """
        Returns the estimated probability of occurence for the wind condition 
        bin that contains the provided specs. Estimate is an empirical 
        probability based on the object's wind attribute (if one is provided) 
        or the experimental data (otherwise)

        Parameters
        ----------
        direction : int or float, optional
            Direction value that you want to know the probability of occurence for.
            The default is None in case you only want to use speed.
        
        speed : int or float, optional
            Speed value that you want to know the probability of occurence for.
            The default is None in case you only want to use direction.

        Returns
        -------
        float
            the estimated probability of occurence for the wind condition 
            bin that contains the provided specs.

        """

        if direction is None and speed is None:
            return 'Specify a speed and/or direction'

        if direction is not None:

            dirIdx = np.digitize(direction, self.directionBins)
            directionBin = self.directionBins[dirIdx]
            key = directionBin
            print("Direction bin:", directionBin)

        if speed is not None:
            speedIdx = np.digitize(speed, self.speedBins)
            speedBin = self.speedBins[speedIdx]
            key = speedBin
            print("Speed bin:", speedBin)

        if direction is not None and speed is not None:
            if dirIdx == (self.directionBins.size-1) and speedIdx == (self.speedBins.size-1):
                "Wind conditions out of bounds of the bins"
                return None

            if dirIdx == (self.directionBins.size-1):
                "Direction out of bounds of wind condition bins,"
            # Joint PMF
            key = (directionBin, speedBin)

        try:
            prob = self.pmf[key]
        except IndexError:
            print("Frequencies not computed for this wind condition.")
            print("Re-define wind condition bins to include this wind condition.")
            prob = np.nan

        return prob

    def scadaLonger(self, turbs='all', df=None):
        
        if df is None:
            df = self.scada.copy()
        
        if turbs == 'all':
            turbs = self.allTurbines
      
        powerColumns = ["pow_{:03.0f}".format(number) for number in turbs]
        keep = powerColumns + [self.wdCol, self.wsCol, "time"]
        df = df[keep].melt(value_vars=powerColumns,
                      value_name="power",
                      var_name="turbine", 
                      id_vars=['time', self.wdCol, self.wsCol])
        df.set_index(df["time"],inplace=True, drop=True)
        return df

    def binAdder(self,
                 copy=True,
                 filterBins = False):
        """
        Add columns for the lower bounds of the wind condition bins to scada data
        (or a copy of the scada data)

        Parameters
        ----------
        stepVars : string or list of strings, optional
            The variable(s) you want to use for the wind condition bins.
            Only options are "speed" and "direction". 
            The default is both: ["direction", "speed"].

        directionBins : numpy array of numerics, optional, optional
            Edges for the wind direction in degrees. Bin membership is 
            left edge inclusive, right edge exclusive. The default None means
            we will ignore wind direction for any calculations that use 
            wind conditions. Must be monotonic INCREASING (bins that share an 
            edge do not need to have their edges be repeated) and on [0,360).
            Rightmost bin edge is expected and will be used as a maximum cutoff for direction.
            Defaults to using object's directionBins attribute.

        speedBins : numpy array of numerics, optional, optional
            Edges for the wind speed in m/s. Bin membership is 
            left edge inclusive, right edge exclusive. The default None means
            we will ignore wind speed for any calculations that use 
            wind conditions. Must be monotonic INCREASING (bins that share an 
            edge do not need to have their edges be repeated) and on [0,360).
            Rightmost bin edge is expected and will be used as a maximum cutoff for speed.
            Defaults to using object's speedBins attribute.

        copy : boolean, optional
            Whether you want to return a copy of the df with the new 
            wind condition columns (copy=True) or add the wind condition bins 
            to the df directly (copy=False). The default is True.

        df : pandas data frame, optional
            The data frame to add the wind condition bins to. 
            Defaults to using object's scada attribute

        Returns
        -------
        df : pandas Data Frame
            scada data wit the new wind condition bin columns.

        """

        stepVars = []
        
        df = self.scada.copy()
        
        if self.directionBins is not None:
            df = df.loc[(df[self.wdCol] < self.directionBins[-1]) &
                        (df[self.wdCol] >= self.directionBins[0])]
            stepVars.append('direction')
            
           
        if self.speedBins is not None:
            df = df.loc[(df[self.wsCol] < self.speedBins[-1]) &
                        (df[self.wsCol] >= self.speedBins[0])]
            stepVars.append('speed')


        if 'direction' in stepVars:
            df['directionBin'] = self.directionBins[np.digitize(df[self.wdCol],
                                                                bins=self.directionBins)]
        if 'speed' in stepVars:
            df['speedBin'] = self.speedBins[np.digitize(df[self.wsCol],
                                                        bins=self.speedBins)]
        
        if not filterBins:
            df = df.merge(self.scada, how='outer')
        
        # Update self.scada if desired
        if not copy:
            self.scada = df

        # Return the copy with the bin columns
        return df

    # def averagePower(self,
    #                  turbineList='all', 
    #                  controlMode=True,
    #                  na_rm=False):
    #     """
    #     Computes the average power for the given list of turbines under the given control mode.
    #     Default computes average power over all turbines ignoring control mode.
    #     This is meant to be a quick operation

    #     Parameters
    #     ----------
    #     directionBins : TYPE, optional
    #         DESCRIPTION. The default is None.
    #     speedBins : TYPE, optional
    #         DESCRIPTION. The default is None.
    #     turbineList : TYPE, optional
    #         DESCRIPTION. The default is None.
    #     controlMode : string, optional
    #         "controlled", "baseline", "both", or None (ignores control mode). 
    #         The default is None.

    #     Returns
    #     -------
    #     TYPE
    #         DESCRIPTION.

    #     """


    #     if turbineList=='all':
    #         turbineList = self.allTurbines

    #     if self.speedBins is None and self.directionBins is None:
    #         return "Need bins for one of the wind conditions"

    #     # Select relevant rows
    #     dfTemp = self.binAdder(copy=True, filterBins=True)
        
    #     if controlMode in ['controlled', 'baseline']:
    #         dfTemp = dfTemp.loc[dfTemp['control_mode']==controlMode]
        
    #     dfLong = self.scadaLonger(turbineList, dfTemp)
        
    #     groupVarCols=[]
        
    #     if controlMode==True:
    #         groupVarCols.append('controlMode')
    #     if self.directionBins is not None:
    #         groupVarCols.append('directionBins')
    #     if self.speedBins is not None:
    #         groupVarCols.append('speedBins')
    #     if not na_rm:
    #         groupVarCols.append('turbine')
        
    #     averages = dfLong.groupby(by=groupVarCols, as_index=False).agg(averagePower=pd.NamedAgg(column='power',
    #                                                                             aggfunc=np.nanmean),
    #                                                    varPower=pd.NamedAgg(column='power',
    #                                                                         aggfunc=lambda x: np.var(x, ddof=1)),
    #                                                    nTurbineObvs=pd.NamedAgg(column="power",
    #                                                                                 aggfunc='count'))
    #     averages.index = averages[groupVarCols]
        
    #     if na_rm:
    #         averages['control_mode'] = controlMode
    #         return averages
        
    #     groupVarCols.pop()
    #     farmAverages = averages.groupby(by=groupVarCols, as_index=False).agg(averageFarmPower=pd.NamedAgg(column='averagePower',
    #                                                                             aggfunc=np.nansum),
    #                                                    nTurbineObvs=pd.NamedAgg(column="count",
    #                                                                                 aggfunc=np.nansum))
    #     farmAverages['control_mode'] = controlMode
    #     return farmAverages

    #     # if controlMode is not None:
    #     #     dfTemp = dfTemp.loc[(dfTemp['control_mode'] == controlMode)]

    #     # # Select only the columns that are for the desired turbines
    #     # # This only works for up to 1000 turbines, otherwise formatting gets messed up
    #     # powerColumns = ["pow_{:03.0f}".format(
    #     #     number) for number in turbineList]
    #     # dfPower = dfTemp[powerColumns]

    #     # # stopped here
    #     # # Add wind condition bins

    #     # if na_rm:
    #     #     dfBinned = dfTemp.groupby(by=groupVarCols).agg(averageTurbinePower=pd.NamedAgg(column='power',
    #     #                                                                                    aggfunc=np.mean),
    #     #                                                    varTurbinePower=pd.NamedAgg(column='power',
    #     #                                                                                aggfunc=lambda x: np.var(x, ddof=1)),
    #     #                                                    nTurbineObvs=pd.NamedAgg(column="power",
    #     #                                                                             aggfunc='count'))

    #     # avg = dfPower.mean(axis=None, skipna=True, numeric_only=True)

    #     # # Reshape
    #     # dfBinnedLong = self.binAll(retainControlMode=rcm,
    #     #                            windDirectionSpecs=windDirectionSpecs,
    #     #                            windSpeedSpecs=windSpeedSpecs,
    #     #                            stepVars=stepVars,
    #     #                            retainTurbineLabel=False,
    #     #                            retainTurbineNumbers=True,
    #     #                            group=False,
    #     #                            df=dfPow,
    #     #                            refTurbines=[])
    #     # # Get turbine-specific summary stats
    #     # dfBinnedLong = dfBinnedLong.sort_values(by=['turbine'])
    #     # dfBinnedTurbineStats = dfBinnedLong.groupby(by=groupVarCols).agg(averageTurbinePower=pd.NamedAgg(column='power',
    #     #                                                                                                  aggfunc=np.mean),
    #     #                                                                  varTurbinePower=pd.NamedAgg(column='power',
    #     #                                                                                              aggfunc=lambda x: np.var(x, ddof=1)),
    #     #                                                                  nTurbineObvs=pd.NamedAgg(column="power",
    #     #                                                                                           aggfunc='count'))

    #     # return avg

    # def powerRatio(self,
    #                controlMode, useReference=None, verbose=False):


    #    if self.speedBins is None and self.directionBins is None:
    #        return "Need bins for one of the wind conditions"

    #    # Select relevant rows
    #    dfTemp = self.binAdder(copy=True, filterBins=True)
       
    #    dfLong = self.scadaLonger(turbineList, dfTemp)
       
    #    stepVars=[]

    #    if 'direction' in stepVars:
    #        groupVarCols.append('directionBins')
    #    if 'speed' in stepVars:
    #        groupVarCols.append('speedBins')

       
    #    if useReference is None:
    #         useReference = self.useReference


    #     # Calculate Ratio
    #     numerator = self.averagePower(turbineList=self.testTurbines, controlMode=controlMode)

    #     if useReference:
    #         denominator = self.averagePower(turbineList=self.referenceTurbines, controlMode=controlMode)
    #         full = numerator.merge(denominator, on=stepVarCols, how='outer')
    #         full['powerRatio']
    #     else:
    #         numerator[f'powerRatio{controlMode}'] = numerator['averagePower'].copy()
    #         return numerator
    #         print("Reference turbines unused; calculating average power.")
       
        
    #    averages = dfLong.groupby(by=groupVarCols, as_index=False).agg(averagePower=pd.NamedAgg(column='power',
    #                                                                            aggfunc=np.nanmean),
    #                                                   varPower=pd.NamedAgg(column='power',
    #                                                                        aggfunc=lambda x: np.var(x, ddof=1)),
    #                                                   nTurbineObvs=pd.NamedAgg(column="power",
    #                                                                                aggfunc='count'))
    #    averages.index = averages[groupVarCols]
       
    #    if na_rm:
    #        return averages
       
    #    groupVarCols.pop()
    #    farmAverages = averages.groupby(by=groupVarCols, as_index=False).agg(averageFarmPower=pd.NamedAgg(column='averagePower',
    #                                                                            aggfunc=np.nansum),
    #                                                   nTurbineObvs=pd.NamedAgg(column="count",
    #                                                                                aggfunc=np.nansum))
       
    #    return farmAverages 
        
        
        
    #     # Assuming valid inputs for now

    #     # Wanted to add this to give flexibility to not use the reference for
    #     # one particular method, but part of me feels like this is just confusing
    #     # or a bad idea. Might take it away and just always use the object attribute
    #     if useReference is None:
    #         useReference = self.useReference


    #     # Calculate Ratio
    #     numerator = self.averagePower(windDirectionBin, windSpeedBin,
    #                                   self.testTurbines, controlMode=controlMode)

    #     if useReference:
    #         denominator = self.averagePower(windDirectionBin, windSpeedBin,
    #                                         self.referenceTurbines, controlMode=controlMode)
    #     else:
    #         denominator = 1
    #         print("Reference turbines unused; calculating average power.")


    #     return numerator/denominator

    # def changeInPowerRatio(self, windDirectionBin=None, windSpeedBin=None, useReference=None, verbose=False):
    #     """
    #     Change in Power Ratio for a specific wind direction bin and wind speed bin.

    #     windDirectionBin: list of length 2
    #     windSpeedBin: list of length 2
    #     """
    #     if useReference is None:
    #         useReference = self.useReference

    #     # Set wind speed if necessary
    #     if self.wsCol is None:
    #         self.setWS()

    #     # Set wind direction if necessary
    #     if self.wdCol is None:
    #         self.setWD()

    #     if windDirectionBin is None:
    #         windDirectionBin = self.defaultWindDirectionSpecs[0:2]

    #     if windSpeedBin is None:
    #         windSpeedBin = self.defaultWindSpeedSpecs[0:2]

    #     if useReference:
    #         # Typical power ratio formula if we are using reference turbines
    #         control = self.powerRatio(
    #             windDirectionBin, windSpeedBin, "controlled", useReference=True)
    #         baseline = self.powerRatio(
    #             windDirectionBin, windSpeedBin, "baseline", useReference=True)
    #     else:
    #         control = self.powerRatio(
    #             windDirectionBin, windSpeedBin, "controlled", useReference=False)
    #         baseline = self.powerRatio(
    #             windDirectionBin, windSpeedBin, "baseline", useReference=False)

    #         # I think this is important so I'm printing this regardless of verbose
    #         FYI = "Change in power ratio is simply change in average power without reference turbines.\n"
    #         FYI += "Returning change in average power. If this isn't what you want, set the useReference argument to True."
    #         print(FYI)

    #     # If either of these are strings,
    #     # there are no observations in this bin to calculate a ratio from
    #     if type(control) is str:
    #         sadMessage = control + "Can't calculate power ratio for controlled mode."
    #         if verbose:
    #             print(sadMessage)
    #         return sadMessage

    #     if type(baseline) is str:
    #         sadMessage = baseline + "Can't calculate power ratio for baseline mode."
    #         if verbose:
    #             print(sadMessage)
    #         return sadMessage

    #     return control - baseline

    # def percentPowerGain(self, windDirectionBin=None, windSpeedBin=None, useReference=None, verbose=False):
    #     """
    #     Percent Power Gain for a specific wind direction bin and wind speed bin.

    #     windDirectionBin: list of length 2
    #     windSpeedBin: list of length 2
    #     useReference: Boolean, wheter to compare Test turbines to Reference 
    #         turbines (True) or Test turbines to themselves in control mode
    #         versus baseline mode (False).
    #     """

    #     # Wanted to add this to give flexibility to not use the reference for
    #     # one particular method, but part of me feels like this is just confusing
    #     # or a bad idea. Might take it away and just always use the object attribute
    #     if useReference is None:
    #         useReference = self.useReference

    #     # Set wind speed if necessary
    #     if self.wsCol is None:
    #         self.setWS()

    #     # Set wind direction if necessary
    #     if self.wdCol is None:
    #         self.setWD()

    #     if windDirectionBin is None:
    #         windDirectionBin = self.defaultWindDirectionSpecs[0:2]

    #     if windSpeedBin is None:
    #         windSpeedBin = self.defaultWindSpeedSpecs[0:2]

    #     # If useReference==False, this simplifies to average power
    #     control = self.powerRatio(
    #         windDirectionBin, windSpeedBin, "controlled", useReference)
    #     baseline = self.powerRatio(
    #         windDirectionBin, windSpeedBin, "baseline", useReference)

    #     # If either of these are strings,
    #     # there are no observations in this bin to calculate a ratio or average power from
    #     if type(control) is str:
    #         sadMessage = control + "Can't calculate power ratio for controlled mode."
    #         if verbose:
    #             print(sadMessage)
    #         return sadMessage

    #     if type(baseline) is str:
    #         sadMessage = baseline + "Can't calculate power ratio for baseline mode."
    #         if verbose:
    #             print(sadMessage)
    #         return sadMessage

    #     return (control - baseline)/baseline

    def binAll(self, retainControlMode=True,
               retainTurbineLabel=True, retainTurbineNumbers=False,
               filterBins=True, long=True):
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


        # Add bins to the data
        df = self.binAdder(copy=True, filterBins=filterBins)
        
        stepVarCols = []
        if self.directionBins is not None:
            stepVarCols.append('directionBin')
        if self.speedBins is not None:
            stepVarCols.append('speedBin')
            
        # Exclude undesirable turbines
        powerColumns = ["pow_{:03.0f}".format(number) for number in self.referenceTurbines + self.testTurbines]
        colsToKeep = stepVarCols[:]
        colsToKeep.append("time")
        if retainControlMode:
            colsToKeep.append("control_mode")
        df = df.loc[:,colsToKeep + powerColumns]
        
        if not long:
            df['totalPower'] = df.loc[:,powerColumns].sum(1, skipna=True)
            return df
        
        # Pivot Longer
        dfLong = df.melt(id_vars=colsToKeep,
                         value_name="power", var_name="turbine")

        # Convert turbine numbers from strings to integers
        dfLong["turbine"] = dfLong["turbine"].str.removeprefix("pow_")
        dfLong["turbine"] = dfLong["turbine"].to_numpy(dtype=int)

        # Add turbine label column
        if retainTurbineLabel:
            labels = [(num in self.testTurbines) for num in dfLong["turbine"]]
            labels = np.where(labels, "test", "reference")
            dfLong["turbineLabel"] = labels
            colsToKeep.append("turbineLabel")

        

        return dfLong

    def averagePower(self, 
                      retainControlMode=True, 
                      retainTurbineLabel=True, 
                      retainTurbineNumbers=False,
                      filterBins=True,
                      dropna=True):

        featuresToRetain = []
        if self.directionBins is not None:
            featuresToRetain.append('directionBin')
        if self.speedBins is not None:
            featuresToRetain.append('speedBin')
        if retainControlMode:
            featuresToRetain.append('control_mode')
        if retainTurbineLabel:
            featuresToRetain.append('turbineLabel')
        if retainTurbineNumbers:
            featuresToRetain.append('turbine')
             
        # Calculating average by group
        breakpoint()
        # if dropna:
            
        #     if retainTurbineNumbers:
        #         dfLong = self.binAll(retainControlMode=retainControlMode, 
        #                           retainTurbineLabel=retainTurbineLabel, 
        #                           retainTurbineNumbers=retainTurbineNumbers,
        #                           filterBins=filterBins,
        #                           long=True)
                
        #         dfGrouped = dfLong.groupby(by=featuresToRetain).agg(averagePower=pd.NamedAgg(column="power",
        #                                                                         aggfunc=np.nanmean),
        #                                               numObvs=pd.NamedAgg(column="power",
        #                                                                   aggfunc='count'))
        #     else:
        #         dfTotal = self.binAll(retainControlMode=retainControlMode, 
        #                           retainTurbineLabel=retainTurbineLabel, 
        #                           retainTurbineNumbers=retainTurbineNumbers,
        #                           filterBins=filterBins,
        #                           long=False)
                
        #         dfGrouped = dfTotal.groupby(by=featuresToRetain).agg(averagePower=pd.NamedAgg(column="totalPower",
        #                                                                         aggfunc=np.nanmean),
        #                                               numObvs=pd.NamedAgg(column="totalPower",
        #                                                                   aggfunc='count'))
        # else:
            
        dfLong = self.binAll(retainControlMode=retainControlMode, 
                              retainTurbineLabel=retainTurbineLabel, 
                              retainTurbineNumbers=retainTurbineNumbers,
                              filterBins=filterBins,
                              long=True)
            
        if not retainTurbineNumbers:
            featuresToRetain.append('turbine')
            
        dfTurbine = dfLong.groupby(by=featuresToRetain).agg(averageTurbinePower=pd.NamedAgg(column="power",
                                                                                aggfunc=np.mean),
                                                      numObvs=pd.NamedAgg(column="power",
                                                                          aggfunc='count'))
        dfGrouped = dfTurbine.copy()
            
        if not retainTurbineNumbers:
            featuresToRetain.pop()
            dfGrouped = dfTurbine.groupby(by=featuresToRetain).agg(averageFarmPower=pd.NamedAgg(column="averageTurbinePower",
                                                                                aggfunc=np.nansum),
                                                      numTurbines=pd.NamedAgg(column="averageTurbinePower",
                                                                          aggfunc='count'),
                                                      numObvs=pd.NamedAgg(column="numObvs",
                                                                          aggfunc=np.sum))
            
        breakpoint()
        # Convert grouping index into columns for easier pivoting
        for var in featuresToRetain:
            dfGrouped[var] = dfGrouped.index.get_level_values(var)

        # # Pivot wider
        # if returnWide:
        #     optionalCols = list(set(colsToKeep) - set(stepVarCols))

        #     dfWide = dfGrouped.pivot(columns=optionalCols, index=stepVarCols,
        #                               values=['averagePower', 'numObvs'])
        #     return dfWide

        # # Don't need these columns anymore since they are a part of the multi-index
        # dfGrouped.drop(columns=colsToKeep, inplace=True)
        return dfGrouped

    # Fix comments later
    def computeAll(self, stepVars=["direction", "speed"],
                   windDirectionSpecs=None, windSpeedSpecs=None,
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

        if windDirectionSpecs is None:
            windDirectionSpecs = self.defaultWindDirectionSpecs

        if windSpeedSpecs is None:
            windSpeedSpecs = self.defaultWindSpeedSpecs

        if type(stepVars) is str:
            stepVars = list([stepVars])

        if df is None:
            df = self.binAll(stepVars=stepVars,
                             windDirectionSpecs=windDirectionSpecs,
                             windSpeedSpecs=windSpeedSpecs,
                             df=df)

        # Sometimes the order of the labels in this tuple seem to change and I haven't figured out why. This should fix the order.
        df = df.reorder_levels([None, "turbineLabel", "control_mode"], axis=1)

        if useReference:
            df["powerRatioBaseline"] = np.divide(df[('averagePower', 'test', 'baseline')],
                                                 df[('averagePower', 'reference', 'baseline')])
            df["powerRatioControl"] = np.divide(df[('averagePower', 'test', 'controlled')],
                                                df[('averagePower', 'reference', 'controlled')])
            df["totalNumObvs"] = np.nansum(np.dstack((df[('numObvs', 'test', 'controlled')],
                                                      df[('numObvs', 'reference',
                                                          'controlled')],
                                                      df[('numObvs', 'test',
                                                          'baseline')],
                                                      df[('numObvs', 'reference', 'baseline')])),
                                           axis=2)[0]

        else:
            df["powerRatioBaseline"] = df[('averagePower', 'test', 'baseline')]
            df["powerRatioControl"] = df[(
                'averagePower', 'test', 'controlled')]

            df["totalNumObvs"] = np.nansum(np.dstack((df[('numObvs', 'test', 'controlled')],
                                                      df[('numObvs', 'test', 'baseline')])),
                                           axis=2)[0]

            df["totalNumObvsInclRef"] = np.nansum(np.dstack((df["totalNumObvs"],
                                                             df[('numObvs', 'reference',
                                                                 'controlled')],
                                                             df[('numObvs', 'reference', 'baseline')])),
                                                  axis=2)[0]

        # Same for both AEP methods
        if self.pmf is None:
            N = np.nansum(df["totalNumObvs"])
            df["freq"] = df["totalNumObvs"]/N
        else:
            df["freq"] = self.pmf(df)

        df["changeInPowerRatio"] = np.subtract(df['powerRatioControl'],
                                               df['powerRatioBaseline'])

        df["percentPowerGain"] = np.divide(df["changeInPowerRatio"],
                                           df['powerRatioControl'])

        # Make columns out of the indices just because it's easier to see sometimes
        stepVarCols = ["{}BinLowerBound".format(var) for var in stepVars]
        for var in stepVarCols:
            df[var] = df.index.get_level_values(var)

        return df

    def aep(self, windDirectionSpecs=None, windSpeedSpecs=None,
            hours=8760, useReference=None, df=None):

        if self.wdCol is None:
            self.setWD()

        # Set wind speed if necessary
        if self.wsCol is None:
            self.setWS()

        if windDirectionSpecs is None:
            windDirectionSpecs = self.defaultWindDirectionSpecs
        windDirectionBin = windDirectionSpecs[0:2]

        if windSpeedSpecs is None:
            windSpeedSpecs = self.defaultWindSpeedSpecs
        windSpeedBin = windSpeedSpecs[0:2]

        if useReference is None:
            useReference = self.useReference

        if df is None:
            df = self.computeAll(stepVars=["speed", "direction"],
                                 windDirectionSpecs=windDirectionSpecs,
                                 windSpeedSpecs=windSpeedSpecs,
                                 df=df,
                                 useReference=useReference)

        df["aepContribution"] = np.multiply(df[('averagePower', 'test', 'baseline')],
                                            df[('freq', '', '')])

        return hours*np.nansum(df["aepContribution"])

    # Fix comments later

    def aepGain(self, windDirectionSpecs=None, windSpeedSpecs=None,
                hours=8760, aepMethod=1, absolute=False, useReference=None, df=None):
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
        if windDirectionSpecs is None:
            windDirectionSpecs = self.defaultWindDirectionSpecs

        if windSpeedSpecs is None:
            windSpeedSpecs = self.defaultWindSpeedSpecs

        if useReference is None:
            useReference = self.useReference

        if not useReference:
            # Both methods are equivalent when reference turbines aren't used,
            aepMethod = 1

        # Calculate nicely formatted df if needed
        if df is None:
            df = self.computeAll(stepVars=["speed", "direction"],
                                 windDirectionSpecs=windDirectionSpecs,
                                 windSpeedSpecs=windSpeedSpecs,
                                 df=df,
                                 useReference=useReference)

        # Different AEP formulas
        if aepMethod == 1:
            if useReference:
                df["aepGainContribution"] = np.multiply(np.multiply(df[('averagePower', 'test', 'baseline')],
                                                                    df[('percentPowerGain', '', '')]),
                                                        df[('freq', '', '')])
            else:
                df["aepGainContribution"] = np.multiply(
                    df["changeInPowerRatio"], df[('freq', '', '')])

            if not absolute:
                denomTerms = np.multiply(
                    df[('averagePower', 'test', 'baseline')], df[('freq', '', '')])

        else:
            # Couldn't find an element-wise weighted mean, so I did this
            sumPowerRefBase = np.multiply(df[('averagePower', 'reference', 'baseline')],
                                          df[('numObvs', 'reference', 'baseline')])
            sumPowerRefcontrolled = np.multiply(df[('averagePower', 'reference', 'controlled')],
                                                df[('numObvs', 'reference', 'controlled')])

            sumPowerRef = np.nansum(
                np.dstack((sumPowerRefBase, sumPowerRefcontrolled)), 2)[0]

            numObvsRef = np.nansum(np.dstack((df[('numObvs', 'reference', 'controlled')], df[(
                'numObvs', 'reference', 'baseline')])), 2)[0]

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
        # print(aep)
        return (df, aep)

    def TNOaverageTurbinePower(self, controlMode,
                               stepVars=['direction', 'speed'],
                               windDirectionSpecs=None,
                               windSpeedSpecs=None, farmStats=True):
        """
        Returns a pandas dataframe with turbine specific summary statistics 
        within a wind condition bin. May also return a dictionary containing
        said data frame and other information helpful for calculating farm 
        statistics if farmStats=True.

        Parameters
        ----------
        controlMode : str or NoneType
            "controlled", "baseline", "both", "neither", "none", or None. 
            "both" will include summary statistics within each control mode. 
            "none", "neither", or None will ignore control mode when calculating summary statistics. 
        stepVars : TYPE, optional
            DESCRIPTION. The default is ['direction','speed'].
        windDirectionSpecs : TYPE, optional
            DESCRIPTION. The default is None.
        windSpeedSpecs : TYPE, optional
            DESCRIPTION. The default is None.
        farmStats : TYPE, optional
            DESCRIPTION. The default is True.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """

        if windDirectionSpecs is None:
            windDirectionSpecs = self.defaultWindDirectionSpecs

        if windSpeedSpecs is None:
            windSpeedSpecs = self.defaultWindSpeedSpecs

        if type(stepVars) is str:
            stepVars = list([stepVars])

        df = self.binAdder(stepVars=stepVars,
                           windDirectionSpecs=windDirectionSpecs,
                           windSpeedSpecs=windSpeedSpecs,
                           copy=True,
                           df=None)

        powerColumns = ["pow_{:03.0f}".format(
            number) for number in self.testTurbines]
        stepVarCols = ["{}BinLowerBound".format(var) for var in stepVars]
        groupVarCols = stepVarCols[:]
        groupVarCols.append('turbine')
        cols = powerColumns[:] + stepVarCols[:] + \
            [self.wdCol, self.wsCol, 'time']
        rcm = False

        dfWithBins = df.copy()

        if controlMode == 'both':
            cols.append("control_mode")
            groupVarCols.append("control_mode")
            rcm = True
        elif controlMode in ['baseline', 'controlled']:
            dfWithBins = dfWithBins.loc[df['control_mode'] == controlMode]

        # Get only the columns with power measurements
        dfPow = dfWithBins[cols]
        # Reshape
        dfBinnedLong = self.binAll(retainControlMode=rcm,
                                   windDirectionSpecs=windDirectionSpecs,
                                   windSpeedSpecs=windSpeedSpecs,
                                   stepVars=stepVars,
                                   retainTurbineLabel=False,
                                   retainTurbineNumbers=True,
                                   group=False,
                                   df=dfPow,
                                   refTurbines=[])
        # Get turbine-specific summary stats
        dfBinnedLong = dfBinnedLong.sort_values(by=['turbine'])
        dfBinnedTurbineStats = dfBinnedLong.groupby(by=groupVarCols).agg(averageTurbinePower=pd.NamedAgg(column='power',
                                                                                                         aggfunc=np.mean),
                                                                         varTurbinePower=pd.NamedAgg(column='power',
                                                                                                     aggfunc=lambda x: np.var(x, ddof=1)),
                                                                         nTurbineObvs=pd.NamedAgg(column="power",
                                                                                                  aggfunc='count'))

        dfBinnedTurbineStats['sdTurbinePower'] = np.sqrt(
            dfBinnedTurbineStats['varTurbinePower'])

        dfBinnedTurbineStats['varAvgTurbinePower'] = np.divide(dfBinnedTurbineStats['varTurbinePower'],
                                                               dfBinnedTurbineStats['nTurbineObvs'])

        dfBinnedTurbineStats['seTurbinePower'] = np.sqrt(
            dfBinnedTurbineStats['varAvgTurbinePower'])

        if farmStats:
            # This dictionary contains information needed to compute other farm stats
            return {'dfTurbine': dfBinnedTurbineStats, 'dfWithBins': dfWithBins,
                    'stepVars': stepVars, 'stepVarCols': stepVarCols, 'powerColumns': powerColumns}

        return dfBinnedTurbineStats

    def TNOaverageFarmPower(self, controlMode, stepVars=['direction', 'speed'],
                            windDirectionSpecs=None,
                            windSpeedSpecs=None, TNOatpDict=None):
        """
        Finds the average farm power for each wind condition bin, as well as the standard error

        Parameters
        ----------
        controlMode : str or NoneType
            "controlled", "baseline", "both", "neither", "none", or None. 
            "both" will include summary statistics within each control mode. 
            "none", "neither", or None will ignore control mode when calculating summary statistics. 
        stepVars : TYPE, optional
            DESCRIPTION. The default is ['direction','speed'].
        windDirectionSpecs : TYPE, optional
            DESCRIPTION. The default is None.
        windSpeedSpecs : TYPE, optional
            DESCRIPTION. The default is None.
        TNOatpDict : dict, optional
            The output of TNOaverageTurbinePower when farmStats=True. The default is None.

        Returns
        -------
        None.

        """

        if windDirectionSpecs is None:
            windDirectionSpecs = self.defaultWindDirectionSpecs

        if windSpeedSpecs is None:
            windSpeedSpecs = self.defaultWindSpeedSpecs

        if type(stepVars) is str:
            stepVars = list([stepVars])

        if TNOatpDict is None:
            TNOatpDict = self.TNOaverageTurbinePower(controlMode=controlMode,
                                                     stepVars=stepVars,
                                                     windDirectionSpecs=windDirectionSpecs,
                                                     windSpeedSpecs=windSpeedSpecs,
                                                     farmStats=True)

        dfTurbine = TNOatpDict['dfTurbine']
        # The last index will be for the turbines, which we don't want to include here
        groupByCols = dfTurbine.index.names[:-1]
        dfFarm = dfTurbine.groupby(groupByCols).agg(averageFarmPower=pd.NamedAgg(column='averageTurbinePower',
                                                                                 aggfunc=np.sum),
                                                    nTurbs=pd.NamedAgg(column='averageTurbinePower',
                                                                       aggfunc='count'),
                                                    sumVarAvgTurbinePower=pd.NamedAgg(column='varAvgTurbinePower',
                                                                                      aggfunc=np.sum))

        dfFarmVar = self.__TNOvarFarmPower__(TNOatpDict)

        dfFarm = dfFarm.merge(dfFarmVar, left_index=True, right_index=True)
        dfFarm['sdFarmPower'] = np.sqrt(np.asarray(dfFarm['varFarmPower']))
        dfFarm['seFarmPower'] = np.sqrt(np.asarray(dfFarm['varAvgFarmPower']))

        return dfFarm

    def TNOturbinePowerCovarianceMatrix(self, df):
        """
        Computes the covariance matrix of all turbine power measurements in df. 

        Parameters
        ----------
        df : TYPE
            Must contain only the columns of the power measurements for 
            turbines you want to compute the covariance between (including missing power entries).
            If the covariance within a specific wind condition bin is desired, df must already be filtered for those conditions.

        Returns
        -------
        covTurbPowerMat : TYPE
            DESCRIPTION.
        nTurbPowerPairsMat : TYPE
            DESCRIPTION.

        """

        nTurbs = len(list(df))

        # Initialize empty covariance matrix for this wind condition bin
        covTurbPowerMat = np.full(
            shape=(nTurbs, nTurbs), fill_value=np.nan, dtype=float)
        nTurbPowerPairsMat = np.full(
            shape=(nTurbs, nTurbs), fill_value=0, dtype=int)

        # Indices of all entries in upper triangle of covariance matrix
        colComboIdxs = np.triu_indices(n=nTurbs, m=nTurbs, k=0)
        numCombos = colComboIdxs[0].size

        # This will leave the diagonal alone for now
        for combo in range(numCombos):
            col1idx = colComboIdxs[0][combo]
            col2idx = colComboIdxs[1][combo]

            # Could have used the k=1 argument in triu_indices but I still needed to access the number of rows for the individual turbines
            if col1idx == col2idx:
                turbMeasurements = df.iloc[:, col1idx].dropna()
                nTurbPairs = turbMeasurements.shape[0]
            else:
                # Data frame of just the two turbines in question
                df2Turbs = df.iloc[:, [col1idx, col2idx]]
                df2Turbs = df2Turbs.dropna()
                covTurbPowerMat2Turbs = df2Turbs.cov(ddof=1)
                # Covariance between these two turbines
                turbCov = covTurbPowerMat2Turbs.iloc[0, 1]
                nTurbPairs = df2Turbs.shape[0]

                # Insert covariance in Upper triangle of covariance matrix
                covTurbPowerMat[col1idx, col2idx] = turbCov
                # Insert covariance in Lower triangle
                covTurbPowerMat[col2idx, col1idx] = turbCov

            # Save number of non-mising pairs
            nTurbPowerPairsMat[col2idx, col1idx] = nTurbPairs
            nTurbPowerPairsMat[col1idx, col2idx] = nTurbPairs

        # Fill the diagonal of this bin's covariance matrix with turbine variances
        variances = df.var(axis=0, skipna=True, ddof=1)
        np.fill_diagonal(covTurbPowerMat, val=np.asarray(variances))

        return {'turbine power covariance matrix': covTurbPowerMat,
                'number non-missing pairs matrix': nTurbPowerPairsMat,
                'columns used': list(df)}

    def TNOaverageTurbinePowerCovarianceMatrix(self, df=None, covTurbPowerMat=None,
                                               TurbPowerPairsMat=None, variances=None, returnCovTurbPower=None):

        if returnCovTurbPower is None:
            returnCovTurbPower = False

        if covTurbPowerMat is None:
            dct = self.TNOturbinePowerCovarianceMatrix(df)
            covTurbPowerMat = dct['turbine power covariance matrix']
            nTurbPowerPairsMat = dct['number non-missing pairs matrix']
            returnCovTurbPower = True
        if variances is None:
            variances = np.diag(covTurbPowerMat)

        covMatAvgTurbPower = np.divide(covTurbPowerMat, nTurbPowerPairsMat)

        if returnCovTurbPower:
            return {'turbine average power covariance matrix': covMatAvgTurbPower,
                    'turbine power covariance matrix': covTurbPowerMat,
                    'number non-missing pairs matrix': nTurbPowerPairsMat}

        return covMatAvgTurbPower

    def __TNOvarFarmPower__(self, TNOatpDict):

        dfBinnedTurbineStats = TNOatpDict['dfTurbine']
        stepVars = TNOatpDict['stepVars']
        stepVarCols = TNOatpDict['stepVarCols']
        powerColumns = TNOatpDict['powerColumns']
        dfWithBins = TNOatpDict['dfWithBins']

        nvars = len(stepVars)
        # Make a list of the wind condition bins
        binList = [i[:nvars] for i in dfBinnedTurbineStats.index]

        # Need to compute the sum of all pairwise covariances for all the turbines in a  given wind direction
        farmPowerVar = {}
        farmAvgPowerVar = {}

        if nvars == 2:
            speedIDX = dfBinnedTurbineStats.index.names.index('speedBins')
            directionIDX = dfBinnedTurbineStats.index.names.index(
                'directionBins')

        # Go through each wind condition bin
        for condition in binList:
            if nvars == 2:
                direction = condition[directionIDX]
                speed = condition[speedIDX]
                # Filter for power observations in this wind condition bin
                dfCurrent = dfWithBins.loc[(dfWithBins['directionBins'] == direction) &
                                           (dfWithBins['speedBins'] == speed)]
                key = f'{direction},{speed}'
            else:
                val = condition[0]
                # Filter for power observations in this wind condition bin
                dfCurrent = dfWithBins.loc[(dfWithBins[stepVarCols[0]] == val)]
                key = f'{val}'

            # Select only the columns with the relevant power measurements
            dfCurrent = dfCurrent[powerColumns]

            # Get the covariance matrices for turbine power and average turbine power
            results = self.TNOaverageTurbinePowerCovarianceMatrix(
                df=dfCurrent, returnCovTurbPower=True)

            avgPowerCov = results['turbine average power covariance matrix']
            powerCov = results['turbine power covariance matrix']

            farmAvgPowerVar[key] = np.sum(avgPowerCov)
            farmPowerVar[key] = np.sum(powerCov)

        # Convert dictionaries to a pandas data frame
        dfFarmPowerVar = self.__TNOvarFarmPowerDictToDf__(dctFarmPower=farmPowerVar,
                                                          dctFarmAvgPower=farmAvgPowerVar,
                                                          stepVars=stepVars)

        return dfFarmPowerVar

    def __TNOvarFarmPowerDictToDf__(self, dctFarmPower, dctFarmAvgPower, stepVars):
        """
        Converts the intermediate dictionary from TNOvarFarmPower into a pandas data frame

        Parameters
        ----------
        dctFarmPower : TYPE
            DESCRIPTION.
        dctAvgFarmPower : TYPE
            DESCRIPTION.
        stepVars : TYPE
            DESCRIPTION.

        Returns
        -------
        farmPowerVarDF : TYPE
            DESCRIPTION.

        """

        directionBins = np.ndarray(0, dtype=float)
        speedBins = np.ndarray(0, dtype=float)
        varFarmPower = np.ndarray(0, dtype=float)
        varAvgFarmPower = np.ndarray(0, dtype=float)

        # Creating arrays that will become series in the df we're constructing
        for key in dctFarmPower.keys():
            variables = key.split(',')

            # Figure out what wind condition bin this belongs in
            if len(variables) == 2:
                directionBins = np.append(directionBins, float(variables[0]))
                speedBins = np.append(speedBins, float(variables[1]))
            elif 'direction' in stepVars:
                directionBins = np.append(directionBins, float(variables[0]))
            else:
                speedBins = np.append(speedBins, float(variables[0]))

            # Store the power variance for this wind condition bin
            varFarmPower = np.append(varFarmPower, dctFarmPower[key])

            # Store the average power variance for this wind condition bin
            if key in dctFarmAvgPower.keys():
                varAvgFarmPower = np.append(
                    varAvgFarmPower, dctFarmAvgPower[key])
            else:
                varAvgFarmPower = np.append(varAvgFarmPower, np.nan)

        # After going through all bins, convert the arrays into a data frame
        dct = {'varFarmPower': varFarmPower,
               'varAvgFarmPower': varAvgFarmPower}

        if 'direction' in stepVars:
            dct['directionBins'] = directionBins

        if 'speed' in stepVars:
            dct['speedBins'] = speedBins

        farmPowerVarDF = pd.DataFrame(dct)

        # Merging later is easier if this is an index
        if len(stepVars) == 2:
            farmPowerVarDF.index = pd.MultiIndex.from_arrays(
                [farmPowerVarDF[f'{var}BinLowerBound'] for var in stepVars])
        else:

            farmPowerVarDF.index = pd.Index(
                farmPowerVarDF[f'{stepVars[0]}BinLowerBound'])

        return farmPowerVarDF

    def TNOpowerRatio(self, stepVars=['direction', 'speed'],
                      windDirectionSpecs=None,
                      windSpeedSpecs=None,
                      one='controlled',
                      two='baseline', seMultiplier=2):

        if windDirectionSpecs is None:
            windDirectionSpecs = self.defaultWindDirectionSpecs

        if windSpeedSpecs is None:
            windSpeedSpecs = self.defaultWindSpeedSpecs

        if type(stepVars) is str:
            stepVars = list([stepVars])

        bin_specificFarmStats1 = self.TNOaverageFarmPower(stepVars=stepVars,
                                                          windDirectionSpecs=windDirectionSpecs,
                                                          windSpeedSpecs=windSpeedSpecs,
                                                          controlMode=one)

        bin_specificFarmStats2 = self.TNOaverageFarmPower(stepVars=stepVars,
                                                          windDirectionSpecs=windDirectionSpecs,
                                                          windSpeedSpecs=windSpeedSpecs,
                                                          controlMode=two)

        farmStats = bin_specificFarmStats1.merge(
            bin_specificFarmStats2, how='outer', left_index=True, right_index=True, suffixes=('_1', '_2'))

        farmStats['powerRatioEstimate'] = np.divide(
            farmStats['averageFarmPower_1'], farmStats['averageFarmPower_2'])
        varPowerRatioNumerator = np.add(farmStats['varFarmPower_1'],
                                        np.multiply(np.multiply(farmStats['powerRatioEstimate'], farmStats['powerRatioEstimate']),
                                                    farmStats['varFarmPower_2']
                                                    )
                                        )
        farmStats['varPowerRatio'] = np.divide(varPowerRatioNumerator,
                                               np.multiply(
                                                   farmStats['averageFarmPower_2'], farmStats['averageFarmPower_2'])
                                               )
        varPowerRatioEstNumerator = np.add(farmStats['varAvgFarmPower_1'],
                                           np.multiply(np.multiply(farmStats['powerRatioEstimate'], farmStats['powerRatioEstimate']),
                                                       farmStats['varAvgFarmPower_2']
                                                       )
                                           )
        farmStats['varPowerRatioEst'] = np.divide(varPowerRatioEstNumerator,
                                                  np.multiply(
                                                      farmStats['averageFarmPower_2'], farmStats['averageFarmPower_2'])
                                                  )

        farmStats['sdPowerRatio'] = np.sqrt(farmStats['varPowerRatio'])
        farmStats['sePowerRatio'] = np.sqrt(farmStats['varPowerRatioEst'])
        farmStats['powerRatioCIlower'] = farmStats['powerRatioEstimate'] - \
            seMultiplier*farmStats['sePowerRatio']
        farmStats['powerRatioCIupper'] = farmStats['powerRatioEstimate'] + \
            seMultiplier*farmStats['sePowerRatio']

        return farmStats

    def plot2DTNOpowerRatio(self, TNOprDF, windDirectionSpecs=None, windSpeedSpecs=None):
        if windDirectionSpecs is None:
            windDirectionSpecs = self.defaultWindDirectionSpecs

        if windSpeedSpecs is None:
            windSpeedSpecs = self.defaultWindSpeedSpecs

        # Empty matrix to hold all possible combinations
        directionEdges = np.arange(*windDirectionSpecs)
        speedEdges = np.arange(*windSpeedSpecs)
        matrix = np.full(
            shape=(speedEdges.size, directionEdges.size), fill_value=np.nan, dtype=float)
        idxs = [(iS, iD) for iS in range(speedEdges.size)
                for iD in range(directionEdges.size)]
        # Fill matrix
        for idx in idxs:
            try:

                if TNOprDF.loc[(directionEdges[idx[1]], speedEdges[idx[0]])]['powerRatioCIupper'] < 1:
                    matrix[idx[0], idx[1]] = -1

                elif TNOprDF.loc[(directionEdges[idx[1]], speedEdges[idx[0]])]['powerRatioCIlower'] > 1:
                    matrix[idx[0], idx[1]] = 1

                elif pd.isna(TNOprDF.loc[(directionEdges[idx[1]], speedEdges[idx[0]])]['powerRatioCIlower']) == False and pd.isna(TNOprDF.loc[(directionEdges[idx[1]], speedEdges[idx[0]])]['powerRatioCIlower']) == False:
                    matrix[idx[0], idx[1]] = 0
            except KeyError:
                continue
        # plot matrix
        plt.clf()
        # sns.set_theme(style="whitegrid")
        fig, ax = plt.subplots(figsize=(10, 5), layout='constrained')

        ax = sns.heatmap(matrix, center=0, square=False, linewidths=1,
                         ax=ax, cbar=True, cbar_kws={"shrink": .8},
                         annot=False, cmap=sns.color_palette("coolwarm_r", as_cmap=True), robust=True)

        # ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.5))
        ax.invert_yaxis()
        # ax.xaxis.set_minor_locator(mticker.MultipleLocator(0.5))

        ax.tick_params(which="major", bottom=True, length=0, color='#4C4C4C',
                       grid_linewidth=0)
        ax.tick_params(which="minor", bottom=True, length=10, color='black')
        # #ax.set_xticklabels(xlabels, rotation=-90)
        # #ax.set_yticklabels(ylabels, rotation=0)
        ax.grid(which='major', visible=False, color='black', linestyle='-',
                linewidth=0)
        ax.grid(which='minor', visible=True, color='black', linestyle='-',
                linewidth=1)

        ax.tick_params(which="minor", bottom=True, length=0, color='white')

        # Labels
        # unicode formatted
        fig.supxlabel(u"Wind Direction (\N{DEGREE SIGN})", fontsize=15)
        fig.supylabel("Wind Speed (m/s)", fontsize=15)
        fig.suptitle("title", fontsize=17)
        plt.show()

        return None

    def TNOexpectedPowerProduction(self, dfTNOpowerRatio, controlModeNumber=1, narm=False):
        """
        This essentailly calculate the numerator or denominator for TNO's 
        version of AEP

        Some differences from the TNO report:

        The TNO report calls this a 'weighted average power production', 
        but the formula they use is essentially one for the expected value of a 
        random variable with a defined probability function, so I'm calling this
        Expected Power Production (see equations 4.23-4.25 in the TNO report).
        I should maybe also mention that I haven't been too concious of consistent 
        terminology for expected values versus averages in the rest of this module.

        The TNO report suggests two methods for computing the weights, one of 
        which is based on the wind condition distributions for  the duraction of the AWC campaign.
        Here, the weights are essemtially whatever is assigned to the PMF attribute.
        This **could** be emprically derived from the campaign, but is empirically 
        derived from long term site-specific wind data in the SMARTE-OLE example.


        The TNO report seems to require that the weights be based on the joint
        distribution of both wind speed and direction. The PMF attribute can 
        be a marginal distribution, so we don't have that requirement here. 
        Of course, a marginal distribution is only appropriate when the bins are 
        incrementing over one variable and aggregated over the rest. 
        There is currently not a built-in check for this.

        Parameters
        ----------
        dfTNOpowerRatio : pandas data frame
            The data frame that is returned by a call to TNOpowerRatio.


        Returns
        -------
        Float

        """
        # If there is no joint PMF provided, use the empirical frequencies for the campaign
        if self.pmf is None:
            # empirical PMF behavioris currently less than ideal
            pmf = self.__empiricalPMF__(dfTNOpowerRatio)
            pd.dfTNOpowerRatio.merge(pmf, how='left',
                                     left_on=[
                                         f'{var}_1' for var in dfTNOpowerRatio.index.names],
                                     right_on=[var for var in dfTNOpowerRatio.index.names])

        else:
            dfTNOpowerRatio['binDensity'] = self.pmf(dfTNOpowerRatio)

        if narm:
            dfTNOpowerRatio = dfTNOpowerRatio.loc[(~dfTNOpowerRatio['averageFarmPower_1'].isna()) &
                                                  (~dfTNOpowerRatio['averageFarmPower_2'].isna())]

        avgAEPterms = np.multiply(dfTNOpowerRatio['binDensity'],
                                  dfTNOpowerRatio[f'averageFarmPower_{controlModeNumber}'])

        return np.nansum(avgAEPterms)

    def TNOannualPowerRatio(self, dfTNOpowerRatio):

        # Annual Power Ratio
        aap1 = self.TNOexpectedPowerProduction(
            dfTNOpowerRatio, controlModeNumber=1, narm=True)
        aap2 = self.TNOexpectedPowerProduction(
            dfTNOpowerRatio, controlModeNumber=2, narm=True)
        apr = aap1/aap2

        if self.pmf is None:
            # empirical PMF behavioris currently less than ideal
            pmf = self.__empiricalPMF__(dfTNOpowerRatio)
            pd.dfTNOpowerRatio.merge(pmf, how='left',
                                     left_on=[
                                         f'{var}_1' for var in dfTNOpowerRatio.index.names],
                                     right_on=[var for var in dfTNOpowerRatio.index.names])
        else:
            dfTNOpowerRatio['binDensity'] = self.pmf(dfTNOpowerRatio)

        # Variance, TNO equation 4.28
        dfTNOpowerRatio['binDensitySquared'] = np.multiply(dfTNOpowerRatio['binDensity'],
                                                           dfTNOpowerRatio['binDensity'])
        firstProduct = np.multiply(dfTNOpowerRatio['varFarmPower_1'],
                                   dfTNOpowerRatio['binDensitySquared'])
        secondProduct = np.multiply(dfTNOpowerRatio['varFarmPower_2'],
                                    dfTNOpowerRatio['binDensitySquared'])*(apr**2)
        summation = np.nansum(np.add(firstProduct, secondProduct))
        var = summation/(aap2**2)
        sd = np.sqrt(var)

        # Standard error
        firstProduct2 = np.multiply(dfTNOpowerRatio['varAvgFarmPower_1'],
                                    dfTNOpowerRatio['binDensitySquared'])
        secondProduct2 = np.multiply(dfTNOpowerRatio['varAvgFarmPower_2'],
                                     dfTNOpowerRatio['binDensitySquared'])*(apr**2)
        summation2 = np.nansum(np.add(firstProduct2, secondProduct2))
        se2 = summation2/(aap2**2)
        se = np.sqrt(se2)

        dct = {"Annual Power Ratio estimate": apr,
               "variance of Annual Power Ratio": var,
               "standard deviation of Annual Power Ratio": sd,
               "variance of Annual Power Ratio estimate": se2,
               "standard error of Annual Power Ratio": se}

        for key in dct:
            print(f'{key}: {dct[key]}')

        return dct

    def bootstrapSamples(self, B=1000, seed=None, pooled=True):

        start = default_timer()
        samples = np.full(B, None, dtype=pd.core.frame.DataFrame)

        prng = np.random.default_rng(seed=seed)

        nrow = self.scada.shape[0]

        if pooled:
            dfPooled = self.scada.sample(n=nrow*B,
                                         replace=True,
                                         random_state=prng)
            dfPooled = dfPooled.reset_index(drop=True)

            duration = default_timer() - start
            print("Sampling Time:", duration)

            dfPooled['repID'] = np.repeat(
                np.arange(0, B, 1, dtype=int), repeats=nrow)

            return dfPooled

        else:
            samples = np.full(B, None, dtype=pd.core.frame.DataFrame)
            for rep in range(B):
                dfTemp = self.scada.sample(n=nrow,
                                           replace=True,
                                           random_state=prng)
                dfTemp.reset_index(drop=True, inplace=True)
                samples[rep] = dfTemp

        duration = default_timer() - start
        print("Sampling Time:", duration)

        return samples

    def bootstrapEstimate(self, stepVars=["direction", "speed"],
                          windDirectionSpecs=None, windSpeedSpecs=None,
                          B=1000, seed=None, useReference=True,
                          seMultiplier=2, lowerPercentile=2.5, upperPercentile=97.5,
                          retainReps=True, diagnose=True, repsPooled=None,
                          **AEPargs):  # figure out how to use kwargs here for hours, aepmethod, and absolute, etc. for metricMethod
        """
        Compute summary statistics of bootsrapped samples based on your metric of choice


        windDirectionSpecs: list of length 3, specifications for wind direction
            bins-- [lower bound (inclusive), upper bound (exclusive), bin width]
            Only used if nDim=2
        windSpeedSpecs: list of length 3, specifications for wind speed bins--
            [lower bound (inclusive), upper bound (exclusive), bin width]
            Only used if nDim=2
        **AEPargs: args for the AEP method
        """
        if windDirectionSpecs is None:
            windDirectionSpecs = self.defaultWindDirectionSpecs

        if windSpeedSpecs is None:
            windSpeedSpecs = self.defaultWindSpeedSpecs

        if type(stepVars) is str:
            stepVars = list([stepVars])

        start = default_timer()

        # Get an array of the bootstrap samples
        if repsPooled is None:

            bootstrapPooled = self.bootstrapSamples(
                B=B, seed=seed, pooled=True)

        else:
            bootstrapPooled = repsPooled
            B = bootstrapPooled[['repID']].iloc[-1][0]

        finalCols = ['percentPowerGain', 'changeInPowerRatio']
        finalColsMultiIdx = [('percentPowerGain', '', ''),
                             ('changeInPowerRatio', '', '')]

        for var in stepVars:
            name = f'{var}BinLowerBound'
            finalCols.append(name)
            finalColsMultiIdx.append((name, '', ''))

        # Setup empty array to hold the binned versions of each bootstrap simulation
       # bootstrapDFbinned = np.full(B, None, dtype=pd.core.frame.DataFrame)

        nrow = self.scada.shape[0]
        rowidx = np.arange(0, (nrow*B)+1, nrow)
        for bootstrap in range(B):

            # Get current bootstrap sample
            startIDX = rowidx[bootstrap]
            stopIDX = rowidx[bootstrap+1]
            currentDF = bootstrapPooled.iloc[startIDX:stopIDX, :]

            # get into correct format
            binadder = self.binAdder(
                stepVars, windDirectionSpecs, windSpeedSpecs, df=currentDF)
            binall = self.binAll(stepVars, windDirectionSpecs,
                                 windSpeedSpecs, df=binadder, filterBins=True)
            computeall = self.computeAll(stepVars, windDirectionSpecs,
                                         windSpeedSpecs, useReference, df=binall)

            binStats = computeall[finalColsMultiIdx]
            # Multi-index on the columns makes indexing annoying when all but one level is empty
            dfTemp = binStats.reset_index(drop=True).droplevel(
                level=[1, 2], axis="columns")
            # Make sure columns are in the right order
            dfTemp = dfTemp[finalCols]
            dfTemp['repID'] = bootstrap

            aepTemp = np.full(shape=(8, 4), fill_value=np.nan, dtype=float)
            i = 0
            for method in (1, 2):
                for abso in (0, 1):
                    for useRef in (0, 1):
                        aep = self.aepGain(aepMethod=method,
                                           absolute=bool(abso),
                                           useReference=bool(useRef),
                                           windDirectionSpecs=windDirectionSpecs,
                                           windSpeedSpecs=windSpeedSpecs,
                                           df=computeall)

                        aepTemp[i] = np.asarray([method, abso, useRef, aep[1]])
                        i += 1

            if bootstrap == 0:
                metricMatrix = np.asarray(dfTemp)
                aepMatrix = aepTemp.copy()
            else:
                metricMatrix = np.concatenate(
                    (metricMatrix, np.asarray(dfTemp)), axis=0)
                aepMatrix = np.concatenate(
                    (aepMatrix, np.asarray(aepTemp)), axis=0)

        aepSamplingDist = pd.DataFrame(data=aepMatrix, columns=[
                                       "aepMethod", "absoluteAEP", "useReference", "aepGain"])

        aepSummary = aepSamplingDist.groupby(by=["aepMethod", "absoluteAEP", "useReference"]).agg(mean=pd.NamedAgg(column="aepGain",
                                                                                                                   aggfunc=np.mean),
                                                                                                  se=pd.NamedAgg(column="aepGain",
                                                                                                                 aggfunc=np.nanstd),
                                                                                                  median=pd.NamedAgg(column="aepGain",
                                                                                                                     aggfunc=np.nanmedian),
                                                                                                  upperPercentile=pd.NamedAgg(column="aepGain",
                                                                                                                              aggfunc=lambda x: np.nanpercentile(x, upperPercentile)),
                                                                                                  lowerPercentile=pd.NamedAgg(column="aepGain",
                                                                                                                              aggfunc=lambda x: np.nanpercentile(x, lowerPercentile)),
                                                                                                  nObvs=pd.NamedAgg(column="aepGain",
                                                                                                                    aggfunc='count'),
                                                                                                  firstQuartile=pd.NamedAgg(column="aepGain",
                                                                                                                            aggfunc=lambda x: np.nanpercentile(x, 25)),
                                                                                                  thirdQuartile=pd.NamedAgg(column="aepGain",
                                                                                                                            aggfunc=lambda x: np.nanpercentile(x, 75)))
        seMultdAEP = seMultiplier*aepSummary["se"]
        aepSummary["meanMinusSE"] = np.subtract(aepSummary["mean"], seMultdAEP)
        aepSummary["meanPlusSE"] = np.add(aepSummary["mean"], seMultdAEP)
        aepSummary["iqr"] = np.subtract(
            aepSummary['thirdQuartile'], aepSummary['firstQuartile'])
        # For convenience
        aepSummary["metric"] = 'aepGain'
        aepSummary["nReps"] = B

        aepSummary = aepSummary[["mean", 'meanMinusSE', 'meanPlusSE',
                                 'median', 'lowerPercentile', 'upperPercentile',
                                 'se', 'iqr', 'nObvs', 'metric', 'nReps']]

        # Save sampling distributions
        metricDF = pd.DataFrame(data=metricMatrix, columns=finalCols+['repID'])
        ppgSamplingDists = metricDF.copy().drop(
            columns='changeInPowerRatio', inplace=False)
        cprSamplingDists = metricDF.copy().drop(
            columns='percentPowerGain', inplace=False)

        # Compute Sample statistics for each wind condition bin
        metricDFlong = metricDF.melt(value_vars=['percentPowerGain', 'changeInPowerRatio'],
                                     value_name='value',
                                     var_name='metric',
                                     id_vars=finalCols[2:])

        finalCols = list(metricDFlong)[:-1]  # last column is value
        dfSummary = metricDFlong.groupby(by=finalCols).agg(mean=pd.NamedAgg(column='value',
                                                                            aggfunc=np.mean),
                                                           se=pd.NamedAgg(column='value',
                                                                          aggfunc=np.nanstd),
                                                           median=pd.NamedAgg(column='value',
                                                                              aggfunc=np.nanmedian),
                                                           upperPercentile=pd.NamedAgg(column='value',
                                                                                       aggfunc=lambda x: np.nanpercentile(x, upperPercentile)),
                                                           lowerPercentile=pd.NamedAgg(column='value',
                                                                                       aggfunc=lambda x: np.nanpercentile(x, lowerPercentile)),
                                                           nObvs=pd.NamedAgg(column='value',
                                                                             aggfunc='count'),
                                                           firstQuartile=pd.NamedAgg(column='value',
                                                                                     aggfunc=lambda x: np.nanpercentile(x, 25)),
                                                           thirdQuartile=pd.NamedAgg(column='value',
                                                                                     aggfunc=lambda x: np.nanpercentile(x, 75)))

        pctPwrGain = dfSummary.iloc[dfSummary.index.get_level_values(
            'metric') == "percentPowerGain"]
        chngPwrRatio = dfSummary.iloc[dfSummary.index.get_level_values(
            'metric') == "changeInPowerRatio"]

        for df in [pctPwrGain, chngPwrRatio]:
            seMultd = seMultiplier*df["se"]
            df["meanMinusSE"] = np.subtract(df["mean"], seMultd)
            df["meanPlusSE"] = np.add(df["mean"], seMultd)
            df["iqr"] = np.subtract(df['thirdQuartile'], df['firstQuartile'])
            # For convenience
            df["nReps"] = B
            df["metric"] = df.index.get_level_values('metric')

            df = df[["mean", 'meanMinusSE', 'meanPlusSE',
                    'median', 'lowerPercentile', 'upperPercentile',
                     'se', 'iqr', 'nObvs', 'metric', 'nReps']]

        pctPwrGain.index = pctPwrGain.index.droplevel('metric')
        chngPwrRatio.index = chngPwrRatio.index.droplevel('metric')

        duration = default_timer() - start
        print("Overall:", duration)

        resultDict = {"percent power gain": pctPwrGain,
                      "change in power ratio": chngPwrRatio,
                      "aep gain": aepSummary,
                      'ppg sampling distributions': ppgSamplingDists,
                      'cpr sampling distributions': cprSamplingDists,
                      'aep sampling distribution': aepSamplingDist,
                      'reps': bootstrapPooled}

        if diagnose:
            dfBinned = self.binAdder(
                stepVars, windDirectionSpecs, windSpeedSpecs)

            self.bootstrapDiagnostics(bsEstimateDict=resultDict,
                                      dfBinned=dfBinned,
                                      windDirectionSpecs=windDirectionSpecs,
                                      windSpeedSpecs=windSpeedSpecs)

        if not retainReps:
            resultDict.pop('reps')
            return resultDict
        return resultDict

    def bootstrapDiagnostics(self, bsEstimateDict, dfBinned,
                             windDirectionSpecs=None, windSpeedSpecs=None,
                             histplotKWS=None):
        # Check for other commonalities that can be moved here

        stepVars = []
        for var in bsEstimateDict['percent power gain'].index.names:
            stepVars.append(var)

        if len(stepVars) == 2:
            if histplotKWS is None:
                histplotKWS = {'linewidth': 1,
                               'cmap': sns.color_palette("rocket_r",
                                                         as_cmap=True)}
            self.__bsDiagnostics2d__(bsEstimateDict=bsEstimateDict,
                                     dfBinned=dfBinned,
                                     stepVars=stepVars,
                                     windDirectionSpecs=windDirectionSpecs,
                                     windSpeedSpecs=windSpeedSpecs,
                                     histplotKWS=histplotKWS)
        else:  # (if len(stepVars)==1)
            if histplotKWS is None:
                histplotKWS = {'thresh': None}
            self.__bsDiagnostics1d__(bsEstimateDict=bsEstimateDict,
                                     stepVar=stepVars,
                                     dfBinned=dfBinned,
                                     windDirectionSpecs=windDirectionSpecs,
                                     windSpeedSpecs=windSpeedSpecs,
                                     histplotKWS=histplotKWS)

        return None

    def __bsDiagnostics1d__(self, bsEstimateDict,
                            dfBinned,
                            stepVar,
                            windDirectionSpecs=None, windSpeedSpecs=None,
                            kdeKWS={'bw_adjust': 2},
                            histplotKWS={'thresh': None}):

        ppgSamplingDists = bsEstimateDict['ppg sampling distributions']
        ppgSummary = bsEstimateDict["percent power gain"]
        bsPooled = bsEstimateDict['reps']

        if windDirectionSpecs is None:
            windDirectionSpecs = self.defaultWindDirectionSpecs

        if windSpeedSpecs is None:
            windSpeedSpecs = self.defaultWindSpeedSpecs

        stepVar = stepVar[0]

        if stepVar == 'directionBins':
            width = windDirectionSpecs[2]
            edges = np.arange(*windDirectionSpecs)
            xLabel = u"Wind Direction (\N{DEGREE SIGN})"
            col = self.wdCol
        else:  # (if stepVar=='speed'):
            width = windSpeedSpecs[2]
            edges = np.arange(*windSpeedSpecs)
            xLabel = "Wind Speed (m/s)"
            col = self.wsCol

        # Histograms
        plt.clf()
        sns.set_theme(style="whitegrid")

        fig, axs = plt.subplots(nrows=1, ncols=2, sharex=True,
                                sharey=True, figsize=(10, 5),
                                layout='constrained')

        h1 = sns.histplot(self.scada, x=col,
                          stat='density',
                          binwidth=width, ax=axs[1],
                          kde_kws=kdeKWS, **histplotKWS)

        h0 = sns.histplot(bsPooled, x=col,
                          stat='density',
                          binwidth=width, ax=axs[0],
                          kde_kws=kdeKWS, **histplotKWS)

        # Tick marks at multiples of 5 and 1
        for ax in axs:
            ax.tick_params(which="major", bottom=True, length=7,
                           color='#4C4C4C', axis='x', left=True)
            ax.tick_params(which="minor", bottom=True, length=3,
                           color='#4C4C4C', axis='x', left=True)
            ax.set_xlabel("", fontsize=0)
            ax.set_ylabel("", fontsize=0)
            ax.grid(which='minor', visible=True, linestyle='-',
                    linewidth=0.5, axis='y')

        axs[0].tick_params(which="minor", bottom=True, length=3,
                           color='#4C4C4C', axis='y', left=True)
        axs[0].tick_params(which="major", bottom=True, length=7,
                           color='#4C4C4C', axis='y', left=True)

        # Labels
        axs[1].set_title(f"Real Data (size={self.scada.shape[0]})")
        axs[0].set_title(
            f"Pooled Bootstrap Samples (size={self.scada.shape[0]}; nReps = {ppgSummary['nReps'].iloc[1]})")
        fig.supxlabel(xLabel, fontsize=15)  # unicode formatted
        fig.suptitle("Densities ", fontsize=17)
        plt.show()

        # Sampling distribution by bin
        #sns.violinplot(data=df, x="age", y="alive", cut=0)

        return None

    def __bsDiagnostics2d__(self, bsEstimateDict,
                            dfBinned,
                            stepVars,
                            windDirectionSpecs, windSpeedSpecs,
                            histplotKWS={'linewidth': 1,
                                         'cmap': sns.color_palette("rocket_r",
                                                                   as_cmap=True)}):
        #####
        colors = ['cmr.iceburn',
                  'cubehelix',
                  sns.color_palette("coolwarm_r", as_cmap=True),
                  'cubehelix']

        ppgSamplingDists = bsEstimateDict['ppg sampling distributions']

        ppgSummary = bsEstimateDict['percent power gain']
        cprSummary = bsEstimateDict['change in power ratio']
        aepSummary = bsEstimateDict['aep gain']

        bsPooled = bsEstimateDict['reps']

        # 2d Histogram

        directionEdges = np.arange(*windDirectionSpecs)
        speedEdges = np.arange(*windSpeedSpecs)

        #####
        # Plotting
        fig, axs = plt.subplots(nrows=1, ncols=2,
                                sharex=True, sharey=True,
                                figsize=(10, 5), layout='constrained')
        # Histograms
        width = np.min(np.asarray([windDirectionSpecs[2], windSpeedSpecs[2]]))

        h1 = sns.histplot(self.scada, x=self.wdCol, y=self.wsCol,
                          cbar=True, stat='density', thresh=None,
                          binwidth=width, ax=axs[1], **histplotKWS)

        h0 = sns.histplot(bsPooled, x=self.wdCol, y=self.wsCol,
                          cbar=True, stat='density', thresh=None,
                          binwidth=width, ax=axs[0], **histplotKWS)

        for i in range(2):
            axs[i].tick_params(which="major", bottom=True, length=5)
            axs[i].tick_params(which="minor", bottom=True,
                               length=3, labelsize=0)
            axs[i].set_xlabel("", fontsize=0)
            axs[i].set_ylabel("", fontsize=0)

        # Labels
        axs[1].set_title(f"Real Data (size={self.scada.shape[0]})")
        axs[0].set_title(
            f"Pooled Bootstrap Samples (size={self.scada.shape[0]}; nReps = {ppgSummary['nReps'].iloc[1]})")
        # unicode formatted
        fig.supxlabel(u"Wind Direction (\N{DEGREE SIGN})", fontsize=15)
        fig.supylabel("Wind Speed (m/s)", fontsize=15)
        fig.suptitle("Densities ", fontsize=17)
        plt.show()
        # Heatmaps

        # Not the correct way to do this but it works for now
        xlabels = []

        for num in np.arange(*windDirectionSpecs):
            if num % 5 == 0:
                xlabels.append(str(num))
                continue
            xlabels.append(" ")

        ylabels = []
        for num in np.arange(*windSpeedSpecs):
            if num % 5 == 0:
                ylabels.append(str(num))
                continue
            ylabels.append(" ")

        # Getting the data
        # 4 types of plots: Centers, variance, CI width, and CI sign )pos/neg)
        pArray = np.asarray([["Centers", "Mean", "Median"],
                             ["Variance", "SE", "IQR"],
                             ["Interval Coverage", "SE Method", "Percentile Method"],
                             ["Confidence Interval Widths", "SE Method", "Percentile Method"]])

        idxs = [(iS, iD) for iS in range(speedEdges.size)
                for iD in range(directionEdges.size)]
        #############

        start1096 = default_timer()
        for dfSummary in [ppgSummary, cprSummary]:
            print("Doing stuff for new metric")
            mCenterMean, mCenterMed, mVarSE, mVarIQR, mPosNegCI, mPosNegPerc, mIWperc, mIWci = [np.full(
                shape=(speedEdges.size, directionEdges.size), fill_value=np.nan, dtype=float) for i in range(8)]
            # Just in case indices are out of order
            dfSummary.index = dfSummary.index.reorder_levels(
                order=['directionBins', 'speedBins'])

            print("filling matrices")
            start1105 = default_timer()
            for idx in idxs:
                try:
                    mVarSE[idx[0], idx[1]] = dfSummary.loc[(
                        directionEdges[idx[1]], speedEdges[idx[0]])]['se']
                    mVarIQR[idx[0], idx[1]] = dfSummary.loc[(
                        directionEdges[idx[1]], speedEdges[idx[0]])]['iqr']
                    mIWperc[idx[0], idx[1]] = dfSummary.loc[(directionEdges[idx[1]],
                                                             speedEdges[idx[0]])]['upperPercentile'] - dfSummary.loc[(directionEdges[idx[1]],
                                                                                                                      speedEdges[idx[0]])]['lowerPercentile']
                    mIWci[idx[0], idx[1]] = dfSummary.loc[(directionEdges[idx[1]],
                                                           speedEdges[idx[0]])]['meanPlusSE'] - dfSummary.loc[(directionEdges[idx[1]],
                                                                                                               speedEdges[idx[0]])]['meanMinusSE']
                    mCenterMean[idx[0], idx[1]] = dfSummary.loc[(
                        directionEdges[idx[1]], speedEdges[idx[0]])]['mean']
                    mCenterMed[idx[0], idx[1]] = dfSummary.loc[(
                        directionEdges[idx[1]], speedEdges[idx[0]])]['median']

                    if dfSummary.loc[(directionEdges[idx[1]], speedEdges[idx[0]])]['meanPlusSE'] < 0:
                        mPosNegCI[idx[0], idx[1]] = -1
                    elif dfSummary.loc[(directionEdges[idx[1]], speedEdges[idx[0]])]['meanMinusSE'] > 0:
                        mPosNegCI[idx[0], idx[1]] = 1
                    elif dfSummary.loc[(directionEdges[idx[1]], speedEdges[idx[0]])]['meanMinusSE']*dfSummary.loc[(directionEdges[idx[1]], speedEdges[idx[0]])]['meanPlusSE'] < 0:
                        mPosNegCI[idx[0], idx[1]] = 0

                    if dfSummary.loc[(directionEdges[idx[1]], speedEdges[idx[0]])]['upperPercentile'] < 0:
                        mPosNegPerc[idx[0], idx[1]] = -1
                    elif dfSummary.loc[(directionEdges[idx[1]], speedEdges[idx[0]])]['lowerPercentile'] > 0:
                        mPosNegPerc[idx[0], idx[1]] = 1
                    elif dfSummary.loc[(directionEdges[idx[1]], speedEdges[idx[0]])]['lowerPercentile']*dfSummary.loc[(directionEdges[idx[1]], speedEdges[idx[0]])]['upperPercentile'] < 0:
                        mPosNegPerc[idx[0], idx[1]] = 0

                except KeyError:
                    continue
            duration1105 = default_timer() - start1105
            print("Done filling matrices:", duration1105)

            mArray = np.asarray([[mCenterMean, mCenterMed],
                                 [mVarSE, mVarIQR],
                                 [mPosNegCI, mPosNegPerc],
                                 [mIWperc, mIWci]])

            # Plotting

            # Set up the plotting field
            for plot in range(pArray.shape[0]):
                start1152 = default_timer()
                print('starting new plot')
                plt.clf()
                fig, axs = plt.subplots(nrows=1, ncols=2,
                                        sharex=True, sharey=True,
                                        figsize=(10, 5), layout='constrained')

                # Set up the individual heatmaps
                for i in range(2):
                    # Get data
                    M = mArray[plot][i]

                    axs[i] = sns.heatmap(M, center=0, square=False, linewidths=0,
                                         ax=axs[i], cbar=bool(i), cbar_kws={"shrink": .8},
                                         annot=False, cmap=colors[plot], robust=True)

                    axs[i].yaxis.set_minor_locator(
                        mticker.MultipleLocator(0.5))
                    axs[i].invert_yaxis()
                    axs[i].xaxis.set_minor_locator(
                        mticker.MultipleLocator(0.5))
                    axs[i].xaxis.tick_bottom()

                    axs[i].tick_params(
                        which="minor", bottom=True, length=6, color='#4C4C4C')
                    axs[i].tick_params(which="major", bottom=True, length=0, color='#4C4C4C',
                                       grid_linewidth=0)
                    axs[i].set_xticklabels(xlabels, rotation=-90)
                    axs[i].set_yticklabels(ylabels, rotation=0)

                    axs[i].grid(which='minor', visible=True, color='#d9d9d9', linestyle='-',
                                linewidth=1)

                    axs[i].set_title(pArray[plot][i+1], fontsize=13)

                axs[1].tick_params(which="minor", bottom=True,
                                   length=0, color='white')

                # Labels
                # unicode formatted
                fig.supxlabel(u"Wind Direction (\N{DEGREE SIGN})", fontsize=15)
                fig.supylabel("Wind Speed (m/s)", fontsize=15)
                fig.suptitle(
                    f"{pArray[plot][0]} ({dfSummary['metric'].iloc[1]}; nReps = {dfSummary['nReps'].iloc[1]})", fontsize=17)
                plt.show()

                duration1152 = default_timer() - start1152
                print("Moving on to next plot", duration1152)

            duration1096 = default_timer() - start1096
            print("Moving on to next metric", duration1096)

        return None

    # Seems inefficient
    def lineplotBE(self, dfSummary=None, repsArray=None, windDirectionSpecs=None,
                   windSpeedSpecs=None, repsPooled=None,
                   stepVar="direction", useReference=True, **BEargs):
        """
        windDirectionSpecs: list of length 3, specifications for wind direction
            bins-- [lower bound (inclusive), upper bound (exclusive), bin width]
        windSpeedSpecs: list of length 3, specifications for wind speed bins--
            [lower bound (inclusive), upper bound (exclusive), bin width]
        """
        if windDirectionSpecs is None:
            windDirectionSpecs = self.defaultWindDirectionSpecs

        if windSpeedSpecs is None:
            windSpeedSpecs = self.defaultWindSpeedSpecs

        fig, axs = plt.subplots(nrows=1, ncols=2,
                                sharex=True, sharey=True,
                                figsize=(20, 7), layout='constrained')

        # put other scenarios in here
        if (dfSummary is None) and (repsArray is not None):
            dfSummary = self.bootstrapEstimate(stepVars=stepVar,
                                               windDirectionSpecs=windDirectionSpecs,
                                               windSpeedSpecs=windSpeedSpecs,
                                               repsPooled=repsPooled,
                                               metric="percentPowerGain",
                                               useReference=useReference,
                                               diagnose=False,
                                               retainReps=False,
                                               **BEargs)

        metric = dfSummary['metric'].iloc[1]

        X = dfSummary.index.get_level_values(f'{stepVar}BinLowerBound')

        yMean = dfSummary['mean']
        yUpperSE = dfSummary['meanPlusSE']
        yLowerSE = dfSummary['meanMinusSE']

        yMed = dfSummary['median']
        yUpperPerc = dfSummary['upperPercentile']
        yLowerPerc = dfSummary['lowerPercentile']

        axs[0].axhline(0, color='black', linestyle='dotted')
        axs[0].plot(X, yMean, linestyle='-', marker='.', markersize=10)
        axs[0].fill_between(x=X, y1=yUpperSE, y2=yLowerSE, alpha=0.4)
        axs[0].grid()
        axs[0].set_title("SE Method", fontsize=13)

        axs[1].axhline(0, color='black', linestyle='dotted')
        axs[1].plot(X, yMed, linestyle='-', marker='.', markersize=10)
        axs[1].fill_between(x=X, y1=yUpperPerc, y2=yLowerPerc, alpha=0.4)
        axs[1].grid()
        axs[1].set_title("Percentile Method", fontsize=13)

        if stepVar == 'speed':
            xAxis = "Wind Speed (m/s)"
            title = f"Wind Directions {windDirectionSpecs[0]}" + \
                u"\N{DEGREE SIGN}" + \
                    f" - {windDirectionSpecs[1]}" + u"\N{DEGREE SIGN}"
        else:
            xAxis = u"Wind Direction (\N{DEGREE SIGN})"
            title = f"Wind Speeds {windSpeedSpecs[0]} to {windSpeedSpecs[1]} m/s"
        fig.supxlabel(xAxis, fontsize=13)
        fig.supylabel(f"{metric} bootstrap centers", fontsize=13)
        fig.suptitle(title, fontsize=17)
        plt.show()

        return None

    def windRose(self, data=None,
                 directionBins=None,
                 speedBins=None):
        """
        INCOMPLETE - do not use

        Parameters
        ----------
        data : TYPE, optional
            DESCRIPTION. The default is None.
        directionBins : TYPE, optional
            DESCRIPTION. The default is None.
        speedBins : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """

        if directionBins is None and speedBins is None:
            return 'Specify direction and/or speed bins'

        if directionBins is not None:
            hist = np.histogram(self.scada[self.wdCol], bins=directionBins)
            counts = hist[0]

            directionBins = directionBins[:]

            widths = [directionBins[i] - directionBins[i-1]
                      for i in range(1, len(directionBins))]
            binMidpoints = [(widths[i]/2)+directionBins[i]
                            for i in range(len(widths))]

            ax = plt.subplot(projection='polar')
            ax.set_theta_zero_location("N", offset=(90)*np.pi/180)
            ax.set_theta_direction("clockwise")
            ax.set_thetagrids(angles=[0, 45, 90, 135, 180, 225, 270, 315],
                              labels=["N", "N E", "E", "S E", "S", "S W", "W", "N W"])

            ax.bar([b*(np.pi/180) for b in binMidpoints],
                   heights=counts,
                   width=[(np.pi/180)*w for w in widths],
                   bottom=0.0,
                   lw=1,
                   ec="black")

            if speedBins is not None:
                None

            plt.show()

        return None
