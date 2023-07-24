# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 12:24:38 2023

@author: ctodd
"""
import pandas as pd
import numpy as np

def jointPMF_albaincourt(df):
    """
    Joint wind speed/direction PMF for the SMARTE-OLE sight, based on long-run frequency data.
        Long run frequency data was used to essentially create 1 joint and 2 marginal PMF lookup tables.
        The data is not publibly availible, but this function is public so that users can see what
        type of PMF the energyGain module expects: a function that will take a dataframe that has
        wind conditions as indices and returns an ordered array of PMF values.
    
    This particular PMF is less than ideal in at least one way: 
        it cannot dynamially resize the wind condition bins and recalculate the PMF based on the new bins.
        For now, I'm just accepting this since, for the majority of the duration of my work, 
        we focused on fixed bin widths of 1.

    Parameters
    ----------
    df : pandas data frame
        Must have the wind condition bins as a pandas index or MultiIndex, and the levels 
        must be named 'speedBinLowerBound' and/or 'directionBinLowerBound'

    Returns
    -------
    newCol : numpy array
        Contains the PMF values for each wind condition that appears in df, in the order that
        it appears in the df (so it can be concatenated as a column directly onto df)

    """
    # This lets the function infer whether the wind condition bins are by speed, direction, or both
    indices = df.index.names
    
    ## Load the appropriate PMF
    # Joint PMF
    if len(indices)==2:
        df = df.reorder_levels(["directionBinLowerBound", "speedBinLowerBound"], axis=0)
        pmf = pd.read_pickle("jointPMFdf_albaincourt")
        pmf = pmf.reorder_levels(["directionBinLowerBound", "speedBinLowerBound"])
    # Marginal PMF for wind speed
    elif indices[0]=='speedBinLowerBound':
        pmf = np.load("speedPMFdf_albaincourt",allow_pickle=True)
    # Marginal PMF for wind direction
    else:
        pmf = np.load("directionPMFdf_albaincourt",allow_pickle=True)
        
    # Array of all possible wind conditions in the input data    
    X = np.asarray(df.index)
    length = X.size
    
    # Create array that will hold the PMF value for each wind condition that appears in the data
    # This will be added to the input data as a new column
    newCol = np.full(length, np.nan)
    
    # Go through each wind condition present in the data
    for i in range(length):
        try:
            # see if the wind condition from the data is present in the PMF's support
            # Perhaps a more sophisticated version of this would return a marginal probability if only one of the jointly distributed variables is outside of the marginal support
            prob = pmf.loc[X[i], 'freq']
        except KeyError:
            # If the wind condition that appeared in the data is not in the support, return a probability of zero
            prob = 0
        
        # Save the probability
        newCol[i] = prob
            
    return newCol