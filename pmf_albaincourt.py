# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 12:24:38 2023

@author: ctodd
"""
import pandas as pd
import numpy as np

def jointPMF_albaincourt(df):
    indices = df.index.names
    if len(indices)==2:
        df = df.reorder_levels(["directionBinLowerBound", "speedBinLowerBound"], axis=0)
        pmf = pd.read_pickle("jointPMFdf_albaincourt")
        pmf = pmf.reorder_levels(["directionBinLowerBound", "speedBinLowerBound"])
    elif indices[0]=='speedBinLowerBound':
        pmf = np.load("speedPMFdf_albaincourt",allow_pickle=True)
    else:
        pmf = np.load("directionPMFdf_albaincourt",allow_pickle=True)
        
    X = np.asarray(df.index)
    length = X.size
    newCol = np.full(length, np.nan)
    for i in range(length):
        try:
            prob = pmf.loc[X[i], 'freq']
        except KeyError:
            prob = 0
        newCol[i] = prob
            
    return newCol