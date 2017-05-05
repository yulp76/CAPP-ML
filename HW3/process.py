import numpy as np
import pandas as pd

def check_missing_value(df):
    '''
    Prints out the variables that
    have missing values.

    -> so that user can determine which
    method to adopt for filling missing values.
    '''
    n = len(df.index)
    for item in df.columns:
        if df[item].count() < n:
            print(item+" has missing values.")


def fill_in(df, var, d, method=None, value=None):
    '''
    Methods (str) refer to filling in missing values with
    "mean", "mode", "median", "forward"(filling),
    or "backward"(filling).

    If you want to fill in with a particular value,
    specifiy a particular value.
    
    *Refer to diff_in_mean function in describe.py
    for requirements of var_lst and d.
    '''
    assert (method is None and value is not None) or (method is not None and value is None),\
    "supply a method (mean, median, mode, backward, forward) or a specific value"

    name = d['x'+str(var)]
    if value is not None:
        df[name] = df[name].fillna(value)

    else:
        if method == 'mean':
            df[name] = df[name].fillna(df[name].mean())
        if method == 'mode':
            df[name] = df[name].fillna(df[name].mode())
        if method == 'median':
            df[name] = df[name].fillna(df[name].median())
        if method == 'forward':
            df[name] = df[name].ffill()
        else:
            df[name] = df[name].bfill()