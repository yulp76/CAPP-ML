import pandas as pd

def discretize(df, var, d, n):
    '''
    For a given continuous variable x,
    want to split them into n groups given equal
    range of percentiles.

    E.g., n=4 -> 4 groups: 0-25%tile, 25-50%tile,
    50-75%tile, and 75-100%tile

    *Refer to diff_in_mean function in describe.py
    for requirements of var_lst and d.

    Returns a list of headers for the sake of creating
    dummy variables.
    '''
    name = d['x'+str(var)]
    percentiles = []
    headers = []
    for i in range(n+1):
        percentiles.append(i/n)
        if i < n:
            headers.append(name+": "+str(i/n*100)+" to "+str((i+1)/n*100)+"%tile")
    boundries = list(df[name].quantile(percentiles))
    l = list(range(n))
    df[name+'_cat'] = pd.cut(df[name],boundries,labels=l)    
    return headers


def create_dummy(df, var, d, headers, drop=True):
    '''
    df: a pandas dataframe
    *Refer to diff_in_mean function in describe.py
    for requirements of var_lst and d
    drop: a bool. If true, drop the original category columns
    Returns: the modified dataframe
    '''
    name = d['x'+str(var)]
    dummies = pd.get_dummies(df[name+'_cat'])
    dummies.columns = headers
    df = pd.merge(df, dummies, left_index=True, right_index=True, how='inner')
    if drop:
        df.drop(name+'_cat', inplace=True, axis=1)
    
    #update variable dictionary
    n = len(d)
    i= 0
    for header in headers:
        d['x'+str(n+i)] = header
        i += 1
    
    return df