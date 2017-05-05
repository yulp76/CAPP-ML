import pandas as pd
import matplotlib.pyplot as plt
import seaborn

#Explore independent variables
def create_var_dict(df, i):
    '''
    ***Dependent variable y (binary) occupies the i-th
    column of the DataFrame***

    Prints out the names of independent variables.

    Returns corresponding dictionary.
    '''
    d = {}
    d['y'] = df.columns[i-1]
    print('y: '+d['y'])
    x = 1
    for k in range(i, len(df.columns)):
        var = 'x'+str(x)
        d[var]=df.columns[k]
        x += 1
        print(var+": "+d[var])
    return d


#Graph 1:
#Figure out the difference in mean of x's of interest
#between y=0 and y=1
def diff_in_mean(df, var_lst, d):
    '''
    var_lst: list of integers corresponding to
        independent variables of interest.
        E.g., interested in comparing x3,x5,x8
        then, var_lst = [3,5,8]

    d: the independent variable dictionary from
        create_var_dict function.
    '''
    col_names = [d['y']]
    for i in var_lst:
        col_names.append(d['x'+str(i)]) 
    a = pd.DataFrame(df[col_names].groupby(d['y']).mean())
    return a

#Graph 2:
#Show the distribution of middle 99%tile of x's of interest
#using a violin plot.
def x_dist(df, var_lst, d):
    '''
    Plots violin charts.
    
    *Refer to diff_in_mean function
    for requirements of var_lst and d.
    '''
    for i in var_lst:
        plt.figure()
        name = d['x'+str(i)]
        low, high = df[name].quantile([0.005,0.995])
        seaborn.violinplot(df[name][(low < df[name]) & (df[name] < high)])
        plt.title('Violin plot - Middle 99% of '+ name)


#Graph 3:
#For an x's of interest
#show the percentage of y=0 vs. y=1
#for a certain value of x.
def comparison_all_values(df, var_lst, d):
    '''
    Prints bar charts.
    
    *Refer to diff_in_mean function
    for requirements of var_lst and d.
    '''
    for i in var_lst:
        plt.figure()
        name = d['x'+str(i)]
        ct = pd.crosstab(df[name],df[d['y']].astype(bool))
        ct.div(ct.sum(1).astype(float), axis=0).plot(
            kind='bar', figsize=(20,5), stacked=True,
            title = "Distribution of y across all values of "+name)
        plt.legend(loc='best', frameon=True)