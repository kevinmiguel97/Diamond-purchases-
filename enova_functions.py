# Libraries
# Standard libraries
import numpy as np
import pandas as pd
# Plotting libraries
import matplotlib.pyplot as plt 
import seaborn as sns
# Machine learning libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import ks_2samp
# Date and time libraries
import datetime as dt
import time as tm
# Keep graphs in line
# Show all columns in pandas
pd.set_option('display.max_columns', 500)
# Graphing style
plt.style.use('seaborn-colorblind')

#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# Function that explotes data shapes
def shapes_exploration(df_train, df_test):
    """
    Explores the shapes and columns of training and testing dataset

    Parameters:
    -----------
    df_train : dataframe
        Training dataframe
    
    df_test : dataframe
        Training dataframe

    Returns:
    --------
    None
    """
    # Training datset shape
    print('Training dataset')
    print('- Observations:', str(df_train.shape[0]))
    print('- Attributes:', str(df_train.shape[1] - 4))                # Price logprice and retail and logretail are not explicative variables
    print('- Target:', str(2))
    # Testing dataset
    print('Offers dataset')
    print('- Observations:', str(df_test.shape[0]))
    print('- Attributes:', str(df_test.shape[1] - 1))                  # Offer column is not an explicative variable                         
    # Verifying variables
    if list(df_train.columns[0:-1]) == list(df_test.columns): 
        print('Same columns: No issues.')
    return None
#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# Function to summarize data
def summary_stats(data, title):  
    import pandas as pd
    """
    Generates a Summary table containing the most relevant information of a dataset

    Parameters:
    ----------
    data : dataframe
        Data to summarize
    title : str
        Title of the graph

    Returns:
    --------
    Dataframe
    """ 
    # Generate a general summary of the variables
    df_missingval = pd.DataFrame(data.isna().any(), columns=['Missing vals'])                   # Check if there are any missing values
    df_types = pd.DataFrame(data.dtypes, columns=['Variable type'])                             # Obtain the datatypes of all colums
    df_describe = data.describe().round(decimals=2).transpose()                                 # Generate summary statistics
    _ = pd.merge(df_missingval, df_types, how='inner', left_index=True, right_index=True)       # Intermediate merge types and missing val
    df_var_summary = pd.merge(df_describe, _ , how='outer', left_index=True, right_index=True)  # Final merge 
    # df_var_summary.loc['date_of_birth', 'count'] = len(data.index)                             # Replace count of date_of_birth
    print(title.center(120))

    return df_var_summary

# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
