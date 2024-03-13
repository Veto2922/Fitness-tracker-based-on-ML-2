import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy
from sklearn.neighbors import LocalOutlierFactor 

# --------------------------------------------------------------
# Load data

df =pd.read_pickle('../../data/interim/01_data_processed.pkl')

outlier_columns =list(df.columns[:6])

# --------------------------------------------------------------
# Plotting outliers


plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"]=(20,5)  #rcparams : runtime configration parameters
plt.rcParams["figure.dpi"]=100

df[['acc_x']].boxplot()
df[['acc_x' , 'label']].boxplot(by ='label', figsize=(20,10)) #plotting boxplot for acc_x values for each label
plt.show()

df[outlier_columns[:3] +['label']].boxplot(by ='label',figsize=(20,10),layout=(1,3))
df[outlier_columns[3:] +['label']].boxplot(by ='label',figsize=(20,10),layout=(1,3))
plt.show()


def plot_binary_outliers(dataset, col, outlier_col, reset_index):
    """ Plot outliers in case of a binary outlier score. Here, the col specifies the real data
    column and outlier_col the columns with a binary value (outlier or not).

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): Column that you want to plot
        outlier_col (string): Outlier column marked with true/false
        reset_index (bool): whether to reset the index for plotting
    """

    # Taken from: https://github.com/mhoogen/ML4QS/blob/master/Python3Code/util/VisualizeDataset.py

    dataset = dataset.dropna(axis=0, subset=[col, outlier_col])
    dataset[outlier_col] = dataset[outlier_col].astype("bool")

    if reset_index:
        dataset = dataset.reset_index()

    fig, ax = plt.subplots()

    plt.xlabel("samples")
    plt.ylabel("value")

    # Plot non outliers in default color
    ax.plot(
        dataset.index[~dataset[outlier_col]],
        dataset[col][~dataset[outlier_col]],
        "+",
    )
    # Plot data points that are outliers in red
    ax.plot(
        dataset.index[dataset[outlier_col]],
        dataset[col][dataset[outlier_col]],
        "r+",
    )

    plt.legend(
        ["outlier " + col, "no outlier " + col],
        loc="upper center",
        ncol=2,
        fancybox=True,
        shadow=True,
    )
    plt.show()
# --------------------------------------------------------------
# Interquartile range (distribution based)


# Insert IQR function
def mark_outliers_iqr(dataset, col):
    """Function to mark values as outliers using the IQR method.

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): The column you want apply outlier detection to

    Returns:
        pd.DataFrame: The original dataframe with an extra boolean column 
        indicating whether the value is an outlier or not.
    """

    dataset = dataset.copy()

    Q1 = dataset[col].quantile(0.25)
    Q3 = dataset[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    dataset[col + "_outlier"] = (dataset[col] < lower_bound) | (
        dataset[col] > upper_bound
    )

    return dataset

# Plot a single column
col="acc_x"
dataset = mark_outliers_iqr(df,col)
plot_binary_outliers(dataset,col=col,outlier_col=col+"_outlier",reset_index=True)


# Loop over all columns
for col in outlier_columns:
    dataset = mark_outliers_iqr(df,col)
    plot_binary_outliers(dataset,col=col,outlier_col=col+"_outlier",reset_index=True)
    

# --------------------------------------------------------------
# Chauvenets criteron (distribution based)


# Check for normal distribution


# Insert Chauvenet's function


# Loop over all columns


# --------------------------------------------------------------
# Local outlier factor (distance based)
# --------------------------------------------------------------

# Insert LOF function


# Loop over all columns


# --------------------------------------------------------------
# Check outliers grouped by label
# --------------------------------------------------------------


# --------------------------------------------------------------
# Choose method and deal with outliers
# --------------------------------------------------------------

# Test on single column


# Create a loop

# --------------------------------------------------------------
# Export new dataframe
# --------------------------------------------------------------
