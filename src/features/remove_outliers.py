import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mlp
import math
import scipy
from sklearn.neighbors import LocalOutlierFactor  # pip install scikit-learn

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

df = pd.read_pickle("../../data/interim/01_data_processed.pkl")


# --------------------------------------------------------------
# Adjust plot settings
# --------------------------------------------------------------

# Set the plotting style to 'fivethirtyeight' for Matplotlib plots
mlp.style.use("fivethirtyeight")
# Set the figure size to (20, 5) inches
mlp.rcParams["figure.figsize"] = (20,5)
# Set the resolution of the figure to 100 dots per inch (dpi) for exporting
mlp.rcParams["figure.dpi"] = 100


# --------------------------------------------------------------
# Define outliers
# --------------------------------------------------------------

# Define the list of columns to analyze for outliers, the first 6 numeric features
oultier_cols = list(df.columns[:6])

# --------------------------------------------------------------
# Plotting outliers
# --------------------------------------------------------------

# Boxplot of accelerometer features by label
df[oultier_cols[:3] + ["label"]].boxplot(by="label",figsize = (20,10),layout = (1,3))

# Boxplot of gyroscope features by label
df[oultier_cols[3:] + ["label"]].boxplot(by="label",figsize = (20,10),layout = (1,3))



# --------------------------------------------------------------
# Poltting outlier depeneds on the marking method we use
# --------------------------------------------------------------

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
# --------------------------------------------------------------

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
col = "acc_x"
oultier_col = col + "_outlier"

# Detect outliers using IQR method
IQR_outlier = mark_outliers_iqr(df,col)

# Plot binary outliers
plot_binary_outliers(IQR_outlier, col, oultier_col, True)

# Loop through each column to detect outliers and plot binary outliers
for col in oultier_cols:
    # Detect outliers using IQR method
    IQR_outlier = mark_outliers_iqr(df,col)
    # Plot binary outliers
    plot_binary_outliers(IQR_outlier, col, col + "_outlier", True)
    


# --------------------------------------------------------------
# Chauvenets criteron (distribution based)
# --------------------------------------------------------------

# Check for normal distribution

# Histograms of accelerometer features by label
df[oultier_cols[:3] + ["label"]].hist(by="label",figsize = (20,20),layout = (3,3))

# Histograms of gyrscope features by label
df[oultier_cols[3:] + ["label"]].hist(by="label",figsize = (20,20),layout = (3,3))


# Insert Chauvenet's function
def mark_outliers_chauvenet(dataset, col, C=2):
    """Finds outliers in the specified column of datatable and adds a binary column with
    the same name extended with '_outlier' that expresses the result per data point.
    
    Taken from: https://github.com/mhoogen/ML4QS/blob/master/Python3Code/Chapter3/OutlierDetection.py

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): The column you want apply outlier detection to
        C (int, optional): Degree of certainty for the identification of outliers given the assumption 
                           of a normal distribution, typicaly between 1 - 10. Defaults to 2.

    Returns:
        pd.DataFrame: The original dataframe with an extra boolean column 
        indicating whether the value is an outlier or not.
    """

    dataset = dataset.copy()
    # Compute the mean and standard deviation.
    mean = dataset[col].mean()
    std = dataset[col].std()
    N = len(dataset.index)
    criterion = 1.0 / (C * N)

    # Consider the deviation for the data points.
    deviation = abs(dataset[col] - mean) / std

    # Express the upper and lower bounds.
    low = -deviation / math.sqrt(C)
    high = deviation / math.sqrt(C)
    prob = []
    mask = []

    # Pass all rows in the dataset.
    for i in range(0, len(dataset.index)):
        # Determine the probability of observing the point
        prob.append(
            1.0 - 0.5 * (scipy.special.erf(high[i]) - scipy.special.erf(low[i]))
        )
        # And mark as an outlier when the probability is below our criterion.
        mask.append(prob[i] < criterion)
    dataset[col + "_outlier"] = mask
    return dataset

# Loop over all columns
for col in oultier_cols:
    chauv_outlier = mark_outliers_chauvenet(df,col)
    plot_binary_outliers(chauv_outlier, col, col + "_outlier", True)

# --------------------------------------------------------------
# Local outlier factor (distance based)
# --------------------------------------------------------------

# Insert LOF function
def mark_outliers_lof(dataset, columns, n=5):
    """Mark values as outliers using LOF

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): The column you want apply outlier detection to
        n (int, optional): n_neighbors. Defaults to 20.
    
    Returns:
        pd.DataFrame: The original dataframe with an extra boolean column
        indicating whether the value is an outlier or not.
    """
    
    dataset = dataset.copy()

    lof = LocalOutlierFactor(n_neighbors=n)
    data = dataset[columns]
    outliers = lof.fit_predict(data)
    X_scores = lof.negative_outlier_factor_

    dataset["outlier_lof"] = outliers == -1
    return dataset, outliers, X_scores

# Loop over all columns

LOF_outlier, outliers, X_scores = mark_outliers_lof(df,oultier_cols)
for col in oultier_cols:
    plot_binary_outliers(LOF_outlier, col, "outlier_lof", True)
    
# --------------------------------------------------------------
# Check outliers grouped by label
# --------------------------------------------------------------

label = "bench"

# Using IQR method
for col in oultier_cols:
    chauv_outlier = mark_outliers_iqr(df[df["label"] == label],col)
    plot_binary_outliers(chauv_outlier, col, col + "_outlier", True)

# Using Chauvenet's criterion
for col in oultier_cols:
    chauv_outlier = mark_outliers_chauvenet(df[df["label"] == label],col)
    plot_binary_outliers(chauv_outlier, col, col + "_outlier", True)


# Using Local Outlier Factor (LOF)
LOF_outlier, outliers, X_scores = mark_outliers_lof(df[df["label"] == label],oultier_cols)
for col in oultier_cols:
    plot_binary_outliers(LOF_outlier, col, "outlier_lof", True)

# --------------------------------------------------------------
# Choose method and deal with outliers
# --------------------------------------------------------------

# Test on single column using Chauvenet's criterion
col = "gyr_y"

# Mark outliers using Chauvenet's criterion
dataset = mark_outliers_chauvenet(df,col)

# Select rows where outliers are detected
dataset[dataset["gyr_y_outlier"]]

# Set outliers to NaN
dataset.loc[dataset["gyr_y_outlier"],col] = np.nan


# Copy the original DataFrame for outlier removal
chauvenet_oultier_removed_df = df.copy()
LOF_oultier_removed_df = df.copy()

# Initialize variables to count the number of outliers removed
no_chauvenet_outliers = 0
no_LOF_outliers  = 0

# Iterate through outlier columns
for col in oultier_cols:
    # Iterate through unique labels
    for label in df["label"].unique():
        # Mark outliers using Chauvenet's criterion
        dataset = mark_outliers_chauvenet(df[df["label"] == label],col)
        # Set outliers to NaN
        dataset.loc[dataset[col + "_outlier"], col] = np.nan
        # Update the outlier-removed DataFrame
        chauvenet_oultier_removed_df.loc[(chauvenet_oultier_removed_df["label"] == label ), col] = dataset[col]
         # Count the number of outliers removed for each column
        n_outliers = len(dataset) - len(dataset[col].dropna())
        print(f"removed {n_outliers} from {col} for {label}")
    # Update the total number of outliers removed
    no_chauvenet_outliers+= n_outliers

# Iterate through unique labels
for label in df["label"].unique():
    # Mark outliers using Local Outlier Factor (LOF)
    dataset, outliers, X_scores = mark_outliers_lof(df[df["label"] == label],oultier_cols)
    # Iterate through outlier columns
    for col in oultier_cols:
        # Set outliers to NaN
        dataset.loc[dataset["outlier_lof"], col] = np.nan
        # Update the outlier-removed DataFrame
        LOF_oultier_removed_df.loc[(LOF_oultier_removed_df["label"] == label ), col] = dataset[col]
        # Count the number of outliers removed for each column
        n_outliers = len(dataset) - len(dataset[col].dropna())
        print(f"removed {n_outliers} from {col} for {label}")
    # Update the total number of outliers removed
    no_LOF_outliers+= n_outliers
print(no_LOF_outliers)
# --------------------------------------------------------------
# Export new dataframe
# --------------------------------------------------------------

# Export new DataFrame with Chauvenet's criterion outlier removed
chauvenet_oultier_removed_df.to_pickle("../../data/interim/02_outliers_removed_chauvenets.pkl")
# Export new DataFrame with Local Outlier Factor (LOF) outlier removed
LOF_oultier_removed_df.to_pickle("../../data/interim/03_outliers_removed_LOF.pkl")