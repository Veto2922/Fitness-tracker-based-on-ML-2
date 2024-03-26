import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter
from scipy.signal import argrelextrema
from sklearn.metrics import mean_absolute_error

pd.options.mode.chained_assignment = None


# Plot settings
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2


# --------------------------------------------------------------
# Load data

df = pd.read_pickle("../../data/interim/01_data_processed.pkl")
#df["label"].unique()
df =df[df["label"] != "rest"]


#count sum of squares 
acc_r = df["acc_x"] ** 2 + df["acc_y"] ** 2 + df["acc_z"] ** 2
gyr_r = df["gyr_x"] ** 2 + df["gyr_y"] ** 2 + df["gyr_z"] ** 2

df["acc_r"] = np.sqrt(acc_r)
df["gyr_r"] = np.sqrt(gyr_r)


# --------------------------------------------------------------
# Split data
df_bench =df[df["label"] == "bench"]
df_ohp =df[df["label"] == "ohp"]
df_squat =df[df["label"] == "squat"]
df_dead =df[df["label"] == "dead"]
df_row =df[df["label"] == "row"]

# --------------------------------------------------------------
# Visualize data to identify patterns
plot_df =df_bench
plot_df[plot_df['set'] == plot_df['set'].unique()[0]]['acc_x'].plot()
plot_df[plot_df['set'] == plot_df['set'].unique()[0]]['acc_y'].plot()
plot_df[plot_df['set'] == plot_df['set'].unique()[0]]['acc_z'].plot()
plot_df[plot_df['set'] == plot_df['set'].unique()[0]]['acc_r'].plot()


plot_df[plot_df['set'] == plot_df['set'].unique()[0]]['gyr_x'].plot()
plot_df[plot_df['set'] == plot_df['set'].unique()[0]]['gyr_y'].plot()
plot_df[plot_df['set'] == plot_df['set'].unique()[0]]['gyr_z'].plot()
plot_df[plot_df['set'] == plot_df['set'].unique()[0]]['gyr_r'].plot()


# --------------------------------------------------------------
# Configure LowPassFilter
# --------------------------------------------------------------
fs = 1000 / 200

lowpass =  LowPassFilter()

# --------------------------------------------------------------
# Apply and tweak LowPassFilter
# --------------------------------------------------------------
bench_set =  df_bench[df_bench['set'] == df_bench['set'].unique()[0]]
ohe_set =  df_ohp[df_ohp['set'] == df_ohp['set'].unique()[11]]
squat_set =  df_squat[df_squat['set'] == df_squat['set'].unique()[0]]
dead_set =  df_dead[df_dead['set'] == df_dead['set'].unique()[0]]
row_set =  df_row[df_row['set'] == df_row['set'].unique()[0]]


# --------------------------------------------------------------
# Create function to count repetitions
# --------------------------------------------------------------


# --------------------------------------------------------------
# Create benchmark dataframe
# --------------------------------------------------------------


# --------------------------------------------------------------
# Evaluate the results
# --------------------------------------------------------------
