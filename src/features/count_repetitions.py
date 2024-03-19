import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mlp
from DataTransformation import LowPassFilter
from scipy.signal import argrelextrema
from sklearn.metrics import mean_absolute_error

pd.options.mode.chained_assignment = None


# --------------------------------------------------------------
# Adjust plot settings
# --------------------------------------------------------------
# Set the plotting style to 'fivethirtyeight' for Matplotlib plots
mlp.style.use("fivethirtyeight")
# Set the figure size to (20, 5) inches
mlp.rcParams["figure.figsize"] = (20,5)
# Set the resolution of the figure to 100 dots per inch (dpi) for exporting
mlp.rcParams["figure.dpi"] = 100
# Set the linewidth for lines in matplotlib plots to 2
mlp.rcParams["lines.linewidth"] = 2


# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------
df = pd.read_pickle("../../data/interim/01_data_processed.pkl")
df = df[df["label"] != "rest"]

# Calculate the squared magnitude for accelerometer and gyroscope readings
acc_r = df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2 
gyr_r = df['gyr_x']**2 + df['gyr_y']**2 + df['gyr_z']**2 

# Calculate the magnitude by taking the square root of the sum of squared components
df['acc_r'] = np.sqrt(acc_r)
df['gyr_r'] = np.sqrt(gyr_r)

# --------------------------------------------------------------
# Split data
# --------------------------------------------------------------
bench_df = df[df["label"] == "bench"]
squats_df = df[df["label"] == "squat"]
row_df = df[df["label"] == "row"]
ohp_df = df[df["label"] == "ohp"]
dead_df = df[df["label"] == "dead"]

# --------------------------------------------------------------
# Visualize data to identify patterns
# --------------------------------------------------------------
plot_df = bench_df

plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["acc_x"].plot()
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["acc_y"].plot()
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["acc_z"].plot()
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["acc_r"].plot()

plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["gyr_x"].plot()
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["gyr_y"].plot()
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["gyr_z"].plot()
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["gyr_r"].plot()

# --------------------------------------------------------------
# Configure LowPassFilter
# --------------------------------------------------------------

fs = 1000 / 200
LowPass= LowPassFilter()


# --------------------------------------------------------------
# Apply and tweak LowPassFilter
# --------------------------------------------------------------

bench_set = bench_df[bench_df["set"] == bench_df["set"].unique()[0]]
squat_set = squats_df[squats_df["set"] == squats_df["set"].unique()[0]]
row_set = row_df[row_df["set"] == row_df["set"].unique()[0]]
ohp_set = ohp_df[ohp_df["set"] == ohp_df["set"].unique()[0]]
dead_set = dead_df[dead_df["set"] == dead_df["set"].unique()[0]]

#taking the misssy pattern and cut out all the noise and converted it to somentihg  we can use to count the number of the rep
#as we have the info that the heavy is 5 reptition and the med is 5
bench_set["acc_r"].plot()


col = "acc_r"
LowPass.low_pass_filter(data_table=bench_set,col = col, sampling_frequency=fs,
                        cutoff_frequency=0.4,order = 5)

bench_set["acc_r_lowpass"].plot()


# --------------------------------------------------------------
# Create function to count repetitions
# --------------------------------------------------------------

def count_reps(dataset ,cutoff = 0.4 ,order = 10 ,column = "acc_r"):
    data = LowPass.low_pass_filter(data_table = dataset,col = column,
                                   sampling_frequency = fs,
                                   cutoff_frequency = cutoff,order = order)


    index = argrelextrema(data[col + "_lowpass"].values, np.greater)
    peaks = data.iloc[index]
    
    fig, ax = plt.subplots()
    plt.plot(dataset[f"{column}_lowpass"])
    plt.plot(peaks[f"{column}_lowpass"], "o", color="red")
    ax.set_ylabel(f"{column}_lowpass")
    exercise = dataset["label"].iloc[0].title()
    category = dataset["category"].iloc[0].title()
    plt.title(f"{category} {exercise}: {len(peaks)} Repetitions")
    plt.show()
    return len(peaks)

peaks = count_reps(dead_set)
    

count_reps(bench_set, cutoff=0.4)
count_reps(squat_set , cutoff=1.12)
count_reps(row_set , cutoff=0.2 , column='acc_x') ##why it's not changi?
count_reps(ohp_set , cutoff=0.35)
count_reps(dead_set , cutoff=0.4)


# --------------------------------------------------------------
# Create benchmark dataframe
# --------------------------------------------------------------

df["reps"] = df["category"].apply(lambda x: 5 if x == "heavy" else 10)

rep_df = df.groupby(["label", "category","set"])["reps"].max().reset_index()

rep_df["reps_pred"] = 0

for set in df["set"].unique():
    subset = df[df["set"] == set]
    
    column = "acc_r"
    cutoff = 0.4
    
    if subset["label"].iloc[0] == "squat":
        cutoff = 0.34
    
    elif subset["label"].iloc[0] == "row":
        cutoff = 0.65
    
    elif subset["label"].iloc[0] == "ohp":
        cutoff = 0.35
    
    reps = count_reps(subset, cutoff=cutoff, column=column)
    rep_df.loc[rep_df["set"] == set, "reps_pred"] = reps
 
# --------------------------------------------------------------
# Evaluate the results
# --------------------------------------------------------------

error = mean_absolute_error(rep_df["reps"], rep_df["reps_pred"]).round(2)
rep_df.groupby(["label", "category"])[["reps","reps_pred"]].mean().plot(kind="bar")
