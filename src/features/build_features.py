import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mlp
from DataTransformation import LowPassFilter , PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction
from FrequencyAbstraction import FourierTransformation
from sklearn.cluster import KMeans


# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

df = pd.read_pickle("../../data/interim/02_outliers_removed_chauvenets.pkl")

predictor_columns = list(df.columns[:6])


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
# Dealing with missing values (imputation)
# --------------------------------------------------------------
df[df["set"] == 20]["gyr_y"].plot()

df.info()

# We try removing all NaN values and filling gaps with the default method, which is linear interpolation. 
# For each predictor column, we perform interpolation to fill the missing values. 
# By default, interpolation computes the mean between the next and last available values in the same row.
for i in predictor_columns:
    df[i] = df[i].interpolate()
    

# --------------------------------------------------------------
# Calculating set duration
# --------------------------------------------------------------

df[df["set"] == 1].index[-1].second - df[df["set"] == 1].index[0].second

# Calculating set durations
for set in df["set"].unique():
    
    start = df[df["set"] == set].index[0]
    end = df[df["set"] == set].index[-1]
    
    duration = end - start
    
    df.loc[(df["set"] == set) , "duration"] = duration.seconds

# Calculate mean duration for each category
duration_df = df.groupby("category")["duration"].mean()

# Duration for heavy sets: Mean duration divided by 5.
duration_df.iloc[0]  / 5 

# Duration for medium sets: Mean duration divided by 10.
duration_df.iloc[1]  / 10

# --------------------------------------------------------------
# Butterworth lowpass filter
# --------------------------------------------------------------
# Create a copy of the DataFrame to apply the lowpass filter
lowpass_df = df.copy()

# Instantiate the LowPassFilter class
LowPass = LowPassFilter()

freq_sample  = 1000 / 200 # Calculates repetitions per second

cutoff = 1.3 

# Apply the lowpass filter to the 'acc_y' column of the DataFrame
LowPass.low_pass_filter(lowpass_df , 'acc_y' , freq_sample  , cutoff)

subset = lowpass_df[lowpass_df["set"] == 45]
print(subset["label"][0])

fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))
# Plot the raw 'acc_y' data in the first subplot
ax[0].plot(subset["acc_y"].reset_index(drop=True), label="raw data")
# Plot the lowpass filtered 'acc_y' data in the second subplot
ax[1].plot(subset["acc_y_lowpass"].reset_index(drop=True), label="butterworth filter")
# Add legend to the first subplot
ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True)
# Add legend to the second subplot
ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True)

# Apply the lowpass filter to all predictor columns
for col in predictor_columns:
    lowpass_df = LowPass.low_pass_filter(lowpass_df, col, freq_sample , cutoff, order=5)
    lowpass_df[col] = lowpass_df[col + "_lowpass"]
    del lowpass_df[col + "_lowpass"]

# --------------------------------------------------------------
# Principal component analysis PCA
# --------------------------------------------------------------
# Create a copy of the lowpass-filtered DataFrame for PCA analysis
pca_df = lowpass_df.copy()

# Instantiate the PrincipalComponentAnalysis class
PCA = PrincipalComponentAnalysis()

# Determine the explained variance for each principal component
pc_values = PCA.determine_pc_explained_variance(pca_df, predictor_columns)

# Plot the explained variance for each principal component
plt.plot(range(1, 7), pc_values)

# Apply PCA to reduce dimensionality to 3 principal components
pca_df = PCA.apply_pca(pca_df, predictor_columns, 3)

# Select a subset of data for a specific set (e.g., set 35)
subset = pca_df[pca_df["set"] == 35]

# Plot the three principal components for the subset
subset[['pca_1', 'pca_2', 'pca_3']].plot()



# --------------------------------------------------------------
# Sum of squares attributes
# --------------------------------------------------------------

# Create a copy of the DataFrame containing PCA results for further analysis
squares_df = pca_df.copy()

# Calculate the squared magnitude for accelerometer and gyroscope readings
acc_r = squares_df['acc_x']**2 + squares_df['acc_y']**2 + squares_df['acc_z']**2 
gyr_r = squares_df['gyr_x']**2 + squares_df['gyr_y']**2 + squares_df['gyr_z']**2 

# Calculate the magnitude by taking the square root of the sum of squared components
squares_df['acc_r'] = np.sqrt(acc_r)
squares_df['gyr_r'] = np.sqrt(gyr_r)

# Select a subset of data for a specific set (e.g., set 18)
subset = squares_df[squares_df["set"] == 18] 

# Plot the magnitude of acceleration and rotation for the subset
subset[['acc_r', 'gyr_r']].plot(subplots=True)


# --------------------------------------------------------------
# Temporal abstraction
# --------------------------------------------------------------
# Create a copy of the DataFrame containing PCA results for temporal abstraction
temporal_df = squares_df.copy()

# Add magnitude columns 'acc_r' and 'gyr_r' to the predictor columns list
predictor_columns = predictor_columns + ['acc_r', 'gyr_r']

# Instantiate the NumericalAbstraction class
NumAbs = NumericalAbstraction()

# Initialize a list to store temporally abstracted subsets
temporal_df_list = []

# Iterate over unique sets in the DataFrame
for set_id in temporal_df['set'].unique():
    # Select subset corresponding to the current set
    subset = temporal_df[temporal_df['set'] == set_id].copy()
    
    # Perform temporal abstraction for each predictor column
    for col in predictor_columns:
        # Calculate the mean and standard deviation with a window size of 5
        subset = NumAbs.abstract_numerical(subset, predictor_columns, window_size=5, aggregation_function='mean')
        subset = NumAbs.abstract_numerical(subset, predictor_columns, window_size=5, aggregation_function='std')

    # Append the abstracted subset to the list
    temporal_df_list.append(subset)

# Concatenate all abstracted subsets into a single DataFrame
temporal_df = pd.concat(temporal_df_list)

# Plot the temporal abstraction results for 'acc_y'
subset[['acc_y', 'acc_y_temp_mean_ws_5', 'acc_y_temp_std_ws_5']].plot()

# Plot the temporal abstraction results for 'gyr_y'
subset[['gyr_y', 'gyr_y_temp_mean_ws_5', 'gyr_y_temp_std_ws_5']].plot()


# --------------------------------------------------------------
# Frequency features
# --------------------------------------------------------------
# Create a copy of the temporally abstracted DataFrame and reset the index
df_freq = temporal_df.copy().reset_index()

# Instantiate the FourierTransformation class
FreqAbs = FourierTransformation()

# Define the sampling frequency (fs) and window size (ws)
fs = int(1000 / 200)  # Sampling frequency (samples per second)
ws = int(2800 / 200)   # Window size (number of samples)

# Apply Fourier transformations to 'acc_y' column
df_freq = FreqAbs.abstract_frequency(df_freq, ["acc_y"], ws, fs)

# Select a subset of data for a specific set (e.g., set 15)
subset = df_freq[df_freq["set"] == 15]

# Plot 'acc_y' data
subset[["acc_y"]].plot()

# Plot frequency features of 'acc_y'
subset[['acc_y_max_freq', 'acc_y_freq_weighted', 'acc_y_pse',
        'acc_y_freq_0.0_Hz_ws_14', 'acc_y_freq_0.357_Hz_ws_14',
        'acc_y_freq_0.714_Hz_ws_14', 'acc_y_freq_1.071_Hz_ws_14']].plot()

# Initialize a list to store Fourier-transformed subsets
df_freq_list = []

# Iterate over unique sets in the DataFrame
for s in df_freq["set"].unique():
    print(f"Applying Fourier transformations to set {s}")
    # Select subset corresponding to the current set
    subset = df_freq[df_freq["set"] == s].reset_index(drop=True).copy()
    # Apply Fourier transformations to predictor columns
    subset = FreqAbs.abstract_frequency(subset, predictor_columns, ws, fs)
    # Append the transformed subset to the list
    df_freq_list.append(subset)

# Concatenate all transformed subsets into a single DataFrame and set index
df_freq = pd.concat(df_freq_list).set_index("epoch (ms)", drop=True)


# --------------------------------------------------------------
# Dealing with overlapping windows
# --------------------------------------------------------------
# Remove rows with any NaN values from the DataFrame
df_freq = df_freq.dropna()

# Select every other row in the DataFrame
df_freq = df_freq.iloc[::2]

# --------------------------------------------------------------
# Clustering
# --------------------------------------------------------------
# Create a copy of the DataFrame containing frequency features for clustering
cluster_df = df_freq.copy()

# Define the columns to be used for clustering
cluster_columns = ['acc_x', 'acc_y', 'acc_z']

# Initialize a list to store the inertia values for different values of K
inertias = []

# Iterate over different values of K for KMeans clustering
for k in range(2, 10):
    # Select subset of data for clustering
    subset = cluster_df[cluster_columns]
    # Instantiate KMeans with current value of K
    kmeans = KMeans(n_clusters=k, n_init=20, random_state=0)
    # Fit KMeans to the subset and predict cluster labels
    cluster_label = kmeans.fit_predict(subset)
    # Append the inertia value to the list
    inertias.append(kmeans.inertia_)

# Plot the elbow curve to determine the optimal number of clusters (K)
plt.figure(figsize=(10, 10))
plt.plot(range(2, 10), inertias) 
plt.xlabel("K")
plt.ylabel("Sum of squared distances")
plt.show()

# Perform KMeans clustering with optimal number of clusters (e.g., 5)
kmeans = KMeans(n_clusters=5, n_init=20, random_state=0)
subset = cluster_df[cluster_columns]
# Predict cluster labels and assign them to the DataFrame
cluster_df["cluster_label"] = kmeans.fit_predict(subset)

# Visualize clusters in 3D space
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(projection="3d")
for cluster in cluster_df["cluster_label"].unique():
    subset = cluster_df[cluster_df["cluster_label"] == cluster]
    ax.scatter(subset["acc_x"], subset["acc_y"], subset["acc_z"], label=cluster)
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
plt.legend()
plt.show()


"""selected = ['acc_y_freq_0.0_Hz_ws_14',
                            'acc_z_freq_0.0_Hz_ws_14',
                            'pca_1',
                            'acc_y_temp_mean_ws_5',
                            'cluster_label',
                            'acc_y',
                            'gyr_r_freq_0.0_Hz_ws_14',
                            'pca_2',
                            'acc_z_temp_mean_ws_5',
                            'acc_x_freq_0.0_Hz_ws_14',
                            'acc_z',
                            'acc_x_temp_mean_ws_5',
                            'gyr_z_temp_std_ws_5',
                            'gyr_r_temp_mean_ws_5',
                            'acc_y_max_freq']

input_data = cluster_df[selected].iloc[3000]
input_data = input_data.to_numpy().reshape(1, -1)
"""
# Create a 3D plot for visualization
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(projection="3d")

# Iterate over unique labels in the DataFrame
for label in cluster_df["label"].unique():
    # Select subset of data for the current label
    subset = cluster_df[cluster_df["label"] == label]
    # Scatter plot of data points in 3D space
    ax.scatter(subset["acc_x"], subset["acc_y"], subset["acc_z"], label=label)

# Set labels for x, y, and z axes
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")

# Display legend
plt.legend()
plt.show()

# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------
# Save the DataFrame with clustering results to a pickle file
cluster_df.to_pickle("../../data/interim/04_data_features.pkl")