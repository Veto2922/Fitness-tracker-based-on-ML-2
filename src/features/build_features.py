import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction
from FrequencyAbstraction import FourierTransformation
from sklearn.cluster import KMeans

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------
df = pd.read_pickle("../../data/interim/02_outliers_removed_chauvenets.pkl")
predictor_columns = list(df.columns[:6])


#plot settings
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"]=(20,5)  
plt.rcParams["figure.dpi"]=100
plt.rcParams["lines.linewidth"]=2
# --------------------------------------------------------------
# Dealing with missing values (imputation)
# --------------------------------------------------------------
for col in predictor_columns:
    df[col]=df[col].interpolate()

df.info()
# --------------------------------------------------------------
# Calculating set duration
# --------------------------------------------------------------
df[df["set"]==25]['acc_y'].plot()
df[df["set"]==50]['acc_y'].plot()

duration = df[df["set"]==1].index[-1]-df[df["set"]==1].index[0]
duration.seconds  #calculating the duration of a single set


for s in df["set"].unique():
    start =df[df["set"]==s].index[0]
    stop = df[df["set"]==s].index[-1]
    
    duration = stop-start
    df.loc[(df["set"]==s),"duration"]=duration.seconds
    
df_duration =df.groupby(["category"])["duration"].mean()

df_duration[0]/5      #the avarage duration of a single repetaion in heavy set
df_duration[1] /10    #the avarage duration of a single repetaion in medium set
# --------------------------------------------------------------
# Butterworth lowpass filter

df_lowpass = df.copy()
Low_pass = LowPassFilter()

fs=1000/200 #resulting in 5 instances per second , 
            #(200)is the diffrences between records
            #1000 one second
            #fs : sampling frequency

cutoff =1.3 
df_lowpass=Low_pass.low_pass_filter(df_lowpass,"acc_y",fs,cutoff,order=5)

subset = df_lowpass[df_lowpass["set"] == 45]
print(subset["label"][0])

fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))
ax[0].plot(subset["acc_y"].reset_index(drop=True), label="raw data")
ax[1].plot(subset["acc_y_lowpass"].reset_index(drop=True), label="butterworth filter")
ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True)
ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True)


#loop over all column and update thier values with low pass values
for col in predictor_columns:
    df_lowpass = Low_pass.low_pass_filter(df_lowpass, col, fs, cutoff, order=5)
    df_lowpass[col] = df_lowpass[col + "_lowpass"]
    del df_lowpass[col + "_lowpass"]
# --------------------------------------------------------------
# Principal component analysis PCA

#PCA is a dimensionality reduction technique
df_pca =df_lowpass.copy()
PCA =PrincipalComponentAnalysis()

pca_values = PCA.determine_pc_explained_variance(df_pca,predictor_columns)

plt.figure(figsize=(10,10))
plt.plot(range(1, len(predictor_columns)+1) , pca_values)
plt.xlabel("principle component numbers")
plt.ylabel("explained variance")
plt.show()

#from the figure the optimal number of principle component is 3
df_pca=PCA.apply_pca(df_pca,predictor_columns,3)

subset = df_pca[df_pca["set"] == 35]
subset[["pca_1","pca_2","pca_3"]].plot()
# --------------------------------------------------------------
# Sum of squares attributes

df_squared = df_pca.copy()

acc_r = df_squared["acc_x"] ** 2 + df_squared["acc_y"] ** 2 + df_squared["acc_z"] ** 2
gyr_r = df_squared["gyr_x"] ** 2 + df_squared["gyr_y"] ** 2 + df_squared["gyr_z"] ** 2

df_squared["acc_r"] = np.sqrt(acc_r)
df_squared["gyr_r"] = np.sqrt(gyr_r)

subset = df_squared[df_squared["set"] == 14]
subset[["acc_r", "gyr_r"]].plot(subplots=True)


# --------------------------------------------------------------
# Temporal abstraction

temporal_df =df_squared.copy()
predictor_columns =predictor_columns +["acc_r","gyr_r"]

Numabs = NumericalAbstraction()
ws = int(1000/200)

for col in predictor_columns:
    temporal_df = Numabs.abstract_numerical(temporal_df,[col],ws,'mean')
    temporal_df = Numabs.abstract_numerical(temporal_df,[col],ws,'std')


#this will introduce wrong values when we move from set to another so , to solve it we calculate this to each set 

df_temporal_list =[]

for s in temporal_df["set"].unique():
    subset=temporal_df[temporal_df["set"]==s].copy()
    for col in predictor_columns:
        subset = Numabs.abstract_numerical(subset,[col],ws,'mean')
        subset = Numabs.abstract_numerical(subset,[col],ws,'std')
    df_temporal_list.append(subset)
    
temporal_df=pd.concat(df_temporal_list)
temporal_df.info( )

subset[["acc_y", "acc_y_temp_mean_ws_5", "acc_y_temp_std_ws_5"]].plot()
subset[["gyr_y", "gyr_y_temp_mean_ws_5", "gyr_y_temp_std_ws_5"]].plot()
    
# --------------------------------------------------------------
# Frequency features
df_freq = temporal_df.copy().reset_index()
FreqAbs = FourierTransformation()

fs = int(1000 / 200)  #number of samples per second
ws = int(2800 / 200)  # the average length of repetation

df_freq = FreqAbs.abstract_frequency(df_freq,["acc_y"],ws,fs)

df_freq_list = []
for s in df_freq["set"].unique():
    print(f"Applying Fourier transformations to set {s}")
    subset = df_freq[df_freq["set"] == s].reset_index(drop=True).copy()
    subset = FreqAbs.abstract_frequency(subset, predictor_columns, ws, fs)
    df_freq_list.append(subset)

df_freq = pd.concat(df_freq_list).set_index("epoch (ms)", drop=True)


# --------------------------------------------------------------
# Dealing with overlapping windows
# --------------------------------------------------------------
df_freq.dropna()
df_freq = df_freq.iloc[::2] #to reduce the correlation between data

# --------------------------------------------------------------
# Clustering
# --------------------------------------------------------------
df_cluster = df_freq.copy()

cluster_columns = ["acc_x", "acc_y", "acc_z"]
k_values = range(2, 10)
inertias = []

for k in k_values:
    subset = df_cluster[cluster_columns]
    kmeans = KMeans(n_clusters=k, n_init=20, random_state=0)
    cluster_labels = kmeans.fit_predict(subset)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(10, 10))
plt.plot(k_values, inertias)
plt.xlabel("k")
plt.ylabel("Sum of squared distances")
plt.show()

kmeans = KMeans(n_clusters=5, n_init=20, random_state=0)
subset = df_cluster[cluster_columns]
df_cluster["cluster"] = kmeans.fit_predict(subset)

# Plot Clusters
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(projection="3d")
for c in df_cluster["cluster"].unique():
    subset = df_cluster[df_cluster["cluster"] == c]
    ax.scatter(subset["acc_x"], subset["acc_y"], subset["acc_z"], label=c)
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
plt.legend()
plt.show()


# Plot Labels
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(projection="3d")
for l in df_cluster["label"].unique():
    subset = df_cluster[df_cluster["label"] == l]
    ax.scatter(subset["acc_x"], subset["acc_y"], subset["acc_z"], label=l)
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
plt.legend()
plt.show()



# --------------------------------------------------------------
# Export dataset

df_cluster.to_pickle("../../data/interim/03_data_features.pkl")