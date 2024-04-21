import numpy as np
import math
import scipy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
    precision_score,
)
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.signal import butter, lfilter, filtfilt
import copy
import pandas as pd
import pickle


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


def low_pass_filter(
    data_table,
    col,
    sampling_frequency,
    cutoff_frequency,
    order=5,
    phase_shift=True,
):
    # http://stackoverflow.com/questions/12093594/how-to-implement-band-pass-butterworth-filter-with-scipy-signal-butter
    # Cutoff frequencies are expressed as the fraction of the Nyquist frequency, which is half the sampling frequency
    nyq = 0.5 * sampling_frequency
    cut = cutoff_frequency / nyq

    b, a = butter(order, cut, btype="low", output="ba", analog=False)
    if phase_shift:
        data_table[col + "_lowpass"] = filtfilt(b, a, data_table[col])
    else:
        data_table[col + "_lowpass"] = lfilter(b, a, data_table[col])
    return data_table


class PrincipalComponentAnalysis:

    pca = []

    def __init__(self):
        self.pca = []

    def normalize_dataset(self, data_table, columns):
        dt_norm = copy.deepcopy(data_table)
        for col in columns:
            dt_norm[col] = (data_table[col] - data_table[col].mean()) / (
                data_table[col].max()
                - data_table[col].min()
                # data_table[col].std()
            )
        return dt_norm

    # Perform the PCA on the selected columns and return the explained variance.
    def determine_pc_explained_variance(self, data_table, cols):

        # Normalize the data first.
        dt_norm = self.normalize_dataset(data_table, cols)

        # perform the PCA.
        self.pca = PCA(n_components=len(cols))
        self.pca.fit(dt_norm[cols])
        # And return the explained variances.
        return self.pca.explained_variance_ratio_

    # Apply a PCA given the number of components we have selected.
    # We add new pca columns.
    def apply_pca(self, data_table, cols, number_comp):

        # Normalize the data first.
        dt_norm = self.normalize_dataset(data_table, cols)

        # perform the PCA.
        self.pca = PCA(n_components=number_comp)
        self.pca.fit(dt_norm[cols])

        # Transform our old values.
        new_values = self.pca.transform(dt_norm[cols])

        # And add the new ones:
        for comp in range(0, number_comp):
            data_table["pca_" + str(comp + 1)] = new_values[:, comp]

        return data_table


class NumericalAbstraction:

    # This function aggregates a list of values using the specified aggregation
    # function (which can be 'mean', 'max', 'min', 'median', 'std')
    def aggregate_value(self, aggregation_function):
        # Compute the values and return the result.
        if aggregation_function == "mean":
            return np.mean
        elif aggregation_function == "max":
            return np.max
        elif aggregation_function == "min":
            return np.min
        elif aggregation_function == "median":
            return np.median
        elif aggregation_function == "std":
            return np.std
        else:
            return np.nan

    # Abstract numerical columns specified given a window size (i.e. the number of time points from
    # the past considered) and an aggregation function.
    def abstract_numerical(self, data_table, cols, window_size, aggregation_function):

        # Create new columns for the temporal data, pass over the dataset and compute values
        for col in cols:
            data_table[
                col + "_temp_" + aggregation_function + "_ws_" + str(window_size)
            ] = (
                data_table[col]
                .rolling(window_size)
                .apply(self.aggregate_value(aggregation_function))
            )

        return data_table


# This class performs a Fourier transformation on the data to find frequencies that occur
# often and filter noise.
class FourierTransformation:

    # Find the amplitudes of the different frequencies using a fast fourier transformation. Here,
    # the sampling rate expresses the number of samples per second (i.e. Frequency is Hertz of the dataset).
    def find_fft_transformation(self, data, sampling_rate):
        # Create the transformation, this includes the amplitudes of both the real
        # and imaginary part.
        transformation = np.fft.rfft(data, len(data))
        return transformation.real, transformation.imag

    # Get frequencies over a certain window.
    def abstract_frequency(self, data_table, cols, window_size, sampling_rate):

        # Create new columns for the frequency data.
        freqs = np.round((np.fft.rfftfreq(int(window_size)) * sampling_rate), 3)

        for col in cols:
            data_table[col + "_max_freq"] = np.nan
            data_table[col + "_freq_weighted"] = np.nan
            data_table[col + "_pse"] = np.nan
            for freq in freqs:
                data_table[
                    col + "_freq_" + str(freq) + "_Hz_ws_" + str(window_size)
                ] = np.nan

        # Pass over the dataset (we cannot compute it when we do not have enough history)
        # and compute the values.
        for i in range(window_size, len(data_table.index)):
            for col in cols:
                real_ampl, imag_ampl = self.find_fft_transformation(
                    data_table[col].iloc[
                        i - window_size : min(i + 1, len(data_table.index))
                    ],
                    sampling_rate,
                )
                # We only look at the real part in this implementation.
                for j in range(0, len(freqs)):
                    data_table.loc[
                        i, col + "_freq_" + str(freqs[j]) + "_Hz_ws_" + str(window_size)
                    ] = real_ampl[j]
                # And select the dominant frequency. We only consider the positive frequencies for now.

                data_table.loc[i, col + "_max_freq"] = freqs[
                    np.argmax(real_ampl[0 : len(real_ampl)])
                ]
                data_table.loc[i, col + "_freq_weighted"] = float(
                    np.sum(freqs * real_ampl)
                ) / np.sum(real_ampl)
                PSD = np.divide(np.square(real_ampl), float(len(real_ampl)))
                PSD_pdf = np.divide(PSD, np.sum(PSD))
                data_table.loc[i, col + "_pse"] = -np.sum(np.log(PSD_pdf) * PSD_pdf)

        return data_table


class FitnessTracker:

    def __init__(self, acc_df, gyr_df) -> None:
        self.acc_df = acc_df
        self.gyr_df = gyr_df
        self.model = pickle.load(open("../../notebooks/model.pkl", "rb"))
        self.test = self._make_test_data()

    def _make_test_data(self):
        acc_df = self.acc_df
        gyr_df = self.gyr_df
        # set datetime as index to convert to a time series database
        acc_df.index = pd.to_datetime(acc_df["time (01:00)"], format="mixed")
        gyr_df.index = pd.to_datetime(gyr_df["time (01:00)"], format="mixed")

        # now we can get rid of all features referencing time
        acc_df.drop(columns=["epoch (ms)", "time (01:00)", "elapsed (s)"], inplace=True)
        gyr_df.drop(columns=["epoch (ms)", "time (01:00)", "elapsed (s)"], inplace=True)

        # Merge the data
        data_merged = pd.merge(
            acc_df.iloc[:, :3], gyr_df, right_index=True, left_index=True, how="outer"
        )

        data_merged.index.rename("epoch (ms)", inplace=True)

        data_merged.columns = [
            "acc_x",
            "acc_y",
            "acc_z",
            "gyr_x",
            "gyr_y",
            "gyr_z",
        ]

        # Resampling the data
        sampling = {
            "acc_x": "mean",
            "acc_y": "mean",
            "acc_z": "mean",
            "gyr_x": "mean",
            "gyr_y": "mean",
            "gyr_z": "mean",
        }

        # group by day
        days = [g for n, g in data_merged.groupby(pd.Grouper(freq="D"))]
        days_resampled = pd.concat(
            [df.resample(rule="200ms").apply(sampling).dropna() for df in days]
        )

        ############### Remove outliers ###############
        outlier_removed_df = days_resampled.copy()

        for col in outlier_removed_df.columns:
            dataset = mark_outliers_chauvenet(days_resampled, col)

            # Replace values marked as outlier with Nan
            dataset.loc[dataset[f"{col}_outlier"], col] = np.nan

            # Update the columns in the original dataframe
            outlier_removed_df.loc[:, col] = dataset[col]

        ############### Build Features ###############
        df = outlier_removed_df.copy()

        # Dealing with missing values (imputation)
        for col in df.columns:
            df[col] = df[col].interpolate()

        # Calculating set duration
        duration = df.index[-1] - df.index[0]
        df.loc[:, "duration"] = duration.seconds

        # Butterworth lowpass filter
        fs = 1000 / 200
        cutoff = 1.3

        for col in df.columns:
            df_lowpass = low_pass_filter(df, col, fs, cutoff, order=5)
            df_lowpass[col] = df_lowpass[col + "_lowpass"]
            del df_lowpass[col + "_lowpass"]

        # Principal component analysis PCA
        df_pca = df_lowpass.copy()
        PCA = PrincipalComponentAnalysis()

        columns = df.columns

        PCA.apply_pca(df_pca, columns, 3)

        # Sum of squares attributes
        df_squared = df_pca.copy()

        acc_r = (
            df_squared["acc_x"] ** 2
            + df_squared["acc_y"] ** 2
            + df_squared["acc_z"] ** 2
        )
        gyr_r = (
            df_squared["gyr_x"] ** 2
            + df_squared["gyr_y"] ** 2
            + df_squared["gyr_z"] ** 2
        )

        df_squared["acc_r"] = np.sqrt(acc_r)
        df_squared["gyr_r"] = np.sqrt(gyr_r)

        # Temporal abstraction
        df_temporal = df_squared.copy()
        NumAbs = NumericalAbstraction()

        presictor_columns = list(columns) + ["acc_r", "gyr_r"]

        ws = int(1000 / 200)  # 1000ms / 200 steps in the data

        df_temporal_list = []
        subset = df_temporal.copy()

        for col in presictor_columns:
            subset = NumAbs.abstract_numerical(subset, [col], ws, "mean")
            subset = NumAbs.abstract_numerical(subset, [col], ws, "std")
            df_temporal_list.append(subset)

        df_temporal = pd.concat(df_temporal_list)

        # Frequency features
        df_freq = df_temporal.copy().reset_index()
        FreqAbs = FourierTransformation()

        fs = int(1000 / 200)
        ws = int(2800 / 200)  # 2800 -> time for repetition

        subset = df_freq.reset_index(drop=True).copy()
        subset = FreqAbs.abstract_frequency(subset, presictor_columns, ws, fs)

        df_freq = subset.set_index("epoch (ms)", drop=True)

        # Dealing with overlapping windows
        df_freq = df_freq.dropna()
        df_freq = df_freq.iloc[::2]

        # Clustering
        df_cluster = df_freq.copy()
        cluster_columns = ["acc_x", "acc_y", "acc_z"]

        kmeans = KMeans(n_clusters=5, n_init=20, random_state=0)
        subset = df_cluster[cluster_columns]
        df_cluster["cluster"] = kmeans.fit_predict(subset)

        # Selected Features

        selected_feature = [
            "acc_z",
            "duration",
            "acc_x_temp_mean_ws_5",
            "acc_x_freq_weighted",
            "acc_y_max_freq",
            "acc_y_freq_0.0_Hz_ws_14",
            "gyr_x_freq_0.0_Hz_ws_14",
            "gyr_y_freq_2.5_Hz_ws_14",
            "acc_r_max_freq",
            "gyr_r_freq_0.0_Hz_ws_14",
        ]

        return df_cluster[selected_feature]


acc_df = pd.read_csv(
    "../../data/raw/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv"
)
gyr_df = pd.read_csv(
    "../../data/raw/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Gyroscope_25.000Hz_1.4.4.csv"
)


mm = FitnessTracker(acc_df, gyr_df)


dataset = mm.test


acc_df = pd.read_csv(
    "../../data/raw/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv"
)
gyr_df = pd.read_csv(
    "../../data/raw/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Gyroscope_25.000Hz_1.4.4.csv"
)


# set datetime as index to convert to a time series database
acc_df.index = pd.to_datetime(acc_df["time (01:00)"], format="mixed")
gyr_df.index = pd.to_datetime(gyr_df["time (01:00)"], format="mixed")

# now we can get rid of all features referencing time
acc_df.drop(columns=["epoch (ms)", "time (01:00)", "elapsed (s)"], inplace=True)
gyr_df.drop(columns=["epoch (ms)", "time (01:00)", "elapsed (s)"], inplace=True)

# Merge the data
data_merged = pd.merge(
    acc_df.iloc[:, :3], gyr_df, right_index=True, left_index=True, how="outer"
)

data_merged.index.rename("epoch (ms)", inplace=True)

data_merged.columns = [
    "acc_x",
    "acc_y",
    "acc_z",
    "gyr_x",
    "gyr_y",
    "gyr_z",
]

# Resampling the data
sampling = {
    "acc_x": "mean",
    "acc_y": "mean",
    "acc_z": "mean",
    "gyr_x": "mean",
    "gyr_y": "mean",
    "gyr_z": "mean",
}

# group by day
days = [g for n, g in data_merged.groupby(pd.Grouper(freq="D"))]
days_resampled = pd.concat(
    [df.resample(rule="200ms").apply(sampling).dropna() for df in days]
)

############### Remove outliers ###############
outlier_removed_df = days_resampled.copy()

for col in outlier_removed_df.columns:
    dataset = mark_outliers_chauvenet(days_resampled, col)

    # Replace values marked as outlier with Nan
    dataset.loc[dataset[f"{col}_outlier"], col] = np.nan

    # Update the columns in the original dataframe
    outlier_removed_df.loc[:, col] = dataset[col]


############### Build Features ###############
df = outlier_removed_df.copy()

# Dealing with missing values (imputation)
for col in df.columns:
    df[col] = df[col].interpolate()

# Calculating set duration
duration = df.index[-1] - df.index[0]
df.loc[:, "duration"] = duration.seconds

# Butterworth lowpass filter
fs = 1000 / 200
cutoff = 1.3

for col in df.columns:
    df_lowpass = low_pass_filter(df, col, fs, cutoff, order=5)
    df_lowpass[col] = df_lowpass[col + "_lowpass"]
    del df_lowpass[col + "_lowpass"]

# Principal component analysis PCA
df_pca = df_lowpass.copy()

columns = df.columns

pca = PrincipalComponentAnalysis()
columns = df.columns


# Normalize the data first.
dt_norm = self.normalize_dataset(df_pca, columns)

# perform the PCA.
self.pca = PCA(n_components=3)
self.pca.fit(dt_norm[columns])

# Transform our old values.
new_values = self.pca.transform(dt_norm[columns])

# And add the new ones:
for comp in range(0, 3):
    df_pca["pca_" + str(comp + 1)] = new_values[:, comp]


pca.apply_pca(df_pca, columns, 3)

# Sum of squares attributes
df_squared = df_pca.copy()

acc_r = df_squared["acc_x"] ** 2 + df_squared["acc_y"] ** 2 + df_squared["acc_z"] ** 2
gyr_r = df_squared["gyr_x"] ** 2 + df_squared["gyr_y"] ** 2 + df_squared["gyr_z"] ** 2

df_squared["acc_r"] = np.sqrt(acc_r)
df_squared["gyr_r"] = np.sqrt(gyr_r)

# Temporal abstraction
df_temporal = df_squared.copy()
NumAbs = NumericalAbstraction()

presictor_columns = list(columns) + ["acc_r", "gyr_r"]

ws = int(1000 / 200)  # 1000ms / 200 steps in the data

df_temporal_list = []
subset = df_temporal.copy()

for col in presictor_columns:
    subset = NumAbs.abstract_numerical(subset, [col], ws, "mean")
    subset = NumAbs.abstract_numerical(subset, [col], ws, "std")
    df_temporal_list.append(subset)

df_temporal = pd.concat(df_temporal_list)

# Frequency features
df_freq = df_temporal.copy().reset_index()
FreqAbs = FourierTransformation()

fs = int(1000 / 200)
ws = int(2800 / 200)  # 2800 -> time for repetition

subset = df_freq.reset_index(drop=True).copy()
subset = FreqAbs.abstract_frequency(subset, presictor_columns, ws, fs)

df_freq = subset.set_index("epoch (ms)", drop=True)

# Dealing with overlapping windows
df_freq = df_freq.dropna()
df_freq = df_freq.iloc[::2]

# Clustering
df_cluster = df_freq.copy()
cluster_columns = ["acc_x", "acc_y", "acc_z"]

kmeans = KMeans(n_clusters=5, n_init=20, random_state=0)
subset = df_cluster[cluster_columns]
df_cluster["cluster"] = kmeans.fit_predict(subset)


# Selected Features

selected_feature = [
    "acc_z",
    "duration",
    "acc_x_temp_mean_ws_5",
    "acc_x_freq_weighted",
    "acc_y_max_freq",
    "acc_y_freq_0.0_Hz_ws_14",
    "gyr_x_freq_0.0_Hz_ws_14",
    "gyr_y_freq_2.5_Hz_ws_14",
    "acc_r_max_freq",
    "gyr_r_freq_0.0_Hz_ws_14",
]
