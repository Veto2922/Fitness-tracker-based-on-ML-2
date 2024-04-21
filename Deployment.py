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

# from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.signal import butter, lfilter, filtfilt
import copy
import pandas as pd
import pickle

from scipy.signal import argrelextrema
from src.features.DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from src.features.FrequencyAbstraction import FourierTransformation
from src.features.TemporalAbstraction import NumericalAbstraction

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


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


class FitnessTracker:

    def __init__(self, acc_df, gyr_df) -> None:
        self.acc_df = acc_df
        self.gyr_df = gyr_df
        self.model = pickle.load(open("notebooks/model.pkl", "rb"))
        self.test = self._make_test_data()

    def _make_dataset(self):
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

        return days_resampled

    def _make_test_data(self):

        ############### Remove outliers ###############
        df = self._make_dataset()
        outlier_removed_df = df.copy()

        for col in outlier_removed_df.columns:
            dataset = mark_outliers_chauvenet(df, col)

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

        LowPass = LowPassFilter()

        for col in df.columns:
            df_lowpass = LowPass.low_pass_filter(
                df, col, sampling_frequency=fs, cutoff_frequency=cutoff, order=5
            )
            df_lowpass[col] = df_lowpass[col + "_lowpass"]
            del df_lowpass[col + "_lowpass"]

        # Principal component analysis PCA
        df_pca = df_lowpass.copy()
        PCA = PrincipalComponentAnalysis()

        columns = df.columns[:6]

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
            "acc_y_temp_mean_ws_5",
            "gyr_z_temp_std_ws_5",
            "gyr_r_temp_mean_ws_5",
            "acc_x_freq_0.0_Hz_ws_14",
            "acc_y_freq_0.0_Hz_ws_14",
            "acc_z_freq_0.0_Hz_ws_14",
            "acc_z_freq_2.143_Hz_ws_14",
            "gyr_z_freq_0.714_Hz_ws_14",
            "acc_r_freq_1.786_Hz_ws_14",
            "gyr_r_freq_0.0_Hz_ws_14",
        ]

        return df_cluster[selected_feature]

    def predict(self):
        test = self.test
        model = self.model

        result = model.predict(test)

        return max(set(list(result)), key=list(result).count)

    def count_rep(self):
        training = self.predict()
        dataset = self._make_dataset()

        # Sum of squares attributes
        acc_r = dataset["acc_x"] ** 2 + dataset["acc_y"] ** 2 + dataset["acc_z"] ** 2
        gyr_r = dataset["gyr_x"] ** 2 + dataset["gyr_y"] ** 2 + dataset["gyr_z"] ** 2
        dataset["acc_r"] = np.sqrt(acc_r)
        dataset["gyr_r"] = np.sqrt(gyr_r)

        if training == "bench":
            cutoff = 0.5
            col = "gyr_y"

        if training == "squat":
            cutoff = 0.4
            col = "acc_r"

        if training == "row":
            cutoff = 0.7
            col = "gyr_r"

        if training == "ohp":
            cutoff = 0.5
            col = "acc_y"

        if training == "dead":
            cutoff = 0.5
            col = "acc_y"

        fs = 1000 / 200
        LowPass = LowPassFilter()
        data = LowPass.low_pass_filter(
            dataset, col=col, sampling_frequency=fs, cutoff_frequency=cutoff, order=10
        )
        indexes = argrelextrema(data[col + "_lowpass"].values, np.greater)
        peaks = data.iloc[indexes]

        return len(peaks)


acc_df = pd.read_csv(
    "data/raw/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv"
)
gyr_df = pd.read_csv(
    "data/raw/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Gyroscope_25.000Hz_1.4.4.csv"
)


mm = FitnessTracker(acc_df, gyr_df)

dataset = mm.test

result = mm.predict()

rep = mm.count_rep()

data = mm._make_dataset()


training = mm.predict()
dataset = mm._make_dataset()

# Sum of squares attributes
acc_r = dataset["acc_x"] ** 2 + dataset["acc_y"] ** 2 + dataset["acc_z"] ** 2
gyr_r = dataset["gyr_x"] ** 2 + dataset["gyr_y"] ** 2 + dataset["gyr_z"] ** 2
dataset["acc_r"] = np.sqrt(acc_r)
dataset["gyr_r"] = np.sqrt(gyr_r)

if training == "bench":
    cutoff = 0.5
    col = "gyr_y"

if training == "squat":
    cutoff = 0.4
    col = "acc_r"

if training == "row":
    cutoff = 0.7
    col = "gyr_r"

if training == "ohp":
    cutoff = 0.5
    col = "acc_y"

if training == "dead":
    cutoff = 0.5
    col = "acc_y"

fs = 1000 / 200
LowPass = LowPassFilter()
data = LowPass.low_pass_filter(
    dataset, col=col, sampling_frequency=fs, cutoff_frequency=cutoff, order=10
)
indexes = argrelextrema(data[col + "_lowpass"].values, np.greater)
peaks = data.iloc[indexes]

len(peaks)
