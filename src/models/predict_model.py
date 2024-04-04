import pandas as pd
import numpy as np
from glob import glob
import math
import scipy
import sys

#sys.path.append('../..')

from src.features.DataTransformation import LowPassFilter , PrincipalComponentAnalysis
from src.features.TemporalAbstraction import NumericalAbstraction
from src.features.FrequencyAbstraction import FourierTransformation
from sklearn.cluster import KMeans
from sklearn.neighbors import LocalOutlierFactor  # pip install scikit-learn


from sklearn.model_selection import train_test_split

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.model_selection import cross_val_score , RandomizedSearchCV, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from scipy.signal import argrelextrema


import pickle

import warnings
warnings.filterwarnings("ignore")


class Tracker:
    
    def __init__(self, acc_path,gyr_path):
        self.acc_path = acc_path
        self.gyr_path = gyr_path
        self.predicted_exersice = None

    def read_data_from_files(self):
    
        
        acc_df = pd.read_csv(self.acc_path)
        gyr_df = pd.read_csv(self.gyr_path)
        
        
        
        acc_df.index = pd.to_datetime(acc_df["epoch (ms)"], unit="ms")
        gyr_df.index = pd.to_datetime(gyr_df["epoch (ms)"], unit="ms")
        
        del acc_df["epoch (ms)"]
        del acc_df["time (01:00)"]
        del acc_df["elapsed (s)"]

        del gyr_df["epoch (ms)"]
        del gyr_df["time (01:00)"]
        del gyr_df["elapsed (s)"]

        acc_set = 1
        gyr_set = 1


        acc_label = self.acc_path.split("-")[1]
        gyr_label = self.gyr_path.split("-")[1]


        acc_df["label"] = acc_label
        gyr_df["label"] = gyr_label
        
        if "Accelerometer" in self.acc_path:
            acc_df["set"] = acc_set
            acc_set+= 1
            
        if "Gyroscope" in self.gyr_path:
            gyr_df["set"] = gyr_set
            gyr_set+= 1
        

        return acc_df, gyr_df

    def merge_and_clean_data(self):
        
        acc_df, gyr_df = self.read_data_from_files()
        
        data_merged = pd.concat([acc_df.iloc[:, :3], gyr_df], axis=1)

        data_merged.columns = [
            "acc_x",
            "acc_y",
            "acc_z",
            "gyr_x",
            "gyr_y",
            "gyr_z",
            "label",
            "set",
            
        ]
        
        sampling = {
            'acc_x':"mean",
            'acc_y':"mean",
            'acc_z':"mean",
            'gyr_x':"mean",
            'gyr_y':"mean",
            'gyr_z':"mean",
            'label':"last",
            'set':"last",
            
        }

        # Resampling the first 100 data points to a 200ms frequency and applying the defined sampling methods.
        # This is done to illustrate the resampling process.
        days = [g for n,g in data_merged.groupby(pd.Grouper(freq="D"))]
        data_resampled = pd.concat([df.resample(rule="200ms").apply(sampling).dropna() for df in days])
        #del data_resampled["set"]
       
        return data_resampled
    
    
    

    def remove_outliers(self):
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

        df = self.merge_and_clean_data()
        oultier_cols = list(df.columns[:6])
        
        chauvenet_oultier_removed_df = df.copy()

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
            
        return chauvenet_oultier_removed_df
    
    
    def low_pass(self):
        df = self.remove_outliers()
        predictor_columns = list(df.columns[:6])
        # We try removing all NaN values and filling gaps with the default method, which is linear interpolation. 
        # For each predictor column, we perform interpolation to fill the missing values. 
        # By default, interpolation computes the mean between the next and last available values in the same row.
        for i in predictor_columns:
            df[i] = df[i].interpolate()
            
        lowpass_df = df.copy()
        LowPass = LowPassFilter()
        
        freq_sample  = 1000 / 200 
        cutoff = 1.3 
        # Apply the lowpass filter to all predictor columns
        for col in predictor_columns:
            lowpass_df = LowPass.low_pass_filter(lowpass_df, col, freq_sample , cutoff, order=5)
            lowpass_df[col] = lowpass_df[col + "_lowpass"]
            del lowpass_df[col + "_lowpass"]
        
        return lowpass_df
    
    def pca(self):
        pca_df = self.low_pass().copy()
        predictor_columns = list(pca_df.columns[:6])
        pca = PrincipalComponentAnalysis()
        pca_df = pca.apply_pca(pca_df, predictor_columns, 3)
        return pca_df
    
    def sum_square(self):
        squares_df = self.pca().copy()
        # Calculate the squared magnitude for accelerometer and gyroscope readings
        acc_r = squares_df['acc_x']**2 + squares_df['acc_y']**2 + squares_df['acc_z']**2 
        gyr_r = squares_df['gyr_x']**2 + squares_df['gyr_y']**2 + squares_df['gyr_z']**2 

        # Calculate the magnitude by taking the square root of the sum of squared components
        squares_df['acc_r'] = np.sqrt(acc_r)
        squares_df['gyr_r'] = np.sqrt(gyr_r)
        return squares_df
    
    def temporal_abstraction(self):
        temporal_df = self.sum_square().copy()
        predictor_columns = list(temporal_df.columns[:6])
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
        
        return temporal_df
    
    def frequency(self):
        frequency_df = self.temporal_abstraction().copy().reset_index()
        predictor_columns = list(frequency_df.columns[:6])
        predictor_columns = predictor_columns + ['acc_r', 'gyr_r']
        # Instantiate the FourierTransformation class
        FreqAbs = FourierTransformation()

        # Define the sampling frequency (fs) and window size (ws)
        fs = int(1000 / 200)  # Sampling frequency (samples per second)
        ws = int(2800 / 200)   # Window size (number of samples)

        # Initialize a list to store Fourier-transformed subsets
        frequency_df_list = []

        # Iterate over unique sets in the DataFrame
        for s in frequency_df["set"].unique():
            # Select subset corresponding to the current set
            subset = frequency_df[frequency_df["set"] == s].reset_index(drop=True).copy()
            # Apply Fourier transformations to predictor columns
            subset = FreqAbs.abstract_frequency(subset, predictor_columns, ws, fs)
            # Append the transformed subset to the list
            frequency_df_list.append(subset)

        # Concatenate all transformed subsets into a single DataFrame and set index
        frequency_df = pd.concat(frequency_df_list).set_index("epoch (ms)", drop=True)

        # --------------------------------------------------------------
        # Dealing with overlapping windows
        # --------------------------------------------------------------
        # Remove rows with any NaN values from the DataFrame
        frequency_df = frequency_df.dropna()

        # Select every other row in the DataFrame
        frequency_df = frequency_df.iloc[::2]
        
        return frequency_df
    
    def clusters(self):
        cluster_df = self.frequency().copy()
        # Perform KMeans clustering with optimal number of clusters (e.g., 5)
        cluster_columns = ['acc_x', 'acc_y', 'acc_z']
        kmeans = KMeans(n_clusters=5, n_init=20, random_state=0)
        subset = cluster_df[cluster_columns]
        # Predict cluster labels and assign them to the DataFrame
        cluster_df["cluster_label"] = kmeans.fit_predict(subset)
                    
        return cluster_df

    def model(self):
        selected_features = ['acc_y_temp_mean_ws_5',
                            'acc_y_freq_0.0_Hz_ws_14',
                            'cluster_label',
                            'acc_z_freq_0.0_Hz_ws_14',
                            'pca_1',
                            'gyr_r_freq_0.0_Hz_ws_14',
                            'acc_y',
                            'acc_x_freq_0.0_Hz_ws_14',
                            'acc_z_temp_mean_ws_5',
                            'acc_z',
                            'acc_x_temp_mean_ws_5',
                            'gyr_z_temp_std_ws_5',
                            'pca_2',
                            'acc_z_pse',
                            'gyr_r_temp_mean_ws_5']

        input_data = self.clusters()[selected_features]
        
        with open("C:/Users/HP/Fitness-tracker-based-on-ML-2/models/04_Model.pkl", "rb") as f:
            model = pickle.load(f)
        
        model_output = model.predict(input_data)
        self.predicted_exersice = model_output[0]
        return model_output[0]
        
    def count_rep(self):
        
        exercise = self.predicted_exersice
        df = self.merge_and_clean_data()

        
        acc_r = df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2 
        gyr_r = df['gyr_x']**2 + df['gyr_y']**2 + df['gyr_z']**2 

        df['acc_r'] = np.sqrt(acc_r)
        df['gyr_r'] = np.sqrt(gyr_r)
        
        fs = 1000 / 200
        LowPass= LowPassFilter()


        col = "acc_r"

        def count_reps(dataset ,cutoff = 0.4 ,order = 10 ,column = "acc_r"):
            data = LowPass.low_pass_filter(data_table = dataset,col = column,
                                        sampling_frequency = fs,
                                        cutoff_frequency = cutoff,order = order)


            index = argrelextrema(data[col + "_lowpass"].values, np.greater)
            peaks = data.iloc[index]
            return len(peaks)
                
        column = "acc_r"
        cutoff = 0.4
        
        if exercise == "squat":
            cutoff = 0.34
        
        elif exercise == "row":
            column = "gyr_X"
            cutoff = 0.65
        
        elif exercise == "ohp":
            cutoff = 0.35
            
        reps = count_reps(df, cutoff=cutoff, column=column)
                
        return reps
        
        
        

#tracker = Tracker("../../data/raw/MetaMotion/MetaMotion/E-squat-heavy_MetaWear_2019-01-15T20.14.03.633_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv","../../data/raw/MetaMotion/MetaMotion/E-squat-heavy_MetaWear_2019-01-15T20.14.03.633_C42732BE255C_Gyroscope_25.000Hz_1.4.4.csv")
#model_output = tracker.model()
#counts = tracker.count_rep()