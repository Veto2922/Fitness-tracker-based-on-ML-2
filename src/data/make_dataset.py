import pandas as pd
from glob import glob

# --------------------------------------------------------------
# Read single CSV file
# --------------------------------------------------------------
single_file_acc = pd.read_csv("../../data/raw/MetaMotion/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv")
single_file_gyr = pd.read_csv("../../data/raw/MetaMotion/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Gyroscope_25.000Hz_1.4.4.csv")

# --------------------------------------------------------------
# List all data in data/raw/MetaMotion
# --------------------------------------------------------------

files = glob("../../data/raw/MetaMotion/MetaMotion/*.csv")
len(files)

# --------------------------------------------------------------
# Extract features from filename
# --------------------------------------------------------------
data_path = "../../data/raw/MetaMotion/MetaMotion/*.csv"
f = files[0]

participate = f.split("-")[0][-1]
label = f.split("-")[1]
category = f.split("-")[2].rstrip('1234567890').rstrip('_MetaWear_')

df = pd.read_csv(f)

df["participate"] = participate
df["label"] = label
df["category"] = category

# --------------------------------------------------------------
# Read all files
# --------------------------------------------------------------

acc_df = pd.DataFrame()
gyr_df = pd.DataFrame()

acc_set = 1
gyr_set = 1

for f in files:
    participate = f.split("-")[0][-1]
    label = f.split("-")[1]
    category = f.split("-")[2].rstrip('123').rstrip('_MetaWear_2019')

    df = pd.read_csv(f)

    df["participate"] = participate
    df["label"] = label
    df["category"] = category
    
    if "Accelerometer" in f:
        df["acc_set"] = acc_set
        acc_set+= 1
        acc_df = pd.concat([acc_df,df])
    else:
        gyr_df = pd.concat([gyr_df,df])

##gyr_df["category"].unique()
    
# --------------------------------------------------------------
# Working with datetimes
# --------------------------------------------------------------

acc_df.index = pd.to_datetime(acc_df["epoch (ms)"],unit="ms")
gyr_df.index = pd.to_datetime(gyr_df["epoch (ms)"],unit="ms")


del acc_df["epoch (ms)"]
del acc_df["time (01:00)"]
del acc_df["elapsed (s)"]

del gyr_df["epoch (ms)"]
del gyr_df["time (01:00)"]
del gyr_df["elapsed (s)"]


# --------------------------------------------------------------
# Turn into function
# --------------------------------------------------------------

# Fetching file paths using glob module.
files = glob("../../data/raw/MetaMotion/MetaMotion/*.csv")

def read_data_from_files(files):
    """
    Reads data from provided files and organizes it into DataFrames for accelerometer and gyroscope data.

    Args:
        files (list): A list of file paths containing data.

    Returns:
        tuple: Two pandas DataFrames. The first DataFrame contains accelerometer data, and the second DataFrame contains gyroscope data.
    """
    acc_df = pd.DataFrame()
    gyr_df = pd.DataFrame()

    acc_set = 1
    gyr_set = 1

    for f in files:
        # Extracting participant, label, and category information from file names
        participate = f.split("-")[0][-1]
        label = f.split("-")[1]
        category = f.split("-")[2].rstrip('123').rstrip('_MetaWear_2019')

        # Reading data from CSV file
        df = pd.read_csv(f)

        # Adding participant, label, and category columns to the DataFrame
        df["participate"] = participate
        df["label"] = label
        df["category"] = category
        
        # Organizing data based on whether it is accelerometer or gyroscope data
        if "Accelerometer" in f:
            df["set"] = acc_set
            acc_set+= 1
            acc_df = pd.concat([acc_df,df])
        else:
            df["set"] = gyr_set
            gyr_set+= 1
            gyr_df = pd.concat([gyr_df,df])
        
    # Setting datetime index for both DataFrames
    acc_df.index = pd.to_datetime(acc_df["epoch (ms)"],unit="ms")
    gyr_df.index = pd.to_datetime(gyr_df["epoch (ms)"],unit="ms")

    # Dropping unnecessary columns which are timing as we already set the index "epoch (ms)"
    del acc_df["epoch (ms)"]
    del acc_df["time (01:00)"]
    del acc_df["elapsed (s)"]

    del gyr_df["epoch (ms)"]
    del gyr_df["time (01:00)"]
    del gyr_df["elapsed (s)"]

    return acc_df,gyr_df

acc_df,gyr_df = read_data_from_files(files)   


# --------------------------------------------------------------
# Merging datasets
# --------------------------------------------------------------

# Combining accelerometer and gyroscope data into a single DataFrame.
# We only select the first three columns (acc_x, acc_y, acc_z) from the accelerometer DataFrame (acc_df)
# as the participant, label, and category information are repeated for each of the two DataFrames.
data_merged = pd.concat([acc_df.iloc[:,:3],gyr_df],axis = 1)
data_merged.dropna()


# renaming columns 
data_merged.columns = [
    "acc_x",
    "acc_y",
    "acc_z",
    "gyr_x",
    "gyr_y",
    "gyr_z",
    "participant",
    "label",
    "category",
    "set",
]

# --------------------------------------------------------------
# Resample data (frequency conversion)
# --------------------------------------------------------------

# Accelerometer:    12.500HZ
# Gyroscope:        25.000Hz

# Defining sampling methods for resampling accelerometer and gyroscope data
sampling = {
    'acc_x':"mean",
    'acc_y':"mean",
    'acc_z':"mean",
    'gyr_x':"mean",
    'gyr_y':"mean",
    'gyr_z':"mean",
    'participant':"last",
    'label':"last",
    'category':"last",
    'set':"last",
    
}

# Resampling the first 100 data points to a 200ms frequency and applying the defined sampling methods.
# This is done to illustrate the resampling process.
data_merged[:100].resample(rule="200ms").apply(sampling)

# Grouping the data by days and resampling each group to a 200ms frequency using the defined sampling methods.
# This process ensures that data from each day is resampled appropriately.
days = [g for n,g in data_merged.groupby(pd.Grouper(freq="D"))]
data_resampled = pd.concat([df.resample(rule="200ms").apply(sampling).dropna() for df in days])

# Converting the 'set' column to integer type and displaying information about the resampled data.
data_resampled["set"] = data_resampled["set"].astype("int")
data_resampled.info()


# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------

# Saving the resampled data to a pickle file for further processing or analysis.
data_resampled.to_pickle("../../data/interim/01_data_processed.pkl")