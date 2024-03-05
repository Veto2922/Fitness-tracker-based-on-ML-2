import pandas as pd
from glob import glob

# --------------------------------------------------------------
# Read single CSV file
# --------------------------------------------------------------

single_file_acc = pd.read_csv(
    "../../data/raw/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv"
)

single_file_gyr = pd.read_csv(
    "../../data/raw/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Gyroscope_25.000Hz_1.4.4.csv"
)

# --------------------------------------------------------------
# List all data in data/raw/MetaMotion
# --------------------------------------------------------------

files = glob("../../data/raw/MetaMotion/*.csv")
len(files)

# --------------------------------------------------------------
# Extract features from filename
# --------------------------------------------------------------

data_path = "../../data/raw/MetaMotion\\"
f = files[0]

participant = f.split("-")[0].replace(data_path, "")
label = f.split("-")[1]
category = f.split("-")[2].rstrip("123").rstrip("_MetaWear_2019")

df = pd.read_csv(f)

df["participant"] = participant
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

    participant = f.split("-")[0].replace(data_path, "")
    label = f.split("-")[1]
    category = f.split("-")[2].rstrip("123").rstrip("_MetaWear_2019")

    df = pd.read_csv(f)

    df["participant"] = participant
    df["label"] = label
    df["category"] = category

    if "Accelerometer" in f:
        df["set"] = acc_set
        acc_set += 1
        acc_df = pd.concat([acc_df, df])

    elif "Gyroscope" in f:
        df["set"] = gyr_set
        gyr_set += 1
        gyr_df = pd.concat([gyr_df, df])

# --------------------------------------------------------------
# Working with datetimes
# --------------------------------------------------------------

# acc_df.info()

# pd.to_datetime(df["epoch (ms)"], unit="ms")

pd.to_datetime(acc_df["time (01:00)"], format="mixed")
pd.to_datetime(gyr_df["time (01:00)"], format="mixed")

acc_df.index = pd.to_datetime(acc_df["time (01:00)"], format="mixed")
gyr_df.index = pd.to_datetime(gyr_df["time (01:00)"], format="mixed")

acc_df.drop(columns=["epoch (ms)", "time (01:00)", "elapsed (s)"], inplace=True)
gyr_df.drop(columns=["epoch (ms)", "time (01:00)", "elapsed (s)"], inplace=True)

# --------------------------------------------------------------
# Turn into function
# --------------------------------------------------------------

files = glob("../../data/raw/MetaMotion/*.csv")


def read_data_from_files(files):
    data_path = "../../data/raw/MetaMotion\\"

    acc_df = pd.DataFrame()
    gyr_df = pd.DataFrame()

    acc_set = 1
    gyr_set = 1

    for file in files:
        split_items = file.split("-")

        participant = split_items[0].replace(data_path, "")
        label = split_items[1]
        # remove the number from our workout category
        category = split_items[2].rstrip("123").rstrip("_MetaWear_2019")

        curr_df = pd.read_csv(file)

        curr_df["participant"] = participant
        curr_df["label"] = label
        curr_df["category"] = category

        # continue to add new rows (series) to the dataframe
        # 'set' is just an arbitrary identifier
        if "Accelerometer" in file:
            curr_df["set"] = acc_set
            acc_set += 1
            acc_df = pd.concat([acc_df, curr_df])

        if "Gyroscope" in file:
            curr_df["set"] = gyr_set
            gyr_set += 1
            gyr_df = pd.concat([gyr_df, curr_df])

    # set datetime as index to convert to a time series database
    acc_df.index = pd.to_datetime(acc_df["time (01:00)"], format="mixed")
    gyr_df.index = pd.to_datetime(gyr_df["time (01:00)"], format="mixed")

    # now we can get rid of all features referencing time
    acc_df.drop(columns=["epoch (ms)", "time (01:00)", "elapsed (s)"], inplace=True)
    gyr_df.drop(columns=["epoch (ms)", "time (01:00)", "elapsed (s)"], inplace=True)

    # another way to delete the columns
    # del acc_df["epoch (ms)"]
    # del acc_df["time (01:00)"]
    # del acc_df["elapsed (s)"]

    # del gyr_df["epoch (ms)"]
    # del gyr_df["time (01:00)"]
    # del gyr_df["elapsed (s)"]

    return acc_df, gyr_df


acc_df, gyr_df = read_data_from_files(files)

# --------------------------------------------------------------
# Merging datasets
# --------------------------------------------------------------

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


sampling = {
    "acc_x": "mean",
    "acc_y": "mean",
    "acc_z": "mean",
    "gyr_x": "mean",
    "gyr_y": "mean",
    "gyr_z": "mean",
    "participant": "last",
    "label": "last",
    "category": "last",
    "set": "last",
}

# group by day
days = [g for n, g in data_merged.groupby(pd.Grouper(freq="D"))]
days_resampled = pd.concat(
    [df.resample(rule="200ms").apply(sampling).dropna() for df in days]
)

days_resampled["set"] = days_resampled["set"].astype("int")

days_resampled.info()

# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------

days_resampled.to_pickle("../../data/interim/01_data_processed.pkl")
