import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from IPython.display import display
import matplotlib as mpl

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------
#Read the data from a pickle file into a Pandas DataFrame
df = pd.read_pickle("../../data/interim/01_data_processed.pkl")

# --------------------------------------------------------------
# Plot single columns
# --------------------------------------------------------------
# Plot the set column 
# Select rows from the DataFrame where the 'set' column equals 1
set_df = df[df["set"] == 1]

## Approach 1: Plotting the raw 'acc_y' values to visualize the duration of a set
plt.plot(set_df["acc_y"])

# Approach 2: Plotting the 'acc_y' values with index reset to visualize the number of samples in the set
plt.plot(set_df["acc_y"].reset_index(drop = True))

# --------------------------------------------------------------
# Plot all exercises
# --------------------------------------------------------------

# Plot all exercises in one plot
for ex in df["label"].unique():
    exercise_set = df[df["label"] == ex]
    # Plot the first 100 values of 'acc_x' for each exercise, with labels
    plt.plot(exercise_set[:100]["acc_x"].reset_index(drop = True),label=ex)
    plt.legend()


# Plot each exercise in separate plots
for ex in df["label"].unique():
    subset = df[df["label"] == ex]
    # Create a new subplot for each exercise
    fig, ax = plt.subplots()
    # Plot the first 100 values of 'acc_x' for each exercise, with labels
    plt.plot(subset[:100]["acc_x"].reset_index(drop = True),label = ex)
    plt.legend()
    plt.show()
    
# --------------------------------------------------------------
# Adjust plot settings
# --------------------------------------------------------------
# Set the figure size to (20, 5) inches
mpl.rcParams["figure.figsize"] = (20,5)

# Set the resolution of the figure to 100 dots per inch (dpi) for exporting
mpl.rcParams["figure.dpi"] = 100

# --------------------------------------------------------------
# Compare medium vs. heavy sets
# --------------------------------------------------------------

# Selecting data for squats performed by participant D
category_df = df.query("label == 'squat'").query("participant == 'D'").reset_index()
# Grouping the data by category and plotting the 'acc_y' values
category_df.groupby(["category"])["acc_y"].plot()

plt.legend(title='category')  # Add a legend with the title 'category'
# Adding labels for the x and y axes
plt.xlabel("acc_y")  
plt.ylabel("samples")
plt.show()  # Displaying the plot


# --------------------------------------------------------------
# Analyzing the distribution of categories for each exercise label
# --------------------------------------------------------------

# Grouping the DataFrame by 'label' and 'category', then counting the occurrences of each category
# The result is reshaped into a more readable format using unstack()
dfg = df.groupby('label')[['category']].value_counts().unstack()
# Plotting the distribution of categories for each exercise label as a stacked bar chart
dfg.plot(kind='bar',stacked = True)

# Adding labels for the x and y axes
plt.xlabel('Participant')  
plt.ylabel('Count')  

# Adding a title to the plot
plt.title('Distribution of category by exersice')  # Customize the plot title
plt.legend(title='category')  # Add a legend with the title 'category'
plt.show()  # Displaying the plot


# --------------------------------------------------------------
# Compare participants
# --------------------------------------------------------------
# Selecting rows from the DataFrame where the exercise label is 'bench'
participant_df = df[df["label"] == 'bench'].sort_values(by="participant").reset_index()

# Grouping the data by participant and plotting the 'acc_y' values for each participant
participant_df.groupby("participant")["acc_y"].plot()

# --------------------------------------------------------------
# Plot multiple axis
# --------------------------------------------------------------
label = "squat"
participant = 'A'
all_axis_df = df.query(f"label == '{label}'").query(f"participant == '{participant}'").reset_index()

fig , ax = plt.subplots()  # Create a new figure and axis
# Plotting the 'acc_x', 'acc_y', and 'acc_z' values against the sample index
all_axis_df[["acc_x","acc_y","acc_z"]].plot(ax=ax)
# Adding labels for the y-axis and x-axis
ax.set_ylabel("acc_y")
ax.set_xlabel("samples")
ax.legend()  # Adding a legend to the plot


# --------------------------------------------------------------
# Create a loop to plot all combinations per sensor
# --------------------------------------------------------------
for lab in df["label"].unique():
    for par in df["participant"].unique():
        all_axis_df = df.query(f"label == '{lab}'").query(f"participant == '{par}'").reset_index()
        
        if len(all_axis_df) > 0:
            fig , ax = plt.subplots()
            all_axis_df[["acc_x","acc_y","acc_z"]].plot(ax=ax)
            ax.set_ylabel("acc_y")
            ax.set_xlabel("samples")
            ax.set_title(f"{lab}: ({par})")
            plt.legend()
            
            
for lab in df["label"].unique():
    for par in df["participant"].unique():
        all_axis_df = df.query(f"label == '{lab}'").query(f"participant == '{par}'").reset_index()
        
        if len(all_axis_df) > 0 :
            fig , ax = plt.subplots()
            all_axis_df[["gyr_x","gyr_y","gyr_z"]].plot(ax=ax)
            ax.set_ylabel("gyr_y")
            ax.set_xlabel("samples")
            plt.title(f"{lab}: ({par})".title())
            plt.legend()

# --------------------------------------------------------------
# Combine plots in one figure
# --------------------------------------------------------------
label = "row"
participant = 'D'
combined_df = df.query(f"label == '{label}'").query(f"participant == '{participant}'").reset_index(drop=True)

# Create a new figure with two subplots arranged in a vertical layout
fig , ax = plt.subplots(nrows=2, figsize= (20,10),sharex=True)

# Plot accelerometer data ('acc_x', 'acc_y', 'acc_z') on the first subplot (top)
combined_df[["acc_x","acc_y","acc_z"]].plot(ax=ax[0])
# Plot gyroscope data ('gyr_x', 'gyr_y', 'gyr_z') on the second subplot (bottom)
combined_df[["gyr_x","gyr_y","gyr_z"]].plot(ax=ax[1])
       
# Add legends to each subplot
ax[0].legend(loc="upper center",ncol=3,fancybox=True,shadow=True,bbox_to_anchor=(0.5,1.15))
ax[1].legend(loc="upper center",ncol=3,fancybox=True,shadow=True,bbox_to_anchor=(0.5,1.15))

ax[0].set_xlabel("samples")  # Add xlabel shared by both subplots

# --------------------------------------------------------------
# Loop over all combinations and export for both sensors
# --------------------------------------------------------------
# Get unique exercise labels and participant IDs
label = df["label"].unique()
participant = df["participant"].unique()

# Iterate over each combination of exercise label and participant
for lab in label:
    for par in participant:
        # Select data for the current combination of exercise label and participant
        combined_df = df.query(f"label == '{lab}'").query(f"participant == '{par}'").reset_index(drop=True)
        # Check if there is data available for the current combination        
        if len(combined_df) > 0:
            fig , ax = plt.subplots(nrows=2, figsize= (20,10),sharex=True)
            combined_df[["acc_x","acc_y","acc_z"]].plot(ax=ax[0])
            combined_df[["gyr_x","gyr_y","gyr_z"]].plot(ax=ax[1])
            ax[0].legend(loc="upper center",ncol=3,fancybox=True,shadow=True,bbox_to_anchor=(0.5,1.15))
            ax[1].legend(loc="upper center",ncol=3,fancybox=True,shadow=True,bbox_to_anchor=(0.5,1.15))
            ax[1].set_xlabel("samples")
            # Save the figure as an image with a filename indicating the exercise label and participant            
            plt.savefig(f"../../reports/figures/{lab.title()} ({par}).png")
            plt.show()
            
            

            

        