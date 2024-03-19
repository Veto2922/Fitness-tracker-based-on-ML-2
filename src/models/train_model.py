import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib as mlp
from LearningAlgorithms import ClassificationAlgorithms
import seaborn as sns
import itertools
from sklearn.metrics import accuracy_score, confusion_matrix

# --------------------------------------------------------------
# Adjust plot settings
# --------------------------------------------------------------
# Set the plotting style to 'fivethirtyeight' for Matplotlib plots
mlp.style.use("fivethirtyeight")
# Set the figure size to (20, 5) inches
mlp.rcParams["figure.figsize"] = (20,5)
# Set the resolution of the figure to 100 dots per inch (dpi) for exporting
mlp.rcParams["figure.dpi"] = 100

mlp.rcParams["lines.linewidth"] = 2


# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

df = pd.read_pickle("../../data/interim//04_data_features.pkl")

# --------------------------------------------------------------
# Create a training and test set
# --------------------------------------------------------------

# Remove non-feature columns from the DataFrame to create the training set
# "participant", "category", "set", and "duration" 
# are dropped as they are not considered features
df_train = df.drop(["participant","category","set","duration"],axis = 1)

# Separate features (X) and labels (y) from the training set
X = df_train.drop("label",axis = 1) # Features
y = df_train["label"] # Target

# Split the data into training and test sets using stratified sampling to maintain class proportions
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=42,stratify=y)

# Plot the distribution of labels in the total dataset, training set, and test set
fig, ax = plt.subplots(figsize = (10,5))
df_train["label"].value_counts().plot(kind = "bar",ax = ax ,label = "Total",color="lightblue")
y_train.value_counts().plot(kind = "bar",ax = ax ,label = "Train",color = "royalblue")
y_test.value_counts().plot(kind = "bar",ax = ax ,label = "Test",color = "blue")
plt.legend()
plt.show()
# --------------------------------------------------------------
# Split feature subsets
# --------------------------------------------------------------
# Define different subsets of features based on their characteristics
# These subsets depend on the feature engineering performed earlier
basic_features = ["acc_x","acc_y","acc_z","gyr_x","gyr_y","gyr_z"]
sqaured_features = ["acc_r","gyr_r"]
pca_features = ["pca_1","pca_2","pca_3"]
time_features = [f for f in df_train.columns if "_temp" in f]
freq_features = [f for f in df_train.columns if ("_freq" in f) or ("_pse" in f)]
cluster_features = ["cluster_label"]

# Basic sensor features (accelerometer and gyroscope)
basic_features = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]

# Squared sensor features (derived from basic features)
squared_features = ["acc_r", "gyr_r"]

# PCA-transformed features
pca_features = ["pca_1", "pca_2", "pca_3"]

# Time-domain features
time_features = [f for f in df_train.columns if "_temp" in f]

# Frequency-domain features
freq_features = [f for f in df_train.columns if ("_freq" in f) or ("_pse" in f)]

# Cluster-based features
cluster_features = ["cluster_label"]

# Print the number of features in each subset for reference
print("basic_features: ", len(basic_features))
print("squared_features: ", len(squared_features))
print("pca_features: ", len(pca_features))
print("time_features: ", len(time_features))
print("freq_features: ", len(freq_features))
print("cluster_features: ", len(cluster_features))

# Define different combinations of feature subsets
# These combinations are based on the feature engineering results
feature_1 = list(set(basic_features))  # Features from basic sensor readings only
feature_2 = list(set(basic_features + squared_features + pca_features))  # Additional derived and transformed features
feature_3 = list(set(feature_2 + time_features))  # Includes time-domain features
feature_4 = list(set(feature_3 + freq_features + cluster_features))  # Includes frequency-domain and cluster-based features
# --------------------------------------------------------------
# Perform forward feature selection using simple decision tree
# --------------------------------------------------------------
# Initialize the classification algorithms class
learner = ClassificationAlgorithms()

# Set the maximum number of features to select
max_features = 10

# Perform forward feature selection and retrieve selected features, ordered features, and their scores
selected_features, ordered_features, ordered_scores = learner.forward_selection(max_features,
                                                              X_train,
                                                              y_train)
# Selected features results:
selected_features = ['pca_1',
                    'acc_x_freq_0.0_Hz_ws_14',
                    'acc_z_freq_0.0_Hz_ws_14',
                    'gyr_r_freq_0.0_Hz_ws_14',
                    'acc_r_freq_0.357_Hz_ws_14',
                    'acc_r_temp_mean_ws_5',
                    'acc_x_freq_1.429_Hz_ws_14',
                    'acc_z_freq_2.5_Hz_ws_14',
                    'gyr_r',
                    'acc_z_freq_1.429_Hz_ws_14']

# Plot the accuracy against the number of features selected
plt.figure(figsize=(10,5))
plt.plot(np.arange(1, max_features+ 1,1), ordered_scores, marker = "o")
plt.xlabel("Number of features")
plt.ylabel("Accuracy")
plt.title("Forward feature selection")
plt.show()

# --------------------------------------------------------------
# Grid search for best hyperparameters and model selection
# --------------------------------------------------------------
# Define possible feature sets and corresponding names
possible_feature_sets = [feature_1,feature_2,feature_3,feature_4,selected_features]
feature_names = ["feature_1","feature_2","feature_3","feature_4","selected_features"]

# Define the number of iterations for averaging non-deterministic classifiers' scores
iterations = 1

# Initialize an empty dataframe to store model performance
score_df = pd.DataFrame()


for i, f in zip(range(len(possible_feature_sets)), feature_names):
    print("Feature set:", i)
    selected_train_X = X_train[possible_feature_sets[i]]
    selected_test_X = X_test[possible_feature_sets[i]]

    # First run non deterministic classifiers to average their score.
    performance_test_nn = 0
    performance_test_rf = 0

    for it in range(0, iterations):
        print("\tTraining neural network,", it)
        (
            class_train_y,
            class_test_y,
            class_train_prob_y,
            class_test_prob_y,
        ) = learner.feedforward_neural_network(
            selected_train_X,
            y_train,
            selected_test_X,
            gridsearch=False,
        )
        performance_test_nn += accuracy_score(y_test, class_test_y)

        print("\tTraining random forest,", it)
        (
            class_train_y,
            class_test_y,
            class_train_prob_y,
            class_test_prob_y,
        ) = learner.random_forest(
            selected_train_X, y_train, selected_test_X, gridsearch=True
        )
        performance_test_rf += accuracy_score(y_test, class_test_y)

    performance_test_nn = performance_test_nn / iterations
    performance_test_rf = performance_test_rf / iterations

    # And we run our deterministic classifiers:
    print("\tTraining KNN")
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.k_nearest_neighbor(
        selected_train_X, y_train, selected_test_X, gridsearch=True
    )
    performance_test_knn = accuracy_score(y_test, class_test_y)

    print("\tTraining decision tree")
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.decision_tree(
        selected_train_X, y_train, selected_test_X, gridsearch=True
    )
    performance_test_dt = accuracy_score(y_test, class_test_y)

    print("\tTraining naive bayes")
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.naive_bayes(selected_train_X, y_train, selected_test_X)

    performance_test_nb = accuracy_score(y_test, class_test_y)

    # Save results to dataframe
    models = ["NN", "RF", "KNN", "DT", "NB"]
    new_scores = pd.DataFrame(
        {
            "model": models,
            "feature_set": f,
            "accuracy": [
                performance_test_nn,
                performance_test_rf,
                performance_test_knn,
                performance_test_dt,
                performance_test_nb,
            ],
        }
    )
    score_df = pd.concat([score_df, new_scores])

# --------------------------------------------------------------
# Create a grouped bar plot to compare the results
# --------------------------------------------------------------
# Sort the score dataframe by accuracy in descending order
score_df.sort_values(by="accuracy", ascending=False)

# Plotting the grouped bar plot
plt.figure(figsize=(10,10))
# Using seaborn to create a bar plot with grouped bars based on model and feature set
sns.barplot(data=score_df, x="model", y="accuracy", hue="feature_set")
# Set the y-axis limits to ensure clarity of comparison
plt.ylim(0.7, 1)
plt.title("Model performance")
plt.legend(loc="lower left")
plt.show()

# Best model: RF (Random Forest) with feature set 'feature_4' and accuracy 0.994829.


# --------------------------------------------------------------
# Select best model and evaluate results
# --------------------------------------------------------------
# Train the Random Forest model using the best feature set selected previously
# and perform grid search for hyperparameter tuning

(
class_train_y,
class_test_y,
class_train_prob_y,
class_test_prob_y,
) = learner.random_forest(
X_train[feature_4], y_train, X_test[feature_4], gridsearch=True
)

# Calculate the accuracy of the model on the test set
accuracy = accuracy_score(y_test, class_test_y)

# Extract the unique classes from the predicted probabilities
classes = class_test_prob_y.columns

# Generate the confusion matrix for further evaluation
cm = confusion_matrix(y_test, class_test_y, labels=classes)

# create confusion matrix for cm
plt.figure(figsize=(10, 10))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion matrix")
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

thresh = cm.max() / 2.0
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(
        j,
        i,
        format(cm[i, j]),
        horizontalalignment="center",
        color="white" if cm[i, j] > thresh else "black",
    )
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.grid(False)
plt.show()
# --------------------------------------------------------------
# Select train and test data based on participant
# --------------------------------------------------------------
# When evaluating the performance of a machine learning model, it's crucial to ensure that the model's effectiveness
# extends beyond the specific individuals or entities used during training. In this section, we split the dataset based 
# on participants, with the aim of testing the model's generalization capabilities on unseen participants. By training 
# the model on data from all participants except A participant and testing it on the A participant, we assess its ability to perform 
# accurately on individuals not encountered during the training phase.


# Remove unnecessary columns from the dataframe for participant-based split
df_participant = df.drop(["category","set","duration"],axis = 1)

# Separate training data from participants other than "A"
X_train = df_participant[df_participant["participant"] != "A"].drop("label",axis = 1)
y_train = df_participant[df_participant["participant"] != "A"]["label"]

# Separate test data for participant "A"
X_test = df_participant[df_participant["participant"] == "A"].drop("label",axis = 1)
y_test = df_participant[df_participant["participant"] == "A"]["label"]

# Remove the participant column from both training and test sets
X_train = X_train.drop("participant",axis = 1)
X_test = X_test.drop("participant",axis = 1)

# Plot the distribution of labels in the dataset and the split between train and test sets
fig, ax = plt.subplots(figsize = (10,5))
df_participant["label"].value_counts().plot(kind = "bar",ax = ax ,label = "Total",color="lightblue")
y_train.value_counts().plot(kind = "bar",ax = ax ,label = "Train",color = "royalblue")
y_test.value_counts().plot(kind = "bar",ax = ax ,label = "Test",color = "blue")
plt.legend()
plt.show()


# --------------------------------------------------------------
# Use best model again and evaluate results
# --------------------------------------------------------------
# After selecting the best-performing model, we apply it again to the test data to evaluate its performance further.
# Here, we utilize the Random Forest model, which demonstrated superior accuracy during our model selection process.
# We assess its effectiveness in classifying the test data by computing various evaluation metrics, including accuracy
# and confusion matrix.

(
class_train_y,
class_test_y,
class_train_prob_y,
class_test_prob_y,
) = learner.random_forest(
X_train[feature_4], y_train, X_test[feature_4], gridsearch=True
)

accuracy = accuracy_score(y_test, class_test_y)

classes = class_test_prob_y.columns

cm = confusion_matrix(y_test, class_test_y, labels=classes)

# create confusion matrix for cm
plt.figure(figsize=(10, 10))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion matrix")
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

thresh = cm.max() / 2.0
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(
        j,
        i,
        format(cm[i, j]),
        horizontalalignment="center",
        color="white" if cm[i, j] > thresh else "black",
    )
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.grid(False)
plt.show()

# --------------------------------------------------------------
# Try a complex model with the selected features
# --------------------------------------------------------------
# In this section, we experiment with a more complex model, specifically a feedforward neural network,
# using the features selected during the feature selection process. The aim is to investigate whether
# a more sophisticated model can further improve classification performance compared to the Random Forest model.

# Apply a feedforward neural network to the test data using the selected features
(
class_train_y,
class_test_y,
class_train_prob_y,
class_test_prob_y,
) = learner.feedforward_neural_network(
X_train[selected_features], y_train, X_test[selected_features], gridsearch=True
)

accuracy = accuracy_score(y_test, class_test_y)

classes = class_test_prob_y.columns

cm = confusion_matrix(y_test, class_test_y, labels=classes)

# create confusion matrix for cm
plt.figure(figsize=(10, 10))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion matrix")
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

thresh = cm.max() / 2.0
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(
        j,
        i,
        format(cm[i, j]),
        horizontalalignment="center",
        color="white" if cm[i, j] > thresh else "black",
    )
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.grid(False)
plt.show()
