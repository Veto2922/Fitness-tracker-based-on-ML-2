import pandas as pd
import matplotlib.pyplot as plt

df =pd.read_pickle("../../data/interim/01_data_processed.pkl")

#plot single column

set_df=df[df["set"]==1]
plt.plot(set_df["acc_y"])

plt.plot(set_df["acc_y"].reset_index(drop=True))


#plot all excersices
for label in df['label'].unique():
    subset =df[df["label"]==label]
    fig,ax=plt.subplots()
    plt.plot(subset["acc_y"].reset_index(drop=True),label=label)
    plt.legend()
    plt.show()
    
for label in df['label'].unique():
    subset =df[df["label"]==label]
    fig,ax=plt.subplots()
    plt.plot(subset["acc_y"][:100].reset_index(drop=True),label=label)
    plt.legend()
    plt.show()