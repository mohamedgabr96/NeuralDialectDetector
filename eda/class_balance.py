import os 
import matplotlib.pyplot as plt 
import seaborn as sns 
import numpy as np 
import pandas as pd 


if __name__ == "__main__":
    
    dset = "train"
    title = "[Training Data] Samples Per Country"
    path = f"../NADI2021_DEV.1.0/NADI2021_DEV.1.0/Subtask_1.2+2.2_DA/DA_{dset}_labeled.tsv"
    df = pd.read_csv(path, sep='\t', header=0)

    count = []
    countries = df["#3_country_label"].unique().tolist()
    for country in countries:
        count += [len(df[df["#3_country_label"]==country])]

    freq, labels = list(zip(*sorted(zip(count, countries), reverse=True)))
    class_weights = (len(df) / (np.array(freq) * len(labels)))

    plot = sns.barplot(x=list(labels), y=list(freq))
    for item in plot.get_xticklabels():
        item.set_rotation(45)
    plt.title(title)
    plt.show()

    # with open('../dataset_dummy/classes_w_weights.txt', 'w') as fout:
    #         for country, weight in zip(labels, class_weights):
    #             fout.write(f"{country}\t{weight}\n")