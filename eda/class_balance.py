import os 
import matplotlib.pyplot as plt 
import seaborn as sns 
import numpy as np 
import pandas as pd 

if __name__ == "__main__":
    
    dset = "dev"
    country_label = "#3_country_label"
    title = "[Dev Data] Samples Per Region"
    path = f"../NADI2021_DEV.1.0/NADI2021_DEV.1.0/Subtask_1.2+2.2_DA/DA_{dset}_labeled.tsv"
    df = pd.read_csv(path, sep='\t', header=0)

    r_path = "../dataset_dummy/region_classes.txt"
    
    with open(r_path, 'r') as fin: lines = fin.readlines()
    region_map = {line.split(',')[0]: line.split(',')[1].strip() for line in lines}

    df["region_label"] = df.apply(lambda row: region_map[row[country_label]], axis=1)

    # count = []
    # countries = df[country_label].unique().tolist()
    # for country in countries:
    #     count += [len(df[df[country_label]==country])]

    count = []
    regions = df["region_label"].unique().tolist()
    for region in regions:
        count += [len(df[df["region_label"]==region])]

    freq, labels = list(zip(*sorted(zip(count, regions), reverse=True)))
    class_weights = (len(df) / (np.array(freq) * len(labels)))

    plot = sns.barplot(x=list(labels), y=list(freq))
    for item in plot.get_xticklabels():
        item.set_rotation(45)
    plt.title(title)
    plt.show()

    # with open('../dataset_dummy/classes_w_weights.txt', 'w') as fout:
    #         for country, weight in zip(labels, class_weights):
    #             fout.write(f"{country}\t{weight}\n")