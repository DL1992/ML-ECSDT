import load_datasets
from collections import Counter
import pandas as pd
import numpy as np
def return_model():
    datasets={}
    data_dir = "DataSets/"
    cancer = pd.read_csv(data_dir + 'breast_cancer.csv',
                         names=["id", "Clump Thickness", "Uniformity of Cell Size", "Uniformity of Cell Shape",
                                "Marginal Adhesion", "Single Epithelial Cell Size", "Bare Nuclei", "Bland Chromatin",
                                "Normal Nucleoli", "Mitoses", "Class"])
    cancer.loc[cancer['Class'] == 2] = 0
    cancer.loc[cancer['Class'] == 4] = 1
    for col in cancer.columns:
        if cancer[col].dtype == object:
            cancer[col] = cancer[col].astype('category')
            cancer[col] = cancer[col].cat.codes
    data = cancer[["Clump Thickness", "Uniformity of Cell Size", "Uniformity of Cell Shape", "Marginal Adhesion",
                "Single Epithelial Cell Size", "Bare Nuclei", "Bland Chromatin", "Normal Nucleoli", "Mitoses"]]
    target = cancer[["Class"]]
    test = target['Class']
    counts=dict(sorted(Counter(test).items(),key=lambda x: x[1]))
    cost_mat=[]
    sumy = sum(counts.values())
    for x in test:
        relation=100000*(float(counts[x])/(sumy))
        cost_mat.append([relation,relation,0,0])
    cost_mat=np.array(cost_mat)
    datasets['cancer'] = dict([("data",data), ("target",target),("cost_mat",cost_mat)])
    return datasets



