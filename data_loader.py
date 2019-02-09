import load_datasets
from collections import Counter
import pandas as pd
import numpy as np


def get_all_datasets():
    datasets={}
    data_dir = "DataSets/"
    cancer = pd.read_csv(data_dir + 'breast_cancer.csv',names=['att'+str(i) for i in range(1,12)])
    cancer.loc[cancer['att11'] == 2] = 0
    cancer.loc[cancer['att1'] == 4] = 1
    datasets['cancer'] = create_dataset(cancer.shape[1],cancer)



    return datasets

def create_dataset(att_num,dataset):
    for col in dataset.columns:
        if dataset[col].dtype == object:
            dataset[col] = dataset[col].astype('category')
            dataset[col] = dataset[col].cat.codes
    data = dataset[['att'+str(i) for i in range(1,att_num)]]
    target = dataset[['att'+str(att_num)]]
    test = target['att'+str(att_num)]
    counts=dict(sorted(Counter(test).items(),key=lambda x: x[1]))
    cost_mat=[]
    sumy = sum(counts.values())
    for x in test:
        relation=np.random.randint(90000,100000,1)*(float(counts[x])/sumy)
        cost_mat.append([relation,relation,0,0])
    cost_mat=np.array(cost_mat)
    return dict([("data",data), ("target",target),("cost_mat",cost_mat)])

