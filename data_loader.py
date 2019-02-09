import load_datasets
from collections import Counter
import pandas as pd
import numpy as np


def get_all_datasets():
    datasets={}
    data_dir = "DataSets/"

    # 1) data set = cancer  https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(original)
    cancer = pd.read_csv(data_dir + 'cancer/breast_cancer.csv',names=['att'+str(i) for i in range(1,11)])
    cancer.loc[cancer['att10'] == 2] = 0
    cancer.loc[cancer['att10'] == 4] = 1
    for col in cancer.columns:
        if cancer[col].dtype == 'object':
            cancer[col].fillna("Unknown", inplace=True)
        else:
            col_mean = cancer[col].mean()
            cancer[col].fillna(col_mean, inplace=True)
    datasets['cancer'] = create_dataset(cancer.shape[1],cancer)

    # 2) data set = bank  https://archive.ics.uci.edu/ml/datasets/Bank+Marketing
    bank = pd.read_csv(data_dir + 'bank marketing/bank.csv',names=["att" + str(i) for i in range(1, 18)])
    datasets['bank'] = create_dataset(bank.shape[1], bank)

    # 3) data set = blood  https://archive.ics.uci.edu/ml/datasets/Blood+Transfusion+Service+Center
    blood = pd.read_csv(data_dir + 'blood_transfusion/blood.csv', names=["att" + str(i) for i in range(1, 6)])
    datasets['blood'] = create_dataset(blood.shape[1], blood)

    # 4) data set = magic04 https://archive.ics.uci.edu/ml/datasets/MAGIC+Gamma+Telescope
    magic = pd.read_csv(data_dir + 'magic04/magic.csv', names=["att" + str(i) for i in range(1, 12)])
    datasets['magic'] = create_dataset(magic.shape[1], magic)

    # 5) data set = default_ccc https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients
    default_ccc = pd.read_csv(data_dir + 'defualt_credit_cards_clients/default_of_credit_card_clients.csv', names=["att" + str(i) for i in range(1, 25)])
    datasets['default_ccc'] = create_dataset(default_ccc.shape[1], default_ccc)

    # 6) data set = default_ccc  https: // archive.ics.uci.edu/ml/datasets/adult
    adult = pd.read_csv(data_dir + 'adult/adult.csv', names=["att" + str(i) for i in range(1, 25)])
    for col in adult.columns:
        if adult[col].dtype == 'object':
            adult[col].fillna("Unknown", inplace=True)
        else:
            col_mean = adult[col].mean()
            adult[col].fillna(col_mean, inplace=True)
    datasets['adult'] = create_dataset(adult.shape[1], adult)

    # 7) data set = default_ccc  https://archive.ics.uci.edu/ml/datasets/banknote+authentication
    banknote = pd.read_csv(data_dir + 'banknote_authentication/banknote_authentication.csv', names=["att" + str(i) for i in range(1, 6)])
    datasets['banknote'] = create_dataset(banknote.shape[1], banknote)

    return datasets


def create_dataset(att_num,dataset):
    for col in dataset.columns:
        if dataset[col].dtype == object:
            dataset[col] = dataset[col].astype('category')
            dataset[col] = dataset[col].cat.codes
    data = dataset[['att'+str(i) for i in range(1,att_num)]]
    target = dataset[['att'+str(att_num)]]
    test = target['att'+str(att_num)]
    counts = dict(sorted(Counter(test).items(),key=lambda x: x[1]))
    cost_mat = []
    sumy = sum(counts.values())
    for x in test:
        relation = np.random.randint(90000,100000,1)*(float(counts[x])/sumy)
        cost_mat.append([relation,relation,0,0])
    cost_mat = np.array(cost_mat)
    return dict([("data",data), ("target",target),("cost_mat",cost_mat)])

