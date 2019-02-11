from costcla.datasets import load_creditscoring1, load_creditscoring2, load_bankmarketing
from costcla.metrics import savings_score
from costcla.models import cost_tree
from sklearn.metrics import f1_score
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import data_loader
import ECSDT
import pandas as pd
import numpy as np
import pickle
import os


def eval_model(model, model_name, x_train, y_train, x_test, y_test, cost_mat_train, cost_mat_test, dataset_name,
               where_to_pickle_path, cost_flag=True):
    file_name = '{}_{}.sav'.format(model_name, dataset_name)
    print(dataset_name, model_name)
    start_time = time.time()
    if cost_flag:
        model.fit(x_train, y_train, cost_mat_train)
    else:
        model.fit(x_train, y_train)
    file_path = os.path.join(where_to_pickle_path, file_name)
    pickle.dump(model, open(file_path, 'wb'))
    end_time = time.time()
    fit_time = end_time - start_time
    start_time = time.time()
    pred = model.predict(x_test)
    end_time = time.time()
    pred_time = end_time - start_time
    inducer, combiner, num_of_iterations, ne, nf = model_name.split("_")
    return [dataset_name, model_name, inducer, combiner, num_of_iterations, ne, nf, fit_time, pred_time,
            max(0.0, savings_score(y_test, pred, cost_mat_test)),
            f1_score(np.append(y_test, [0, 1]), np.append(pred, [0, 1]))]


def eval_models_on_data(dataset, dataset_name, models, where_to_pickle_path, flag=False):
    if flag:
        dataset["data"] = dataset["data"].values
        dataset["target"] = dataset["target"].values
    x_train, x_test, y_train, y_test, cost_mat_train, cost_mat_test = train_test_split(dataset["data"],
                                                                                       dataset["target"],
                                                                                       dataset["cost_mat"],
                                                                                       test_size=0.2)
    out = []
    if flag:
        y_test = y_test.T[0]
    # RF baseline
    print(dataset_name, "Random Forest")
    model_name = "{}_{}_{}_{}_{}".format("RF", "MV", "0", x_train.shape[0], x_train.shape[1])
    model = RandomForestClassifier()
    out.append(eval_model(model, model_name, x_train, y_train, x_test, y_test, cost_mat_train, cost_mat_test,
                          dataset_name, where_to_pickle_path, False))
    # CSDT baseline
    print(dataset_name, "CSDT")
    model_name = "{}_{}_{}_{}_{}".format("CSDT", "-", "-", x_train.shape[0], x_train.shape[1])
    model = cost_tree.CostSensitiveDecisionTreeClassifier()
    out.append(eval_model(model, model_name, x_train, y_train, x_test, y_test, cost_mat_train, cost_mat_test,
                          dataset_name, where_to_pickle_path))
    # models results
    for m in models.keys():
        model_name = m
        model = models[model_name]
        out.append(eval_model(model, model_name, x_train, y_train, x_test, y_test, cost_mat_train, cost_mat_test,
                              dataset_name, where_to_pickle_path))
    return out


def get_models_dict(Ne, Nf):
    inducers = {"Bagging": 0, "Pasting": 1, "RF": 2, "RP": 3}
    combiners = {"MV": 0, "CSWV": 1, "CSS": 2}
    models = {}
    for num_of_iteration in [10,20,30]:
        for inducer in inducers:
            for combiner in combiners:
                name = "{}_{}_{}_{}_{}".format(inducer, combiner, str(num_of_iteration), str(int(Ne * 0.2)),
                                               str(int(Nf * 0.7)))
                models[name] = ECSDT.ECSDT(num_of_iteration, int(Ne * 0.2), int(Nf * 0.7), combiner, inducer)
    return models


# start of running main
out = []
headers = ["DataName", "ModelName", "Inducer", "Combiner", "num_of_iteration", "NE", "NF", "FitTIme", "PredictTIme",
           "SavingScore", "F1Score"]

creditscoring1 = load_creditscoring1()
try:
    out = out + eval_models_on_data(creditscoring1, "credit scoring1",
                                    get_models_dict(creditscoring1["data"].shape[0], creditscoring1["data"].shape[1]),
                                    r'C:\School\workspace\ECSDT\test')
except:
    print("credit scoring1 fail")

try:
    bankmarketing = load_bankmarketing()
    out = out + eval_models_on_data(bankmarketing, "bankmarketing",
                                    get_models_dict(bankmarketing["data"].shape[0], bankmarketing["data"].shape[1]),
                                    r'C:\School\workspace\ECSDT\test')
except:
    print("bankmarketing fail")

try:
    creditscoring2 = load_creditscoring2()
    out = out + eval_models_on_data(creditscoring2, "credit scoring2",
                                    get_models_dict(creditscoring2["data"].shape[0], creditscoring2["data"].shape[1]),
                                    r'C:\School\workspace\ECSDT\test')
except:
    print("credit scoring2 fail")


loaded_datasets = data_loader.get_all_datasets()
for key, value in loaded_datasets.items():
    try:
        out = out + eval_models_on_data(value, key, get_models_dict(value["data"].shape[0], value["data"].shape[1]),
                                        r'C:\School\workspace\ECSDT\test', True)
    except:
        print(key + "fail")

df = pd.DataFrame(out, columns=headers)
writer = pd.ExcelWriter('ECSDT_results.xlsx')
df.to_excel(writer)
writer.save()
