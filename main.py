from costcla.datasets import load_bankmarketing
from costcla.datasets import load_creditscoring1
from costcla.datasets import load_creditscoring2
from costcla.metrics import savings_score
from costcla.models import cost_tree
import csv
from sklearn.metrics import f1_score
import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import data_loader
import ECSDT
import pandas as pd


def eval_models_on_data(dataset, dataset_name, models, flag=False):
    if flag:
        dataset["data"] = dataset["data"].values
        dataset["target"] = dataset["target"].values
    x_train, x_test, y_train, y_test, cost_mat_train, cost_mat_test = train_test_split(dataset["data"],
                                                                                       dataset["target"],
                                                                                       dataset["cost_mat"],
                                                                                       test_size=0.33,
                                                                                       random_state=42)
    out = []
    if flag:
        y_test= y_test.T[0]
    ##RF baseline
    print(dataset_name, "Random Forest")
    start_time = datetime.datetime.now()
    rf_clf = RandomForestClassifier(random_state=42)
    rf_clf.fit(x_train, y_train)
    end_time = datetime.datetime.now()
    fit_time = end_time - start_time
    start_time = datetime.datetime.now()
    y_pred_test_rf = rf_clf.predict(x_test)
    end_time = datetime.datetime.now()
    pred_time = end_time - start_time
    out.append(
        [dataset_name, "Random Forest", "RF", "MV", "0", fit_time, pred_time,
         savings_score(y_test, y_pred_test_rf, cost_mat_test), f1_score(y_test, y_pred_test_rf)])
    ##CSDT baseline
    print(dataset_name, "CSDT")
    start_time = datetime.datetime.now()
    csdt_clf = cost_tree.CostSensitiveDecisionTreeClassifier()
    csdt_clf.fit(x_train, y_train, cost_mat_train)
    end_time = datetime.datetime.now()
    fit_time = end_time - start_time
    start_time = datetime.datetime.now()
    y_pred_test_csdt = csdt_clf.predict(x_test)
    end_time = datetime.datetime.now()
    pred_time = end_time - start_time
    out.append(
        [dataset_name, "CSDT", "-", "-", "-", fit_time, pred_time,
         savings_score(y_test, y_pred_test_csdt, cost_mat_test), f1_score(y_test, y_pred_test_csdt)])
    ## models results
    for m in models.keys():
        print(dataset_name, m)
        model = models[m]
        start_time = datetime.datetime.now()
        model.fit(X=x_train, y=y_train, cost_mat=cost_mat_train)
        end_time = datetime.datetime.now()
        fit_time = end_time - start_time
        start_time = datetime.datetime.now()
        pred = model.predict(x_test)
        end_time = datetime.datetime.now()
        pred_time = end_time - start_time
        inducer, combiner, num_of_iterations, ne, nf = m.split("_")
        out.append([dataset_name, m, inducer, combiner, num_of_iterations, fit_time, pred_time,
                    savings_score(y_test, pred, cost_mat_test), f1_score(y_test, pred)])
    return out


def get_models_dict(Ne, Nf):
    inducers = {"Bagging": 0, "Pasting": 1, "RF": 2, "RP": 3}
    combiners = {"MV": 0, "CSWV": 1, "CSS": 2}
    models = {}
    for num_of_iteration in [5]:
        for inducer in inducers:
            for combiner in combiners:
                name = "{}_{}_{}_{}_{}".format(inducer, combiner, str(num_of_iteration), str(int(Ne * 0.2)),
                                               str(int(Nf * 0.5)))
                models[name] = ECSDT.ECSDT(num_of_iteration, int(Ne * 0.2), int(Nf * 0.5), combiner, inducer)
    return models


path = "results.csv"
out= []
headers = ["DataName", "ModelName", "Inducer", "Combiner", "num_of_iteration", "FitTIme", "PredictTIme", "SavingScore",
        "F1Score"]
data = load_creditscoring1()
out = out + eval_models_on_data(data, "credit scoring1", get_models_dict(data["data"].shape[0], data["data"].shape[1]))

# data1 = data_loader.return_model()
# data = data1['cancer']
# out = out + eval_models_on_data(data, "cancer", get_models_dict(data["data"].shape[0], data["data"].shape[1]),True)
df = pd.DataFrame(out, columns=headers)
writer = pd.ExcelWriter('results2.xlsx')
df.to_excel(writer)
writer.save()