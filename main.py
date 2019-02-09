from costcla.datasets import load_creditscoring1, load_creditscoring2, load_bankmarketing
from costcla.metrics import savings_score
from costcla.models import cost_tree
from sklearn.metrics import f1_score
import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import data_loader
import ECSDT
import pandas as pd


def eval_model(model, model_name, x_train, y_train, x_test, y_test, cost_mat_train, cost_mat_test, dataset_name,
               cost_flag=True):
    print(dataset_name, model_name)
    start_time = datetime.datetime.now()
    if cost_flag:
        model.fit(x_train, y_train, cost_mat_train)
    else:
        model.fit(x_train, y_train)
    end_time = datetime.datetime.now()
    fit_time = end_time - start_time
    start_time = datetime.datetime.now()
    pred = model.predict(x_test)
    end_time = datetime.datetime.now()
    pred_time = end_time - start_time
    inducer, combiner, num_of_iterations, ne, nf = model_name.split("_")
    return [dataset_name, model_name, inducer, combiner, num_of_iterations, ne, nf, fit_time, pred_time,
            max(0.0, savings_score(y_test, pred, cost_mat_test)), f1_score(y_test, pred)]


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
        y_test = y_test.T[0]
    # RF baseline
    print(dataset_name, "Random Forest")
    model_name = "{}_{}_{}_{}_{}".format("RF", "MV", "0", x_train.shape[0], x_train.shape[1])
    model = RandomForestClassifier(random_state=42)
    out.append(eval_model(model, model_name, x_train, y_train, x_test, y_test, cost_mat_train, cost_mat_test,
                          dataset_name, False))
    ##CSDT baseline
    print(dataset_name, "CSDT")
    model_name = "{}_{}_{}_{}_{}".format("-", "-", "-", x_train.shape[0], x_train.shape[1])
    model = cost_tree.CostSensitiveDecisionTreeClassifier()
    out.append(eval_model(model, model_name, x_train, y_train, x_test, y_test, cost_mat_train, cost_mat_test,
                          dataset_name))
    ## models results
    for m in models.keys():
        model_name = m
        model = models[model_name]
        out.append(eval_model(model, model_name, x_train, y_train, x_test, y_test, cost_mat_train, cost_mat_test,
                              dataset_name))
    return out


def get_models_dict(Ne, Nf):
    inducers = {"Bagging": 0, "Pasting": 1, "RF": 2, "RP": 3}
    combiners = {"MV": 0, "CSWV": 1, "CSS": 2}
    models = {}
    for num_of_iteration in [5,15,25]:
        for inducer in inducers:
            for combiner in combiners:
                name = "{}_{}_{}_{}_{}".format(inducer, combiner, str(num_of_iteration), str(int(Ne * 0.2)),
                                               str(int(Nf * 0.5)))
                models[name] = ECSDT.ECSDT(num_of_iteration, int(Ne * 0.2), int(Nf * 0.7), combiner, inducer)
    return models


path = "results.csv"
out = []
headers = ["DataName", "ModelName", "Inducer", "Combiner", "num_of_iteration", "NE", "NF", "FitTIme", "PredictTIme",
           "SavingScore",
           "F1Score"]
data = load_creditscoring1()
out = out + eval_models_on_data(data, "credit scoring1", get_models_dict(data["data"].shape[0], data["data"].shape[1]))

# data1 = data_loader.return_model()
# data = data1['cancer']
# out = out + eval_models_on_data(data, "cancer", get_models_dict(data["data"].shape[0], data["data"].shape[1]),True)
df = pd.DataFrame(out, columns=headers)
writer = pd.ExcelWriter('results4.xlsx')
df.to_excel(writer)
writer.save()
