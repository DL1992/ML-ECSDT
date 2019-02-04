__author__ = 'amir'

from costcla.datasets import load_bankmarketing
from costcla.datasets import load_creditscoring1
from costcla.datasets import load_creditscoring2

import numpy
import csv
import sklearn.metrics
import datetime, time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import ECSDTa
import load_datasets


def evaluate_ESCDT_RF(data,combinator="RF",inducer="MV",num_estimators=10,samples=2000,max_features=7,pruned=True):
    sets = train_test_split(data.data, data.target, data.cost_mat, test_size=0.33, random_state=0)
    X_train, X_test, y_train, y_test, cost_mat_train, cost_mat_test = sets
    y_pred_test_rf = RandomForestClassifier(random_state=0).fit(X_train, y_train).predict(X_test)
    model=ECSDTa.ECSDT(combinator, inducer, num_estimators, samples, max_features, pruned) #(combinator,inducer,num_estimators,samples,max_features,pruned):
    d=datetime.datetime.now()
    model.fit(X=X_train, y=y_train, cost_mat=cost_mat_train)
    d1=datetime.datetime.now()-d
    pred=model.predict(X_test,cost_mat_test)
    return ECSDTa.saving(y_test, y_pred_test_rf, cost_mat_test), ECSDTa.saving(y_test, pred, cost_mat_test), d1.total_seconds()


def eval_models_on_data(data,dataName,models):
    sets = train_test_split(data["data"], data["target"], data["cost_mat"], test_size=0.33, random_state=0)
    X_train, X_test, y_train, y_test, cost_mat_train, cost_mat_test = sets
    # out=[["dataName","modelName","score","learn time"]]
    out=[]
    d=datetime.datetime.now()
    RF_learn = RandomForestClassifier(random_state=0).fit(X_train, y_train)
    d1=datetime.datetime.now()-d
    time=d1.total_seconds()
    y_pred_test_rf = RF_learn.predict(X_test)
    out.append([dataName,"scikit random forest","RF","MV","0", ECSDTa.saving(y_test, y_pred_test_rf, cost_mat_test), time])

    for m in models.keys():
        print (dataName,m)
        model=models[m]
        d=datetime.datetime.now()
        model.fit(X=X_train, y=y_train, cost_mat=cost_mat_train)
        d1=datetime.datetime.now()-d
        time=d1.total_seconds()
        pred=model.predict(X_test,cost_mat_test)
        inducer,combinator,num_estimators=m.split("_")
        out.append([dataName, m, inducer, combinator, num_estimators, ECSDTa.saving(y_test, pred, cost_mat_test), time])
    return out


def get_models_dict():
    inducers={"Bagging":0,"Pasting":1,"RF":2,"RP":3}
    combinators={"MV":0,"CSWV":1,"CSS":2}


    models={}
    for num_estimators in [5]:
        for inducer in inducers:
            for combinator in combinators:
                name=inducer+"_"+combinator+"_"+str(num_estimators)
                models[name]=ECSDTa.ECSDT(inducer, combinator, num_estimators, 2000, 7, True)
    return models


# data = load_creditscoring1()
# # numpy.set_printoptions(threshold=numpy.nan)
# for x in zip(data["cost_mat"],data["target"]):
#     s=""
#     for i in x[0]:
#         s+= str(i) + "\t"
#     s+=str(x[1])
#     print s
# exit()


out=[["dataName","modelName","inducer","combinator","num_estimators","score","learn time"]]
models_dict=get_models_dict()
data = load_creditscoring1()
out =out+eval_models_on_data(data,"load_creditscoring1", get_models_dict())
data = load_creditscoring2()
out =out+eval_models_on_data(data,"load_creditscoring2", get_models_dict())
data = load_bankmarketing()
out =out+eval_models_on_data(data,"bankmarketing", get_models_dict())


data_dir = 'E:\Dropbo\Dropbox\Amir & Dadi\\rokach\project_results\datasets'
datasets=load_datasets.load_arff_from_dir(data_dir)
for ds in datasets:
    out =out+eval_models_on_data(datasets[ds],ds, get_models_dict())




writer=csv.writer(open("E:\Dropbo\Dropbox\Amir & Dadi\\rokach\project_results\\resultsAll.csv","wb"))
writer.writerows(out)

#
# data = load_creditscoring1()
# print evaluate_ESCDT_RF(data,"RF","MV",10,2000,7,True)
# print evaluate_ESCDT_RF(data,"Bagging","MV",10,2000,7,True)
# print evaluate_ESCDT_RF(data,"Bagging","CSWV",10,2000,7,True)
# data = load_creditscoring2()
# print evaluate_ESCDT_RF(data,"RF","MV",10,2000,7,True)
# print evaluate_ESCDT_RF(data,"Bagging","MV",10,2000,7,True)
# print evaluate_ESCDT_RF(data,"Bagging","CSWV",10,2000,7,True)
# data = load_bankmarketing()
# print evaluate_ESCDT_RF(data,"RF","MV",10,2000,7,True)
# print evaluate_ESCDT_RF(data,"Bagging","MV",10,2000,7,True)
# print evaluate_ESCDT_RF(data,"Bagging","CSWV",10,2000,7,True)
# #
# sets = train_test_split(data.data, data.target, data.cost_mat, test_size=0.33, random_state=0)
# X_train, X_test, y_train, y_test, cost_mat_train, cost_mat_test = sets
# # y_pred_test_rf = RandomForestClassifier(random_state=0).fit(X_train, y_train).predict(X_test)
# # f = CostSensitiveBaggingClassifier()
# # y_pred_test_csdt = f.fit(X_train, y_train, cost_mat_train).predict(X_test)
# # # Savings using only RandomForest
# # print(savings_score(y_test, y_pred_test_rf, cost_mat_test))
# # # 0.12454256594
# # # Savings using CostSensitiveRandomForestClassifier
# # print(savings_score(y_test, y_pred_test_csdt, cost_mat_test))
# # # 0.478964004931
#
# model=ECSDT.ECSDT("RF","MV",10,2000,7,True) #(combinator,inducer,num_estimators,samples,max_features,pruned):
# print X_train.shape
# model.fit(X=X_train, y=y_train, cost_mat=cost_mat_train)
# pred=model.predict(X_test,cost_mat_test)
#
# print(savings_score(y_test, pred, cost_mat_test))
#
#
#

