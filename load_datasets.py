# __author__ = 'amir'
#
# from collections import Counter
# import numpy as np
# import arff
# import os
#
# data_arff = 'E:\Dropbo\Dropbox\Amir & Dadi\\rokach\project_results\datasets\creditApproval\\crx.data.arff'
# # data_arff = 'E:\Dropbo\Dropbox\Amir & Dadi\\rokach\project_results\datasets\\example.arff'
#
# def convert_data(attributes, data):
#     new_attributes = []
#     attributes_convert = []
#     for name, values in attributes:
#         if values == "NUMERIC":
#             new_attributes.append([name, values])
#             attributes_convert.append(values)
#         else:
#             new_attributes.append([name, [ind for ind, x in enumerate(values)]])
#             attributes_convert.append(dict([(x, ind) for ind, x in enumerate(values)]))
#     new_data = []
#     for instance in data:
#         if None in instance:
#             continue
#         new_instance = []
#         for ind, x in enumerate(instance):
#             if attributes_convert[ind] == "NUMERIC":
#                 new_instance.append(x)
#             else:
#                 new_instance.append(attributes_convert[ind][x])
#         new_data.append(new_instance)
#     return new_data
#
#
# def load_arff(data_arff):
#     arff_load = arff.load(open(data_arff, 'rb'))
#     attributes = arff_load['attributes']
#     data = arff_load['data']
#     new_data=convert_data(attributes, data)
#
#     data = np.array([x[:-1]  for x in new_data])
#     target = np.array([x[-1]  for x in new_data])
#     counts=dict(sorted(Counter(target).items(),key=lambda x: x[1]))
#     cost_mat=[]
#     for x in target:
#         relation=100000*(float(counts[1-x])/(counts[x]+counts[1-x]))
#         cost_mat.append([relation,relation,0,0])
#     cost_mat=np.array(cost_mat)
#     return dict([("data",data), ("target",target),("cost_mat",cost_mat)])
#
# def load_arff_from_dir(dir):
#     datasets={}
#     for root, dirs, files in os.walk(dir):
#         for name in files:
#             if ".arff" in name:
#                 path = os.path.join(root, name)
#                 print (path)
#                 dirName=path.split("\\")[-2]
#                 datasets[dirName]=load_arff(path)
#     return datasets
#
#
