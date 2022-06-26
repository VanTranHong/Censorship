import pathlib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import linear_model
from sklearn.ensemble import IsolationForest
import datetime
from xgboost import XGBClassifier
import pickle
from sklearn.model_selection import StratifiedKFold
import glob

result_folder = "./ML_runs_results/results_not_include_induced_features/XGB/"


original = []
sampled = []

for file in glob.glob(result_folder+"*.csv"):
    if "US" not in file and "prediction" not in file:
        if "_1" in file:
            sampled.append(file)
        else:
            original.append(file)
            
print(original)
# original = ['./ML_runs_results/results_not_include_induced_features/IF/CNclean_GFCN.csv','./ML_runs_results/results_not_include_induced_features/IF/CNclean_CNclean.csv','./ML_runs_results/results_not_include_induced_features/IF/CNclean_OONICN.csv']
# sampled = ['./ML_runs_results/results_not_include_induced_features/IF/CNclean_GFCN.csv','./ML_runs_results/results_not_include_induced_features/IF/CNclean_CNclean_1.csv','./ML_runs_results/results_not_include_induced_features/IF/CNclean_OONICN_1.csv']
        
original_df = []
sampled_df = []
for ori in original:
    print(ori)
    names = ori.split("/")[-1].split(".")[0].split("_")
    print(names)
    train = names[0]
    validate = names[1]
    df = pd.read_csv(ori)
    df["Validate"] = [validate for i in range(df.shape[0])]
    original_df.append(df)
for sam in sampled:
    df = pd.read_csv(sam)
    names = sam.split("/")[-1].split(".")[0].split("_")
    train = names[0]
    validate = names[1]
    df = pd.read_csv(ori)
    df["Validate"] = [validate for i in range(df.shape[0])]
    sampled_df.append(df)
    
print(original_df)
original_df = pd.concat(original_df)
original_df["Validate_Test"] = [list(original_df["Validate"])[i] +"_"+ list(original_df["Test"])[i] for i in range(original_df.shape[0])]

sampled_df = pd.concat(sampled_df)
sampled_df["Validate_Test"] = [list(sampled_df["Validate"])[i] +"_"+ list(sampled_df["Test"])[i] for i in range(sampled_df.shape[0])]
print(sampled_df)
print(original_df)

original_acc = list(original_df["Accuracy"])
original_tpr = list(original_df["TPR"])

sampled_acc = list(sampled_df["Accuracy"])
sampled_tpr = list(sampled_df["TPR"])

val_test_ori = list(original_df["Validate_Test"])
o_acc = []
o_tpr =[]
s_acc = []
s_tpr = []

sorted(val_test_ori)

for name in val_test_ori:
    ori_df = original_df[original_df["Validate_Test"] == name]
#     print(ori_df)
    o_acc.append(list(ori_df["Accuracy"])[0])
    o_tpr.append(list(ori_df["TPR"])[0])
    sam_df = sampled_df[sampled_df["Validate_Test"]==name]
    print(sam_df)
    
    
    
    
    
    s_acc.append(list(sam_df["Accuracy"])[0])
    s_tpr.append(list(sam_df["TPR"])[0])

    
    




# original_acc = [original_acc[i] for i in sorted_name]
# original_tpr = [original_tpr[i] for i in sorted_name]

# val_test_sam = list(sampled_df["Validate_Test"])
# sorted_name = np.argsort(val_test_sam)
# sampled_acc = [sampled_acc[i] for i in sorted_name]
# sampled_tpr = [sampled_tpr[i] for i in sorted_name]
# print(sampled_df)

import numpy as np
import matplotlib.pyplot as plt
 
# set width of bar
barWidth = 0.25
fig = plt.subplots(figsize =(12, 8))
 
# # set height of bar
# original = list(original_df["Accuracy"])
# sampled = list(sampled_df["\Accuracy"])
 
# Set position of bar on X axis
br1 = np.arange(len(o_acc))
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]
 
# Make the plot
plt.bar(br1, o_acc, color ='green', width = barWidth,
        edgecolor ='grey', label ='Full')
plt.bar(br2, s_acc, color ='blue', width = barWidth,
        edgecolor ='grey', label ='10%')
# plt.bar(br3, CSE, color ='b', width = barWidth,
#         edgecolor ='grey', label ='CSE')
 
# Adding Xticks
plt.xlabel('Validation_Test set', fontweight ='bold', fontsize = 15)
plt.ylabel('Test Accuracy', fontweight ='bold', fontsize = 15)
plt.xticks([r + barWidth for r in range(len(original_acc))],
        val_test_ori, rotation =45)
plt.title("Test Accuracy with clean train set of different size for XGB classifier")
 
plt.legend()
plt.show()
