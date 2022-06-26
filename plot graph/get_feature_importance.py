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
import os
import glob

def get_feature_importance(model, model_name):
    pickle_model = pickle.load(open(model,"rb"))
    if model_name == "OCSVM":
        return pickle_model.coef_
    elif model_name == "IF":
        return pickle_model.estimators_features_
    elif model_name == "XGB":
        return pickle_model.feature_importances_
import shap
model_folder = "./ML_runs_models/results_not_include_induced_features/"
XGB = []
IF = []
OCSVM = []
for filename  in glob.glob(model_folder+"*/*.sav"):

   
    if "IF" in filename:
        IF.append(filename)
    elif "XGB" in filename:
        XGB.append(filename)
    elif "OCSVM" in filename:
        OCSVM.append(filename)
print(XGB)
# ls = []

# model_name = "IF"

# dat = []
# trains = []
# tests = []
# for model in chosen_folder:
#     features_impt = list(get_feature_importance(model, model_name))
#     print(model)
#     print(features_impt)
#     feature_frequency = [0 for i in range(95)]
#     for item in features_impt:
#         for i in item:
#             feature_frequency[i]+=1
#     frequency = [item/len(features_impt) for item in feature_frequency]
#     dat.append(features_impt)
        

    
#     name = model.split("/")[-1].split(".")[0]
    
#     train = name.split("_")[0]
#     validate = "_".join(name.split("_")[1:])
#     trains.append(train)
#     tests.append(validate)
# # print(trains)
# # print(tests)
# # print(trains)
# print(dat)
    
# # trains = [item[:-5] for item in trains]
# # tests = [item[:-8] for item in tests]
# train_test = [trains[i]+"_"+tests[i] for i in range(len(trains))]
# columns = ["Feature "+ str(i+1) for i in range(len(dat[0]))]
# df = pd.DataFrame(dat, columns = columns)
# df["Train_Val"] = [trains[i]+"_"+tests[i] for i in range(len(trains)) ] 

# print(df)
# df.to_csv("./ML_runs_models/results_not_include_induced_features/"+model_name+"/"+"feature_importance.csv")
# print("./ML_runs_models/results_not_include_induced_features/"+model_name+"/"+"feature_importance.csv")
    
    




# feature_importances = list(get_feature_importance(model, model_name))
# sorted_acc_index = np.argsort(feature_importances)
# print(sorted_acc_index[-1])
# print(max(feature_importances))
CN = pd.read_csv("./data_after_preprocess/CN_temp_encoded.csv")
CN = CN.drop(columns=['Unnamed: 0'])

US = pd.read_csv("./data_after_preprocess/US_temp_encoded_sampled.csv")
print(US.shape)
US = US.drop(columns=["Unnamed: 0"])

clean_CN = CN[CN["blocking"]==0]
clean_CN = clean_CN[clean_CN["GFWatchblocking_truth"]==0]

OONI_CN = CN[CN["blocking"]==1]
GF_CN = CN[CN["GFWatchblocking_truth"] == 1]

clean_US = US[US["blocking"]==0]
clean_US = clean_US[clean_US["GFWatchblocking_truth"]==0]

OONI_US = US[US["blocking"]==1]
GF_US = US[US["GFWatchblocking_truth"] == 1]
X_USclean_train, X_USclean_rest, y_USclean_train, y_USclean_rest = train_test_split(clean_US.drop(columns = ["blocking","GFWatchblocking_truth"]),clean_US["blocking"] , test_size=0.33)
X_USclean_validate, X_USclean_test, y_USclean_validate, y_USclean_test = train_test_split(X_USclean_rest,y_USclean_rest, test_size=0.33)

X_CNclean_train, X_CNclean_rest, y_CNclean_train, y_CNclean_rest = train_test_split(clean_CN.drop(columns = ["blocking","GFWatchblocking_truth"]),clean_CN["blocking"] , test_size=0.33)
X_CNclean_validate, X_CNclean_test, y_CNclean_validate, y_CNclean_test = train_test_split(X_CNclean_rest,y_CNclean_rest , test_size=0.33)


X_GFCN_train, X_GFCN_rest, y_GFCN_train, y_GFCN_rest = train_test_split(GF_CN.drop(columns = ["blocking","GFWatchblocking_truth"]),GF_CN["GFWatchblocking_truth"] , test_size=0.33)
X_GFCN_validate, X_GFCN_test, y_GFCN_validate, y_GFCN_test = train_test_split(X_GFCN_rest,y_GFCN_rest , test_size=0.33)

X_OONICN_train, X_OONICN_rest, y_OONICN_train, y_OONICN_rest = train_test_split(OONI_CN.drop(columns = ["blocking","GFWatchblocking_truth"]),OONI_CN["blocking"] , test_size=0.33)
X_OONICN_validate, X_OONICN_test, y_OONICN_validate, y_OONICN_test = train_test_split(X_OONICN_rest,y_OONICN_rest , test_size=0.33)

folder = "./ML_runs_results/results_not_include_induced_features/"
# X_GFCN_Test = pd.concat([X_CNclean_test,X_GFCN_test])
# y_GFCN_Test = pd.concat([y_CNclean_test,y_GFCN_test])

# X_test_US = X_USclean_test
# y_test_US = y_USclean_test

# X_OONICN_Test = pd.concat([X_CNclean_test,X_OONICN_test])
# y_OONICN_Test = pd.concat([y_CNclean_test,y_OONICN_test])

# X_test_GFCN= np.array(X_GFCN_Test)
# y_test_GFCN= np.array(y_GFCN_Test)
# X_test_OONICN= np.array(X_OONICN_Test)
# y_test_OONICN= np.array(y_OONICN_Test)



# X_test_US= np.array(X_test_US)
# y_test_US= np.array(y_test_US)
sample_X_GF_train = X_GFCN_validate.sample(frac=0.03, replace=True, random_state=1)
sample_y_GF_train = y_GFCN_validate.sample(frac=0.03, replace=True, random_state=1)

sample_X_OONI_train = X_OONICN_validate.sample(frac=0.03, replace=True, random_state=1)
sample_y_OONI_train = y_OONICN_validate.sample(frac=0.03, replace=True, random_state=1)

X_train_clean = X_CNclean_train
y_train_clean = y_CNclean_train


X_train_GF = pd.concat([X_CNclean_train,sample_X_GF_train])
y_train_GF = pd.concat([y_CNclean_train,sample_y_GF_train])

X_train_OONI = pd.concat([X_CNclean_train,sample_X_OONI_train])
y_train_OONI = pd.concat([y_CNclean_train,sample_y_OONI_train])

X_validate_clean = X_CNclean_validate
y_validate_clean = y_CNclean_validate

X_validate_GF = pd.concat([X_CNclean_validate,X_GFCN_validate])
y_validate_GF = pd.concat([y_CNclean_validate,y_GFCN_validate])

X_validate_OONI = pd.concat([X_CNclean_validate,X_OONICN_validate])
y_validate_OONI = pd.concat([y_CNclean_validate,y_OONICN_validate])


X_train_clean = np.array(X_train_clean)
y_train_clean= np.array(y_train_clean)

X_train_OONI = np.array(X_train_OONI)
y_train_OONI= np.array(y_train_OONI)

X_train_GF = np.array(X_train_GF)
y_train_GF= np.array(y_train_GF)

X_validate_clean = np.array(X_validate_clean)
y_validate_clean= np.array(y_validate_clean)

X_validate_OONI = np.array(X_validate_OONI)
y_validate_OONI= np.array(y_validate_OONI)

X_validate_GF = np.array(X_validate_GF)
y_validate_GF= np.array(y_validate_GF)
chosen_folder = IF
index = 0
model = chosen_folder[index]
model = "./ML_runs_models/results_not_include_induced_features/IF/CNclean_OONICN.sav"
print(model)
pickle_model = pickle.load(open(model,"rb"))
train = model.split("/")[-1].split("_")[0]
if train == "GFCN":# model = './ML_runs_models/results_not_include_induced_features/XGB/OONICN_OONICN.sav'

model = "./ML_runs_models/results_not_include_induced_features/IF/CNclean_GFCN.sav"
feature_importance = get_feature_importance(model, "IF")
# index_feature =[ "Feature "+ str(i) for i in range(len(feature_importance))]
# sorted_index = np.argsort(feature_importance)[-1::-1]

feature_frequency = [0 for i in range(88)]
print("Number of trees")
print(len(feature_importance))
for arr in feature_importance:
    for i in arr:
        feature_frequency[i]+=1

# top_10_importance = feature_importance[sorted_index][:10]
# top_10_index = [index_feature[i] for i in sorted_index[:10]]

# print(feature_importance)



feature_sorted = np.argsort(feature_frequency)[-1::-1]





df_features = pd.read_csv('features_encoded.csv')
features = list(df_features["name"])


top_features = [features[i] for i in feature_sorted]
top_frequency = [feature_frequency[i] for i in feature_sorted]
# for i in top_features:
#     print(i)
# # print()
# for i in top_frequency:
#     print(i)
# print(sorted_frequency)

Features = ["http_experiment_failure","headers_match","dns_consistency","probe_asn","test_keys_asn","title_match",
            "body_length_match","resolver_asn","status_code_match","body_proportion"
,"probe_network_name","resolver_network_name","measurement_start_time","test_start_time","test_runtime",
            "dns_experiment_failure","test_keys_as_org_name"]

scores =[]
frequency = []
for i in range(17):
    feat = Features[i]
    fre = 0
    score =0
    for j in range(len(top_features)):
        feature = top_features[j]
        if feat in feature:
            score +=  top_frequency[j]
            fre+=1
    frequency.append(fre)
    scores.append(score)
for i in frequency:
    print(i)
    
print("hello")
for i in scores:
    print(i)

feature_frequency = [0 for i in range(88)]
for arr in feature_importance:
    for i in arr:
        feature_frequency[i]+=1

feature_sorted = np.argsort(feature_frequency)[-1::-1]
print(feature_sorted[:10])


sorted_frequency = [feature_frequency[i]/88 for i in feature_sorted][:10]
print(sorted_frequency)
import pandas as pd
from matplotlib import pyplot as plt
 
    
df_features = pd.read_csv('features_encoded.csv')
features = list(df_features["name"])
# print(df_features)
model = "./ML_runs_models/results_not_include_induced_features/XGB/GFCN_GFCN.sav"



# Figure Size

feature_importance = get_feature_importance(model, "XGB")
# index_feature =[ "Feature "+ str(i) for i in range(len(feature_importance))]
sorted_index = np.argsort(feature_importance)[-1::-1]



top_features = [features[i] for i in sorted_index]
top_frequency = [feature_importance[i] for i in sorted_index]
# for i in top_features:
#     print(i)
# # print()
# for i in top_frequency:
#     print(i)
    
    





Features = ["http_experiment_failure","headers_match","dns_consistency","probe_asn","test_keys_asn","title_match",
            "body_length_match","resolver_asn","status_code_match","body_proportion"
,"probe_network_name","resolver_network_name","measurement_start_time","test_start_time","test_runtime",
            "dns_experiment_failure","test_keys_as_org_name"]
print(len(Features))
scores =[]
frequency = []
for i in range(17):
    feat = Features[i]
    fre = 0
    score =0
    for j in range(len(top_features)):
        feature = top_features[j]
        if feat in feature:
            score +=  top_frequency[j]
            fre+=1
    frequency.append(fre)
    scores.append(score)
for i in frequency:
    print(i)
for i in scores:
    print(i)















# top_10_importance = feature_importance[sorted_index][:10]
# top_10_features = [features[i] for i in sorted_index[:10]]
# fig, ax = plt.subplots(figsize =(16, 9)) 
# # Horizontal Bar Plot
# ax.barh(top_10_features, top_10_importance)
# plt.ylabel('Feature Name')
# plt.xlabel("|Feature importance|")
 
# ax.set_title("Top 10 features selected by XGB when trained and validated on data labeled by OONI")


# df = pd.DataFrame()
# df["Feature Name"] = top_10_features
# df["Feature Importance"] = top_10_importance
# print(df)

# print(top_10_features)
# print(top_10_importance)



# plt.show()
chosen_folder = XGB
index = 0
model = chosen_folder[index]
model = "./ML_runs_models/results_not_include_induced_features/XGB/GFCN_GFCN.sav"
print(model)
pickle_model = pickle.load(open(model,"rb"))
train = model.split("/")[-1].split("_")[0]
if train == "GFCN":
    X = np.array(X_train_GF)
elif train=="OONICN":
    X = np.array(X_train_OONI)
else:
    X = np.array(X_train_clean)
import pandas as pd
from matplotlib import pyplot as plt
 
feature_importance = get_feature_importance(model, "XGB")
index_feature =[ "Feature "+ str(i) for i in range(len(feature_importance))]
sorted_index = np.argsort(feature_importance)[-1::-1]


top_10_importance = feature_importance[sorted_index][:10]
top_10_index = [index_feature[i] for i in sorted_index[:10]]
# Figure Size
fig, ax = plt.subplots(figsize =(16, 9))
 
# Horizontal Bar Plot
ax.barh(top_10_index, top_10_importance)
plt.ylabel('Feature Name')
plt.xlabel("|Feature importance|")
 
ax.set_title("Top 10 features selected by XGB when the train data is clean data with a bit of censored data labled by GFWatch")

plt.show()
chosen_folder = IF
index = 0
model = chosen_folder[index]
model = "./ML_runs_models/results_not_include_induced_features/IF/CNclean_GFCN.sav"
print(model)
pickle_model = pickle.load(open(model,"rb"))
train = model.split("/")[-1].split("_")[0]
if train == "GFCN":
    X = np.array(X_train_GF)
elif train=="OONICN":
    X = np.array(X_train_OONI)
else:
    X = np.array(X_train_clean)
def get_model(model_name):
    models = []
    if model_name == "OCSVM":
        ###### OCSVM #####
        params={ "max_iter":[20,40,60,80]}
        models = [] 

        for j in params["max_iter"]:
            model = linear_model.SGDOneClassSVM(random_state=42, max_iter = j)
            models.append(model)
    elif model_name == "IF":
        ###### IF #######
# Model parameters
        params={"max_features":[10],"n_estimators":[10], "contamination":[0.001] }
        for n in params["contamination"]:
            for j in params["max_features"]:
                for t in params["n_estimators"]:

                    model = IsolationForest(random_state=0, max_features = j, contamination=n, n_estimators = t)
                    models.append(model)
    elif model_name=="XGB":
        # Model parameters
        params={"max_depth":[15,30,45], "n_estimators":[15,30,50,75,100,200] }


        models = [] 
        for n in params["max_depth"]:
            for j in params["n_estimators"]:

                model = XGBClassifier(max_depth=n, n_estimators=j)
                models.append(model)
    return models
  model = "./ML_runs_models/results_not_include_induced_features/XGB/GFCN_GFCN.sav"
pickle_model = pickle.load(open(model,"rb"))

train = "GFCN"
if train == "GFCN":
    X = np.array(X_train_GF)
elif train=="OONICN":
    X = np.array(X_train_OONI)
else:
    X = np.array(X_train_clean)
explainer = shap.Explainer(pickle_model)
shap_values = explainer(X)

print(shap_values.values.shape)
print(len(shap_values.values))
print(shap_values.values[0][0])

# values = []
# for i in range(shap_values.shape[1]):
#     start = 0
#     for j in range(shap_values.shape[0]):
#         start+=shap_values[j][i]
#     values.append(start)
# print(values)
        
# for i in range(len(shap_values)):
#     value = shap_values[i]
#     print(value)
    



# visualize the first prediction's explanation
# shap.plots.waterfall(shap_values[0])
transpose = shap_values.values.transpose()
values = []
for i in range(len(transpose)):
    abs_ = [abs(j) for j in transpose[i]]
    sum_ = sum(abs_)
    values.append(sum_)
print(values)
values = [i/324040 for i in values]
print(values)
sorted_index = np.argsort(values)[-1::-1]



top_features = [features[i] for i in sorted_index]
top_frequency = [values[i] for i in sorted_index]
for i in top_features:
    print(i)
# print()
for i in top_frequency:
    print(i)
Features = ["http_experiment_failure","headers_match","dns_consistency","probe_asn","test_keys_asn","title_match",
            "body_length_match","resolver_asn","status_code_match","body_proportion"
,"probe_network_name","resolver_network_name","measurement_start_time","test_start_time","test_runtime",
            "dns_experiment_failure","test_keys_as_org_name"]
print(len(Features))
scores =[]
frequency = []
for i in range(17):
    feat = Features[i]
    fre = 0
    score =0
    for j in range(len(top_features)):
        feature = top_features[j]
        if feat in feature:
            score +=  top_frequency[j]
            fre+=1
    frequency.append(fre)
    scores.append(score)
for i in frequency:
    print(i)
for i in scores:
    print(i)
# print(scores)
model = "./ML_runs_models/results_not_include_induced_features/XGB/GFCN_GFCN.sav"
pickle_model = pickle.load(open(model,"rb"))

train = "GFCN"
if train == "GFCN":
    X = np.array(X_train_GF)
elif train=="OONICN":
    X = np.array(X_train_OONI)
else:
    X = np.array(X_train_clean)
explainer = shap.Explainer(pickle_model)
shap_values = explainer(X)

print(shap_values.values.shape)
print(len(shap_values.values))
print(shap_values.values[0][0])

values = []
for i in range(shap_values.shape[1]):
    start = 0
    for j in range(shap_values.shape[0]):
        start+=shap_values[j][i]
    values.append(start)
    print("Good")
print(values)
        
# for i in range(len(shap_values)):
#     value = shap_values[i]
#     print(value)
    



# visualize the first prediction's explanation
# shap.plots.waterfall(shap_values[0])
chosen_folder = XGB
index = 1
model = chosen_folder[index]

print(model)
pickle_model = pickle.load(open(model,"rb"))
train = model.split("/")[-1].split("_")[0]
if train == "GFCN":
    X = np.array(X_train_GF)
elif train=="OONICN":
    X = np.array(X_train_OONI)
    
explainer = shap.Explainer(pickle_model)
shap_values = explainer(X)

# for i in range(len(shap_values)):
#     value = shap_values[i]
#     print(value)
    



# visualize the first prediction's explanation
# shap.plots.waterfall(shap_values[0])
shap.plots.beeswarm(shap_values)
chosen_folder = XGB
index = 2
model = chosen_folder[index]

print(model)
pickle_model = pickle.load(open(model,"rb"))
train = model.split("/")[-1].split("_")[0]
if train == "GFCN":
    X = np.array(X_train_GF)
elif train=="OONICN":
    X = np.array(X_train_OONI)
explainer = shap.Explainer(pickle_model)
shap_values = explainer(X)

# for i in range(len(shap_values)):
#     value = shap_values[i]
#     print(value)
    



# visualize the first prediction's explanation
# shap.plots.waterfall(shap_values[0])
shap.plots.beeswarm(shap_values)
# # Set position of bar on X axis
import numpy as np
import matplotlib.pyplot as plt
fig = plt.subplots(figsize =(12, 8))
barWidth = 0.9
br1 = np.arange(len(dat[0]))
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]
br4 = [x + barWidth for x in br3]
br5 = [x + barWidth for x in br4]
br6 = [x + barWidth for x in br5]
# print(col1)
index = 0
# # Make the plot
plt.bar(br1, dat[index], color ='grey', width = barWidth,
        edgecolor ='black', label = train_test[index])
# plt.bar(br2, dat[1], color ='maroon', width = barWidth,
#         edgecolor ='grey', label = train_test[1])
# plt.bar(br3, dat[2], color ='magenta', width = barWidth,
#         edgecolor ='grey', label = train_test[2])
# plt.bar(br4, dat[3], color ='y', width = barWidth,
#         edgecolor ='grey', label = train_test[3])
# plt.bar(br5, dat[4], color ='cyan', width = barWidth,
#         edgecolor ='grey', label = train_test[4])
# plt.bar(br6, dat[5], color ='g', width = barWidth,
#         edgecolor ='grey', label = train_test[5])
 

    
# Adding Xticks
plt.xlabel('Feature', fontweight ='bold', fontsize = 15)
plt.ylabel("Importance score", fontweight ='bold', fontsize = 15)

plt.title("Feature importance of XGB for "+ train_test[index])
plt.xticks([5*i for i in range(20)],
        [5*i for i in range(20)])

 
plt.legend()
plt.show()




    
    X = np.array(X_train_GF)
elif train=="OONICN":
    X = np.array(X_train_OONI)
else:
    X = np.array(X_train_clean)
