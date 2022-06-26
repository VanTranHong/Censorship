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

seed = 777

VERBOSITY = 6
rng = np.random.default_rng(seed)


#### Train clean data, validate using clean China data

def get_accuracy_unsupervised(predictions, y_test):


    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i in range(len(predictions)):
        if predictions[i]==1:
            if y_test[i]==0:
                tn+=1
            else:
                fn+=1
        else:
            if y_test[i]==1:
                tp +=1
            else:
                fp+=1        
    return tp,fp,tn,fn



def get_accuracy(predictions, y_test):

    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i in range(len(predictions)):
        if predictions[i]==1:
            if y_test[i]==1:
                tp+=1
            else:
                fp+=1
        else:
            if y_test[i]==0:
                tn +=1
            else:
                fn+=1            
    return tp,fp,tn,fn

    
def runSKFold(splits, X, y):

    runs = []
    skf = StratifiedKFold(n_splits=splits, random_state=0, shuffle=True)
    for train, test in skf.split(X, y):
#         print(train)
        X_train, X_test = X.iloc[train], X.iloc[test]
        y_train, y_test = y.iloc[train], y.iloc[test]
        arr = [X_train, X_test, y_train, y_test]
        runs.append(arr)
    return runs
def replace_nan(df, column):
    new_labels = []
    for val in df[column]:
        if pd.isna(val):
            new_labels.append("")
        else:
            new_labels.append(val)
    df[column]=new_labels
    return df

def relabel(df, col_name, base_cat):
    new_label = []
    for i in df[col_name]:
        if i==base_cat:
            new_label.append(0)
        else:
            new_label.append(1)
    df[col_name]=new_label
    return df

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
        params={"max_features":[15,30,50],"n_estimators":[100,300,700], "contamination":[0.001,0.0025,0.005,0.007] }
        index = 0
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
def accuracy(lst):
    accuracy =[]
    for item in lst:
        tp = item[0]
        fp = item[1]
        tn = item[2]
        fn = item[3]
        accuracy.append((tp+tn)/(tp+tn+fp+fn))
    return accuracy

def run_unsupervised(model_name,models,X_train,X_validation,y_validation,X_test,y_test):
    results = []
    for model in models:
        model.fit(X_train)
        predictions = model.predict(X_validation)
        tp,fp,tn,fn = get_accuracy_unsupervised(list(predictions), list(y_validation))
        results.append([tp,fp,tn,fn])
    
    accuracy_score = accuracy(results)
    print(accuracy_score)
    
    df_val = pd.DataFrame()
    df_val["Accuracy"] = accuracy_score
    df_val["model"] = [model_name for score in accuracy_score]
    df_val["val/test"]=["Validate" for score in accuracy_score]
    df_val["True Positive"]=[item[0] for item in results]
    df_val["False Positive"]=[item[1] for item in results]
    df_val["True Negative"]=[item[2] for item in results]
    df_val["False Negative"]=[item[3] for item in results]
    

    test_result = []
    sorted_acc_index = np.argsort(accuracy_score)
    best_model = models[sorted_acc_index[-1]]
    test_predictions = best_model.predict(X_test)
    tp,fp,tn,fn = get_accuracy_unsupervised(list(test_predictions), list(y_test))
    
    test_result.append([tp,fp,tn,fn])
    accuracy_score = accuracy(test_result)
    df_test = pd.DataFrame()
    df_test["Accuracy"] = accuracy_score
    df_test["model"] = [model_name for score in accuracy_score]
    df_test["val/test"]=["Test" for score in accuracy_score]
    df_test["True Positive"]=[item[0] for item in test_result]
    df_test["False Positive"]=[item[1] for item in test_result]
    df_test["True Negative"]=[item[2] for item in test_result]
    df_test["False Negative"]=[item[3] for item in test_result]
    df_final = pd.concat([df_val, df_test])
    return best_model, df_final
    
    
    
    
    
def run_supervised(model_name,models,X_train,y_train,X_validation,y_validation,X_test,y_test):
    results = []
    for model in models:
        model.fit(X_train, y_train)
        predictions = model.predict(X_validation)
        tp,fp,tn,fn = get_accuracy(np.array(predictions), y_validation)
        results.append([tp,fp,tn,fn])
    accuracy_score = accuracy(results)

    df_val = pd.DataFrame()
    df_val["Accuracy"] = accuracy_score
    df_val["model"] = [model_name for score in accuracy_score]
    df_val["val/test"]=["Validate" for score in accuracy_score]
    df_val["True Positive"]=[item[0] for item in results]
    df_val["False Positive"]=[item[1] for item in results]
    df_val["True Negative"]=[item[2] for item in results]
    df_val["False Negative"]=[item[3] for item in results]
    

    test_result = []
    sorted_acc_index = np.argsort(accuracy_score)
    best_model = models[sorted_acc_index[-1]]
    test_predictions = best_model.predict(np.array(X_test))
    tp,fp,tn,fn = get_accuracy(np.array(test_predictions), np.array(y_test))
    
    test_result.append([tp,fp,tn,fn])
    accuracy_score = accuracy(test_result)
    df_test = pd.DataFrame()
    df_test["Accuracy"] = accuracy_score
    df_test["model"] = [model_name for score in accuracy_score]
    df_test["val/test"]=["Test" for score in accuracy_score]
    df_test["True Positive"]=[item[0] for item in test_result]
    df_test["False Positive"]=[item[1] for item in test_result]
    df_test["True Negative"]=[item[2] for item in test_result]
    df_test["False Negative"]=[item[3] for item in test_result]
    df_final = pd.concat([df_val, df_test])
    return best_model, df_final
  
US = ["./ML_runs_models/temporal_data/US_month_7.csv","./ML_runs_models/temporal_data/US_month_8.csv","./ML_runs_models/temporal_data/US_month_9.csv","./ML_runs_models/temporal_data/US_month_10.csv","./ML_runs_models/temporal_data/US_month_11.csv","./ML_runs_models/temporal_data/US_month_12.csv","./ML_runs_models/temporal_data/US_month_1.csv"]
CN = ["./ML_runs_models/temporal_data/CN_month_7.csv","./ML_runs_models/temporal_data/CN_month_8.csv","./ML_runs_models/temporal_data/CN_month_9.csv","./ML_runs_models/temporal_data/CN_month_10.csv","./ML_runs_models/temporal_data/CN_month_11.csv","./ML_runs_models/temporal_data/CN_month_12.csv","./ML_runs_models/temporal_data/CN_month_1.csv"]

CN = pd.read_csv("./data_after_preprocess/CN_dataset_final.csv")

CN = CN.drop(columns=['Unnamed: 0'])

US = pd.read_csv("./data_after_preprocess/US_dataset_final.csv")
print(US.columns)
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
X_GFCN_Test = pd.concat([X_CNclean_test,X_GFCN_test])
y_GFCN_Test = pd.concat([y_CNclean_test,y_GFCN_test])

X_test_US = X_USclean_test
y_test_US = y_USclean_test

X_OONICN_Test = pd.concat([X_CNclean_test,X_OONICN_test])
y_OONICN_Test = pd.concat([y_CNclean_test,y_OONICN_test])

X_test_GFCN= np.array(X_GFCN_Test)
y_test_GFCN= np.array(y_GFCN_Test)
X_test_OONICN= np.array(X_OONICN_Test)
y_test_OONICN= np.array(y_OONICN_Test)



X_test_US= np.array(X_test_US)
y_test_US= np.array(y_test_US)
folder_model = "./ML_runs_models/results_not_include_induced_features/"+model_name+"/"
model = "Train_month_11_CNOONI_censor_include.sav"
test_dataset=CN

test = "GF"
model_path = folder_model+model

best_model = pickle.load(open(model_path, 'rb'))

test_results = []
for test_month in test_months:


    test_source = test_dataset[test_month-7]
    test_df = pd.read_csv(test_source)
    test_df = test_df.drop(columns = ["Unnamed: 0"])
    
    clean = test_df[test_df["blocking"]==0]
    clean= clean[clean["GFWatchblocking_truth"]==0]

    OONI = test_df[test_df["blocking"]==1]
    GF = test_df[test_df["GFWatchblocking_truth"] == 1]
#     X_ = test_df.drop(columns = ["blocking","GFWatchblocking_truth"])

    if test == "OONI":
        X_ = pd.concat([clean, OONI])
        y_ = X_["blocking"]
        X_ = X_.drop(columns = ["blocking","GFWatchblocking_truth"])
        
        
    elif test == "GF":
        X_ = pd.concat([clean, GF])
        y_ = X_["GFWatchblocking_truth"]
        X_ = X_.drop(columns = ["blocking","GFWatchblocking_truth"])
    else:
        X_ = clean
        y_ = X_["GFWatchblocking_truth"]
        X_ = X_.drop(columns = ["blocking","GFWatchblocking_truth"])
        
    predictions = best_model.predict(X_)
    tp,fp,tn,fn = get_accuracy(np.array(predictions), np.array(y_))
    test_results.append([tp,fp,tn,fn])




# predictions_US = best_model.predict(X_test_US)
# tp,fp,tn,fn = get_accuracy_unsupervised(np.array(predictions_US), np.array(y_test_US))
# test_results.append([tp,fp,tn,fn])


# predictions_GF = best_model.predict(X_test_GFCN)
# tp,fp,tn,fn = get_accuracy_unsupervised(np.array(predictions_GF), np.array(y_test_GFCN))
# test_results.append([tp,fp,tn,fn])

# predictions_OONI = best_model.predict(X_test_OONICN)
# tp,fp,tn,fn = get_accuracy_unsupervised(np.array(predictions_OONI), np.array(y_test_OONICN))
# test_results.append([tp,fp,tn,fn])




accuracy_score = accuracy(test_results)
print(accuracy_score)

df = pd.DataFrame()
folder = "./ML_runs_results/temporal/"+model_name+"/"
name = model.split(".")[0]+".csv"


df["Train month"] = [train_month for i in test_months]
df["Test month"]=test_months


df["Test_acc"]=accuracy_score

df["Test True Positive"]=[item[0] for item in test_results ]
df["Test False Positive"]=[item[1] for item in test_results]
df["Test True Negative"]=[item[2] for item in test_results]
df["Test False Negative"]=[item[3] for item in test_results]
df["Train type"] = [train for i in range(df.shape[0])]
df["Test type"] = [test for i in range(df.shape[0])]





# df["Train set"] = [train for i in range(df_size)]
# df["Test set"] = ["US","GFCN","OONICN"]
print(df)

df.to_csv(folder+"Train_month_"+str(train_month)+"_"+train+"_"+test+ "_"+train_name+"_"+test_name+".csv")

    
model_name="IF"
folder_model = "./ML_runs_models/results_not_include_induced_features/"+model_name+"/"
model = "USclean_USclean.sav"
model_path=folder_model+model
best_model = pickle.load(open(model_path, 'rb'))
test_result = []
test_predictions = best_model.predict(X_test_US)
tp,fp,tn,fn = get_accuracy_unsupervised(list(test_predictions), list(y_test_US))

test_result.append([tp,fp,tn,fn])
accuracy_score = accuracy(test_result)
df_test = pd.DataFrame()
df_test["Accuracy"] = accuracy_score
df_test["model"] = [model_name for score in accuracy_score]
df_test["val/test"]=["Test" for score in accuracy_score]
df_test["True Positive"]=[item[0] for item in test_result]
df_test["False Positive"]=[item[1] for item in test_result]
df_test["True Negative"]=[item[2] for item in test_result]
df_test["False Negative"]=[item[3] for item in test_result]
name = model.split(".")[0]+"_US.csv"
folder_result = "./ML_runs_results/results_not_include_induced_features/"+model_name+"/"
df_test.to_csv(folder_result+name)
print(df_test)

X_train = np.array(X_USclean_train)
y_train = np.array(y_USclean_train)
X_validation = np.array(X_USclean_validate)
y_validation = np.array(y_USclean_validate)
                        


results = []
model_name = "IF"
##### MODEL = OCSVM
models = get_model(model_name)
for model in models:
    model.fit(X_train)
    predictions = model.predict(X_validation)
    tp,fp,tn,fn = get_accuracy_unsupervised(list(predictions), list(y_validation))
    results.append([tp,fp,tn,fn])

accuracy_score = accuracy(results)
print(accuracy_score)

sorted_acc_index = np.argsort(accuracy_score)
best_model = models[sorted_acc_index[-1]]
pickle.dump(best_model, open(folder_model+"USclean_USclean.sav", 'wb'))
    
  
test_predictions = best_model.predict(np.array(X_test))
tp,fp,tn,fn = get_accuracy(np.array(test_predictions), np.array(y_test))

