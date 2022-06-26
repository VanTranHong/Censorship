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
seed = 777
folder = "./ML_runs_results/temporal/not_include_induced/"
folder_model = "./ML_runs_models/temporal/not_include_induced/"
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
        params={"max_features":[15,25,45], "contamination":[0.001,0.003,0.005,0.007,0.009,0.01] }
        for n in params["contamination"]:
            for j in params["max_features"]:

                model = IsolationForest(random_state=0, max_features = j, contamination=n)
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
US = pd.read_csv("./data_after_preprocess/US_temp_encoded_sampled.csv")

US = replace_nan(US,"body_proportion")

US = US[US["body_proportion"]!= ""]
US = US.drop(columns = ["Unnamed: 0",'x_status0', 'x_status1', 'x_status2', 'x_status3',
       'x_status4','accessible0','accessible1'])
US.columns

CN = pd.read_csv("./data_after_preprocess/CN_temp_encoded.csv")

CN = replace_nan(CN,"blocking")
CN = replace_nan(CN,"GFWatchblocking_truth")
CN = replace_nan(CN,"body_proportion")

CN = pd.concat([CN[CN["blocking"]=='False'],CN[CN["blocking"]=='dns']])
CN = pd.concat([CN[CN["GFWatchblocking_truth"]==''],CN[CN["GFWatchblocking_truth"]=='Confirmed']])
CN = CN[CN["body_proportion"]!= ""]
CN = relabel(CN, "blocking","False")
CN = relabel(CN, "GFWatchblocking_truth","")
CN = CN.drop(columns = ["Unnamed: 0","Unnamed: 0.1",'x_status0', 'x_status1', 'x_status2', 'x_status3',
       'x_status4','accessible0','accessible1'])

benchmark = datetime.datetime(2021,6,20)
upper_benchmark = datetime.datetime(2021,7,1)
difference = upper_benchmark-benchmark
difference_in_s = difference.total_seconds()
### making the data start from 2021, July,1
CN=CN[CN["measurement_start_time"]>difference_in_s]
CN.to_csv("./data_after_preprocess/CN_temp_encoded.csv")
US.to_csv("./data_after_preprocess/US_temp_encoded_sampled.csv")

new_benchmark = datetime.datetime(2021,7,1)
upper_benchmark_7 = (datetime.datetime(2021,7,31) - new_benchmark).total_seconds()
upper_benchmark_8 = (datetime.datetime(2021,8,31)- new_benchmark).total_seconds()
upper_benchmark_9 = (datetime.datetime(2021,9,30)- new_benchmark).total_seconds()
upper_benchmark_10 = (datetime.datetime(2021,10,31)- new_benchmark).total_seconds()
upper_benchmark_11 = (datetime.datetime(2021,11,30)- new_benchmark).total_seconds()
upper_benchmark_12 = (datetime.datetime(2021,12,31)- new_benchmark).total_seconds()
upper_benchmark_1 = (datetime.datetime(2022,1,31)- new_benchmark).total_seconds()

index_7=[]
index_8=[]
index_9=[]
index_10=[]
index_11=[]
index_12=[]
index_1=[]

index = 0
for row in CN.iterrows():
    time  = row[1]["measurement_start_time"]
    if time> upper_benchmark_1:
        index_1.append(index)
    elif time> upper_benchmark_12:
        index_12.append(index)
    elif time> upper_benchmark_11:
        index_11.append(index)
    elif time> upper_benchmark_10:
        index_10.append(index)
    elif time> upper_benchmark_9:
        index_9.append(index)
    elif time> upper_benchmark_8:
        index_8.append(index)
    elif time> upper_benchmark_7:
        index_7.append(index)
    index +=1
month_7=CN.iloc[index_7]
month_8=CN.iloc[index_8]
month_9=CN.iloc[index_9]
month_10=CN.iloc[index_10]
month_11=CN.iloc[index_11]
month_12=CN.iloc[index_12]
month_1=CN.iloc[index_1]
month_7.to_csv("./ML_runs_models/temporal_data/CN_month_7.csv")
month_8.to_csv("./ML_runs_models/temporal_data/CN_month_8.csv")
month_9.to_csv("./ML_runs_models/temporal_data/CN_month_9.csv")
month_10.to_csv("./ML_runs_models/temporal_data/CN_month_10.csv")
month_11.to_csv("./ML_runs_models/temporal_data/CN_month_11.csv")
month_12.to_csv("./ML_runs_models/temporal_data/CN_month_12.csv")
month_1.to_csv("./ML_runs_models/temporal_data/CN_month_1.csv")
US = ["./ML_runs_models/temporal_data/US_month_7.csv","./ML_runs_models/temporal_data/US_month_8.csv","./ML_runs_models/temporal_data/US_month_9.csv","./ML_runs_models/temporal_data/US_month_10.csv","./ML_runs_models/temporal_data/US_month_11.csv","./ML_runs_models/temporal_data/US_month_12.csv","./ML_runs_models/temporal_data/US_month_1.csv"]
CN = ["./ML_runs_models/temporal_data/CN_month_7.csv","./ML_runs_models/temporal_data/CN_month_8.csv","./ML_runs_models/temporal_data/CN_month_9.csv","./ML_runs_models/temporal_data/CN_month_10.csv","./ML_runs_models/temporal_data/CN_month_11.csv","./ML_runs_models/temporal_data/CN_month_12.csv","./ML_runs_models/temporal_data/CN_month_1.csv"]

###### train on 1 month, test on 1 month for supervised
#### do it for CN data first
train_name = "CN"
train_dataset = CN
model_name = "XGB"
##### MODEL = OCSVM
models = get_model(model_name)

train = "GF"


train_month = 7
train_source = train_dataset[train_month-7]
train_df = pd.read_csv(train_source)
train_df = train_df.drop(columns = ["Unnamed: 0"])
X_ = train_df.drop(columns = ["blocking","GFWatchblocking_truth"])

if train == "OONI":
    y_ = train_df["blocking"]
else:
    y_ = train_df["GFWatchblocking_truth"]
        
X_train, X_val, y_train, y_val = train_test_split(X_,y_ , test_size=0.33, random_state = 1)
# X_train["Label"]=y_train
# X_clean_train = X_train[X_train["Label"]==0].drop(columns = ["Label"])

train_results = []

X_train = np.array(X_train)
y_train = np.array(y_train)
# X_train = np.array(X_clean_train)
X_val = np.array(X_val)
y_val = np.array(y_val)
for model in models:
    
    
    
    model.fit(X_train,y_train)
    predictions = model.predict(X_val)
    tp,fp,tn,fn = get_accuracy(np.array(predictions), np.array(y_val))
    train_results.append([tp,fp,tn,fn])

accuracy_score = accuracy(train_results)
sorted_acc_index = np.argsort(accuracy_score)
best_acc = accuracy_score[sorted_acc_index[-1]]
best_stats = train_results[sorted_acc_index[-1]]

best_model = models[sorted_acc_index[-1]]
print(best_acc)

folder_models = "./ML_runs_models/temporal/"+model_name+"/"
filename_1 = folder_models+"Train_month_"+str(train_month)+"_"+train_name+train+ "_censor_include.sav"
pickle.dump(best_model, open(filename_1, 'wb'))
print(filename_1)
test_dataset = CN

test_name = "CN"
test = "OONI"
test_results = []
test_months = [8,9,10,11,12,13]
for test_month in test_months:

 
    test_source = test_dataset[test_month-7]
    test_df = pd.read_csv(test_source)
    test_df = test_df.drop(columns = ["Unnamed: 0"])
    X_ = test_df.drop(columns = ["blocking","GFWatchblocking_truth"])
    
    if test == "OONI":
        y_ = test_df["blocking"]
    else:
        y_ = test_df["GFWatchblocking_truth"]
    predictions = best_model.predict(X_)
    tp,fp,tn,fn = get_accuracy(np.array(predictions), np.array(y_))
    test_results.append([tp,fp,tn,fn])
    
accuracy_score = accuracy(test_results)
print(accuracy_score)

df = pd.DataFrame()
folder = "./ML_runs_results/temporal/"+model_name+"/"

df_size = len(test_months)
df["Train month"] = [train_month for i in range(df_size)]
df["Test month"]=test_months
df["Train_acc"]=[best_acc for i in range(df_size)]

df["Train True Positive"]=[best_stats[0] for i in range(df_size) ]
df["Train False Positive"]=[best_stats[1] for i in range(df_size)]
df["Train True Negative"]=[best_stats[2] for i in range(df_size)]
df["Train False Negative"]=[best_stats[3] for i in range(df_size)]

df["Test_acc"]=accuracy_score

df["Test True Positive"]=[item[0] for item in test_results ]
df["Test False Positive"]=[item[1] for item in test_results]
df["Test True Negative"]=[item[2] for item in test_results]
df["Test False Negative"]=[item[3] for item in test_results]

df["Train set"] = [train for i in range(df_size)]
df["Test set"] = [test for i in range(df_size)]

df.to_csv(folder+"Train_month_"+str(train_month)+"_"+train+"_"+test+ "_"+train_name+"_"+test_name+"censor_include.csv")
