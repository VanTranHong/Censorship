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
seed = 777
folder = "./ML_runs_results/"
folder_model = "./ML_runs_models/"
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
  
  
CN = pd.read_csv("./data_after_preprocess/CN_temp_encoded.csv")
CN = CN.drop(columns=['Unnamed: 0'])
CN.shape
# CN = replace_nan(CN,"blocking")
# CN = replace_nan(CN,"GFWatchblocking_truth")
# CN = replace_nan(CN,"body_proportion")

# CN = pd.concat([CN[CN["blocking"]=='False'],CN[CN["blocking"]=='dns']])
# CN = pd.concat([CN[CN["GFWatchblocking_truth"]==''],CN[CN["GFWatchblocking_truth"]=='Confirmed']])
# CN = CN[CN["body_proportion"]!= ""]
# CN = relabel(CN, "blocking","False")
# CN = relabel(CN, "GFWatchblocking_truth","")

# CN = CN.drop(columns = ["Unnamed: 0","Unnamed: 0.1",'x_status0', 'x_status1', 'x_status2', 'x_status3',
#        'x_status4','accessible0','accessible1'])

# columns = list(CN.columns)
# index = [i for i in range(len(columns))]
# df = pd.DataFrame()
# df["Index"] = index
# df["Column Name"]=columns

# df.to_csv("./data_after_preprocess/columns_names.csv")
# benchmark = datetime.datetime(2021,6,20)
# upper_benchmark = datetime.datetime(2021,7,1)
# difference = upper_benchmark-benchmark
# difference_in_s = difference.total_seconds()
# ### making the data start from 2021, July,1
# CN=CN[CN["measurement_start_time"]>difference_in_s]

US = pd.read_csv("./data_after_preprocess/US_temp_encoded.csv")
print(US.shape)
US = US.drop(columns=["Unnamed: 0"])
# US = replace_nan(US,"blocking")
US.shape

US = replace_nan(US,"body_proportion")

US = US[US["body_proportion"]!= ""]
US = relabel(US, "blocking","False")
US = relabel(US, "GFWatchblocking_truth","")
print(US.columns)
US = pd.read_csv("US_temp_encoded_sampled.csv")
print(US.shape)

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

##### TRAIN: clean CHINA, VALIDATE: GF-CHINA
##### TEST: GF-CHINA, CLEAN:US, OONI-CHINA
X_train = X_CNclean_train
y_train = y_CNclean_train

X_validate = pd.concat([X_CNclean_validate,X_GFCN_validate])
y_validate = pd.concat([y_CNclean_validate,y_GFCN_validate])


X_GFCN_Test = pd.concat([X_CNclean_test,X_GFCN_test])
y_GFCN_Test = pd.concat([y_CNclean_test,y_GFCN_test])

# X_test_US = X_USclean_test
# y_test_US = y_USclean_test

X_OONICN_Test = pd.concat([X_CNclean_test,X_OONICN_test])
y_OONICN_Test = pd.concat([y_CNclean_test,y_OONICN_test])


X_train = np.array(X_train)
y_train= np.array(y_train)

X_validate= np.array(X_validate)
y_validate= np.array(y_validate)
X_test_GFCN= np.array(X_GFCN_Test)
y_test_GFCN= np.array(y_GFCN_Test)
X_test_OONICN= np.array(X_OONICN_Test)
y_test_OONICN= np.array(y_OONICN_Test)



# X_test_US= np.array(X_test_US)
# y_test_US= np.array(y_test_US)
# print(X_test_US.shape)
# print(X_test_OONICN.shape)

import pickle
model_name = "OCSVM"
##### MODEL = OCSVM
models = get_model(model_name)
folder = "./ML_runs_results/results_not_include_induced_features/"
folder_model = "./ML_runs_models/results_not_include_induced_features/"

model_1, result_1 = run_unsupervised(model_name,models,X_train,X_validate,y_validate,X_test_GFCN,y_test_GFCN)
model_2, result_2 = run_unsupervised(model_name,models,X_train,X_validate,y_validate,X_test_OONICN,y_test_OONICN)
model_1, result_3 = run_unsupervised(model_name,models,X_train,X_validate,y_validate,X_test_US,y_test_US)

result_1["Train"]=["CleanCN" for i in range(result_1.shape[0])]
result_1["Validate"]=["GF_CN" for i in range(result_1.shape[0])]
result_1["Test"]=["GF_CN" for i in range(result_1.shape[0])]

result_2["Train"]=["CleanCN" for i in range(result_2.shape[0])]
result_2["Validate"]=["GF_CN" for i in range(result_2.shape[0])]
result_2["Test"]=["OONI_CN" for i in range(result_2.shape[0])]

result_3["Train"]=["CleanCN" for i in range(result_3.shape[0])]
result_3["Validate"]=["GF_CN" for i in range(result_3.shape[0])]
result_3["Test"]=["CleanUS" for i in range(result_3.shape[0])]

master = pd.concat([result_1, result_2, result_3])



master.to_csv(folder+model_name+"/CNclean_GFCN.csv")
filename_1 = folder_model+model_name+"/CNclean_GFCN.sav"
pickle.dump(model_1, open(filename_1, 'wb'))


