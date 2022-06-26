import pandas as pd
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
CN_name = pd.read_csv("./data_after_preprocess/CN_dataset_final.csv")
CN_name = CN_name.drop(columns = ['Unnamed: 0'])
CN = pd.read_csv("./data_after_preprocess/CN_temp_encoded.csv")
CN = CN.drop(columns=['Unnamed: 0'])
CN["Domain"] = CN_name["input"]
clean_CN = CN[CN["blocking"]==0]
clean_CN = clean_CN[clean_CN["GFWatchblocking_truth"]==0]

OONI_CN = CN[CN["blocking"]==1]
GF_CN = CN[CN["GFWatchblocking_truth"] == 1]
X_CNclean_train, X_CNclean_rest, y_CNclean_train, y_CNclean_rest = train_test_split(clean_CN,clean_CN["blocking"] , test_size=0.33)
X_CNclean_validate, X_CNclean_test, y_CNclean_validate, y_CNclean_test = train_test_split(X_CNclean_rest,y_CNclean_rest , test_size=0.33)


X_GFCN_train, X_GFCN_rest, y_GFCN_train, y_GFCN_rest = train_test_split(GF_CN,GF_CN["GFWatchblocking_truth"] , test_size=0.33)
X_GFCN_validate, X_GFCN_test, y_GFCN_validate, y_GFCN_test = train_test_split(X_GFCN_rest,y_GFCN_rest , test_size=0.33)

X_OONICN_train, X_OONICN_rest, y_OONICN_train, y_OONICN_rest = train_test_split(OONI_CN,OONI_CN["blocking"] , test_size=0.33)
X_OONICN_validate, X_OONICN_test, y_OONICN_validate, y_OONICN_test = train_test_split(X_OONICN_rest,y_OONICN_rest , test_size=0.33)



X_GFCN_Test = pd.concat([X_CNclean_test,X_GFCN_test])
y_GFCN_Test = pd.concat([y_CNclean_test,y_GFCN_test])



X_OONICN_Test = pd.concat([X_CNclean_test,X_OONICN_test])
y_OONICN_Test = pd.concat([y_CNclean_test,y_OONICN_test])



X_GFCN_Test_dup = X_GFCN_Test
X_OONICN_Test_dup = X_OONICN_Test

y_GFCN_Test_dup = y_GFCN_Test
y_OONICN_Test_dup = y_OONICN_Test



# X_GFCN_Test = X_GFCN_Test.drop(columns = ["index"])
# X_OONICN_Test = X_OONICN_Test.drop(columns = ["index"])

X_GFCN_Test = X_GFCN_Test.drop(columns=["Domain","blocking","GFWatchblocking_truth"])
X_OONICN_Test = X_OONICN_Test.drop(columns = ["Domain","blocking","GFWatchblocking_truth"])
X_test_GFCN= np.array(X_GFCN_Test)
y_test_GFCN= np.array(y_GFCN_Test)
X_test_OONICN= np.array(X_OONICN_Test)
y_test_OONICN= np.array(y_OONICN_Test)

print(X_test_GFCN.shape)

model_name="XGB"
folder_model = "./ML_runs_models/results_not_include_induced_features/"+model_name+"/"
model = "OONICN_OONICN.sav"
model_path=folder_model+model
best_model = pickle.load(open(model_path, 'rb'))


train_val = model.split(".")[0]
train = train_val.split("_")[0]
val = train_val.split("_")[1]




# test_result = []
test_predictions = best_model.predict(X_test_GFCN)
X_GFCN_Test_dup["XGB prediction"]=test_predictions

print(test_predictions)
# actual_prediction = []
# for i in test_predictions:
#     if i==-1:
#         actual_prediction.append(0)
#     else:
#         actual_prediction.append(1)
# X_GFCN_Test_dup["GF label"] = list(y_GFCN_Test)
# X_GFCN_Test_dup["prediction"] = actual_prediction
# X_GFCN_Test_dup["Domain"] = list((CN_name["input"][X_GFCN_Test_dup["index"]]))
X_GFCN_Test_dup.to_csv("./ML_runs_results/results_not_include_induced_features/"+model_name+"/" + train_val+"GF_prediction_include.csv")
        





test_predictions = best_model.predict(X_test_OONICN)
X_OONICN_Test_dup["XGB prediction"]=test_predictions
# actual_prediction = []
# for i in test_predictions:
#     if i==-1:
#         actual_prediction.append(0)
#     else:
#         actual_prediction.append(1)
# X_OONICN_Test_dup["OONI label"] = list(y_OONICN_Test)
# X_OONICN_Test_dup["prediction"] = actual_prediction
# X_OONICN_Test_dup["Domain"] = list((CN_name["input"][X_OONICN_Test_dup["index"]]))
X_OONICN_Test_dup.to_csv("./ML_runs_results/results_not_include_induced_features/"+model_name+"/" + train_val+"OONI_prediction_include.csv")
        





# tp,fp,tn,fn = get_accuracy_unsupervised(list(test_predictions), list(y_test_US))



# test_result.append([tp,fp,tn,fn])
# accuracy_score = accuracy(test_result)
# df_test = pd.DataFrame()
# df_test["Accuracy"] = accuracy_score
# df_test["model"] = [model_name for score in accuracy_score]
# df_test["val/test"]=["Test" for score in accuracy_score]
# df_test["True Positive"]=[item[0] for item in test_result]
# df_test["False Positive"]=[item[1] for item in test_result]
# df_test["True Negative"]=[item[2] for item in test_result]
# df_test["False Negative"]=[item[3] for item in test_result]
# name = model.split(".")[0]+"_US.csv"
# folder_result = "./ML_runs_results/results_not_include_induced_features/"+model_name+"/"
# df_test.to_csv(folder_result+name)
# print(df_test)

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


folder_rerun = "./Preprocess/rerun/CN"
month_7.to_csv(folder_rerun + "_month_7_with_domain_name.csv")
month_8.to_csv(folder_rerun + "_month_8_with_domain_name.csv")
month_9.to_csv(folder_rerun + "_month_9_with_domain_name.csv")
month_10.to_csv(folder_rerun + "_month_10_with_domain_name.csv")
month_11.to_csv(folder_rerun + "_month_11_with_domain_name.csv")
month_12.to_csv(folder_rerun + "_month_12_with_domain_name.csv")
month_1.to_csv(folder_rerun + "_month_1_with_domain_name.csv")

CN = ["_month_7_with_domain_name.csv","_month_8_with_domain_name.csv","_month_9_with_domain_name.csv","_month_10_with_domain_name.csv",
     "_month_11_with_domain_name.csv","_month_12_with_domain_name.csv","_month_1_with_domain_name.csv"]
CN = [folder_rerun + i for i in CN]

# model_name = "IF"
# folder_model = "./ML_runs_models/temporal/"+model_name+"/"
# model = "Accumulative_train_month_12_CNclean_censor_include.sav"
# model_path=folder_model+model
# best_model = pickle.load(open(model_path, 'rb'))

# folder_result = "./ML_runs_results/temporal/"+model_name+"/"
# name = model.split(".")[0]



# test_dataset = CN

# test_name = "CN"
# test = "GF"
# test_results = []
# test_months = [13]
# for test_month in test_months:

 
#     test_source = folder_rerun+ test_dataset[test_month-7]
#     test_df = pd.read_csv(test_source)
#     test_df_dup = test_df
#     test_df = test_df.drop(columns = ["Unnamed: 0","index","domain"])
#     X_ = test_df.drop(columns = ["blocking","GFWatchblocking_truth"])
    
#     if test == "OONI":
#         y_ = test_df["blocking"]
#     else:
#         y_ = test_df["GFWatchblocking_truth"]
#     predictions = best_model.predict(X_)
#     actual_prediction = []
#     for i in predictions:
#         if i==-1:
#             actual_prediction.append(0)
#         else:
#             actual_prediction.append(1)
#     test_df_dup["Prediction"]=actual_prediction
#     test_df_dup.to_csv(folder_result+name+"test_month_"+str(test_month)+test+".csv")
    
#     tp,fp,tn,fn = get_accuracy(np.array(predictions), np.array(y_))
#     test_results.append([tp,fp,tn,fn])
    
# accuracy_score = accuracy(test_results)
# print(accuracy_score)

m7 = pd.read_csv(CN[0])
m8 = pd.read_csv(CN[1])
m9 = pd.read_csv(CN[2])
m10 = pd.read_csv(CN[3])
m11 = pd.read_csv(CN[4])
m12 = pd.read_csv(CN[5])
m1 = pd.read_csv(CN[6])

m7_censor = m7[m7["blocking"]==1]
m7_censor = set(m7_censor["Domain"])
m7_uncensor = m7[m7["blocking"]==0]
m7_uncensor = set(m7_uncensor["Domain"])
print(len(m7_censor))
print(len(m7_uncensor))
print(len(m7_censor)/len(m7_uncensor))


m8_censor = m8[m8["blocking"]==1]
m8_censor = set(m8_censor["Domain"])
m8_uncensor = m8[m8["blocking"]==0]
m8_uncensor = set(m8_uncensor["Domain"])
print(len(m8_censor))
print(len(m8_uncensor))
print(len(m8_censor)/len(m8_uncensor))

m9_censor = m9[m9["blocking"]==1]
m9_censor = set(m9_censor["Domain"])
m9_uncensor = m9[m9["blocking"]==0]
m9_uncensor = set(m9_uncensor["Domain"])
print(len(m9_censor))
print(len(m9_uncensor))
print(len(m9_censor)/len(m9_uncensor))

m10_censor = m10[m10["blocking"]==1]
m10_censor = set(m10_censor["Domain"])
m10_uncensor = m10[m10["blocking"]==0]
m10_uncensor = set(m10_uncensor["Domain"])
print(len(m10_censor))
print(len(m10_uncensor))
print(len(m10_censor)/len(m10_uncensor))

m11_censor = m11[m11["blocking"]==1]
m11_censor = set(m11_censor["Domain"])
m11_uncensor = m11[m11["blocking"]==0]
m11_uncensor = set(m11_uncensor["Domain"])
print(len(m11_censor))
print(len(m11_uncensor))
print(len(m11_censor)/len(m11_uncensor))

m12_censor = m12[m12["blocking"]==1]
m12_censor = set(m12_censor["Domain"])
m12_uncensor = m12[m12["blocking"]==0]
m12_uncensor = set(m12_uncensor["Domain"])
print(len(m12_censor))
print(len(m12_uncensor))
print(len(m12_censor)/len(m12_uncensor))

m1_censor = m1[m1["blocking"]==1]
m1_censor = set(m1_censor["Domain"])
m1_uncensor = m1[m1["blocking"]==0]
m1_uncensor = set(m1_uncensor["Domain"])
print(len(m7_censor))
print(len(m1_uncensor))
print(len(m1_censor)/len(m1_uncensor))



Censor = [m7_censor,m8_censor,m9_censor,m10_censor,m11_censor,m12_censor,m1_censor]
Uncensor = [m7_uncensor,m8_uncensor,m9_uncensor,m10_uncensor,m11_uncensor,m12_uncensor,m1_uncensor]
df = pd.DataFrame()
censor_1 = []
censor_2 = []
censor_1_len = []
censor_2_len = []
intersect_1_2 = []

for i in range(len(Censor)-1):
    censor_month_1 = i+7
    censor_set_1 = Censor[i]
    for j in range(i+1, len(Censor)):
        censor_month_2 = j+7
        censor_set_2 = Censor[j]
        intersection = censor_set_1.intersection(censor_set_2)
        censor_1.append(censor_month_1)
        censor_2.append(censor_month_2)
        censor_1_len.append(len(censor_set_1))
        censor_2_len.append(len(censor_set_2))
        intersect_1_2.append(len(intersection))
df["Censor 1"]=censor_1
df["Censor 2"]=censor_2
df["Censor 1 length"] = censor_1_len
df["Censor 2 length"] = censor_2_len
df["Intersection"] = intersect_1_2
df["match percentage"] = [list(df["Intersection"])[i] / list(df["Censor 2 length"])[i] for i in range(df.shape[0])]
df = pd.DataFrame()
censor_1 = []
uncensor_2 = []
censor_1_len = []
uncensor_2_len = []
intersect_1_2 = []

for i in range(len(Censor)-1):
    censor_month_1 = i+7
    censor_set_1 = Censor[i]
    for j in range(i+1, len(Uncensor)):
        uncensor_month_2 = j+7
        uncensor_set_2 = Uncensor[j]
        intersection = censor_set_1.intersection(uncensor_set_2)
        censor_1.append(censor_month_1)
        uncensor_2.append(uncensor_month_2)
        censor_1_len.append(len(censor_set_1))
        uncensor_2_len.append(len(uncensor_set_2))
        intersect_1_2.append(len(intersection))
df["Censor 1"]=censor_1
df["Uncensor 2"]=uncensor_2
df["Censor 1 length"] = censor_1_len
df["Uncensor 2 length"] = uncensor_2_len
df["Intersection"] = intersect_1_2
df["match percentage"] = [list(df["Intersection"])[i] / list(df["Uncensor 2 length"])[i] for i in range(df.shape[0])]
df = pd.DataFrame()
uncensor_1 = []
censor_2 = []
uncensor_1_len = []
censor_2_len = []
intersect_1_2 = []
intersect_dataset = []
for i in range(len(Uncensor)-1):
    uncensor_month_1 = i+7
    uncensor_set_1 = Uncensor[i]
    for j in range(i+1, len(Censor)):

        censor_month_2 = j+7
        print("Uncensor month")
        print(uncensor_month_1)
        print(censor_month_2)
        print("Censor month")
        
        censor_set_2 = Censor[j]
        intersection = uncensor_set_1.intersection(censor_set_2)
        uncensor_1.append(uncensor_month_1)
        censor_2.append(censor_month_2)
        uncensor_1_len.append(len(uncensor_set_1))
        censor_2_len.append(len(censor_set_2))
        intersect_1_2.append(len(intersection))
        intersect_dataset.append(intersection)
        print(intersect_dataset)
        
df["Uncensor 1"]=uncensor_1
df["Censor 2"]=censor_2
df["Uncensor 1 length"] = uncensor_1_len
df["Censor 2 length"] = censor_2_len
df["Intersection"] = intersect_1_2
df["match percentage"] = [list(df["Intersection"])[i] / list(df["Censor 2 length"])[i] for i in range(df.shape[0])]
m7["Domain"] = [ name.split("/")[0] for name in m7["Domain"]]
m7_domains = set(m7["Domain"])

m8["Domain"] = [ name.split("/")[0] for name in m8["Domain"]]
m8_domains = set(m8["Domain"])

m9["Domain"] = [ name.split("/")[0] for name in m9["Domain"]]
m9_domains = set(m9["Domain"])

m10["Domain"] = [ name.split("/")[0] for name in m10["Domain"]]
m10_domains = set(m10["Domain"])

m11["Domain"] = [ name.split("/")[0] for name in m11["Domain"]]
m11_domains = set(m11["Domain"])

m12["Domain"] = [ name.split("/")[0] for name in m12["Domain"]]
m12_domains = set(m12["Domain"])

m1["Domain"] = [ name.split("/")[0] for name in m1["Domain"]]
m1_domains = set(m1["Domain"])
# m7_m8 = m7_domains.intersection(m8_domains)
# m7_m9 = m7_domains.intersection(m9_domains)
# m7_m10 = m7_domains.intersection(m10_domains)
# m7_m11 = m7_domains.intersection(m11_domains)
# m7_m12 =m7_domains.intersection(m12_domains)
# m7_m1 =m7_domains.intersection(m1_domains)



m8_m9 =  m8_domains.intersection(m9_domains)
m8_m10 = m8_domains.intersection(m10_domains)
m8_m11 = m8_domains.intersection(m11_domains)
m8_m12 =m8_domains.intersection(m12_domains)
m8_m1 =m8_domains.intersection(m1_domains)


m9_m10 = m9_domains.intersection(m10_domains)
m9_m11 =  m9_domains.intersection(m11_domains)
m9_m12 = m9_domains.intersection(m12_domains)
m9_m1 = m9_domains.intersection(m1_domains)

m10_m11 = m10_domains.intersection(m11_domains)
m10_m12 = m10_domains.intersection(m12_domains)
m10_m1 =m10_domains.intersection(m1_domains)

m11_m12 =m11_domains.intersection(m12_domains)
m11_m1 =m11_domains.intersection(m1_domains)

m12_m1 =m12_domains.intersection(m1_domains)

common_dataset = []
dataset = [m7,m8,m9,m10,m11,m12,m1]
for data in dataset:
    indexes = []
    index =0
    for row in data.iterrows():
        if row[1]["Domain"] in common_domains:
            indexes.append(index)
        index+=1
    dat_common = data.loc[indexes]
    common_dataset.append(dat_common)
    
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

def run_tests(model_name, train_df, train,   test_df,test):
    models = get_model(model_name)
    clean = train_df[train_df["blocking"]==0]
    clean= clean[clean["GFWatchblocking_truth"]==0]

    OONI = train_df[train_df["blocking"]==1]
    GF = train_df[train_df["GFWatchblocking_truth"] == 1]
    
    
    if train == "OONI":
        X_ = pd.concat([clean, OONI])
        y_ = X_["blocking"]
        X_ = X_.drop(columns = ["blocking","GFWatchblocking_truth"])
        
        
    elif train == "GF":
        X_ = pd.concat([clean, OONI])
        y_ = X_["GFWatchblocking_truth"]
        X_ = X_.drop(columns = ["blocking","GFWatchblocking_truth"])
    else:
        X_ = clean
        y_ = X_["GFWatchblocking_truth"]
        X_ = X_.drop(columns = ["blocking","GFWatchblocking_truth"])
        
    X_train, X_val, y_train, y_val = train_test_split(X_,y_ , test_size=0.33, random_state = 1)
    
    train_results = []

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_val = np.array(X_val)
    
    y_val = np.array(y_val)
    for model in models:
        if model_name == "XGB":
            model.fit(X_train, y_train)
            tp,fp,tn,fn = get_accuracy(np.array(predictions), np.array(y_val))
        else:
            model.fit(X_train)
            predictions = model.predict(X_val)
            tp,fp,tn,fn = get_accuracy_unsupervised(np.array(predictions), np.array(y_val))
        train_results.append([tp,fp,tn,fn])

    accuracy_score = accuracy(train_results)
    sorted_acc_index = np.argsort(accuracy_score)
    best_acc = accuracy_score[sorted_acc_index[-1]]
    best_stats = train_results[sorted_acc_index[-1]]

    best_model = models[sorted_acc_index[-1]]
    print(best_acc)
#     folder_models = "./ML_runs_models/temporal/"+model_name+"/"
#     filename_1 = folder_models+"Train_month_"+str(train_month)+"_"+train_name+train+ "_censor_include.sav"
#     pickle.dump(best_model, open(filename_1, 'wb'))
#     tests = ["GF","OONI"]
#     for test in tests:
#         test_results = []
#         for test_month in test_months:


#             test_source = test_dataset[test_month-7]
#             test_df = pd.read_csv(test_source)
#             test_df = test_df.drop(columns = ["Unnamed: 0"])
    clean = test_df[test_df["blocking"]==0]
    clean= clean[clean["GFWatchblocking_truth"]==0]

    OONI = test_df[test_df["blocking"]==1]
    GF = test_df[test_df["GFWatchblocking_truth"] == 1]

    if test == "OONI":
        y_ = test_df["blocking"]
    else:
        y_ = test_df["GFWatchblocking_truth"]
    predictions = best_model.predict(X_)
    tp,fp,tn,fn = get_accuracy(np.array(predictions), np.array(y_))
    test_results.append([tp,fp,tn,fn])

    accuracy_score = accuracy(test_results)
    print(accuracy_score)

#         df = pd.DataFrame()
#         folder = "./ML_runs_results/temporal/"+model_name+"/"

#         df_size = len(test_months)
#         df["Train month"] = [train_month for i in range(df_size)]
#         df["Test month"]=test_months
#         df["Train_acc"]=[best_acc for i in range(df_size)]

#         df["Train True Positive"]=[best_stats[0] for i in range(df_size) ]
#         df["Train False Positive"]=[best_stats[1] for i in range(df_size)]
#         df["Train True Negative"]=[best_stats[2] for i in range(df_size)]
#         df["Train False Negative"]=[best_stats[3] for i in range(df_size)]

#         df["Test_acc"]=accuracy_score

#         df["Test True Positive"]=[item[0] for item in test_results ]
#         df["Test False Positive"]=[item[1] for item in test_results]
#         df["Test True Negative"]=[item[2] for item in test_results]
#         df["Test False Negative"]=[item[3] for item in test_results]

#         df["Train set"] = [train for i in range(df_size)]
#         df["Test set"] = [test for i in range(df_size)]

#         df.to_csv(folder+"Train_month_"+str(train_month)+"_"+train+"_"+test+ "_"+train_name+"_"+test_name+"censor_include.csv")

