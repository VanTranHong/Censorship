import pandas as pd
import os
import glob

def get_statistic(tp,fp,tn,fn):
    tpr = 0
    fpr = 0
    tnr = 0
    fnr = 0
    if tp+fn>0:
        tpr = tp/(tp+fn)
        fnr = fn/(tp+fn)
    if fp+tn>0:
        fpr = fp/(fp+tn)
        tnr = tn/(fp+tn)
    return tpr,fpr,tnr,fnr

def reduce_df(df):
    df=df.drop(columns =["Unnamed: 0"],axis = 1)
    df["Index"]=[i for i in range(df.shape[0])]

    big_dict = {}
    for row in df.iterrows():
        index = row[1]["Index"]
        acc = row[1]["Accuracy"]
        model = row[1]["model"]
        test_type = row[1]["val/test"]
        train = row[1]["Train"]
        val = row[1]["Validate"]
        test = row[1]["Test"]
        if train not in big_dict.keys():
            big_dict[train]={}
        dict_1 = big_dict[train]
        if val not in dict_1.keys():
            dict_1[val]={}
        dict_2 = dict_1[val]
        if test not in dict_2.keys():
            dict_2[test]={}
        dict_3 = dict_2[test]
        if model not in dict_3.keys():
            dict_3[model]={}
        dict_4 = dict_3[model]
        if test_type not in dict_4.keys():
            dict_4[test_type]={}
        dict_5=dict_4[test_type]
        if acc not in dict_5.keys():
            dict_5[acc]=index
#     print(big_dict)

    indexes = []
    for key_1 in big_dict.keys():
        dict_1 = big_dict[key_1]
        for key_2 in dict_1.keys():
            dict_2 = dict_1[key_2]
            for key_3 in dict_2.keys():
                dict_3 = dict_2[key_3]
                for key_4 in dict_3.keys():
                    dict_4 = dict_3[key_4]
                    for key_5 in dict_4.keys():
                        dict_5 = dict_4[key_5]
                        max_score = max(list(dict_5.keys()))
                        index_to_keep = dict_5[max_score]
                        indexes.append(index_to_keep)
    df = df.iloc[indexes]
    return df
                    
                       
      
# for filename in glob.glob("./ML_runs_results/temporal/IF/*.csv"):
for filename in ["./ML_runs_results/results_not_include_induced_features/IF/USclean_USclean_US.csv"]:
    df = pd.read_csv(filename)
#     print(df)
#     print(filename)
#     df=df[df["val/test"]=="Test"]

    name = filename.split("/")[-1]
    print(name)
    
    TPR=[]
    FPR = []
    TNR =[]
    FNR = []
    
    if "True Positive" in df.columns:
        for row in df.iterrows():

            tp = row[1]["True Positive"]
            fp = row[1]["False Positive"]
            tn = row[1]["True Negative"]
            fn = row[1]["False Negative"]
            tpr,fpr,tnr,fnr = get_statistic(tp,fp,tn,fn)
            TPR.append(tpr)
            FPR.append(fpr)
            TNR.append(tnr)
            FNR.append(fnr)
    else:
        for row in df.iterrows():

            tp = row[1]["Test True Positive"]
            fp = row[1]["Test False Positive"]
            tn = row[1]["Test True Negative"]
            fn = row[1]["Test False Negative"]
            tpr,fpr,tnr,fnr = get_statistic(tp,fp,tn,fn)
            TPR.append(tpr)
            FPR.append(fpr)
            TNR.append(tnr)
            FNR.append(fnr)
        
    df["TPR"]=TPR
    df["FPR"]=FPR
    df["TNR"]=TNR
    df["FNR"]=FNR
#     df["Test"]=[""]
    
#     if "_1" in filename:
#         df["Sampling"]=[0.1 for i in range(df.shape[0])]
#     else:
    df["Sampling"]=[1 for i in range(df.shape[0])]
    for col in ["Unnamed: 0","Unnamed: 0.1","Unnamed: 0.2","Unnamed: 0.3"]:
        if col in df.columns:
            df = df.drop(columns = [col])
        
   
    print(df)
    df.to_csv(filename)
ls = []
for filename in glob.glob("./ML_runs_results/temporal/IF/Train*.csv"):
    df = pd.read_csv(filename)
#     name = filename.split("/")[-1].split(".")[0]
#     train = name.split("_")[0]
#     validate = name.split("_")[1]
#     df["Train"]=[train for i in range(df.shape[0])]
#     df["Validate"]=[validate for i in range(df.shape[0])]
    for col in ["Unnamed: 0","Unnamed: 0.1","Unnamed: 0.2","Unnamed: 0.3"]:
        if col in df.columns:
            df = df.drop(columns = [col])
    ls.append(df)
summary = pd.concat(ls)
print(summary)
summary.to_csv("./ML_runs_results/summary/IF/temporal_month.csv")

# lst = []
# summary_folder = "./ML_runs_results/summary/" 
# folder = "./ML_runs_results/"
# for filename in glob.glob("./ML_runs_results/*.csv"):
#     name = filename.split("/")[-1]
#     train = name.split("_")[0]
#     val = name.split("_")[1]


#     sampling_train = 1
#     sampling_validate = 1
#     if "0_1_0_1" in name:
#         sampling_train = 0.1
#         sampling_validate =0.1
#     elif "0_5_0_5" in name:
#         sampling_train = 0.5
#         sampling_validate = 0.5
#     elif "0_1" in name:
#         sampling_train = 0.1
#     elif "0_5" in name:
#         sampling_train = 0.5
        
    

#     df = pd.read_csv(filename)
#     df["Train"]=[train for i in range(df.shape[0])]
#     df["Validate"]=[val for i in range(df.shape[0])]

#     TPR=[]
#     FPR=[]
#     TNR=[]
#     FNR=[]

#     for row in df.iterrows():

#         tp = row[1]["True Positive"]
#         fp = row[1]["False Positive"]
#         tn = row[1]["True Negative"]
#         fn = row[1]["False Negative"]
#         tpr,fpr,tnr,fnr = get_statistic(tp,fp,tn,fn)
#         TPR.append(tpr)
#         FPR.append(fpr)
#         TNR.append(tnr)
#         FNR.append(fnr)
#     df["TPR"]=TPR
#     df["FPR"]=FPR
#     df["TNR"]=TNR
#     df["FNR"]=FNR

#     name = name.split(".")[0]+"_summary.csv"

#     df = reduce_df(df)
#     if 'Unnamed: 0.1' in df.columns:
#         df = df.drop(columns = ["Unnamed: 0.1"])
        
#     df["Train Sampling"] = [sampling_train for i in range(df.shape[0])]
#     df["Validate Sampling"] = [sampling_validate for i in range(df.shape[0])]
#     lst.append(df)
# master = pd.concat(lst)
# columns_select = ["Accuracy","model","val/test","True Positive","False Positive","True Negative","False Negative","Train",'Test', "Validate",'TPR',
#        'FPR', 'TNR', 'FNR', 'Index', 'Train Sampling', 'Validate Sampling']
# master = master[columns_select]
# master.to_csv(summary_folder+"SUMMARY.csv")
# print(master)

XGB_df = master[master["model"]=="XGB"]
OCSVM_df = master[master["model"]=="OCSVM"]
IF_df = master[master["model"]=="IF"]

print(OCSVM_df["Train"].unique())
OCSVM_original = OCSVM_df[OCSVM_df["Train Sampling"]==1.0]
OCSVM_original = OCSVM_original[OCSVM_original["Validate Sampling"]==1.0]
print(OCSVM_original)
lst = []
for train in OCSVM_original["Train"].unique():
    small_df = OCSVM_original[OCSVM_original["Train"]==train]
    for validate in small_df["Validate"].unique():
        s_s_df = small_df[small_df["Validate"]==validate]
        for test in s_s_df["Test"].unique():
            df = pd.DataFrame()
            smallest_df = s_s_df[s_s_df["Test"]==test]
            Val = smallest_df[smallest_df["val/test"]=="Validate"]
            Test = smallest_df[smallest_df["val/test"]=="Test"]
            df["Val Acc"]=[Val.iloc[0]["Accuracy"]]
            df["Val True Positive"]=[Val.iloc[0]["True Positive"]]
            df["Val False Positive"]=[Val.iloc[0]["False Positive"]]
            df["Val True Negative"]=[Val.iloc[0]["True Negative"]]
            df["Val False Negative"]=[Val.iloc[0]["False Negative"]]
            
            df["Val TPR"]=[Val.iloc[0]["TPR"]]
            df["Val FPR"]=[Val.iloc[0]["FPR"]]
            df["Val TNR"]=[Val.iloc[0]["TNR"]]
            df["Val FNR"]=[Val.iloc[0]["FNR"]]
            
            df["Test Acc"]=[Test.iloc[0]["Accuracy"]]
            df["Test True Positive"]=[Test.iloc[0]["True Positive"]]
            df["Test False Positive"]=[Test.iloc[0]["False Positive"]]
            df["Test True Negative"]=[Test.iloc[0]["True Negative"]]
            df["Test False Negative"]=[Test.iloc[0]["False Negative"]]
            
            df["Test TPR"]=[Test.iloc[0]["TPR"]]
            df["Test FPR"]=[Test.iloc[0]["FPR"]]
            df["Test TNR"]=[Test.iloc[0]["TNR"]]
            df["Test FNR"]=[Test.iloc[0]["FNR"]]
            
            df["Classifier"]=[Test.iloc[0]["model"]]
            df["Train"]=[Test.iloc[0]["Train"]]
            df["Test"]=[Test.iloc[0]["Test"]]
            df["Validate"]=[Test.iloc[0]["Validate"]]
            
            
            
#             print(df)
            lst.append(df)
master = pd.concat(lst)
columns_keep = ["Classifier","Train","Validate","Test","Test True Positive","Test False Positive","Test True Negative","Test False Negative"]
sub = master[columns_keep]
sub.to_csv("./ML_runs_results/summary/IF_full_susmmary.csv")
Train_Test=[]
Train_Validate=[]
Validate_Test=[]
for row in master.iterrows():
    train = row[1]["Train"]
    validate = row[1]["Validate"]
    test = row[1]["Test"]

    Train_Test.append(train+"_"+test)
    Train_Validate.append(train+"_"+validate)
    Validate_Test.append(validate+"_"+test)
master["Train_Test"] =Train_Test
master["Train_Validate"] =Train_Validate
master["Validate_Test"] =Validate_Test

train_val = list(master["Train_Validate"].unique())
train_val.sort()
clean_US = master[master["Test"]=="CleanUS"]
GF  = master[master["Test"]=="GF_CN"]
OONI  = master[master["Test"]=="OONI_CN"]

clean_US_tpr = []
clean_US_fpr = []
clean_US_tnr = []
clean_US_fnr = []
clean_US_acc = []

GF_tpr = []
GF_fpr = []
GF_tnr = []
GF_fnr = []
GF_acc = []

OONI_tpr = []
OONI_fpr = []
OONI_tnr = []
OONI_fnr = []
OONI_acc = []

for tv in train_val:
    if tv in clean_US["Train_Validate"].unique():
        clean_US_tpr.append(clean_US[clean_US["Train_Validate"]==tv].iloc[0]["Test TPR"])
        clean_US_fpr.append(clean_US[clean_US["Train_Validate"]==tv].iloc[0]["Test FPR"])
        clean_US_tnr.append(clean_US[clean_US["Train_Validate"]==tv].iloc[0]["Test TNR"])
        clean_US_fnr.append(clean_US[clean_US["Train_Validate"]==tv].iloc[0]["Test FNR"])
        clean_US_acc.append(clean_US[clean_US["Train_Validate"]==tv].iloc[0]["Test Acc"])
    else:
        clean_US_tpr.append(0)
        clean_US_fpr.append(0)
        clean_US_tnr.append(0)
        clean_US_fnr.append(0)
        clean_US_acc.append(0)
    if tv in GF["Train_Validate"].unique():
        GF_tpr.append(GF[GF["Train_Validate"]==tv].iloc[0]["Test TPR"])
        GF_fpr.append(GF[GF["Train_Validate"]==tv].iloc[0]["Test FPR"])
        GF_tnr.append(GF[GF["Train_Validate"]==tv].iloc[0]["Test TNR"])
        GF_fnr.append(GF[GF["Train_Validate"]==tv].iloc[0]["Test FNR"])
        GF_acc.append(GF[GF["Train_Validate"]==tv].iloc[0]["Test Acc"])
    else:
        GF_tpr.append(0)
        GF_fpr.append(0)
        GF_tnr.append(0)
        GF_fnr.append(0)
        GF_acc.append(0)
    if tv in OONI["Train_Validate"].unique():
        OONI_tpr.append(OONI[OONI["Train_Validate"]==tv].iloc[0]["Test TPR"])
        OONI_fpr.append(OONI[OONI["Train_Validate"]==tv].iloc[0]["Test FPR"])
        OONI_tnr.append(OONI[OONI["Train_Validate"]==tv].iloc[0]["Test TNR"])
        OONI_fnr.append(OONI[OONI["Train_Validate"]==tv].iloc[0]["Test FNR"])
        OONI_acc.append(OONI[OONI["Train_Validate"]==tv].iloc[0]["Test Acc"])
    else:
        OONI_tpr.append(0)
        OONI_fpr.append(0)
        OONI_tnr.append(0)
        OONI_fnr.append(0)
        OONI_acc.append(0)

import numpy as np
import matplotlib.pyplot as plt
 
# set width of bar
barWidth = 0.2
fig = plt.subplots(figsize =(12, 8))
 

 
# Set position of bar on X axis
br1 = np.arange(len(OONI_tpr))
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]
br4 = [x + barWidth for x in br3]
br5 = [x + barWidth for x in br4]
br6 = [x + barWidth for x in br5]
 
# Make the plot
# plt.bar(br1, clean_US_val, color ='r', width = barWidth,
#         edgecolor ='grey', label =train_val[0])
plt.bar(br1, clean_US_tnr, color ='maroon', width = barWidth,
        edgecolor ='grey', label ="Clean US")
# plt.bar(br3, GF_val, color ='magenta', width = barWidth,
#         edgecolor ='grey', label =train_val[2])
plt.bar(br2, GF_tnr, color ='y', width = barWidth,
        edgecolor ='grey', label ="GF China")
# plt.bar(br5, OONI_val, color ='cyan', width = barWidth,
#         edgecolor ='grey', label =train_val[4])
plt.bar(br3, OONI_tnr, color ='g', width = barWidth,
        edgecolor ='grey', label ="OONI China")
 
# Adding Xticks
plt.xlabel('Train and Validation Combination', fontweight ='bold', fontsize = 15)
plt.ylabel('True Negative Rate', fontweight ='bold', fontsize = 15)
plt.xticks([r + barWidth for r in range(len(OONI_tpr))],
        train_val, rotation=45)
plt.title("True Negative Rate for OCSVM classifier, performance on different test sets")

 
plt.legend()
plt.show()

#### Plotting temporal data
temporal_folder = "./ML_runs_results/temporal/"
classifier="IF/"
ls = []

for filename in glob.glob(temporal_folder + classifier+"*.csv"):
    df = pd.read_csv(filename)
    if "CN_US" in filename:
        df["Train_Test_country"]=["CN_US" for i in range(df.shape[0])]

    elif "US_CN" in filename:
        df["Train_Test_country"]=["US_CN" for i in range(df.shape[0])]
    else:
        df["Train_Test_country"]=["CN_CN" for i in range(df.shape[0])]
        
        
    ls.append(df)
master = pd.concat(ls)
master = master.drop(columns = ["Unnamed: 0"])
Tpr = []
Fpr= []
Tnr = []
Fnr = []
for row in master.iterrows():
    
    tp = row[1]["Test True Positive"]
    fp = row[1]["Test False Positive"]
    tn = row[1]["Test True Negative"]
    fn = row[1]["Test False Negative"]
    tpr,fpr,tnr,fnr = get_statistic(tp,fp,tn,fn)
    Tpr.append(tpr)
    Fpr.append(fpr)
    Tnr.append(tnr)
    Fnr.append(fnr)
master["Test TPR"] = Tpr
master["Test FPR"] = Fpr
master["Test TNR"] = Tnr
master["Test FNR"] = Fnr
CN_US = master[master["Train_Test_country"] == "CN_US"]
CN_CN = master[master["Train_Test_country"] == "CN_CN"]
US_CN = master[master["Train_Test_country"] == "US_CN"]
    
master = CN_CN
Train_Test=[]

for row in master.iterrows():
    train = row[1]["Train set"]
    test = row[1]["Test set"]

    Train_Test.append(train+"_"+test)

master["Train_Test"] =Train_Test

train_val = list(master["Train_Test"].unique())
train_val.sort()






# clean_US_tpr = []
# clean_US_fpr = []
# clean_US_tnr = []
# clean_US_fnr = []
# clean_US_acc = []

# GF_tpr = []
# GF_fpr = []
# GF_tnr = []
# GF_fnr = []
# GF_acc = []

# OONI_tpr = []
# OONI_fpr = []
# OONI_tnr = []
# OONI_fnr = []
# OONI_acc = []

tv = "OONI_OONI"
Acc= []
Tpr = []
Fpr = []
Tnr = []
Fnr = []
small_df = master[master["Train_Test"] == tv]
for i in range(7,13):
    acc = []
    tpr = []
    fpr = []
    tnr = []
    fnr = []
    s_s_df = small_df[small_df["Train month"]==i]
    for j in range(8,14):
        if j in s_s_df["Test month"].unique():
            acc.append(s_s_df[s_s_df["Test month"]==j].iloc[0]["Test_acc"])
            tpr.append(s_s_df[s_s_df["Test month"]==j].iloc[0]["Test TPR"])
            fpr.append(s_s_df[s_s_df["Test month"]==j].iloc[0]["Test FPR"])
            tnr.append(s_s_df[s_s_df["Test month"]==j].iloc[0]["Test TNR"])
            fnr.append(s_s_df[s_s_df["Test month"]==j].iloc[0]["Test FNR"])
        else:
            acc.append(0)
            tpr.append(0)
            fpr.append(0)
            tnr.append(0)
            fnr.append(0)
    Acc.append(acc)
    Tpr.append(tpr)
    Fpr.append(fpr)
    Tnr.append(tnr)
    Fnr.append(fnr)


# set width of bar
barWidth = 0.1
fig = plt.subplots(figsize =(12, 8))

rearrange_dat = Acc
col1 = [item[0] for item in rearrange_dat]
col2 = [item[1] for item in rearrange_dat]
col3 = [item[2] for item in rearrange_dat]
col4 = [item[3] for item in rearrange_dat]
col5 = [item[4] for item in rearrange_dat]
col6 = [item[5] for item in rearrange_dat]
 

 
# # Set position of bar on X axis
br1 = np.arange(len(col1))
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]
br4 = [x + barWidth for x in br3]
br5 = [x + barWidth for x in br4]
br6 = [x + barWidth for x in br5]
# print(col1)
 
# # Make the plot
plt.bar(br1, col1, color ='r', width = barWidth,
        edgecolor ='grey', label = "Test month 8")
plt.bar(br2, col2, color ='maroon', width = barWidth,
        edgecolor ='grey', label ="Test month 9")
plt.bar(br3, col3, color ='magenta', width = barWidth,
        edgecolor ='grey', label ="Test month 10")
plt.bar(br4, col4, color ='y', width = barWidth,
        edgecolor ='grey', label ="Test month 11")
plt.bar(br5, col5, color ='cyan', width = barWidth,
        edgecolor ='grey', label ="Test month 12")
plt.bar(br6, col6, color ='g', width = barWidth,
        edgecolor ='grey', label ="Test month 1/2022")
 
metrics = "Accuracy"
    
# Adding Xticks
plt.xlabel('Train month', fontweight ='bold', fontsize = 15)
plt.ylabel(metrics, fontweight ='bold', fontsize = 15)
plt.xticks([r + barWidth for r in range(len(col1))],
        [7,8,9,10,11,12])
plt.title(metrics +" for IF classifier, train data is OONI of China, test data is OONI of China")

 
plt.legend()
plt.show()



        
                
                
                
                
                
#     if tv in clean_US["Train_Validate"].unique():
#         clean_US_tpr.append(clean_US[clean_US["Train_Validate"]==tv].iloc[0]["Test TPR"])
#         clean_US_fpr.append(clean_US[clean_US["Train_Validate"]==tv].iloc[0]["Test FPR"])
#         clean_US_tnr.append(clean_US[clean_US["Train_Validate"]==tv].iloc[0]["Test TNR"])
#         clean_US_fnr.append(clean_US[clean_US["Train_Validate"]==tv].iloc[0]["Test FNR"])
#         clean_US_acc.append(clean_US[clean_US["Train_Validate"]==tv].iloc[0]["Test Acc"])
#     else:
#         clean_US_tpr.append(0)
#         clean_US_fpr.append(0)
#         clean_US_tnr.append(0)
#         clean_US_fnr.append(0)
#         clean_US_acc.append(0)
#     if tv in GF["Train_Validate"].unique():
#         GF_tpr.append(GF[GF["Train_Validate"]==tv].iloc[0]["Test TPR"])
#         GF_fpr.append(GF[GF["Train_Validate"]==tv].iloc[0]["Test FPR"])
#         GF_tnr.append(GF[GF["Train_Validate"]==tv].iloc[0]["Test TNR"])
#         GF_fnr.append(GF[GF["Train_Validate"]==tv].iloc[0]["Test FNR"])
#         GF_acc.append(GF[GF["Train_Validate"]==tv].iloc[0]["Test Acc"])
#     else:
#         GF_tpr.append(0)
#         GF_fpr.append(0)
#         GF_tnr.append(0)
#         GF_fnr.append(0)
#         GF_acc.append(0)
#     if tv in OONI["Train_Validate"].unique():
#         OONI_tpr.append(OONI[OONI["Train_Validate"]==tv].iloc[0]["Test TPR"])
#         OONI_fpr.append(OONI[OONI["Train_Validate"]==tv].iloc[0]["Test FPR"])
#         OONI_tnr.append(OONI[OONI["Train_Validate"]==tv].iloc[0]["Test TNR"])
#         OONI_fnr.append(OONI[OONI["Train_Validate"]==tv].iloc[0]["Test FNR"])
#         OONI_acc.append(OONI[OONI["Train_Validate"]==tv].iloc[0]["Test Acc"])
#     else:
#         OONI_tpr.append(0)
#         OONI_fpr.append(0)
#         OONI_tnr.append(0)
#         OONI_fnr.append(0)
#         OONI_acc.append(0)

