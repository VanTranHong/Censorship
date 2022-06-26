import pandas as pd
import datetime
US = ["./ML_runs_models/temporal_data/US_month_7_with_domain_name.csv","./ML_runs_models/temporal_data/US_month_8_with_domain_name.csv",
     "./ML_runs_models/temporal_data/US_month_9_with_domain_name.csv","./ML_runs_models/temporal_data/US_month_10_with_domain_name.csv",
      "./ML_runs_models/temporal_data/US_month_11_with_domain_name.csv","./ML_runs_models/temporal_data/US_month_12_with_domain_name.csv",
      "./ML_runs_models/temporal_data/US_month_1_with_domain_name.csv"
     ]

month_7=pd.read_csv(US[0])
month_7 = month_7.drop(columns = ["Unnamed: 0"])
month_7 = month_7.drop(columns = ['accessible0',
       'accessible1', 'x_status0', 'x_status1', 'x_status2', 'x_status3',
       'x_status4'])
feature_columns = list(month_7.drop(columns = ['blocking', 'GFWatchblocking_truth']).columns)
print(feature_columns)
df = pd.DataFrame(feature_columns, columns = ["name"])
df["Index"]=[i for i in range(df.shape[0])]
df.to_csv("features_encoded.csv")
CN = pd.read_csv("./Preprocess/CN_temp_encoded_sanitized.csv")
CN = CN.drop(columns = ['Unnamed: 0.3', 'Unnamed: 0.2', 'Unnamed: 0.1', 'Unnamed: 0'])
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
US = pd.read_csv("./Preprocess/US_temp_encoded_sanitized_sampled.csv")
US = US.drop(columns = [ 'Unnamed: 0.2', 'Unnamed: 0.1', 'Unnamed: 0'])


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
for row in US.iterrows():
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
month_7=US.iloc[index_7]
month_8=US.iloc[index_8]
month_9=US.iloc[index_9]
month_10=US.iloc[index_10]
month_11=US.iloc[index_11]
month_12=US.iloc[index_12]
month_1=US.iloc[index_1]
folder_rerun = "./Preprocess/rerun/US"
month_7.to_csv(folder_rerun + "_month_7_with_domain_name.csv")
month_8.to_csv(folder_rerun + "_month_8_with_domain_name.csv")
month_9.to_csv(folder_rerun + "_month_9_with_domain_name.csv")
month_10.to_csv(folder_rerun + "_month_10_with_domain_name.csv")
month_11.to_csv(folder_rerun + "_month_11_with_domain_name.csv")
month_12.to_csv(folder_rerun + "_month_12_with_domain_name.csv")
month_1.to_csv(folder_rerun + "_month_1_with_domain_name.csv")
