import os
import pandas as pd

base_path = os.path.dirname(os.path.abspath(__file__))  # folder where project.py is located
folder_path = os.path.join(base_path, "extradata")

#Preprocessing and joining of all 
df = pd.DataFrame({"REGION": [],"SETTLEMENTDATE":[],"TOTALDEMAND":[],"RRP":[],"PERIODTYPE":[]})
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    root, extension = os.path.splitext(file_path)
    if extension == ".csv":
        dfe = pd.read_csv(file_path)
        df = pd.concat([df, dfe])

df["SETTLEMENTDATE"] = pd.to_datetime(df["SETTLEMENTDATE"])
df_hourly = df[df["SETTLEMENTDATE"].dt.minute.eq(0) & df["SETTLEMENTDATE"].dt.second.eq(0)]
df_hourly.to_csv("sample.csv", index = False)

df1 = pd.read_csv("data_2024-2025.csv", skip_blank_lines=True)

df1["SETTLEMENTDATE"] = pd.to_datetime(
    df1[["YEAR", "MO", "DY", "HR"]]
    .astype(str)
    .agg("-".join, axis=1),
    format="%Y-%m-%d-%H"
)

df1["SETTLEMENTDATE"] = df1["SETTLEMENTDATE"].dt.strftime("%Y/%m/%d %H:%M:%S")


df0 = df1[[
    "SETTLEMENTDATE",
    "ALLSKY_SFC_SW_DWN",
    "T2M",
    "T2MDEW",
    "T2MWET",
    "RH2M",
    "PS",
    "WS2M"
]]

df0.to_csv("processed_weather.csv", index=False)
print(df0.head())

# 2021/06/12 16:00:00

# YEAR,MO,DY,HR,ALLSKY_SFC_SW_DWN,T2M,T2MDEW,T2MWET,RH2M,PS,WS2M
# 2018,1,1,0,0.0,21.54,18.87,20.2,84.69,100.31,3.58