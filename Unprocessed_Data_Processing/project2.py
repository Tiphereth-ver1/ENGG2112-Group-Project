import pandas as pd

df1 = pd.read_csv("processed_weather_2024-2025.csv")
df1 = df1.drop("SETTLEMENTDATE", axis = 1)

df2 = pd.read_csv("sample.csv")
df2 = df2.drop("PERIODTYPE", axis = 1)

df2.to_csv("sample2.csv", index = False)

df3 = pd.concat([df2,df1], axis = 1)
df3.to_csv("finalfile.csv", index = False)
