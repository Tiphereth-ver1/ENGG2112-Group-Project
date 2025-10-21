import pandas as pd

df1 = pd.read_csv("final_processed_weather.csv")
df1 = df1.drop("SETTLEMENTDATE", axis = 1)
df1.to_csv("final_processed_weather1.csv", index = False)

df2 = pd.read_csv("sample.csv")
df2 = df2.drop("PERIODTYPE", axis = 1)

df2.to_csv("sample2.csv", index = False)

df3 = pd.concat([df2,df1], axis = 1)
df3.to_csv("finalfile.csv", index = False)
