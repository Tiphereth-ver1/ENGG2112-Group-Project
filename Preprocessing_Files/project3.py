import datetime
import pandas as pd

file1 = "finalfile copy.csv"
file2 = "extrafeatures.csv"

# Read CSV safely
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2, skip_blank_lines=True)

# Optional: clean column names (this fixes hidden spaces or encoding artifacts)
df2.columns = df2.columns.str.strip()

# Check column names
print(df2.columns)

# Convert the timestamp column
df2['timestamp'] = pd.to_datetime(df2['timestamp'], format='%Y%m%dT%H%M')
df2 = df2.drop("timestamp", axis = 1)


print(df2.head())

df3 = pd.concat([df1,df2], axis = 1)
df3.to_csv("finalfinalfile.csv", index=False)
