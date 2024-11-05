import pandas as pd

df = pd.read_csv('papers.csv')

print(df.head())    
print(df.shape)
print(df.columns)
print(df.info())
print(df.describe())


