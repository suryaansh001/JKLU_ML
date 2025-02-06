import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

file="indiaprice.csv"
df=pd.read_csv(file)
print(df.head())
print(df.info())
print(df.describe())

print(df.columns)