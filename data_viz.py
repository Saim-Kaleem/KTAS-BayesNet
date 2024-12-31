import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('data_cleaned3.csv')
df = df.drop(columns=['Patients number per hour', 'Chief_complain', 'Diagnosis in ED'])

# top 10 correlated features
print(df.corr().unstack().sort_values().drop_duplicates().head(10))

# top 10 anti-correlated features
print(df.corr().unstack().sort_values().drop_duplicates().tail(10))

# plot the correlation matrix
plt.matshow(df.corr())
plt.show()