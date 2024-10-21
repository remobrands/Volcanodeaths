import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("volcano-events-2024-10-12_16-24-07_+0200.tsv", sep='\t')
print(data.head())

plt.plot(data['Year'], data['Deaths'])
plt.xlabel('Year')
plt.ylabel('Deaths')
plt.title('Volcanic Eruption Deaths Over Time')
plt.show()

def filter_data_by_year(dataframe, min_year):
    return dataframe[dataframe['Year'] >= min_year]

data_filtered = filter_data_by_year(data, 1630)
print(data_filtered.head())

plt.plot(data_filtered['Year'], data_filtered['Deaths'])
plt.xlabel('Year')
plt.ylabel('Deaths')
plt.title('Volcanic Eruption Deaths Over Time')
plt.show()

print("Missing values per column (filtered data):\n", data_filtered.isnull().sum())
data_cleaned = data_filtered.drop_duplicates()

def aggregate_deaths_by_decade(dataframe):
    dataframe['Decade'] = (dataframe['Year'] // 10) * 10
    data_by_decade = dataframe.groupby('Decade').agg({'Deaths': 'sum'}).reset_index()
    return data_by_decade

data_by_decade = aggregate_deaths_by_decade(data_cleaned)
print(data_by_decade)

plt.plot(data_by_decade['Decade'], data_by_decade['Deaths'], marker='o')
plt.xlabel('Decade')
plt.ylabel('Deaths')
plt.title('Volcanic Eruption Deaths Over Time (Aggregated by Decade)')
plt.show()