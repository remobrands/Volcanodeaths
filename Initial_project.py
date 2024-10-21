import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

data = pd.read_csv("volcano-events-2024-10-12_16-24-07_+0200.tsv", sep='\t')
print(data.head())

plt.plot(data['Year'], data['Deaths'])
plt.xlabel('Year')
plt.ylabel('Deaths')
plt.title('Volcanic Eruption Deaths Over Time')
plt.show()

def filter_data_by_year(dataframe, min_year, max_year=None):
    if max_year is not None:
        return dataframe[(dataframe['Year'] >= min_year) & (dataframe['Year'] <= max_year)]
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

def remove_outliers(dataframe, column, iqr_multiplier=1.5):
    Q1 = dataframe[column].quantile(0.25)
    Q3 = dataframe[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - iqr_multiplier * IQR
    upper_bound = Q3 + iqr_multiplier * IQR
    filtered_data = dataframe[(dataframe[column] >= lower_bound) & (dataframe[column] <= upper_bound)]
    return filtered_data

data_no_outliers = remove_outliers(data_by_decade, 'Deaths', iqr_multiplier=1.5)
print(data_no_outliers)

plt.plot(data_no_outliers['Decade'], data_no_outliers['Deaths'], marker='o')
plt.xlabel('Decade')
plt.ylabel('Deaths')
plt.title('Volcanic Eruption Deaths Over Time (Without Major Disasters)')
plt.show()

def bin_data(dataframe, bin_size):
    dataframe = dataframe.copy()
    dataframe.loc[:, 'Binned_Decade'] = (dataframe['Decade'] // bin_size) * bin_size
    print("Dataframe after adding Binned_Decade:\n", dataframe.head())  
    binned_data = dataframe.groupby('Binned_Decade').agg({'Deaths': 'sum'}).reset_index()
    return binned_data

binned_data = bin_data(data_no_outliers, 50)
print(binned_data)

plt.figure(figsize=(12, 6))
plt.plot(binned_data['Binned_Decade'], binned_data['Deaths'], marker='o', color='blue', linestyle='-')
plt.xlabel('Binned Decade')
plt.ylabel('Total Deaths')
plt.title('Binned Volcanic Eruption Deaths Over Time')
plt.grid()
plt.show()

data_improved = filter_data_by_year(data, 1850)

plt.plot(data_improved['Year'], data_improved['Deaths'])
plt.xlabel('Year')
plt.ylabel('Deaths')
plt.title('Volcanic Eruption Deaths Over Time')
plt.show()

data_improved2 = data_improved.drop_duplicates()
data_decade = aggregate_deaths_by_decade(data_improved2)
data_no_outlier = remove_outliers(data_decade, 'Deaths', iqr_multiplier=1.5)
data_no_outlier

plt.plot(data_no_outlier['Decade'], data_no_outlier['Deaths'], marker='o')
plt.xlabel('Decade')
plt.ylabel('Deaths')
plt.title('Volcanic Eruption Deaths Over Time (Without Major Disasters)')
plt.show()

def moving_average(dataframe, window_size):
    return dataframe['Deaths'].rolling(window=window_size).mean()

data_no_outlier.loc[:, 'Moving_Average'] = moving_average(data_no_outlier, 3)

plt.figure(figsize=(12, 6))
plt.plot(data_no_outlier['Decade'], data_no_outlier['Deaths'], marker='o', label='Deaths')
plt.plot(data_no_outlier['Decade'], data_no_outlier['Moving_Average'], color='orange', label='Moving Average', linewidth=2)
plt.xlabel('Decade')
plt.ylabel('Deaths')
plt.title('Volcanic Eruption Deaths Over Time (Without Major Disasters) with Moving Average')
plt.legend()
plt.grid()
plt.show()

x_data = data_no_outlier['Decade'].values
y_data = data_no_outlier['Deaths'].values

def linear_model(x, a, b):
    return a * x + b

linear_params, _ = curve_fit(linear_model, x_data, y_data)
y_linear_fit = linear_model(x_data, *linear_params)

plt.figure(figsize=(12, 6))
plt.plot(x_data, y_data, 'o', label='Actual Deaths', markersize=5)
plt.plot(x_data, y_linear_fit, color='orange', label='Fitted Linear Model', linewidth=2)
plt.plot(data_no_outlier['Decade'], data_no_outlier['Moving_Average'], color='blue', label='Moving Average', linewidth=2)
plt.xlabel('Decade')
plt.ylabel('Deaths')
plt.title('Volcanic Eruption Deaths with Fitted Linear Model and Moving Average')
plt.legend()
plt.grid()
plt.show()