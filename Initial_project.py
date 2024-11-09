import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Load and display the dataset
data = pd.read_csv("volcano-events-2024-10-12_16-24-07_+0200.tsv", sep='\t')
print(data.head())

plt.plot(data['Year'], data['Deaths'])
plt.xlabel('Year')
plt.ylabel('Deaths')
plt.title('Volcanic Eruption Deaths Over Time')
plt.show()

def filter_data_by_year(dataframe, min_year, max_year=None):
    """
    Filters the dataset by a minimum year, and optionally a maximum year.

    Parameters:
    dataframe (pd.DataFrame): The input data with a 'Year' column.
    min_year (int): The minimum year to filter data.
    max_year (int, optional): The maximum year to filter data. Defaults to None.

    Returns:
    pd.DataFrame: The filtered data containing only records within the specified year range.
    """
    if max_year is not None:
        return dataframe[(dataframe['Year'] >= min_year) & (dataframe['Year'] <= max_year)]
    return dataframe[dataframe['Year'] >= min_year]

# Filter data from 1630 onwards and plot
data_filtered = filter_data_by_year(data, 1630)
print(data_filtered.head())

plt.plot(data_filtered['Year'], data_filtered['Deaths'])
plt.xlabel('Year')
plt.ylabel('Deaths')
plt.title('Volcanic Eruption Deaths Over Time')
plt.show()

# The following is to check whether there are missing values for important columns. It has been noticed that there are no missing values for 'Year' and 'Deaths'
print("Missing values per column (filtered data):\n", data_filtered.isnull().sum())

# Remove duplicate records
data_cleaned = data_filtered.drop_duplicates()

def aggregate_deaths_by_decade(dataframe):
    """
    Aggregates the number of deaths by decade.

    Parameters:
    dataframe (pd.DataFrame): The input data with 'Year' and 'Deaths' columns.

    Returns:
    pd.DataFrame: Data aggregated by decade, with total deaths per decade.
    """
    dataframe['Decade'] = (dataframe['Year'] // 10) * 10
    data_by_decade = dataframe.groupby('Decade').agg({'Deaths': 'sum'}).reset_index()
    return data_by_decade

# Aggregate deaths by decade and plot
data_by_decade = aggregate_deaths_by_decade(data_cleaned)
print(data_by_decade)

plt.plot(data_by_decade['Decade'], data_by_decade['Deaths'], marker='o')
plt.xlabel('Decade')
plt.ylabel('Deaths')
plt.title('Volcanic Eruption Deaths Over Time (Aggregated by Decade)')
plt.show()

def remove_outliers(dataframe, column, iqr_multiplier=1.5):
    """
    Removes outliers based on the interquartile range (IQR).

    Parameters:
    dataframe (pd.DataFrame): The input data.
    column (str): Column to analyze for outliers.
    iqr_multiplier (float): Multiplier for the IQR to define outliers.

    Returns:
    pd.DataFrame: Data with outliers removed.
    """
    Q1 = dataframe[column].quantile(0.25)
    Q3 = dataframe[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - iqr_multiplier * IQR
    upper_bound = Q3 + iqr_multiplier * IQR
    filtered_data = dataframe[(dataframe[column] >= lower_bound) & (dataframe[column] <= upper_bound)]
    return filtered_data

# Remove outliers from data and plot
data_no_outliers = remove_outliers(data_by_decade, 'Deaths', iqr_multiplier=1.5)
print(data_no_outliers)

plt.plot(data_no_outliers['Decade'], data_no_outliers['Deaths'], marker='o')
plt.xlabel('Decade')
plt.ylabel('Deaths')
plt.title('Volcanic Eruption Deaths Over Time (Without Major Disasters)')
plt.show()

def bin_data(dataframe, bin_size):
    """
    Bins data into intervals of specified size and aggregates deaths.

    Parameters:
    dataframe (pd.DataFrame): The input data with 'Decade' and 'Deaths' columns.
    bin_size (int): Size of each bin in years.

    Returns:
    pd.DataFrame: Binned data with total deaths per bin.
    """
    dataframe = dataframe.copy()
    dataframe.loc[:, 'Binned_Decade'] = (dataframe['Decade'] // bin_size) * bin_size
    print("Dataframe after adding Binned_Decade:\n", dataframe.head())  
    binned_data = dataframe.groupby('Binned_Decade').agg({'Deaths': 'sum'}).reset_index()
    return binned_data

# Bin data by 50 years and plot
binned_data = bin_data(data_no_outliers, 50)
print(binned_data)

plt.figure(figsize=(12, 6))
plt.plot(binned_data['Binned_Decade'], binned_data['Deaths'], marker='o', color='blue', linestyle='-')
plt.xlabel('Binned Decade')
plt.ylabel('Total Deaths')
plt.title('Binned Volcanic Eruption Deaths Over Time')
plt.grid()
plt.show()

#After binning, it has been noticed that the data is too random and unreliable before the year 1850. The same steps as before are taken, now from 1850 unwards.
data_improved = filter_data_by_year(data, 1850)
data_improved

plt.plot(data_improved['Year'], data_improved['Deaths'])
plt.xlabel('Year')
plt.ylabel('Deaths')
plt.title('Volcanic Eruption Deaths Over Time')
plt.show()

data_improved_filtered = data_improved.drop_duplicates()

data_decade = aggregate_deaths_by_decade(data_improved_filtered)

data_no_outlier = remove_outliers(data_decade, 'Deaths', iqr_multiplier=1.5)
data_no_outlier

plt.plot(data_no_outlier['Decade'], data_no_outlier['Deaths'], marker='o')
plt.xlabel('Decade')
plt.ylabel('Deaths')
plt.title('Volcanic Eruption Deaths Over Time (Without Major Disasters)')
plt.show()

def moving_average(dataframe, window_size):
    """
    Calculates a moving average for the 'Deaths' column.

    Parameters:
    dataframe (pd.DataFrame): The input data with 'Deaths' column.
    window_size (int): The window size for the moving average.

    Returns:
    pd.Series: Moving average values.
    """
    return dataframe['Deaths'].rolling(window=window_size).mean()

# Plot a moving average over the actual data
data_no_outlier.loc[:, 'Moving_Average'] = moving_average(data_no_outlier.copy(), 3)

plt.figure(figsize=(12, 6))
plt.plot(data_no_outlier['Decade'], data_no_outlier['Deaths'], marker='o', label='Deaths')
plt.plot(data_no_outlier['Decade'], data_no_outlier['Moving_Average'], color='orange', label='Moving Average', linewidth=2)
plt.xlabel('Decade')
plt.ylabel('Deaths')
plt.title('Volcanic Eruption Deaths Over Time (Without Major Disasters) with Moving Average')
plt.legend()
plt.grid()
plt.show()

def linear_model(x, a, b):
    """Defines a linear model."""
    return a * x + b

def quadratic_model(x, a, b, c):
    """Defines a quadratic model."""
    return a * x**2 + b * x + c

def exponential_model(x, a, b, c):
    """Defines an exponential model."""
    return a * np.exp(b * x) + c
    
def logarithmic_model(x, a, b):
    """Defines a logarithmic model."""
    return a * np.log(b * x)

def fit_and_plot_model(model_func, x_data, y_data, label, color, p0=None):
    """
    Fits a model to data and plots the result.

    Parameters:
    model_func (function): Model function to fit to the data.
    x_data (array-like): Independent variable data.
    y_data (array-like): Dependent variable data.
    label (str): Label for the plot.
    color (str): Color for the plot line.
    p0 (tuple, optional): Initial parameter guesses for the model.

    Returns:
    None
    """
    params, _ = curve_fit(model_func, x_data, y_data, p0=p0, maxfev=1000)
    y_fit = model_func(x_data, *params)
    plt.plot(x_data, y_fit, label=label, color=color, linewidth=2)

# Divide the total time period into three periods (1850-1920, 1920-1950, 1950-present) to be able to use multiple models at once 
period1 = data_no_outlier[(data_no_outlier['Decade'] >= 1850) & (data_no_outlier['Decade'] <= 1920)]
period2 = data_no_outlier[(data_no_outlier['Decade'] > 1920) & (data_no_outlier['Decade'] <= 1950)]
period3 = data_no_outlier[(data_no_outlier['Decade'] > 1950)]

x1, y1 = period1['Decade'].values, period1['Deaths'].values
x2, y2 = period2['Decade'].values, period2['Deaths'].values
x3, y3 = period3['Decade'].values, period3['Deaths'].values

# A fully linear model has been tried and plotted
plt.figure(figsize=(12, 6))
plt.plot(data_no_outlier['Decade'], data_no_outlier['Deaths'], 'o', label='Actual Deaths', markersize=5)
plt.plot(data_no_outlier['Decade'], data_no_outlier['Moving_Average'], color='blue', label='Moving Average', linewidth=2)
fit_and_plot_model(linear_model, x1, y1, 'Linear Model (1850-1920)', 'orange')
fit_and_plot_model(linear_model, x2, y2, 'Linear Model (1920-1950)', 'green')
fit_and_plot_model(linear_model, x3, y3, 'Linear Model (1950-Present)', 'red')

plt.xlabel('Decade')
plt.ylabel('Deaths')
plt.title('Piecewise Modeling of Volcanic Eruption Deaths')
plt.legend()
plt.grid()
plt.show()

# A combined linear and quadratic model has been tried and plotted
plt.figure(figsize=(12, 6))
plt.plot(data_no_outlier['Decade'], data_no_outlier['Deaths'], 'o', label='Actual Deaths', markersize=5)
plt.plot(data_no_outlier['Decade'], data_no_outlier['Moving_Average'], color='blue', label='Moving Average', linewidth=2)
fit_and_plot_model(quadratic_model, x1, y1, 'Quadratic Model (1850-1920)', 'orange')
fit_and_plot_model(linear_model, x2, y2, 'Linear Model (1920-1950)', 'green')
fit_and_plot_model(quadratic_model, x3, y3, 'Quadratic Model (1950-Present)', 'red')

plt.xlabel('Decade')
plt.ylabel('Deaths')
plt.title('Piecewise Modeling of Volcanic Eruption Deaths')
plt.legend()
plt.grid()
plt.show()

def predict_future_deaths(dataframe, model_func, x_data, y_data, future_years, label, color, initial_params=None):
    """
    Predicts future deaths using a specified model function, plots the predictions, 
    and provides a conclusion on the trend.

    Parameters:
    dataframe (pd.DataFrame): Historical data used for plotting.
    model_func (function): The model function to fit for extrapolation.
    x_data (array-like): X data for fitting the model.
    y_data (array-like): Y data for fitting the model.
    future_years (array-like): Array of future years (or decades) for prediction.
    label (str): Label for the prediction plot line.
    color (str): Color for the prediction plot line.
    initial_params (tuple, optional): Initial parameter guesses for the model.

    Returns:
    None
    """
    # Fit the model to current data
    params, _ = curve_fit(model_func, x_data, y_data, p0=initial_params, maxfev=1000)
    
    # Predict future deaths
    future_deaths = model_func(future_years, *params)
    future_deaths = np.maximum(future_deaths, 0)  # Prevent negative predictions
    
    # Plot historical data, moving average, and future predictions
    plt.figure(figsize=(12, 6))
    plt.plot(dataframe['Decade'], dataframe['Deaths'], 'o', label='Actual Deaths', markersize=5)
    plt.plot(dataframe['Decade'], dataframe['Moving_Average'], color='blue', label='Moving Average', linewidth=2)
    
    # Plotting the historical model fit
    fit_and_plot_model(quadratic_model, x1, y1, 'Quadratic Model (1850-1920)', 'orange')
    fit_and_plot_model(linear_model, x2, y2, 'Linear Model (1920-1950)', 'green')
    fit_and_plot_model(quadratic_model, x3, y3, 'Quadratic Model (1950-Present)', 'red')
    
    # Plot future predictions
    plt.plot(future_years, future_deaths, label=label, color=color, linestyle='--')
    
    plt.xlabel('Decade')
    plt.ylabel('Deaths')
    plt.title('Volcanic Eruption Deaths (Including Predictions for Future Decades)')
    plt.legend()
    plt.grid()
    plt.show()

    # Conclusion
    if np.all(future_deaths < y_data.mean()):
        print("The model predicts a continued decline in volcanic eruption deaths. "
              "This supports the hypothesis that education and hazard management improvements "
              "will lead to fewer fatalities.")
    else:
        print("The model suggests fluctuations in volcanic eruption deaths, indicating that "
              "further improvements may be needed to consistently reduce fatalities.")

future_decades = np.arange(2020, 2051, 10)  # Predict in decade intervals up to 2050
predict_future_deaths(data_no_outlier, quadratic_model, x3, y3, future_decades, 
                      label='Quadratic Model (1950-Present) Prediction', color='red', initial_params=[1, 0, 1])