import pytest
import pandas as pd
import numpy as np
from Initial_project import (
    filter_data_by_year, aggregate_deaths_by_decade,
    remove_outliers, bin_data, moving_average,
    linear_model, quadratic_model, exponential_model,
    logarithmic_model, predict_future_deaths
)

# Sample data to use for tests
sample_data = pd.DataFrame({
    'Year': [1600, 1630, 1650, 1700, 1850, 1900, 1950, 2000],
    'Deaths': [0, 5, 20, 15, 10, 30, 60, 5]
})
sample_data['Decade'] = (sample_data['Year'] // 10) * 10

@pytest.fixture
def filtered_data():
    return filter_data_by_year(sample_data, 1630)

@pytest.fixture
def cleaned_data(filtered_data):
    return filtered_data.drop_duplicates()

@pytest.fixture
def data_by_decade(cleaned_data):
    data_by_decade = aggregate_deaths_by_decade(cleaned_data)
    data_by_decade['Moving_Average'] = data_by_decade['Deaths'].rolling(3, min_periods=1).mean()
    return data_by_decade

@pytest.fixture
def no_outliers_data(data_by_decade):
    return remove_outliers(data_by_decade, 'Deaths')

def test_filter_data_by_year():
    filtered = filter_data_by_year(sample_data, 1630, 2000)
    assert filtered['Year'].min() >= 1630
    assert filtered['Year'].max() <= 2000

def test_aggregate_deaths_by_decade(data_by_decade):
    assert 'Decade' in data_by_decade.columns
    assert 'Deaths' in data_by_decade.columns
    assert data_by_decade.loc[data_by_decade['Decade'] == 1630, 'Deaths'].values[0] == 5

def test_remove_outliers(no_outliers_data):
    deaths_std = no_outliers_data['Deaths'].std()
    assert (no_outliers_data['Deaths'] <= deaths_std * 2 + no_outliers_data['Deaths'].mean()).all()

def test_bin_data():
    binned = bin_data(sample_data, 50)
    assert 'Binned_Decade' in binned.columns
    assert binned['Deaths'].sum() == sample_data['Deaths'].sum()

def test_moving_average():
    moving_avg = moving_average(sample_data, 3)
    assert len(moving_avg) == len(sample_data)
    assert moving_avg.isnull().sum() == 2  # First 2 values in MA should be NaN

def test_models():
    x_data = np.array([1, 2, 3, 4, 5])
    y_linear = linear_model(x_data, 2, 3)
    y_quadratic = quadratic_model(x_data, 1, -2, 1)
    y_exponential = exponential_model(x_data, 1, 0.5, 2)
    y_logarithmic = logarithmic_model(x_data, 2, 1.5)

    assert np.allclose(y_linear, np.array([5, 7, 9, 11, 13]))
    assert np.allclose(y_quadratic, np.array([0, 1, 4, 9, 16]), atol=1)
    assert np.all(y_exponential > 0)
    assert np.all(y_logarithmic > 0)

def test_predict_future_deaths(data_by_decade):
    future_decades = np.array([2020, 2030, 2040])
    predict_future_deaths(
        data_by_decade,
        quadratic_model,
        data_by_decade['Decade'],
        data_by_decade['Deaths'],
        future_decades,
        label="Prediction",
        color="blue",
        initial_params=[1, 0, 1])
