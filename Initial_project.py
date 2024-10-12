import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_csv("volcano-events-2024-10-12_16-24-07_+0200.tsv", sep='\t')
print(data.head())