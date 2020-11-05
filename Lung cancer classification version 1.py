# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 10:35:39 2020

@author: Admin
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 10:18:21 2020

@author: Admin
"""


# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 11:36:38 2020

@author: Admin
"""



from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from itertools import cycle, islice
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates
import matplotlib.pyplot as plt
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
data = pd.read_csv('F:/Implementation/lung_stats.csv', sep=',')
print(data.shape)
print(data.head())
print(data.describe().transpose())
print(data.columns)
data=data.drop(columns=["img_id"])
data= data.reset_index()
features = ['lung_area_mm2', 'lung_volume_fraction', 'lung_mean_hu', 'lung_pd95_hu', 
        'lung_pd05_hu']
new_features=data[features]
print(new_features.columns)
X = StandardScaler().fit_transform(new_features)
kmeans = KMeans(n_clusters=12)
model = kmeans.fit(X)
print("model\n", model)
centers = model.cluster_centers_

# Function that creates a DataFrame with a column for Cluster Number

def pd_centers(featuresUsed, centers):
	colNames = list(featuresUsed)
	colNames.append('prediction')

	# Zip with a column called 'prediction' (index)
	Z = [np.append(A, index) for index, A in enumerate(centers)]

	# Convert to pandas data frame for plotting
	P = pd.DataFrame(Z, columns=colNames)
	P['prediction'] = P['prediction'].astype(int)
	return P
# Function that creates Parallel Plots

def parallel_plot(data):
	my_colors = list(islice(cycle(['b', 'r', 'g', 'y', 'k']), None, len(data)))
	plt.figure(figsize=(15,8)).gca().axes.set_ylim([-3,+3])
	parallel_coordinates(data, 'prediction', color = my_colors, marker='o')

P = pd_centers(new_features, centers)
P
parallel_plot(P[P['lung_area_mm2']< 110.761982])