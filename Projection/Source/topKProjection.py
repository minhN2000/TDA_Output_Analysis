import umap as um
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# features data values reading from cameloutput
features = pd.read_csv('cameloutput.csv', header = None);
features = features.loc[0:1515,:] # removing the last summary row

# sort the data features
def sortHelper():
	global features
	features.sort_values(by = [2, 0, 1], inplace = True, ascending = False)
	features = features.reset_index(drop = True)
# Calling the sort helper
sortHelper()
# Create a list for sub features for H0 H1 H2
listOfDimension = [features.loc[1017 : 1515, :], features.loc[159 : 1016, :], features.loc[0 : 158, :]]
# boundary points coodinate data  reading from camel 500
pointsCoor = pd.read_csv('camel500.csv', header = None, names = ['x', 'y', 'z'])
# t-sne plot
def tsnePlot(camel, h):
	camel = camel[camel.dim == h]
	m = TSNE(learning_rate=50)
	camel_val = camel[['x','y','z']]
	tsne_features = m.fit_transform(camel_val)
	print(camel)
	sns.scatterplot(x = tsne_features[:, 0], y = tsne_features[:, 1], data = camel, c = camel['idx'], hue = 'idx', palette = 'PiYG')
	plt.xlabel('tsne 2d x')
	plt.ylabel('tsne 2d y')
	plt.title('tsne for top 5 features in H2')
	plt.show()
	
# pca plot
def pcaPlot(listIndex, camel, h):
	camel1 = camel[camel.dim == h]
	x = camel1[['x', 'y', 'z']]
	pca = PCA()
	components = pca.fit_transform(x)
	labels = {
    		str(i): f"PC {i+1} ({var:.1f}%)"
    		for i, var in enumerate(pca.explained_variance_ratio_ * 100)
		}
	fig = px.scatter_matrix(
    		components,
    		labels=labels,
    		dimensions=range(3),
    		color=camel1["idx"]
		)	
	fig.update_traces(diagonal_visible=False)	
	#fig = px.scatter(components, x = 0, y = 1, color = camel['dim'])
	#fig.update_layout(showlegend = True)
	#fig.write_html("pca_H1_top5.html")
	fig.show()

# umap plot
def umapPlot(k, camel, h):
	sns.set(style='white', context = 'notebook', rc = {'figure.figsize':(14,10)}) # canvas setting
	reducer = um.UMAP()
	listDF = []
	camel = camel[camel.dim == h]
	# color setting
	colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'] 
	j  = 0
	
	# iterating through the data boundary points coordinate data and plot using umap
	for i in range(0,k):
		tempDF = camel[camel.idx == i] 
		tempDF = tempDF[["x","y","z"]].values
		scaledTempDF = StandardScaler().fit_transform(tempDF)
		try:
			embedding = reducer.fit_transform(scaledTempDF)
			embedding.shape
		except ValueError:
			continue
		plt.scatter(embedding[:, 0], embedding [:, 1], c = colors[j], cmap = 'PiYG', label = str(i))
		j += 1
	# canvas setting
	plt.gca().set_aspect('equal', 'datalim')
	plt.title('UMAP_top5feature_H2', fontsize = 16)
	plt.legend(loc = 'upper right')
	plt.show()

# main- user interface
def projection(k, h): # h is dimension
	global features
	global pointsCoor
	global listOfDimension

	listIndex = [0, 1, 2]
	# get index of the boundary points from i-th feature
	list_points = []
	idx = []
	for i in listIndex:
		list_point = []
		list_point.append(i)
		for j in range (0, k):
			temp = listOfDimension[i].iloc[j, :]
			for l in range(5, temp.size):
				if pd.notna(temp[l]):
					list_point.append(int(temp[l]))
					idx.append(j)
		list_points.append(list_point)	
	# get lists of boundary points coordinate for features 
	dim = []
	x = []
	y = []
	z = []
	for feature in list_points:
		for i in range(1, len(feature)):
			dim.append(feature[0])
			x.append(pointsCoor.loc[feature[i], 'x'])
			y.append(pointsCoor.loc[feature[i], 'y'])
			z.append(pointsCoor.loc[feature[i], 'z'])

	# new data frame of boundary points with specific features
	d = {'dim': dim, 'idx' : idx, 'x': x, 'y': y, 'z': z}
	camel = pd.DataFrame(data = d)
	# plotting
	umapPlot(k, camel, h) 
	pcaPlot(listIndex, camel, h)
	tsnePlot(camel, h)

# example
k = int(input('Please enter the top k feature(s) you want to have: '))	
h = int(input('Please enter the dimension for top k faeture(s): '))
projection(k, h)
