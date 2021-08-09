import umap as um
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# features data values reading from cameloutput
features = pd.read_csv('cameloutput.csv', header = None);
features = features.loc[0:1516,:] # removing the last summary row

# sort the data features
def sortHelper():
	global features
	features.sort_values(by = [0], inplace = True, ascending = False)
	features = features.reset_index(drop = True)
	
sortHelper()

# boundary points coodinate data  reading from camel 500
pointsCoor = pd.read_csv('camel500.csv', header = None, names = ['x', 'y', 'z'])

# tnse plot
def tsnePlot(camel):
	m = TSNE(learning_rate=50)
	camel_val = camel[['x','y','z']]
	tsne_features = m.fit_transform(camel_val)

	sns.scatterplot(x = tsne_features[:, 0], y = tsne_features[:, 1], data = camel, c = camel['idx'], hue = 'idx', palette = 'PiYG')
	plt.xlabel('tsne 2d x')
	plt.ylabel('tsne 2d y')
	plt.title('tsne')
	plt.show()
	
# pca plot
def pcaPlot(listIndex, camel):
	x = camel[['x', 'y', 'z']]
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
    		color=camel["idx"]
		)	
	fig.update_traces(diagonal_visible=False)	
	#fig = px.scatter(components, x = 0, y = 1, color = camel['dim'])
	#fig.update_layout(showlegend = True)
	#fig.write_html("pca_H1_top5.html")
	fig.show()

#	umap plot
def umapPlot(listIndex, camel):
	sns.set(style='white', context = 'notebook', rc = {'figure.figsize':(14,10)}) # canvas setting
	reducer = um.UMAP()
	listDF = []
	
	# color setting
	colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'] 
	j  = 0
	
	# iterating through the data boundary points coordinate data and plot using umap
	for i in listIndex:
		tempDF = camel[camel.idx == i] 
		tempDF = tempDF[["x","y","z"]].values
		scaledTempDF = StandardScaler().fit_transform(tempDF)
		try:
			embedding = reducer.fit_transform(scaledTempDF)
			embedding.shape
		except ValueError:
			continue
		plt.scatter(embedding[:, 0], embedding [:, 1], c = colors[j], label = str(i))
		j += 1
	# canvas setting
	plt.gca().set_aspect('equal', 'datalim')
	plt.title('UMAP', fontsize = 12)
	plt.legend(loc = 'upper right')
	plt.show()
# main- user interface
def projection(argv):
	global features
	global pointsCoor
	
	listIndex = argv
	
	# get index of the boundary points from arg-th feature
	list_points = []
	for arg in argv:
		list_point = []
		list_point.append(arg)
		feature = features.loc[arg,:]
		for i in range(4, feature.size):
			if pd.notna(feature[i]):
				list_point.append(int(feature[i]))
		list_points.append(list_point)
	
	
	# get lists of boundary points coordinate for features 
	idx = []
	x = []
	y = []
	z = []
	for feature in list_points:
		for i in range(1, len(feature)):
			idx.append(feature[0])
			x.append(pointsCoor.loc[feature[i], 'x'])
			y.append(pointsCoor.loc[feature[i], 'y'])
			z.append(pointsCoor.loc[feature[i], 'z'])
	# new data frame of boundary points with specific features
	d = {'idx': idx, 'x': x, 'y': y, 'z': z}
	camel = pd.DataFrame(data = d)
	
	# plotting
	umapPlot(listIndex, camel) 
	pcaPlot(listIndex, camel)
	tsnePlot(camel)

# example	
i = [int(i) for i in input("Enter features you want to see their projection: ").split()]
projection(i)
