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

# Read all features output of the camel500 to dataframe
features = pd.read_csv('cameloutput.csv', header = None);
features = features.loc[0:1515,: ]

# Read all vertices coordination to dataframe 
pointsCoor = pd.read_csv('camel500.csv', header = None, names = ['x', 'y', 'z'], index_col = None)
# print(pointsCoor) - Make sure that x, y, z are located in right colums, somtimes the input file has hidden index column,
# which shift x, y, z to the right 1 col 

# sort the features by its dimension -> total vertices -> lifetime in descending order
def sortHelper():
	global features
	features.sort_values(by = [2, 0, 1], inplace = True, ascending = False)
	features = features.reset_index(drop = True)
sortHelper()

# Make the shortcut of the feature in order to draw the barcode
featuresShortcut = pd.DataFrame({'dim' : features[2], 'birth' : features[3], 'death' : features[4]})

# drawing the barcode
fig1, ax1 = plt.subplots()
colors = {"0" : "Blue","1" : "Red","2" : "Green","3" : "Black"}
ax1.scatter(featuresShortcut['birth'], featuresShortcut['death'], c = featuresShortcut['dim'].map(colors), s = 6, picker = True)
ax1.axline([0, 0], [0.07, 0.07],linewidth=1,color="black")

# making interactive function with the barcode graph
# so that when you click on a point (feature) from the graph, it will show 2 figures of T-SNE, PCA and UMAP projection.
# the projection only works for features that have more than 5 data points 
def onpick(event):
	global features
	global pointsCoor
	fig2, (ax2, ax3) = plt.subplots(1,2) # two axes on figure
	ind = event.ind
	if ind.size == 1:
		colors = ["blue", "red", "green"] # sync colors
		list_point = []
		list_point.append(ind[0])
		feature = features.loc[ind[0],:]

		for i in range(5, feature.size):
			if pd.notna(feature[i]):
				list_point.append(int(feature[i]))
		
		dim  = 2
		if ind[0] >= 159 and ind[0] <= 1016:
			dim = 1
		elif ind[0] >= 1017:
			dim = 0
		
		# get lists of boundary points coordinate for features 
		idx = []
		x = []
		y = []
		z = []

		for i in range(1, len(list_point)):
			idx.append(list_point[0])
			x.append(pointsCoor.loc[list_point[i], 'x'])
			y.append(pointsCoor.loc[list_point[i], 'y'])
			z.append(pointsCoor.loc[list_point[i], 'z'])
		# new data frame of boundary points with specific features
		d = {'idx': idx, 'x': x, 'y': y, 'z': z}
		camel = pd.DataFrame(data = d)
		listIndex = [ind[0]]
		
		##############################################################################################
		######################################## UMAP DRAWING ########################################
		##############################################################################################
		sns.set(style='white', context = 'notebook', rc = {'figure.figsize':(14,10)}) # canvas setting
		reducer = um.UMAP()
		listDF = []
		
		# color setting
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
			ax2.scatter(embedding[:, 0], embedding [:, 1], c = colors[dim], label = str(i))
			j += 1
		# canvas setting
		ax2.set_aspect('equal', 'datalim')
		title2 = 'UMAP for ' + str(listIndex[0]) + 'th feature' + ', H' + str(dim)
		ax2.set_title(str(title2), fontsize = 12)
		ax2.legend(loc = 'upper right')
		ax2.grid()
		
		
		##############################################################################################
		######################################## TSNE DRAWING ########################################
		##############################################################################################
		m = TSNE(learning_rate=50)
		camel_val = camel[['x','y','z']]
		tsne_features = m.fit_transform(camel_val)
		ax3.scatter(tsne_features[:, 0],tsne_features[:, 1], c = colors[dim], label = str(listIndex[0]))
		ax3.set_xlabel('tsne 2d x')
		ax3.set_ylabel('tsne 2d y')
		title3 = 'tsne for ' + str(listIndex[0]) + 'th feature' + ', H' + str(dim)
		ax3.set_title(str(title3), fontsize = 12)
		ax3.legend(loc = 'upper right')
		
		
		##############################################################################################
		######################################## PCA DRAWING #########################################	
		##############################################################################################
		x = camel[['x', 'y', 'z']]
		pca = PCA()
		components = pca.fit_transform(x)
		labels = {
	    		str(i): f"PC {i+1} ({var:.1f}%)"
	    		for i, var in enumerate(pca.explained_variance_ratio_ * 100)
			}
		title4 = 'pca for ' + str(listIndex[0]) + 'th feature' + ', H' + str(dim)
		fig3 = px.scatter_matrix(
	    		components,
	    		labels= labels,	
	    		dimensions=range(3),
	    		color_discrete_sequence = [colors[dim]],
	    		title = str(title4)
			)	
		fig3.update_traces(diagonal_visible=False)	
		#fig3.update_layout(showlegend = True)
		fig3.show()
		plt.show()

plt.connect('pick_event', onpick)
plt.show() # show the barcode
