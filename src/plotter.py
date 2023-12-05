# Basic modules 
import numpy as np

# Plotting modules 
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

# Local modules 
from utils import *


def plot_per_object_distribution(per_obj,labels,save_path=None):

	valid_classes = np.where(per_obj['train'] > 0)[0]
	valid_labels = [  labels[i] for i in valid_classes] 


	# Create subplots: use 'domain' type for Pie subplot
	fig = make_subplots(rows=1, cols=3, specs=[[{'type':'domain'}]*3])
	for i,x in enumerate(per_obj):
		fig.add_trace(go.Pie(labels=valid_labels, values=per_obj[x][valid_classes], name=str(x)),1, i+1)
	

	# Use `hole` to create a donut-like pie chart
	fig.update_traces(hole=.4, hoverinfo="label+percent+name",textposition='inside', textinfo='percent+label')

	fig.update_layout(
		title_text="Category Distribution across dataset splits",
		title_x=0.5,
		# Add annotations in the center of the donut pies.
		annotations=[dict(text='Train', x=0.125, y=0.5, font_size=20, showarrow=False),
					 dict(text='Val', x=0.5, y=0.5, font_size=20, showarrow=False),
					 dict(text='Test', x=0.875, y=0.5, font_size=20, showarrow=False)])
	
	if save_path is None:
		fig.show()
	else: 
		fig.write_image(save_path,width=1920, height=1080) 

def plot_num_object_distribution(num_obj,save_path=None):

	# Create subplots: use 'domain' type for Pie subplot
	fig = go.Figure(data=go.Scatter(x=['train','val','test'], 
		y=[ num_obj[x].mean() for x in ['train','val','test'] ],
		error_y=dict(
            type='data', # value of error bar given in data coordinates
            array=[ num_obj[x].std() for x in ['train','val','test'] ],
            visible=True
            )
    	))  

	fig.update_layout(
		title_text="Number of objects for each image across dataset splits",
		title_x=0.5)
	
	if save_path is None:
		fig.show()
	else: 
		fig.write_image(save_path,width=1920, height=1080) 

	

def plot_points_object_distribution(per_obj,labels,save_path=None):

	valid_classes = list(per_obj['train'].keys())
	valid_labels = [  labels[i] for i in valid_classes] 

	for x in valid_classes:
		# Create subplots: use 'domain' type for Pie subplot

		hist_data = [ per_obj[k][x] for k in per_obj ]
		group_labels = [ k for k in per_obj ]

		fig = ff.create_distplot(hist_data, group_labels, bin_size=10)
		
		fig.update_layout(
			title_text=f"Distribution of number of points for category:{labels[x]}",
			title_x=0.5)
		
		if save_path is None:
			fig.show()
		else: 
			fig.write_image(save_path + f"_{labels[x]}.png",width=1920, height=1080) 