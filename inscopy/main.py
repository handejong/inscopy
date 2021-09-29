#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 8 16:23:21 2021

Code for quick analysis of Inscopix miniscope data.

Note, this scripts can be run directly from the command line by passing the file
identifier as the first argument.

Example:
	Files:
		- Mouse3_AC1_test.csv
		- Mouse3_AC1_TTL_test.csv
	Command:
		$ ./inscopix_han.py Mouse3_AC1
		
Another Example:
	
	Use the '-i' flag to drop into a console after loading the data
	
	Command:
		$ python -i inscopix_han.py Mouse3_AC1
		>>> stamps = TLL['GPIO-1'].Start # Looking at whatever happend on GPIO-1
		>>> cell_014 = peri_event(cells['C014'], stamps) # Make Peri event
		>>> cell_014 = normalize_PE(cell_014, method='z-score') # Normalize
		>>> plot_PE(cell_014) # Plot
	
NOTE: Indentation using tabs instead of spaces!

@author: Han de Jong
"""

# Libraries
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Sub libraries of Inscopy
try:
	import normalization as norm
except:
	import inscopy.normalization as norm

# Settings
plt.ion()

# Set here if you want to run some examples
run_examples = True


def load_cells(filename):
	"""
	Loads the data as formatted by Inscopix into a Pandas dataframe.
	"""

	# Load the data
	data = pd.read_csv(filename, header=[0, 1], index_col=0)

	# Deal with leading spaces
	columns = [[i[0][1:] for i in data.columns], [i[1][1:] for i in data.columns]]
	columns = pd.MultiIndex.from_tuples(list(zip(*columns)))
	data.columns = columns

	return data


def load_TTL(filename):
	"""
	Loads TTL data as formatted by Inscopix. Every channel will be converted
	into a DataFrame with start and stop times of the pulses. The channels
	will be storred in a dict with the channel names as keys.

	"""

	# Deal with whitespaces behind the commas.
	data = pd.read_csv(filename, delimiter=', ', engine='python')
	data.fillna(0, inplace=True)
	
	# Grabbing the pulses on every channel
	output = {}
	for channel in data['Channel Name'].unique():
		try:
			# Grab the channel
			temp = data[data['Channel Name']==channel].set_index('Time (s)')
			temp.drop('Channel Name', axis=1, inplace=True)

			# Check if there are any pulses there to begin with
			if temp.Value.sum() == 0:
				continue

			# Above threshold (threshold is just half of the max)
			temp.loc[:, 'High'] = temp.Value>temp.Value.max()/2

			# Find pulse starts
			temp.loc[:, 'Start'] = np.append(False, (temp.High.iloc[1:].values) & (~temp.High.iloc[:-1].values))

			# Find pulse ends
			temp.loc[:, 'End'] = np.append((~temp.High.iloc[1:].values) & (temp.High.iloc[:-1].values), False)

			# Grab starts and stops
			starts = temp.index[temp.Start]
			stops = temp.index[temp.End]

			# Make sure only complete pulses
			if stops[0]<starts[0]:
				stops = stops[1:]
			if starts[-1]>stops[-1]:
				starts = starts[:-1]

			# Check if they are the same length
			if not len(starts) == len(stops):
				print('Unequal pulse starts and ends.')
				raise ValueError('Unequal pulse starts and ends.')

			# Grab the pulses
			temp2 = pd.DataFrame()
			temp2.loc[:, 'Start'] = starts
			temp2.loc[:, 'Stop'] = stops
			temp2.loc[:, 'Duration'] = temp2.Stop - temp2.Start

			output[channel] = temp2

		except:
			print(f'Unable to grab pulses from: {channel}')
			print(temp)

	return output


def plot_cells(cells, TTL, window = None):
	"""
	Will plot all the cells in 'cells' and the pulses in TTL. 'Cells' is a
	DataFrame with the cells as columns and a timeline as index. TTL is a
	DataFrame with the pulse onset, offset and duration. For instance one of
	the outputs of the function 'load_TTL' (but only one value, not the
	entire dictionary.)

	Window can be used to set the X limits.

	Example:

		>>> cells = load_cells(file_ID + '_test.csv')
		>>> TTL = load_TTL(file_ID + '_TTL_test.csv')
		>>> GPIO_1 = TTL['GPIO-1']
		>>> plot_cells(cells, GPIO_1)

	Another Example

		>>> cells = load_cells(file_ID + '_test.csv')
		>>> TTL = load_TTL(file_ID + _TTL_test.csv')
		>>> GPIO_2 = TTL['GPIO-2']
		>>> plot_cells(cells, GPIO_2, window=[100, 200])

	"""

	# Make figure
	fig, ax = plt.subplots(1, tight_layout=True)

	# Plot the cells
	cells.plot(ax=ax, linewidth=1)
	ax.set_xlabel('Time (s)')
	ax.set_ylabel('Arbitrary Units')

	# Plot the TLL pulses
	for i in TTL.Start:
		ax.axvline(x=i, linestyle='--', color='k', linewidth=0.5)
	for i in TTL.Stop:
		ax.axvline(x=i, linestyle='--', color='k', linewidth=0.5)

	# Set X limits
	if not window==None:
		plt.xlim(window)

	# Finish up
	plt.show()


def peri_event(cells, stamps, window = 10):
	"""
	Will cut the 1D data in 'cells' and produce a matrix of (n, m) where n is the
	number of timestamps in 'stamps' and m is the 1 + 2 * window * the framerate. 
	(window in sec). If there are multiple cells (columns) in the DataFrame
	'cells' then the output will be a dict with the column names as the keys.

	NOTE: unexpected results if framerate is not close to constant!
	"""

	# Figure out the framerate. Use it to build the timeline.
	framerate = len(cells)/cells.index[-1] - cells.index[0] 
	timeline = np.linspace(-window, window, np.round(2*window*framerate).astype(int)+1)
	window = int((len(timeline)-1)/2) # The window size as an index

	# Put the output in a dictionary
	output = {}
	for cell in cells.columns:

		# Figure out the key name
		if cell.__class__==tuple:
			cell = cell[0]

		# Build the dataframe with the data
		output[cell] = pd.DataFrame(index = timeline)
		for i, stamp in enumerate(stamps):
			center = np.argmin(np.abs(cells.index-stamp))
			temp = cells.iloc[center-window:center+window+1, :]
			output[cell].loc[:, i] = temp[cell].values.reshape(-1)
			
	# If the user only asked for one cell, return the DataFrame instead of the
	# dict.
	if len(output)==1:
		output = output[list(output.keys())[0]]

	return output


def normalize_PE(PE_data, method='z-score'):
	"""
	Will normalize the data in PE_data. The input should be a DataFrame
	with the trials in columns and a timeline in the index or a dict
	containing multiple of such DataFrames.

	The optional methods are:
		
		z-score: 	Z-score normalization on the basis of the baseline
		min-max: 	Will normalize the entire signal between 0 and 1
		sub_base:	Simple baseline subtraction
		auROC: 	 	Information criterium as in Cohen et al. Nature 2012
		
	Most likely z-score is best if you want to look at one individual cell and
	see trend over different trials. auROC is most likely the best method
	if you want one 1D trace for a single cell that you should be able to
	compare to other cells (even from different animals).
	
	For the z-score, currently the std and mean are calculated over the all the
	data before T=0, however, there will be an option to manually set the
	baseline in the future.
	"""
	
	# Error handeling
	method = method.lower()
	if not method in ('z-score', 'min-max', 'sub_base', 'auroc'):
		print('Please use one of the following methods:')
		print('    - z-score')
		print('    - min-max')
		print('    - sub_base')
		print('    - auROC')
		print('See details in method docstring.')
		raise ValueError
	
	# Figure out if this is only one cell (DataFrame) or a Dict
	if PE_data.__class__ == pd.DataFrame:
		PE_data = {'temp':PE_data}
	
	# Output is a dict (for now)
	output = {}

	# Perform normalization
	for key in PE_data.keys():

		temp = PE_data[key]

		# Z-score normalization
		if method == 'z-score':
			output[key] = norm.z_score(temp)
		
		# Min-max normalization
		if method =='min-max':
			output[key] = norm.min_max(temp)

		# Baseline subtraction
		if method == 'sub_base':
			output[key] = norm.sub_base(temp)

		# auROC normalization
		if method == 'auroc':
			output[key] = norm.auroc(temp)

	# If only one cell, convert back to DataFrame
	if len(output) == 1:
		output = output[list(output.keys())[0]]
	
	return output	


def sort_PE(PE_data, sorter=None, method=None):
	"""
	Responsible for sorting the trials (or mice) in a PE Dataset.


	TODO


	"""

	temp = PE_data.transpose()
	temp['sorter'] = sorter
	temp = temp.sort_values(by='sorter')
	temp = temp.drop('sorter', axis=1).transpose()
	
	return temp



def plot_PE(PE_data, cmap = 'viridis', sorter = None):
	"""
	Will plot the heatmap for individual trials/cells on top and the average
	as well as SEM below.
	
	Input:
		PE_data: A dataframe with individual trials or cells as columns and the
				 timeline as the index.
		cmap: 	 the colormap you want to use see: ....TODO...
		sorter:  If you want the trials/mice in the heatmap in heatmap in a
				 different order. Pass an array that can be easily sorted.

	Output:
		Handles to the figure and the subplots.
	
	"""

	# Prevent leakage
	PE_data = PE_data.copy()

	# Deal with multi-level columns
	if PE_data.columns.nlevels>1:
		PE_data.columns = PE_data.columns.droplevel(1)

	# Should we sort?
	if not sorter.__class__ == None.__class__:
		PE_data = sort_PE(PE_data, sorter=sorter)
	
	# Figure
	fig, axs = plt.subplots(2)
	
	# Figure out if a signal type is defined
	try:
		y_label = PE_data.signal_type
	except:
		y_label = 'Signal'
	
	# Heatmap first
	sns.heatmap(PE_data.transpose(), ax=axs[0], xticklabels=False, 
			 yticklabels=False, cmap = cmap,
			 cbar_kws={"location": "top", "use_gridspec": False, "label": y_label})

	# Melt the data
	temp = PE_data.reset_index().melt('index')
	temp.columns = ['Time (s)', 'i', y_label]
	
	# Plot
	sns.lineplot(y=y_label, x='Time (s)', data=temp, ax=axs[1])
	plt.show()

	return fig, axs


def multi_cell_PE(cells, stamps, norm_method ='auROC', verbose=False):
	"""
	This function will make peri-event plots of all the cells in 'cells' around
	the stamps in 'stamps'. These plots will then be normalized and (if necessary)
	averaged. Normlized peri-event plots are then returned in one DataFrame.

	For an overview of the normalization methods, see normalize_PE.

	Note that the data for every cell will be averaged over axis = 1, meaning
	over the trials. So the output is a 1-D vector for every cell.


	"""

	# Output
	results = pd.DataFrame()

	# Make all peri events
	PE_data = peri_event(cells, stamps)

	# normalize
	PE_data = normalize_PE(PE_data, method = norm_method)

	# Put it in a dict if it's only one cell
	if not PE_data.__class__ == dict:
		PE_data = {'cell':PE_data}

	# For every cell
	for cell in PE_data.keys():

		# Grab the data
		norm_PE = PE_data[cell]

		# Do we have to average?
		if norm_PE.shape[1]>1:
			norm_PE = norm_PE.mean(axis=1)

		# Store the data
		results = pd.concat([results, norm_PE], axis=1)

		# Update the user
		if verbose:
			print(f'Finished with: {cell}')

	# Update the column names
	results.columns = cells.columns

	# Store the norm_method in the dataframe (easy for plotting)
	results.signal_type = norm_method

	return results

		

if __name__ == '__main__':
	try:
		file_ID = sys.argv[1]
	except:
		file_ID = '../example_data/Mouse3_AC1'

	# This will load the cells and TLL pulses with the file_ID
	cells = load_cells(file_ID + '_test.csv')
	TTL = load_TTL(file_ID + '_TTL_test.csv')

	if run_examples:
		print('Running examples')

		# This example will plot the first 2 cells as well as shocks
		plot_cells(cells.iloc[:, :2], TTL['GPIO-1'])
		plt.title('The first two cells, response to shock')

		# This example with plot peri-event plot of cell 1 to CS+onset
		# NOTE: trial onset is signalled by the END of a TTL pulse.
		CS_trials = TTL['GPIO-2']
		CS_trials = CS_trials[CS_trials.Duration>0.6].Stop
		cell_1_PE = peri_event(cells.iloc[:, [0]], CS_trials)
		cell_1_PE = normalize_PE(cell_1_PE, 'Z-score')
		plot_PE(cell_1_PE)
		plt.title('Cell 1 response to CS+ and shock')

		# This example will show the responses of all cells in the dataset
		data = multi_cell_PE(cells, CS_trials, norm_method='Z-score')

		# Sort the cells for going up or down
		goes_up = data[data.index>0].mean(axis=0)>0.5
		cells_that_go_up = data.loc[:, goes_up]
		cells_that_go_down = data.loc[:, ~goes_up]

		# Plot both
		plot_PE(cells_that_go_up)
		plt.title('All cells that go up')
		plot_PE(cells_that_go_down)
		plt.title('All cell that go down')


