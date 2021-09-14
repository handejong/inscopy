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
from sklearn import metrics

# Settings
plt.ion()


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
	Loads TTL data as formatted by Inscopix
	"""

	# Super annoying that Inscopix puts whitepaces after the commas.
	data = pd.read_csv(filename, delimiter=', ', engine='python')
	data.fillna(0, inplace=True)
	
	# Grabbing the pulses on every channel
	output = {}
	for channel in data['Channel Name'].unique():
		try:
			# Grab the channel
			temp = data[data['Channel Name']==channel].set_index('Time (s)')
			temp.drop('Channel Name', axis=1, inplace=True)

			# Above threshold (threshold is just half of the max)
			temp.loc[:, 'High'] = temp.Value>temp.Value.max()/2

			# Find pulse starts
			temp.loc[:, 'Start'] = np.append(False, (temp.High.iloc[1:].values) & (~temp.High.iloc[:-1].values))

			# Find pulse ends
			temp.loc[:, 'End'] = np.append((~temp.High.iloc[1:].values) & (temp.High.iloc[:-1].values), False)

			# Make sure equal number of starts and stops
			# TO DO

			# Grab the pulses
			temp2 = pd.DataFrame()
			temp2.loc[:, 'Start'] = temp.index[temp.Start]
			temp2.loc[:, 'Stop'] = temp.index[temp.End]
			temp2.loc[:, 'Duration'] = temp2.Stop - temp2.Start

			output[channel] = temp2

		except:
			print(f'Unable to grab pulses from: {channel}')

	return output


def plot_cells(cells, TTL):
	"""
	Will plot all the cells in 'cells' and the pulses in TTL. 'Cells' is a
	DataFrame with the cells as columns and a timeline as index. TTL is a
	DataFrame with the pulse onset, offset and duration. For instance one of
	the outputs of the function 'load_TTL' (but only one value, not the
	entire dictionary.)

	Example:

		>>> cells = load_cells(file_ID + '_test.csv')
		>>> TTL = load_TTL(file_ID + '_TTL_test.csv')
		>>> GPIO_1 = TTL['GPIO-1']
		>>> plot_cells(cells, GPIO_1)
	"""

	# Verify input
	# ToDo

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
	Will normalize the data in PE_data the optional methods are:
		
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
		
	# Baseline indexer
	baseline = PE_data.index<0
	
	# Z-score normalization
	if method == 'z-score':
		new_data = PE_data.copy()
		new_data = new_data - new_data.iloc[baseline, :].mean(axis=0)
		new_data = new_data/new_data.iloc[baseline, :].std(axis=0)
		new_data.signal_type = 'Signal (Z-score)'
	
	# Min-max normallization
	if method =='min-max':
		new_data = PE_data.copy()
		new_data = new_data - new_data.min(axis=0)
		new_data = new_data/new_data.max(axis=0)
		new_data.signal_type = 'Signal (Normalized)'

	# Baseline subtraction
	if method == 'sub_base':
		new_data = PE_data.copy()
		new_data = new_data - new_data[baseline]
	
	# auROC normalization
	# For explanation see Cohen et al. Nature 2012. There's an excellent
	# supplementary figure.
	if method == 'auroc':
		new_data  = pd.DataFrame(index = PE_data.index, columns=['auROC'])

		# Build a linspace of threshold we will try
		thresholds = np.linspace(PE_data.min().min(), PE_data.max().max(), 100)

		# Figure out what fraction of baseline above threshold
		base_points = PE_data[baseline].values.reshape(-1)
		l = len(base_points)
		pBaseline = [(sum(base_points>t)/l) for t in thresholds]

		# For every timepoint build ROC
		l = PE_data.shape[1]
		for time in new_data.index:
            
            # Pandas indexing is actually really slow, so we want to do it only
            # once every loop.
			temp = PE_data.loc[time, :].values
            
			# Calculate the fraction of trials that the signal is above any threshold
			pSignal = [(temp>t).sum()/l for t in thresholds]
			
			# Calculate the auROC and store it
			new_data.loc[time, 'auROC'] = metrics.auc(pBaseline, pSignal)

		# Just making sure it's floats not objects or anything
		new_data = new_data.astype(float)
		new_data.signal_type = 'auROC'
		
	return new_data	


def plot_PE(PE_data):
	"""
	Will plot the heatmap for individual trials/cells on top and the average
	as well as SEM below.
	
	Input:
		A dataframe with individual trials or cells as columns and the
		timeline as the index.
		
	Note: plot_PE does not normalize the data.
	
	"""
	
	# Figure
	fig, axs = plt.subplots(2)
	
	# Figure out if a signal type is defined
	try:
		y_label = PE_data.signal_type
	except:
		y_label = 'Signal'
	
	# Heatmap first
	sns.heatmap(PE_data.transpose(), ax=axs[0], xticklabels=False, 
			 yticklabels=False, 
			 cbar_kws={"location": "top", "use_gridspec": False, "label": y_label})

	# Melt the data
	temp = PE_data.reset_index().melt('index')
	temp.columns = ['Time (s)', 'i', y_label]
	
	# Plot
	sns.lineplot(y=y_label, x='Time (s)', data=temp, ax=axs[1])
	plt.show()


def multi_cell_PE(cells, stamps, norm_method ='auROC'):
	"""
	This function will make peri-event plots of all the cells in 'cells' around
	the stamps in 'stamps'. These plots will then be normalized and (if necessary)
	averaged. Normlized peri-event plots are then returned in one DataFrame.

	normalization methods:

		- Z-score:	
			Generally good, but since you have to averaged over all trials,
			information about the inter-trial variation is lost.
		- min-max:
			Usually not as good as Z-score, unless there is high-baseline
			variation or super low baselines.
		- sub_base:
			Sometimes it's nice to not normalize at all so not to lose intuition
			for what the raw data looks like. In this case only the baseline is
			subtracted.
		- auROC:
			If there are sufficient trials. This is most likely the best method.
			This will normalize between 0 and 1 where the baseline is 0.5 and the
			number reffers to the area under the receiver-operant curve. This
			means that the inter-trial variation is taken into consideration and
			single outliers do not have a large effect on the outcome.
	
	"""

	# Output
	results = pd.DataFrame()

	# Make all peri events
	PE_data = peri_event(cells, stamps)

	# For every cell
	for cell in cells.columns:
		
		# Deal with multi-index
		if cell.__class__== tuple:
			cell = cell[0]

		# Grab the PE and normalize
		norm_PE = normalize_PE(PE_data[cell], method=norm_method)
		signal_type = norm_PE.signal_type # inefficient, but fine

		# Do we have to average?
		if norm_PE.shape[1]>1:
			norm_PE = norm_PE.mean(axis=1)

		# Rename the data
		norm_PE.columns = [cell]

		# Store the data
		results = pd.concat([results, norm_PE], axis=1)

		# Update the user
		print(f'Finished with: {cell}')

	# Store the norm_method in the dataframe (easy for plotting)
	results.signal_type = signal_type

	return results

		




if __name__ == '__main__':
	try:
		file_ID = sys.argv[1]
	except:
		file_ID = 'Mouse3_AC1'

	# Set here if you want to run some examples
	run_examples = True

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
		data = multi_cell_PE(cells, CS_trials, norm_method='auROC')

		# Sort the cells for going up or down
		goes_up = data[data.index>0].mean(axis=0)>0.5
		cells_that_go_up = data.loc[:, goes_up]
		cells_that_go_down = data.loc[:, ~goes_up]

		# Plot both
		plot_PE(cells_that_go_up)
		plt.title('All cells that go up')
		plot_PE(cells_that_go_down)
		plt.title('All cell that go down')



