#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 14:25:29 2021

Normalization methods for Inscopy. At the moment 4 normalization methods are included:
	
	z-score: 	Z-score normalization on the basis of the baseline
	min-max: 	Will normalize the entire signal between 0 and 1
	sub_base:	Simple baseline subtraction
	auROC: 	 	Information criterium as in Cohen et al. Nature 2012
	
Most likely z-score is best if you want to look at one individual cell and
see trend over different trials. auROC is most likely the best method
if you want one 1D trace for a single cell that you should be able to
compare to other cells (even from different animals).

For more information about the auROC method, see Cohen et al. Nature 2012.

For the z-score, currently the std and mean are calculated over the all the
data before T=0 unless the baseline parameter is set. (See examples in
function docstring).


	
NOTE: Indentation using tabs instead of spaces!

@author: Han de Jong
"""

def z_score(PE_data, baseline=None):
	"""
	Will perform z-score normalization on the peri-event data in PE_data. PE_data
	should be a Dataframe with the trials as columns and a timeline as index.
	The mean and std will be calculated over the baseline (T<0) unless a specific
	baseline is given.

	Examples:

		>>> PE_norm = z_score(PE_data, baseline = [-10, -5])
		>>> # This will use the interval from T=-10 to T=-5 as the baseline.

	"""

	# Baseline
	if baseline == None:
		baseline = PE_data.index<0
	else:
		baseline = (PE_data.index>=baseline[0]) & (PE_data.index<baseline[1])

	# Perform z-score normalization
	new_data = PE_data.copy()
	new_data = new_data - new_data.iloc[baseline, :].mean(axis=0)
	new_data = new_data/new_data.iloc[baseline, :].std(axis=0)
	new_data.signal_type = 'Signal (Z-score)'

	return new_data


def min_max(PE_data):
	"""
	Will normalize all the data to within the interval 0 to 1. The input 
	should be a Dataframe with the trials as columns and a timeline as index.
	If a different interval is required, simply multiply the data by the width 
	of that interval and add the floor.
	
	Examples:

		>>> PE_norm = min_max_norm(PE_data)

		>>> # If the interval should be -1 to 1
		>>> PE_norm = (PE_norm * 2) - 1

	"""

	new_data = PE_data.copy()
	new_data = new_data - new_data.min(axis=0)
	new_data = new_data/new_data.max(axis=0)
	new_data.signal_type = 'Signal (Normalized)'

	return new_data




def auroc(PE_data, baseline = None):
	"""
	Will normalize the PE_data by calculating the area under the curve of the
	receiver opperant characteristic at every time point. The final result is
	a 1-D array with bounds of 0 and 1 and the baseline at 0.5. A vallue of 1
	at T = X means that there is at least one threshold that perfecty
	distinguises the values at timepoint X from the values of the baseline. It
	is a very powerfull method, but only if there are sufficient trials.

	The input should be a Dataframe with the trials as columns and a timeline 
	as index. All timepoints with T<0 will be defined as baseline unless a
	specific baseline is given.

	See Cohen et al. Nature 2012 for a detailed explanation of this method.

	Examples:

		>>> PE_norm = z_score(PE_data, baseline = [-10, -5])
		>>> # This will use the interval from T=-10 to T=-5 as the baseline.

	"""

	# Baseline
	if baseline == None:
		baseline = PE_data.index<0
	else:
		baseline = (PE_data.index>=baseline[0]) & (PE_data.index<baseline[1])

	# Output data
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