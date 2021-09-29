# Inscopy
Inscopy is a set of Python scripts for quick and convenient analysis of Inscopix miniscope data.

## Background
Inscopix...

## Input data
<figure class="image">
	<img src="assets/Mouse3_AC1_cells.PNG" alt="Example Cells" width=500, height=500 >
	<figcaption>Overview of the cells included in the example data 'Mouse3_AC1' as identified by the Inscopix software.</figcaption>
</figure>
Input data are the .csv files produced by the Inscopix ... software. Usually a recording produces two files. One with the fluorescence of the identified cells and one with TTL stamps.

## Code
All functions and classes used in Inscopy are annotated. The most important functions are in the file 'main.py'. This file can be run as a stand alone library if you prefer to do your analysis from the command line. There is a variable called 'run_example' if this variable is set to True (default) it will run some example analysis on the input data.

```shell
# Will run example analyses on the included example data
$python -i main.py

# Will run example analyses on the files new_mouse_test.csv and new_mouse_TTL_test.csv
$python -i main.py new_mouse

```

## Jupyter Notebook
Most users will prefer to walk trough an analysis of an example mouse using the included Jupyter notebook, 'Inscopy_Jupyter.ipynb'. Jupyter Notebooks can be run on your system after you install the Jupyter Notebook program. The easiest way to do this is by first installing Anaconda and to follow the guidelines [here](https://www.anaconda.com/products/individual).

