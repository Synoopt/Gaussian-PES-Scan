# Gaussian-PES-Scan
A series of simple python batch files to create a 2-dimension gaussian PES scan of single atom's x and y coordiantes.

## 1. inputFileGenerator.py

### What you need to prepare

1. A Gaussian job template file(see /example/template.gjf), in which the cartesian coordinates that you have to scan should be substituted with [$1] and [$2].

> In this example, I want to change the x and y cartesian coordinates of the 6th atom, H.

2. The coordinate range that you want to scan.

> In this example, I want to scan the x coordinate from -1 to 4, with step size of 0.2; And the y coordinate from -2 to 3, with step size of 0.2.

3. An output directory.

### What you need to do
1. Fill in the blank file path in:
> template_path = ""
> output_path = ""

2. Run this script, you'll be asked to type in the scanning range, as mentioned before.

3. It will automatically generate a series of .gjf files as Gaussian job input file into the given output directory. The file will be named as {xcoordinate}{ycoordinate}.gjf, like 1.00.4.gjf, -0.2-1.2.gjf.

## 2. g09TaskSubmit.py

### What you need to prepare

1. This script is based on the macOS Ventura 13.6.1(Intel Chip) and with Gaussian09m installed. It's application on Linux or WSL hasn't been tested.

2. An input directory with all the input files generated from 'inputFileGenerator.py'.

3. An output directory.

### What you need to do
1. Fill in the blank file path in:
> input_dir = ""
> output_dir = ""

2. Run the script, it will automatically run the Gaussian job and give the outputs. The output files will have the same name as the input files.

## 3. filesToExcel.py

### What you need to prepare

1. The output file directory that the script 'g09TaskSubmit.py' gives.

2. The Excel file's name and its path.

### What you need to do

1. Fill in the blank file path in:
> input_directory = ""
> output_file = ""

2. Run the sciript, it will automatically use the last energy in every output file(no matter if it is converged or not)

3. The Excel file it create will use the first and second column as the x and y coordinate, the third column as the energy given.

## neuroNetFit-3D.py
> It's generally the same that you use either the 'neuroNetFit-3D.py' or 'neuroNetFit-3D-2.py', they're just using different NeuroNet fitting function and method.
> 'neuroNetFit-2D.py' is to give a 2-dimension contour graph.

### What you need to prepare

1. The Excel file that the 'fileToExcel.py' gives, which use the first and second column as the x and y coordinate, the third column as the energy given.

### What you need to do

1. Replace the blank file path in:
> file_path = ""

2. JUST RUN IT.

3. Or you can moderate the parameter and the method use in NeuroNet Fit based on your own data.

