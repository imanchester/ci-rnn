# GaitPrediction

 ***********************************************************************
 Predicting Transitioning Walking Gaits from the Motion of Walking Canes 
 ***********************************************************************
 A. Mounir Boudali, Peter J. Sinclair, Richard Smith, Ian R. Manchester 
 -----------------------------------------------------------------------
 
 			Experimental dataset

 					last edited: 03/09/18
 					A. Mounir Boudali
***********************************************************************


Data used in this study are extracted from the Cartesian coordinates of 22 
retro-reflective markers.
For a more detailed description of the experimental setup, please refer to 
the manuscript "Predicting Transitioning Walking Gaits from the Motion of 
Walking Canes"


	Dataset description
There are two folders in this dataset: "Raw" and "Matlab data".

"Raw"
This folder contains the raw data from the MOCAP experiments. There are nine 
sub-folders, each representing one subject.
The files "XX.cap" can be opened with a MOCAP software such as Mokka, 
or simply using Excell. 

"Matlab data"
This folder contains the processed data used in the study reported in 
"Predicting Transitioning Walking Gaits from the Motion of Walking Canes". 
There are nine sub-folders, each representing one subject.
The files "XX.mat" can be opened with Matlab. Each data file is an iddata 
structure, made of 11 experiments. Each experiment has the following:
 - iddata.Ouputs = [hip hip_dot knee knee_dot ankle ankle_dot]_l 
 - iddata.Inputs = [Shoulder Shoulder_dot Elbow Elbow_dot Wrist Wrist_dot 
                Cane Cane_dot hip hip_dot knee knee_dot ankle ankle_dot]_r
 - Columns are joint coordinates
 - Rows are snapshots
 - Sampling rate 100Hz


