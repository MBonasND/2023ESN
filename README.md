# 2021ESN
Public codes for "Calibration of Spatial Forecasts from Citizen Science Urban Air Pollution Data with Sparse Recurrent Neural Networks" by Matthew Bonas and Stefano Castruccio

## Data
Dataset "SimulatedData.RData" with 10 variables (locations) and 500 time points. This data is to be used in conjunction with the R scripts.

## functions.R
R script containing the user created functions used in both longrangeforecasting.R and calibration.R. This script does not need to be run manually for the other files automatically import the functions from this file.

## longrangeforecasting.R
R script containing code and methods used to generate long-range forecasts with the Echo State Network (ESN) using SimulatedData.RData. This script corresponds to the method outlined in Algorithm S1.

## calibration.R
R script containing the code and methods used to calibrate the forecasts from the ESN. This script corresponds to Algorithm 1. 
