###########################################
###########################################
### Calibration of Long-Range Forecasts ###
###########################################
###########################################


#clear environment and load libraries
rm(list = ls())
library(tidyverse)
library(Matrix)
library(abind)
library(scam)

#parallel libraries
library(doParallel)
library(parallel)
library(foreach)

#load functions and data
source('functions.R')
load('SimulatedData.RData')

#specify cores for parallel
#options(cores = 4)



### In the example below we generate 5 windows of forecasts for data 
### with 10 locations (vector elements) for a time horizon of 30 time points.
### Using the forecast windows, it creates a final output of a 10x30
### (locations x horizon) matrix of optimal standard deviation values
### to be used to quantify the uncertainty of future forecasts



########################################
### 'Windows' Long-Range Forecasting ###
########################################

#Parameter specification
#User should perform a CV to determine hyper-parameter values
first.n.h = 150
last.n.h = 40
nu = c(0.4,1.0,1.0)
lambda.r = 0.1
m = 4
alpha = 0.49
reduced.units = 10

#Fixed parameters
tau = 30 #this should be the same as testLen
layers = 3
pi.w = rep(0.1, layers)
pi.win = rep(0.1, layers)
eta.w = rep(0.1, layers)
eta.win = rep(0.1, layers)
start.range = 200 #trainLen of first window forecasts
testLen = 30
locations = 10
iterations = 300
rawData = sim.dat

n.w = 5 #user specified number of 'windows'
WindowForcs = array(NaN, dim = c(locations, 1, iterations))


#Begin Forecasting Windows
for(w in 1:n.w)
{
  
  trainLen = (w-1)*testLen + start.range
  
  #Create training and testing sets
  sets = cttv(rawData, tau, trainLen, testLen)
  
  #Generating input data
  input.dat = gen.input.data(trainLen = trainLen,
                             m = m,
                             tau = tau,
                             yTrain = sets$yTrain,
                             rawData = rawData,
                             locations = locations,
                             xTestIndex = sets$xTestIndex,
                             testLen = testLen)
  y.scale = input.dat$y.scale
  y.train = input.dat$in.sample.y
  designMatrix = input.dat$designMatrix
  designMatrixOutSample = input.dat$designMatrixOutSample
  addScaleMat = input.dat$addScaleMat
  
  
  n.h = c(rep(first.n.h, layers-1), last.n.h)
  #Begin DESN forecasting
  testing = deep.esn(y.train = y.train,
                     x.insamp = designMatrix,
                     x.outsamp = designMatrixOutSample,
                     y.test = sets$yTest,
                     n.h = n.h,
                     nu = nu,
                     pi.w = pi.w, 
                     pi.win = pi.win,
                     eta.w = eta.w,
                     eta.win = eta.win,
                     lambda.r = lambda.r,
                     alpha = alpha,
                     m = m,
                     iter = iterations,
                     future = testLen,
                     layers = layers,
                     reduced.units = reduced.units,
                     startvalues = NULL,
                     activation = 'tanh',
                     distribution = 'Normal',
                     scale.factor = y.scale,
                     scale.matrix = addScaleMat,
                     logNorm = FALSE,
                     fork = FALSE,
                     parallel = FALSE,
                     verbose = TRUE)
  
  #plot best forecasts
  # best.loc = which.min(apply((sets$yTest - testing$forecastmean)^2, 2, mean))
  # plot(sets$yTest[,best.loc], type = 'l', lwd = 2,
  #      ylab = '',
  #      xlab = '',
  #      main = paste('Location:', best.loc))
  # lines(testing$forecastmean[,best.loc], col = 'red', lty = 2, lwd = 2)
  # legend('bottomleft',
  #        legend = c('True Value', 'Mean Forecast'),
  #        col = c('black', 'red'),
  #        lty = c(1,2), lwd = 2, seg.len = 3)


  WindowForcs = abind(WindowForcs, testing$predictions, along = 2)
  #print(w)
  
}
WindowForcs = WindowForcs[,-1,]

###################################################################
### Determine Optimum SD Vector for Each Location - Algorithm 1 ###
###################################################################

locations = 10
n.w = 5
rawData = sim.dat
tau = 30
start.range = 200
horizon = 30
true.range = (start.range + tau + 1):(start.range + n.w*horizon + tau)

optim.sd.mat = matrix(NaN, nrow = locations, ncol = horizon)
#Get optimal sd for each location
for(l in 1:locations)
{
  location = l
  dat = WindowForcs[location,,]
  true.y = rawData[true.range, location]
  means = apply(dat, 1, mean)
  
  #Generate data windows for j-step ahead forecasts
  true.window = list()
  mean.window = list()
  for(i in 1:horizon)
  {
    index = seq(i, (n.w-1)*horizon+i, horizon)
    true.window[[i]] = true.y[index]
    mean.window[[i]] = apply(dat[index,], 1, mean)
  }
  

  ### Optimal SD w/ Monotonic Spline ###
  #Generate optimal SD
  optim.sd = rep(0, horizon)
  for(i in 1:horizon)
  {
    optim.sd[i] = sd(true.window[[i]] - mean.window[[i]])
  }
  
  #Monotonic Spline
  testy = optim.sd
  testx = 1:horizon
  fit = scam(testy~s(testx, k=-1, bs="mpi"), 
             family=gaussian(link="identity"))
  
  #set optimum values from monotonic spline
  optim.sd.mat[l,] = fit$fitted.values
}

