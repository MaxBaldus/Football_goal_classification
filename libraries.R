install_and_load = function(a){
  # install packages
  if(a == TRUE){
    install.packages("randomForest") # random forests
    install.packages("ranger") # faster random forests (C in the backround)
    install.packages('glmnet') # Ridge and Lasso
    install.packages("caret", dependencies = TRUE)
    install.packages('gbm') # gradient boosted machines
    install.packages("xgboost") # extreme gradient boosting
    install.packages("dplyr") # basic data manipulation procedures
    install.packages('tidymodels')
    install.packages("broom")
    install.packages("broom.mixed")
    install.packages("ggplot") # plotting
    install.packages("foreach") 
    install.packages('doParallel')
  } 
  # load packages
  library(randomForest)
  library(ranger)
  library(glmnet)
  library(caret)
  library(gbm)
  library(xgboost)
  library(dplyr)    
  library(tidymodels)
  library(broom)
  library(broom.mixed)
  library(ggplot2)  
  library(foreach)
  library(doParallel)
}



