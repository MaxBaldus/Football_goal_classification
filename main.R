######################### 
# Data preparation
######################### 

# clean environment
rm(list=ls()) # clear out all variables in current session 

# load the data
df = read.csv("Data_Gathering/final_dataset.csv")

# prepare the data 
source("additional_fct.R")
X = data_prep(df)

# data inspection
which(is.na(X))
str(X)
View(X)

# training and test set via function
#X_train = data.frame(train_test(X)[1])
#X_test = data.frame(train_test(X)[2])

set.seed(123)
train = sample(1:nrow(X), size = nrow(X)*0.8) # size = number of times are 80% of "observations" (number of observation into training set)

# training Data
X_train = X[train,] # training Data
# test data
X_test = X[-train,] # test Data

dim(X_train)
dim(X_test)

# target
y_train = X_train$goals
y_test = X_test$goals

# load the libraries
source("libraries.R")
a = FALSE
install_and_load(a)


# load models 
source("models.R")

######################### 
# 1) regularized regression
######################### 

# 1) Lasso: alpha = 1
# 2) Ridge: alpha = 0 
# 3) elastic net regression: combination Lasso and Ridge 

# data preparation: dummy encode qualitative variables
x = model.matrix(goals ~., X)[,-1] # create Dummies in new Matrix, neglect Intercept 

# training and test sample rows
set.seed(123)
train = sample(1:nrow(x), size = nrow(x)*0.8) # size = number of times are 80% of "observations" (number of observation into training set)

# training Data
x_train = x[train,] # training Data
dim(x_train)

# test data
x_test = x[-train,] # test Data
dim(x_test)

#### lasso regression: standardization by default
set.seed(123)
lasso = glmnet(x_train, y_train,
               family = "binomial", 
               alpha = 1)
plot(lasso, xvar = "lambda") # coefficient sizes for lambda values
plot(lasso$lambda) # lambda values used by default of glmnet package

# tuning: cv ridge regression to find best lambda
lasso_tun = cv.glmnet(x_train, y_train,
                      family = "binomial",
                      alpha = 1,
                      seed = 234)
min(lasso_tun$cvm) # lowest mean squared error
# 1.372627
lambda_opt = lasso_tun$lambda.min # optimal lambda
plot(lasso_tun)

# predictions
lass_pred = predict(lasso_tun, s = lambda_opt, x_test, type="class")

# confusion table
table(predicted = lass_pred, actual = y_test)

# accuracy
calc_acc(predicted = lass_pred, actual = y_test)
# 0.5421901

# variable importance plot 
# coef(lasso_tun, s = "lambda.1se") %>%
# tidy() %>%
# filter(row != "(Intercept)") %>%
# ggplot(aes(value, reorder(row, value))) +
# geom_point() +
# ggtitle("Top 25 influential variables") +
# xlab("Coefficient") +
# ylab(NULL)

saveRDS(lasso_tun, file = "output/lasso.rda")


#### ridge regression
set.seed(123)
ridge = glmnet(x_train, y_train,
               family = "binomial", 
               alpha = 0)
plot(ridge, xvar = "lambda") # coefficient sizes for lambda values
plot(ridge$lambda) # lambda values used by default of glmnet package

# tuning: cv ridge regression to find best lambda
ridge_tun = cv.glmnet(x_train, y_train,
                      family = "binomial",
                      alpha = 0,
                      seed = 234)
min(ridge_tun$cvm) # lowest mean squared error
# 1.372627
lambda_opt = ridge_tun$lambda.min # optimal lambda
plot(ridge_tun)

# predictions
ridge_pred = predict(ridge_tun, s = lambda_opt, x_test, type="class")

# confusion table
table(predicted = ridge_pred, actual = y_test)

# accuracy
calc_acc(predicted = ridge_pred, actual = y_test)
# 0.5420716

saveRDS(ridge, file = "output/ridge.rda")

## elastic net: with caret package
# optimal alpha 
cv_5 = caret::trainControl(method = "cv", number = 5) # cv_5 object
# cv the optimal alpha
hit_elnet_int = caret::train(
  goals ~ . ,
  data = X_train,
  method = "glmnet",
  trControl = cv_5,
  tuneLength = 10
)

# extract results of trained object:
source("additional_fct.R")
results = el_result(hit_elnet_int)
results
# alpha = 0.6, hence closer to lasso
results$alpha
results$lambda

# fit elastic net with tuned hyper parameters
el = glmnet(x_train, y_train,
      family = "binomial", 
      alpha = results$alpha,
      lambda = results$lambda)

# predictions
lass_pred = predict(el, s = lambda_opt, x_test, type="class")

# confusion table
table(predicted = lass_pred, actual = y_test)

# accuracy
calc_acc(predicted = lass_pred, actual = y_test)
# 0.5406494

saveRDS(el, file = "output/elastic_net.rda")

######################### 
# 2) bagging
######################### 
source("models.R")

start_time = Sys.time()
print(paste("starting bagging", start_time))
bag = bagging(X_train, 1500, X_test)
end_time = Sys.time()
print(paste("estimation time", end_time - start_time))

saveRDS(bag, "output/bagged_trees.rda")
bag = readRDS("output/bagged_trees.rda")

# accuracy
bagged_tst_acc = calc_acc(predicted = bag[[2]], actual = X_test$goals)
bagged_tst_acc
# 0.5836691

# confusion table
bag[[3]]

# oob error w.r.t. number of trees
plot(1:500,  bag[[1]]$err.rate[,1], type = "l")

######################### 
# 3) random forests
######################### 
source("models.R")

# plain random forests
p_ranFor = round(sqrt(dim(X_train)[2]-1)) # number of variables for random forest
start_time = Sys.time()
print(paste("starting rf_plain", start_time))
rf = rf_fit(X_train, X_test, ntrees = 1000, mtry = p_ranFor, samp_size = 0.7, node_size = 6)
end_time = Sys.time()
print(paste("estimation time", end_time - start_time))

saveRDS(rf, "output/rf_plain_model.rda")
rf = readRDS("output/rf_plain_model.rda")

# accuracy
forest_tst_acc = calc_acc(predicted = rf[[2]]$predictions, actual = X_test$goals)
forest_tst_acc
# 0.5876985

# confusion table
rf[[3]]

# mtry (number of variables)
rf[[1]]$mtry

#### hyper parameter tuning with grid search, using OOB and CV

# hyper parameter combinations
mtry_grid = seq(5, (ncol(X)-1), 2) # start with 5 split variables, than increase up to p (bagging)
samp_size_grid = c(0.55, 0.632, 0.7, 0.8)
node_size_grid = seq(3,9, 2)

# total number of combinations
combi = length(mtry_grid) * length(samp_size_grid) * length(node_size_grid)
combi

##### 1) OOB error
source("models.R")
Sys.time()
start_time = Sys.time()
print(paste("starting rf_combi", start_time))
rf_hyper_params_oob = rf_combination(X_train, mtry_grid, samp_size_grid, node_size_grid, ntree = 500) # estimate all random forests
end_time = Sys.time()
print(paste("estimation time", end_time - start_time))

saveRDS(rf_hyper_params_oob, file = "output/rf_hyper_params_oob.rda") # save the error table 
rf_hyper_params_oob = readRDS("output/rf_hyper_params_oob.rda") # read error table

# extract optimal combination and fit TREE again 
rf_opt_hyper = rf_hyper_params_oob[which.min(rf_hyper_params_oob[,1]),]
rf_opt_hyper

# fit optimal tree again
rf_oos_tuned_model = rf_fit(X_train, X_test, ntrees = 1000, mtry = rf_opt_hyper$mtry, 
                              samp_size = rf_opt_hyper$samp_size, node_size = rf_opt_hyper$node_size)

# accuracy
forest_tst_acc = calc_acc(predicted = rf_oos_tuned_model[[2]]$predictions, actual = X_test$goals)
forest_tst_acc

# confusion table
rf_oos_tuned_model[[3]]

# mtry (number of variables)
rf_oos_tuned_model[[1]]$mtry


##### 2) cv
# now do hyper parameter search again using cv (although oob error amounts to cv)
# use X: cross-validate for each combination and save 
# resample training set  k (e.g. 5) times => then compute rf and evaluate with a test set => then take average 
# set same seed before each sample, s.t. each hyperparameter gets same training data

rf_hyper_params_cv = list() # initialize empty list

source("models.R")
Sys.time()
start_time = Sys.time()
print(paste("starting rf_combi", start_time))
rf_hyper_params_cv = rf_combination_cv(X, mtry_grid, samp_size_grid, node_size_grid, ntree = 500, rf_hyper_params_cv) # estimate all random forests
end_time = Sys.time()
print(paste("estimation time", end_time - start_time))

saveRDS(rf_hyper_params_cv, file = "output/rf_hyper_params_cv.rda")
rf_hyper_params_oob = readRDS("output/rf_hyper_params_cv.rda") # read error table


# get best hyper parameter combination and fit tree with optimal parameters again 


######################### 
# 4) boosting 
######################### 

## data preparation
# need response to be 0 or 1: if few goals => 0, otherwise (if many goals) => 1
train = sample(1:nrow(X), size = nrow(X)*0.8) # size = number of times are 80% of "observations" (number of observation into training set)
X_boost = X
X_boost$goals = as.numeric(ifelse(X_boost$goal == "few", "0", "1"))
X_boost_train = X_boost[train,]
X_boost_test  = X_boost[-train,]

######################### 
# adaboost

# if bag fraction always = 1 => no gradient boosting 

source("models.R")
Sys.time()
start_time = Sys.time()
print(paste("starting boosting", start_time))
ada = gbm_fit(X_boost_train, n.trees = 1000, 
              shrinkage = 0.1, interaction.depth = 2, bag.fraction = 1, n.minobsinnode = 5,
              cv.folds = 5, X_boost_test = X_boost_test)
end_time = Sys.time()
print(paste("estimation time", end_time - start_time))

# optimal number of trees
ada$n_trees_final

calc_acc(predicted = ada$prediction, actual = X_test$goals)
# 0.5145769

#### adaboost grid search via hyper parameter grid as matrix 
ada_hyper_grid = expand.grid(
  shrinkage = c(.01, .1, .3),
  interaction.depth = c(1, 3, 5),
  n.minobsinnode = c(5, 10, 15),
  bag.fraction = c(.65, .8, 1), 
  optimal_trees = 0,               
  min_RMSE = 0                     
)

source("models.R")
Sys.time()
start_time = Sys.time()
print(paste("starting grid search", start_time))
opt_tree_and_error = ada_grid_search(X_boost_train, ada_hyper_grid)
end_time = Sys.time()
print(paste("estimation time", end_time - start_time))

# convert list to df
opt_tree_and_error_df = as.data.frame(opt_tree_and_error)

# add min training error and optimal num of trees to grid
for (i in 1:ncol(opt_tree_and_error_df)) {
  ada_hyper_grid[i,"optimal_trees"] = opt_tree_and_error_df[1,i]
  ada_hyper_grid[i,"min_RMSE"] = opt_tree_and_error_df[2,i]
}

# read error table
saveRDS(ada_hyper_grid, file = "output/ada_hyper_grid.rda")
ada_hyper_grid = readRDS("output/ada_hyper_grid.rda") 

# extract optimal combination and train again 
ada_opt_hyper = ada_hyper_grid[which.min(ada_hyper_grid$min_RMSE),]
ada_opt_hyper

# fit again with optimal hyper params
ada_tuned = gbm_fit(X_boost_train, n.trees = ada_opt_hyper$optimal_trees, 
              shrinkage = ada_opt_hyper$shrinkage, interaction.depth = ada_opt_hyper$interaction.depth, 
              bag.fraction = ada_opt_hyper$bag.fraction, n.minobsinnode = ada_opt_hyper$n.minobsinnode,
              cv.folds = 5, X_boost_test = X_boost_test)

# optimal number of trees
ada_tuned$n_trees_final

calc_acc(predicted = ada_tuned$prediction, actual = X_test$goals)


####### XGBoost (extreme gradient boosting)
# encoding categorical features für CV!
x = model.matrix(goals ~., X)[,-1] # create Dummies in new Matrix without Intercept (and without target goals)

# encode target
goals = as.numeric(ifelse(X$goal == "few", "0", "1"))

# training Data
x_train = x[train,] # training Data
y_train = goals[train]

# test data
x_test = x[-train,] # test Data
y_test = goals[-train]

# convert again to xgb matrix format
xgb_train = xgb.DMatrix(data =  as.matrix(x_train), label =y_train)
# unseen test set:
xgb_dtest = xgb.DMatrix(data =  as.matrix(x_test))

# fitting ensemble with arbitrary hyper parameter (including early stopping)
set.seed(123)

xgb_raw = xgb.cv(
  data = xgb_train,
  nround = 1000,
  nfold = 5,
  objective = "binary:logistic",
  verbose = 0,
  early_stopping_rounds = 100
)

# evaluation
opt_trees = xgb_raw$evaluation_log[ which.min(xgb_raw$evaluation_log$train_logloss_mean), ]
opt_trees 

# plotting error vs. tree
ggplot(xgb_raw$evaluation_log) +
  geom_line(aes(iter, train_logloss_mean), color = "red") +
  geom_line(aes(iter, test_logloss_mean), color = "blue")

xgb_raw_model = xgboost(
  data = xgb_train,
  nrounds = opt_trees$iter,
  verbose = 0
)

# create importance matrix
importance_matrix = xgb.importance(model = xgb_raw_model)
# variable importance plot
xgb.plot.importance(importance_matrix, top_n = 10, measure = "Gain")

# forecasting
xgb_forecast = predict(xgb_raw_model, xgb_dtest, type = 'response')
p_pred = ifelse(xgb_forecast > 0.5, "many", "few")

# 0.5591372
calc_acc(predicted = p_pred, actual = X_test$goals)

### hyper parameter tuning

# parameter starting block
max_depths = max_depth = c(1, 3, 5, 7)
etas = c(.01, .05, .1, .3)
min_child_weight = c(1, 3, 5, 7)
subsample = c(.65, .8, 1)
colsample_bytree = c(.8, .9, 1)

source("models.R")
xgb_boost_grid = xgboost_grid_search(gbtrain, max_depth, etas, min_child_weight,
                         subsample, colsample_bytree)

# save grid, zoom further in for fine tuning
# https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
# finally fit model with optimal hyper params again and forecast


# nn
# svm