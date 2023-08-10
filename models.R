#########################################
# Bagging and Random Forest

bagging = function(X_train, ntrees, X_test){
  # Estimation: Bagging aka aggregated boostrapping with all number of predictors, no cv 
  bagged = randomForest(goals ~ .,
                        data = X_train,
                        mtry = dim(X_train)[2]-1, # using all predictors
                        importance = T, # get importance for each variable
                        ntrees = ntrees)

  # Prediction on test set
  bag_pred_class = predict(bagged, newdata = X_test, type = "class")
  
  # confusion table
  conf_matr = table(predicted = bag_pred_class, actual = X_test$goals)

  return(list(bagged, bag_pred_class, conf_matr))
  
}

# fit rf and predict test set
rf_fit = function(X_train, X_test, ntrees, mtry, samp_size, node_size){
  
  # Random Forest without any prior specifications
  rand_forest = ranger::ranger(goals ~., data = X_train, 
                               mtry = mtry, sample.fraction = samp_size, min.node.size = node_size,
                               num.trees = ntrees , classification = TRUE)
  
  # prediction test set
  forest_pred_class = predict(rand_forest, data = X_test)
  
  # Confusion matrix test set
  conf_matr = table(predicted = forest_pred_class$predictions, actual = X_test$goals)
  
  return(list(rand_forest, forest_pred_class, conf_matr))
}

###### hyper parameter tuning via grid search and using oob error
rf_combination = function(X, mtry_grid, samp_size_grid, node_size_grid, ntree){
  
  count = 1
  
  set.seed(123) 
  
  # initializing
  rf_acc = data.frame(matrix(ncol = 4, nrow = 0))
  colnames(rf_acc) = c("Error", "mtry", "samp_size", "node_size")
  
  for (mtry in mtry_grid){
    for (samp_size in samp_size_grid){
      for (node_size in node_size_grid) {

        rand_forest = ranger::ranger(goals ~., data = X, 
                                     mtry = mtry, sample.fraction = samp_size, min.node.size = node_size,
                                     num.trees = ntree, classification = TRUE)

          
        rf_acc[count,"Error"] = rand_forest$prediction.error # extract oob error
        rf_acc[count,"mtry"] = mtry
        rf_acc[count,"samp_size"] = samp_size
        rf_acc[count,"node_size"] = node_size
        
        count = count + 1
        print(count)
      }
    }
  }
  View(rf_acc)
  return(rf_acc)
}

rf_combination_parallel = function(X, mtry_grid, samp_size_grid, node_size_grid, ntree){

  set.seed(123) 
  
  # cores
  totalCores = detectCores()
  totalCores
  cluster = makeCluster(totalCores[1]-1) # set number of cores to use
  registerDoParallel(cluster)
  
  # for each mtry (number of regressors used) one dataframe 
  # with samp_size, node_size and error 
  list_mtry = foreach(mtry=mtry_grid) %dopar% {
    
    # initializing
    count = 1
    rf_acc_per_mtry = data.frame(matrix(ncol = 3, nrow = 0))
    colnames(rf_acc_per_mtry) = c("Error", "samp_size", "node_size")
    
    for (samp_size in samp_size_grid){
      for (node_size in node_size_grid) {
        
        rand_forest = ranger::ranger(goals ~., data = X, 
                                     mtry = mtry, sample.fraction = samp_size, min.node.size = node_size,
                                     num.trees = ntree, classification = TRUE)
        
        
        rf_acc_per_mtry[count,"Error"] = rand_forest$prediction.error # extract oob error
        rf_acc_per_mtry[count,"samp_size"] = samp_size
        rf_acc_per_mtry[count,"node_size"] = node_size
        
        count = count + 1
        print(count)
      }
    }
  }
  # stop cluster
  stopCluster(cluster)
  
  return(list_mtry)
}

###### hyper parameter tuning via grid search and using cv (compare against oob error)
# using the ranger package
rf_combination_cv = function(X, mtry_grid, samp_size_grid, node_size_grid, ntree, hyper_list){
  
  count = 1
  
  for (mtry in mtry_grid){
    for (samp_size in samp_size_grid){
      for (node_size in node_size_grid) {
        
        k = 1
        set.seed(123) # each hyper parameter pair gets same shuffled df 
        # initializing
        rf_acc = data.frame(matrix(ncol = 4, nrow = 0))
        colnames(rf_acc) = c("accuracy", "mtry", "samp_size", "node_size")
        
        while (k <= 5) {
          
          train = sample(1:nrow(X), size = nrow(X)*0.8) # shuffle df
          X_train = X[train,] # training Data
          X_test = X[-train,] # test Data

          rand_forest = ranger::ranger(goals ~., data = X_train, 
                                       mtry = mtry, sample.fraction = samp_size, min.node.size = node_size,
                                       num.trees = ntree, classification = TRUE)
          
          forest_pred_class = predict(rand_forest, data = X_test, type = 'response')
          forest_acc = calc_acc(predicted = forest_pred_class$predictions, actual = X_test$goals)
          
          rf_acc[k,"accuracy"] = forest_acc
          rf_acc[k,"mtry"] = mtry
          rf_acc[k,"samp_size"] = samp_size
          rf_acc[k, "node_size"] = node_size
          
          k = k + 1
          
        }

        cv_error = mean(rf_acc[,1]) # cv error (mean of the k test errors)
        hyper_list[[count]] = cbind(cv_error,rf_acc[1,-1]) # save cv_error and hyper parameter combination

        count = count + 1
        print(count)
      }
    }
  }
  View(hyper_list)
  return(hyper_list)
}

#########################################
# 2) boosting

# AdaBoost using cv
gbm_fit = function(X_boost_train, n.trees, shrinkage, interaction.depth, n.minobsinnode,
                   bag.fraction, cv.folds, X_boost_test){
  
  out = gbm(goals ~., data=X_boost_train, distribution = "bernoulli", 
            n.trees = n.trees, 
            shrinkage = shrinkage, 
            interaction.depth = interaction.depth, 
            bag.fraction = bag.fraction, 
            n.minobsinnode = n.minobsinnode ,
            cv.folds = cv.folds, 
            n.cores = NULL, verbose = FALSE)
  
  # Check performance using 5-fold cross-validation and get optimal number of trees 
  n_trees_final = gbm.perf(out, method = "cv", plot.it = TRUE)
  
  gbm_predict = predict(out, n.trees = n_trees_final, newdata = X_boost_test, type = "response")
  p_pred = ifelse(gbm_predict > 0.5, "many", "few")
  
  return(list(adaboost = out, n_trees_final = n_trees_final, prediction = p_pred))
}

#### hyper parameter tuning using parallel computing
ada_grid_search = function(X_boost, hyper_grid){
  
  # cores
  totalCores = detectCores()
  print(totalCores)
  cluster = makeCluster(totalCores[1]-2) # set number of cores to use
  registerDoParallel(cluster)
  
  opt_tree_and_error = foreach(i = 1:nrow(hyper_grid)) %dopar% {
    
    set.seed(123) 
    library(gbm)
    
    gbm_tune = gbm(
      formula = goals ~ .,
      distribution = "bernoulli",
      data = X_boost,
      n.trees = 5000,
      interaction.depth = hyper_grid$interaction.depth[i],
      shrinkage = hyper_grid$shrinkage[i],
      n.minobsinnode = hyper_grid$n.minobsinnode[i],
      bag.fraction = hyper_grid$bag.fraction[i],
      train.fraction = .25, # no cv, but train/test split 
      n.cores = NULL, # will use all cores by default
      verbose = FALSE
    )
    # %dopar%
    list(which.min(gbm_tune$valid.error), sqrt(min(gbm_tune$valid.error)))
    
    # %do%
    #hyper_grid$optimal_trees[i] <- which.min(gbm_tune$valid.error)
    #hyper_grid$min_RMSE[i] <- sqrt(min(gbm_tune$valid.error))
    
  }
  # stop cluster
  stopCluster(cluster)
  
  return(opt_tree_and_error)
}

#### hyper parameter tuning using for loops 
# for each hyper parameter combination: do cv: i.e. use all but k fold for training, k for testing
# then averaging error produced => and save for each hyper parameter combination
gbm_grid = function(X_boost, n.trees, shrinkage, interaction.depth, bag.fraction, train.fraction, cv.folds){
  
  count = 1
  
  for( depth in interaction_depth ){
    for( num in shrinkage){
      for (ntree in n.trees){  
        k = 1
        set.seed(123) # each hyper parameter pair gets same shuffled df 
        # initializing
        # initializing
        boost_acc = data.frame(matrix(ncol = 4, nrow = 0))
        colnames(boost_acc) = c("accuracy", "interaction.depth", "shrinkage", "n.trees")
        
        while (k <= 5) {
          train = sample(1:nrow(X), size = nrow(X)*0.8)
          X_boost_train = X_boost[train,]
          X_boost_test  = X_boost[-train,]
          
          boost = gbm(goals ~., data=X_boost_train, distribution = "bernoulli", 
                      n.trees = ntree,
                      shrinkage = num, 
                      interaction.depth = depth, 
                      bag.fraction = 0.5, 
                      # the fraction of the training set observations randomly selected to propose the next tree in the expansion
                      train.fraction = 0.8, 
                      # The firsttrain.fraction * nrows(data)observations are used to fit the gbm and the remainder are used for 
                      # computing out-of-sample estimates of the lossfunction.
                      n.minobsinnode = 100 , # minimal number of observations in terminal nodes
                      cv.folds = 5, keep.data = FALSE,verbose = FALSE)
          
            # do I need opt. number of trees????
            
            # # Check performance using the out-of-bag (OOB) error
            # best.iter_oob = gbm.perf(boost, method = "OOB")
            # # Check performance using the 50% heldout test set : 20% ?!
            # best.iter_test = gbm.perf(boost, method = "test")
            # # Check performance using 5-fold cross-validation
            # best.iter_cv = gbm.perf(boost, method = "cv")
            # 
            # n.trees_avr = round(as.numeric((best.iter_test + best.iter_cv + best.iter_oob) / 3))
            
            boost_pred = predict(boost, n.trees = 1000, newdata = X_boost_test, type = "response")
            p_pred = ifelse(boost_pred > 0.5, "many", "few")
            
            boost_acc[k,1] = calc_acc(predicted = p_pred, actual = X_test$goals)
            boost_acc[k,2] = boost$interaction.depth
            boost_acc[k,3] = boost$shrinkage
            boost_acc[k,4] = n.trees_avr
            
            k = k + 1
        }
        browser()
        cv_error = mean(boost_acc[,1]) # cv error
        hyper_list[[count]] = cbind(cv_error,boost_acc[1,-1]) # save cv_error and hyper parameter combination
        # compute mean
        count = count + 1
        
      }
    }
  }
}

###################################################
# xgboost
xgboost_grid_search = function(gbtrain, max_depth, etas, min_child_weight,
                   subsample, colsample_bytree){
  
  xgb_grid = expand.grid(max_depth = max_depth, 
                         eta = etas,
                         min_child_weight= min_child_weight,
                         subsample = subsample,
                         colsample_bytree = colsample_bytree,
                         optimal_trees = 0,
                         test_logloss_mean = 0)
  
  set.seed(123)
  
  for (i in 1:nrow(xgb_grid)) {
    
    # training
    xgb_cv = xgb.cv(data = gbtrain, 
                    max_depth = xgb_grid$max_depth[i], 
                    eta = xgb_grid$eta[i], 
                    min_child_weight = xgb_grid$min_child_weight[i],
                    subsample = xgb_grid$subsample[i],
                    colsample_bytree = xgb_grid$colsample_bytree[i],
                    nrounds = 5000,
                    nfold = 5,   
                    objective = "binary:logistic", 
                    eval_metric = "error",
                    early_stopping_rounds = 50,  
                    verbose=0)
        
      browser()
      
      # add min training error and trees to grid
      xgb_grid$optimal_trees[i] <- which.min(xgb_cv$evaluation_log$test_error_mean)
      xgb_grid$test_logloss_mean[i] <- min(xgb_cv$evaluation_log$test_error_mean)
        
  }
  return(xgb_grid)
}


#################################
# shrinkage
shrink = function(x_train, y_train, alpha, x_test){
  cv.out = cv.glmnet(x_train, y_train, family = 'binomial', alpha = alpha) #10 fold Cross-Validation for lambda for RIDGE
  # Best cv Lambda
  bestlam_10 = cv.out$lambda.min
  # fit model with optimal lambda
  out = glmnet(x_train, y_train, family = "binomial", alpha=alpha, lambda=bestlam_10)
  # Predictions
  pred = predict(out, newx = x_test, type = "class") # predict classes 
  return(list(cv.out=cv.out, bestlamda = bestlam_10, model = out, pred = pred))
}

### SVM
### NN