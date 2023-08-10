data_prep = function(df){
  
  # delete undefined rows
  X_raw = df
  X = X_raw[!(X_raw$HM5=="M"),] # all lines are deleted where HM5 == M (implies also deleting HM4, HM3,HM2, HM1)
  
  # create target
  goals = X$FTHG + X$FTAG 
  goals = ifelse(goals >= 2.5, 1, 0)
  goals = factor(goals, labels = c("few", "many"))
  
  X = cbind(X, goals)
  
  # FEATURE MATRIX: All variables, with characters as factors
  X[sapply(X, is.character)] = lapply(X[sapply(X, is.character)], as.factor)
  
  X = na.omit(X)
  
  # scale the rest: 
  X$HTGS = X$HTGS / X$MW
  X$ATGS = X$ATGS / X$MW
  X$HTGC = X$HTGC / X$MW
  X$ATGC = X$ATGC / X$MW
  
  # delete non-aggregated variables
  X = subset(X, select = -c(X, Date, HomeTeam, AwayTeam, FTHG, FTAG, MW, Unnamed..0, FTR, HS, 
                            AS, HST, AST, HC, AC, HF, AF, HY, AY, HR, AR, HTFormPtsStr, ATFormPtsStr)) 
  
  return(X)
}

# training and test set
train_test = function(X){
  set.seed(123)
  train = sample(1:nrow(X), size = nrow(X)*0.8) # size = number of times are 80% of "observations" (number of observation into training set)
  
  # training Data
  X_train = X[train,] # training Data
  # test data
  X_test = X[-train,] # test Data
  
  return(list(X_train, X_test))
}

# accuracy function
calc_acc = function(actual, predicted) {
  return(mean(actual == predicted))
}


# cv tuning using care package and elastic net: extracting main information
el_result = function(caret_fit) {
  best = which(rownames(caret_fit$results) == rownames(caret_fit$bestTune)) # define slicer
  best_result = caret_fit$results[best, ] # extract best row (with smallest deveance)
  rownames(best_result) = NULL # delete ROW NA;E
  best_result # show result
}