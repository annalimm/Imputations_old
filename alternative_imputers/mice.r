library(mice)
impute_mice <- function(X_miss, maxit, m, seed, meth=NULL){
  imp <- mice(X_miss, maxit = maxit, meth = meth, m=m, seed = seed, print = FALSE)
  return(complete(imp))
}

