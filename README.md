# Adult-Income-Classifier

Here I have used the adult income dataset from kaggle. 

The objective is to use the xgboost algorithm to train a classifier that can predict whether a person has an income greater than or lesser than 50000.

I have done some minor preprocessing on the dataset before running the xgboost algorithm. 
i) I first replaced the '?' values by nan.
ii) Next I replaced the nan values which are only present in 3 columns with the mode.
iii) Next I used the minmax scaler before using the xgboost algorithm

I was able to obtain a very high accuracy of 87%
