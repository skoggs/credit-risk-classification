# Module 12 Report Template

## Overview of the Analysis

The purpose of this analysis is to test predictive power of financial information when combined with machine learning processes. With loan size, interest rate, borrower income, debt-to-income ratio, number of accounts, derogatory marks, and total debt, we seek to predict whether a given loan should be classified as high-risk. 

The high-risk status is a binary classification (marked as 1's in our data set). Non-high risk status loans are classified as healthy loans (marked as 0's in our data set). There are 75,000 healthy loans versus 2,500 high-risk loans in our data set, meaning the former has more data to work with.

To begin training our machine learning algorithms, we first split the data into both testing and training groups. The training groups include all of our original data, including whether a given loan is high-risk, meaning it has the benefit of hindsight in its modeling. Testing groups only have the other variables available, so they will be making educated guesses about loan risk with the training data but will not "know" the answers. 

The model we will be using is a logistic regression model, which estimates a binary dependent variable (our loan status) by utilizing an assortment of independent variables (our other financial information in the data set). The particular one used in this program is powered by Limited-Memory BFGS, or lbfgs, an optimization algorithm. 

Also used is Imbalanced-Learn's RandomOverSampler module. This generates a new sample of data based on an existing data set, which we use to create a second data set to model with. This is done to compensate for the small volume of high-risk loan data relative to the healthy loan data. 

## Results

Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all machine learning models.

* Machine Learning Model 1 (Original Data):
  Balanced Accuracy Score: .9937
  Precision (Healthy Loan): 1.0
  Recall (Healthy Loan): 0.99
  Precision (High-Risk Loan): 0.85
  Recall (High-Risk Loan): 0.91

* Machine Learning Model 2 (Resampled Data):
  Balanced Accuracy Score: .9937
  Precision (Healthy Loan): 1.0
  Recall (Healthy Loan): 0.99
  Precision (High-Risk Loan): 0.84
  Recall (High-Risk Loan): 0.99
 

## Summary

Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. For example:
* Which one seems to perform best? How do you know it performs best?
* Does performance depend on the problem we are trying to solve? (For example, is it more important to predict the `1`'s, or predict the `0`'s? )

Out of our two models, the resampled model seemed to perform best. While both have accuracy scores of about .9937 and are quite accurate at identifying healthy loans, the resampled benefit has the additional benefit of a very low recall rate for high-risk loans. This means when it identifies a high-risk loan, only a very small percent of the time does it result in a false positive.

The predictive power of the models are surprisingly mixed. Healthy loans can be classified at a near 100% accuracy, so a .9937 balanced accuracy score does not mean a whole lot for our purposes despite looking impressive at first glance. Instead, we want to track down high-risk loans with a very low risk of false negatives, which neither model seems to do with modest Precision scores of high-risk loans in the .84-.85 range. Our models are great for identifying an initial batch of high-risk loans with few false positives, but the high number of false negatives mean that the model will likely need to be supplemented by other methods. 
