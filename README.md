UC Berkeley - Professional Data Science and Machine Learning - Assignment 17.1

# Goal of this exercise:
The goal is to compare the performance of the classifiers (k-nearest neighbors, logistic regression, decision trees, and support vector machines). 

# Link to Jupyter notebook:
 https://github.com/vinay-jain-1/classifier-performance/blob/main/prompt_III.ipynb

# Business problem
Compare the performance of the four classifier models (k-nearest neighbors, logistic regression, decision trees, and support vector machines) learned so far. The idea is to use a Banking dataset that has information for over 40K customers who signed up for a term deposit. The dataset contains 21 columns (including target variable 'y') describing the features for bank client characteristics, campaign information and social and economic context attributes. 

Need to compare both base model training time and scores and the same metrics after tuning the hyperparameters across the 4 models.

Besides that also showcase which attributes have what kind of importance so that appropriate recommendations can be provided to the bank for future campaigns.

# Approach
## Engineering features
1. After reading the data in the dataframe, all the 'unknown' values were handled by replacing with the most popular value in each of the respective columns containing unknowns. These columns were: 'job', 'marital', 'education', 'default', 'housing', 'loan'.
2. One hot encoding was applied for categorical columns.
3. The target column was changed from a character column ('y'/'n') to integer column containing 1 (for yes) and 0 (for no).

## Train/Test split
Data was split into 80-20 for training and testing. Data was then scaled using StandardScaler. 

## Baseline 
A dubm baseline metric was defined that identified the percentage of most popular value. In this case, it was about 89% for 'no'.

## Simple model
LogisticRegression was then used to create a basic model for the data. Its baseline accuracy was recorded at 89.66%. 

## Compare the performance of the four classifier models with default settings
The four models were created, fit and compared for training time, training dataset accuracy and test dataset accuracy. Here is how they fared:<br>
![](https://github.com/vinay-jain-1/classifier-performance/blob/main/images/Default%20models.png)<br>
(Accuracy was identified as the most suitable scoring technique for this dataset).

## Improving the model

### More feature engineering applied to improve the models
1. Feature 'day_of_week' does not make sense to have a significant outcome on the possibility to doing a term deposit. So dropping it.
2. For 'poutcome', if the value is 'nonexistent', then change it to 'failure' if 'pdays' is not 999. (999 means no contact was made).
3. Replace 999 in the 'poutcome' column with the mean value of that column so it does not sway the influence on the outcome with its large value.
4. Do the remaining feature engineering like it was done before:
    a. handle 'unknown' values
    b. apply one-hot encoder on categorical features
    c. Encode the 'y' value using a Label Encoder.
5. Perform the train/test split.
6. Apply the MinMaxScaler (instead of StandardScaler as some of the columns have a negative values after scaling and that does not fit well with all classifiers).
7. Since there are 51 columns (after applying one-hot encoding), we need to reduce the dimensionality. So perform feature selection (using SelectKBest with k=20) to reduce dimensionality.

### Seven models with their hyperparameters and their explainability
Seven models were put through the ringer and have their hyperparameters tweaked to get the best training time, training score and test score. <p>
RandomizedSearchCV was used with cross validation fold of 3 and with 3 iterations.
<p>
After several iterations, these were the hyperparameters and their values chosen for each model:<br>

![](https://github.com/vinay-jain-1/classifier-performance/blob/main/images/Hyperparameters.png)

<p>

This is how the 7 models performed with their hyperparameters tuned:<br>

![](https://github.com/vinay-jain-1/classifier-performance/blob/main/images/Models%20with%20hyperparameters%20tuned.png)

### Comparing the default models to the ones that are hypertuned
#### Training time
Training times generally became much faster in all models except for the KNN model.<br>

![](https://github.com/vinay-jain-1/classifier-performance/blob/main/images/Models-Train%20time.png)

![](https://github.com/vinay-jain-1/classifier-performance/blob/main/images/SVM%20Models-Train%20time.png)

#### Train score (accuracy)
Training scores remained largely the same for all models except Decision Tree which is a good sign of not overfitting.

![](https://github.com/vinay-jain-1/classifier-performance/blob/main/images/Models-Train%20score.png)

![](https://github.com/vinay-jain-1/classifier-performance/blob/main/images/SVM%20Models-Train%20score.png)

#### Test score (accuracy)
Test scores remained largely the same for all models except that it increased by a good 5% for Decision Tree model.

![](https://github.com/vinay-jain-1/classifier-performance/blob/main/images/Models-Test%20score.png)

![](https://github.com/vinay-jain-1/classifier-performance/blob/main/images/SVM%20Models-Test%20score.png)

### Explanability of features and their importance with each model
#### Logistic Regression - Feature explanability
                        |-------------------------------|-------------|
                        |                      Feature  | Coefficient |
                        |-------------------------------|-------------|
                        |             poutcome_success  |   1.807072  |
                        |                cons.price.idx |   1.496229  |
                        |                    month_mar  |   1.023291  |
                        |                     euribor3m |   0.798893  |
                        |         poutcome_nonexistent  |   0.550798  |
                        |                     previous  |   0.484996  |
                        |                  job_retired  |   0.275786  |
                        |             contact_cellular  |   0.275294  |
                        |                  job_student  |   0.231307  |
                        |                    month_dec  |   0.206200  |
                        |  education_university.degree  |   0.105353  |
                        |                    month_oct  |   0.105111  |
                        |               marital_single  |   0.065226  |
                        |                    month_apr  |  -0.062489  |
                        |                    month_sep  |  -0.089932  |
                        |              job_blue-collar  |  -0.114341  |
                        |            contact_telephone  |  -0.273395  |
                        |                    month_may  |  -0.654663  |
                        |                  nr.employed  |  -1.705810  |
                        |                 emp.var.rate  |  -2.603888  |
                        |-------------------------------|-------------|

<p>
Logistic regression provides a good understanding of not just the weights but also how a feature may negatively impact the outcome. This ability to have negative coefficients was the standout factor about this classifier model.
<p>
This allows for a much easier interpretation and helps provide a very specific recommendation to the bank. 
<br>
For example, reaching out to customers where the previous campaign outcome was successful is one of the best ways to get a customer to get their term deposit again. Similarly, the higher the consumer price index (cons.price.idx) - which is a measure of inflation or banks offering higher interest rate, the higher the probability of having a customer subscribing for term deposit.
<br>
On the other hand, the quarterly indicator of employment variation rate has a inversely proportion impact such that the larger the value of emp.var.rate, the lower the changes of landing the term deposit.

### KNN - Feature explanability
KNN model also provides weights for each of the features that can then be used for providing specific recommendations to the bank. KNN's feature importance chart looks like this: <br>

                        |----------------|-----------------------------|
                        | Weight         | Feature                     |
                        |----------------|-----------------------------|
                        | 0.0113±0.0005  | poutcome_success            |
                        | 0.0091±0.0006  | euribor3m                   |
                        | 0.0087±0.0009  | education_university.degree |
                        | 0.0057±0.0004  | marital_single              |
                        | 0.0054±0.0009  | month_may                   |
                        | 0.0051±0.0004  | poutcome_nonexistent        |
                        | 0.0045±0.0008  | month_oct                   |
                        | 0.0034±0.0005  | job_retired                 |
                        | 0.0028±0.0004  | previous                    |
                        | 0.0027±0.0004  | cons.price.idx              |
                        | 0.0023±0.0006  | month_apr                   |
                        | 0.0020±0.0002  | job_blue-collar             |
                        | 0.0018±0.0002  | month_sep                   |
                        | 0.0016±0.0002  | month_mar                   |
                        | 0.0012±0.0004  | job_student                 |
                        | 0.0004±0.0001  | month_dec                   |
                        | 0.0003±0.0003  | contact_cellular            |
                        | 0.0002±0.0001  | nr.employed                 |
                        | 0.0001±0.0005  | contact_telephone           |
                        | 0.0000±0.0002  | emp.var.rate                |
                        |----------------|-----------------------------|

### Decision Tree - Feature explanability
Decision Tree model also provides a simplistic view of the features and their importance. Considering that this model gave the highest training and test scores, I would treat this feature importance matrix with highest confidence.
<p>

                        |-----------------------------|------------|
                        | Feature                     | Importance |
                        |-----------------------------|------------|
                        | nr.employed                 | 0.664706   |
                        | poutcome_success            | 0.128195   |
                        | month_apr                   | 0.046748   |
                        | month_mar                   | 0.044270   |
                        | euribor3m                   | 0.035452   |
                        | month_oct                   | 0.026354   |
                        | contact_cellular            | 0.020129   |
                        | cons.price.idx              | 0.014823   |
                        | poutcome_nonexistent        | 0.010510   |
                        | education_university.degree | 0.007886   |
                        | marital_single              | 0.000927   |
                        | month_may                   | 0.000000   |
                        | month_dec                   | 0.000000   |
                        | previous                    | 0.000000   |
                        | contact_telephone           | 0.000000   |
                        | emp.var.rate                | 0.000000   |
                        | job_student                 | 0.000000   |
                        | job_retired                 | 0.000000   |
                        | job_blue-collar             | 0.000000   |
                        | month_sep                   | 0.000000   |
                        |-----------------------------|------------|

### SVM (Linear) - Feature explanability
SVM (with a kernel of Linear) is the only model in the SVM family that provides feature explanability. Here is its feature explanability:<p>

        |------------------|-----------------------------|    
        |          Weight  | Feature                     |
        |------------------|-----------------------------|
        | 0.0265 ± 0.0006  | nr.employed                 |
        | 0.0171 ± 0.0002  | euribor3m                   |
        | 0.0130 ± 0.0005  | poutcome_success            |
        | 0.0044 ± 0.0008  | emp.var.rate                |
        | 0.0011 ± 0.0003  | poutcome_nonexistent        |
        | 0.0006 ± 0.0004  | contact_telephone           |
        | 0.0005 ± 0.0003  | month_mar                   |
        | 0.0004 ± 0.0005  | cons.price.idx              |
        | 0.0001 ± 0.0005  | month_may                   |
        | 0.0001 ± 0.0002  | marital_single              |
        | 0.0001 ± 0.0001  | contact_cellular            |
        | 0.0001 ± 0.0002  | education_university.degree |
        | 0.0000 ± 0.0001  | month_dec                   |
        | 0.0000 ± 0.0000  | month_oct                   |
        | 0.0000 ± 0.0001  | job_retired                 |
        | 0.0000 ± 0.0001  | job_blue-collar             |
        | -0.0000 ± 0.0001 | month_sep                   |
        | -0.0000 ± 0.0002 | previous                    |
        | -0.0001 ± 0.0000 | job_student                 |
        | -0.0002 ± 0.0001 | month_apr                   |
        |------------------|-----------------------------|

# Model Recommendation
Considering that the Decision Tree model tuned with hyperparameters provided a very high training and test acurracy score, coupled with the fact that it has an easy to explain list of features, the improved Decision Tree model would be the choice for this dataset.
<p>
It is also a good idea to use the Logistic Regression's coefficient sign to explain the positive or negative effect of the features on the outcome.

# Recommendation to the bank
These are the recommendations to improve the probability of having a customer subscribe to the term deposit at bank (listed in the order of decreasing importance-- first is the most important):
1. The highest swing factor is the number of employees - quarterly indicator. The higher the number, the more negatively it affects the probability of getting customers to subscribe to term deposits.
2. Having a successful previous outcome is a good boost to improving the probability.
3. Making the last contact in April has not turned out to be favorable.
4. However, the last contact in March turned out to be favorable.
5. The euribor3m interest rate is a good leading indicator of positively increasing the probability of term deposits. The higher the interest rate, the greater the chances.
6. Having the last contact in October showed improved probability.
7. Making contact over cellular phone has also showed improved probability.
8. cons.price.idx has showed to be a positive factor.
9. Customers who have not been contacted in the previous marketing campaign are the next positive factor.
10. Customers with an education of university degree are inclined to subsribe to term deposits.
11. Customers that have a marital status of single are the last one in the list.

# Questions that I pondered about and intend to do my own research:
1. I spent quite some time tuning the hyperparameters for each of the models. However, in some cases, I could not reach the same scores as the default model. In some cases, I have even tried to take the default parameter values to try things out but it still did not consistently match. Is it because of RandomizedGridSearch with cv=3? (that is the best explanation I could think of).
<p>
2. Is usage of SelectKBest(chi2, k=20) to narrow down the dimensionality appropriate in this case? What is a typical number to apply for k for commercial purposes? Using PCA could be an option too but then I would lose the ability to explain the features hence I did not choose PCA. Are there other such techniques that can be used for reducing dimensionality?
<p>