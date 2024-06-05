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

<p>

### Explanability of feature importance with each model
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

<table class="eli5-weights eli5-feature-importances" style="border-collapse:collapse;border:none;margin-top:0em;table-layout:auto;">
    <thead>
        <tr style="border:none;">
            <th style="padding:0 1em 0 0.5em;text-align:right;border:none;">Weight</th>
            <th style="padding:0 0.5em 0 0.5em;text-align:left;border:none;">Feature</th>
        </tr>
    </thead>
    <tbody>
        <tr style="background-color:hsl(120,100.00%,80.00%);border:none;">
            <td style="padding:0 1em 0 0.5em;text-align:right;border:none;">0.0113&plusmn;0.0005</td>
            <td style="padding:0 0.5em 0 0.5em;text-align:left;border:none;">poutcome_success</td>
        </tr>
        <tr style="background-color:hsl(120,100.00%,82.81%);border:none;">
            <td style="padding:0 1em 0 0.5em;text-align:right;border:none;">0.0091&plusmn;0.0006</td>
            <td style="padding:0 0.5em 0 0.5em;text-align:left;border:none;">euribor3m</td>
        </tr>
        <tr style="background-color:hsl(120,100.00%,83.35%);border:none;">
            <td style="padding:0 1em 0 0.5em;text-align:right;border:none;">0.0087&plusmn;0.0009</td>
            <td style="padding:0 0.5em 0 0.5em;text-align:left;border:none;">education_university.degree</td>
        </tr>
        <tr style="background-color:hsl(120,100.00%,87.70%);border:none;">
            <td style="padding:0 1em 0 0.5em;text-align:right;border:none;">0.0057&plusmn;0.0004</td>
            <td style="padding:0 0.5em 0 0.5em;text-align:left;border:none;">marital_single</td>
        </tr>
        <tr style="background-color:hsl(120,100.00%,88.03%);border:none;">
            <td style="padding:0 1em 0 0.5em;text-align:right;border:none;">0.0054&plusmn;0.0009</td>
            <td style="padding:0 0.5em 0 0.5em;text-align:left;border:none;">month_may</td>
        </tr>
        <tr style="background-color:hsl(120,100.00%,88.59%);border:none;">
            <td style="padding:0 1em 0 0.5em;text-align:right;border:none;">0.0051&plusmn;0.0004</td>
            <td style="padding:0 0.5em 0 0.5em;text-align:left;border:none;">poutcome_nonexistent</td>
        </tr>
        <tr style="background-color:hsl(120,100.00%,89.45%);border:none;">
            <td style="padding:0 1em 0 0.5em;text-align:right;border:none;">0.0045&plusmn;0.0008</td>
            <td style="padding:0 0.5em 0 0.5em;text-align:left;border:none;">month_oct</td>
        </tr>
        <tr style="background-color:hsl(120,100.00%,91.37%);border:none;">
            <td style="padding:0 1em 0 0.5em;text-align:right;border:none;">0.0034&plusmn;0.0005</td>
            <td style="padding:0 0.5em 0 0.5em;text-align:left;border:none;">job_retired</td>
        </tr>
        <tr style="background-color:hsl(120,100.00%,92.44%);border:none;">
            <td style="padding:0 1em 0 0.5em;text-align:right;border:none;">0.0028&plusmn;0.0004</td>
            <td style="padding:0 0.5em 0 0.5em;text-align:left;border:none;">previous</td>
        </tr>
        <tr style="background-color:hsl(120,100.00%,92.69%);border:none;">
            <td style="padding:0 1em 0 0.5em;text-align:right;border:none;">0.0027&plusmn;0.0004</td>
            <td style="padding:0 0.5em 0 0.5em;text-align:left;border:none;">cons.price.idx</td>
        </tr>
        <tr style="background-color:hsl(120,100.00%,93.41%);border:none;">
            <td style="padding:0 1em 0 0.5em;text-align:right;border:none;">0.0023&plusmn;0.0006</td>
            <td style="padding:0 0.5em 0 0.5em;text-align:left;border:none;">month_apr</td>
        </tr>
        <tr style="background-color:hsl(120,100.00%,94.01%);border:none;">
            <td style="padding:0 1em 0 0.5em;text-align:right;border:none;">0.0020&plusmn;0.0002</td>
            <td style="padding:0 0.5em 0 0.5em;text-align:left;border:none;">job_blue-collar</td>
        </tr>
        <tr style="background-color:hsl(120,100.00%,94.38%);border:none;">
            <td style="padding:0 1em 0 0.5em;text-align:right;border:none;">0.0018&plusmn;0.0002</td>
            <td style="padding:0 0.5em 0 0.5em;text-align:left;border:none;">month_sep</td>
        </tr>
        <tr style="background-color:hsl(120,100.00%,94.92%);border:none;">
            <td style="padding:0 1em 0 0.5em;text-align:right;border:none;">0.0016&plusmn;0.0002</td>
            <td style="padding:0 0.5em 0 0.5em;text-align:left;border:none;">month_mar</td>
        </tr>
        <tr style="background-color:hsl(120,100.00%,95.75%);border:none;">
            <td style="padding:0 1em 0 0.5em;text-align:right;border:none;">0.0012&plusmn;0.0004</td>
            <td style="padding:0 0.5em 0 0.5em;text-align:left;border:none;">job_student</td>
        </tr>
        <tr style="background-color:hsl(120,100.00%,97.95%);border:none;">
            <td style="padding:0 1em 0 0.5em;text-align:right;border:none;">0.0004&plusmn;0.0001</td>
            <td style="padding:0 0.5em 0 0.5em;text-align:left;border:none;">month_dec</td>
        </tr>
        <tr style="background-color:hsl(120,100.00%,98.50%);border:none;">
            <td style="padding:0 1em 0 0.5em;text-align:right;border:none;">0.0003&plusmn;0.0003</td>
            <td style="padding:0 0.5em 0 0.5em;text-align:left;border:none;">contact_cellular</td>
        </tr>
        <tr style="background-color:hsl(120,100.00%,98.89%);border:none;">
            <td style="padding:0 1em 0 0.5em;text-align:right;border:none;">0.0002&plusmn;0.0001</td>
            <td style="padding:0 0.5em 0 0.5em;text-align:left;border:none;">nr.employed</td>
        </tr>
        <tr style="background-color:hsl(120,100.00%,99.28%);border:none;">
            <td style="padding:0 1em 0 0.5em;text-align:right;border:none;">0.0001&plusmn;0.0005</td>
            <td style="padding:0 0.5em 0 0.5em;text-align:left;border:none;">contact_telephone</td>
        </tr>
        <tr style="background-color:hsl(120,100.00%,99.73%);border:none;">
            <td style="padding:0 1em 0 0.5em;text-align:right;border:none;">0.0000&plusmn;0.0002</td>
            <td style="padding:0 0.5em 0 0.5em;text-align:left;border:none;">emp.var.rate</td>
        </tr>
    </tbody>
</table>

    <style>table.eli5-weights tr:hover{filter:brightness(85%)}</style><table class="eli5-weights eli5-feature-importances" style="border-collapse:collapse;border:none;margin-top:0em;table-layout:auto;"><thead><tr style="border:none;"><th style="padding:0 1em 0 0.5em;text-align:right;border:none;">Weight</th><th style="padding:0 0.5em 0 0.5em;text-align:left;border:none;">Feature</th></tr></thead><tbody><tr style="background-color:hsl(120,100.00%,80.00%);border:none;"><td style="padding:0 1em 0 0.5em;text-align:right;border:none;">0.0113&plusmn;0.0005</td><td style="padding:0 0.5em 0 0.5em;text-align:left;border:none;">poutcome_success</td></tr><tr style="background-color:hsl(120,100.00%,82.81%);border:none;"><td style="padding:0 1em 0 0.5em;text-align:right;border:none;">0.0091&plusmn;0.0006</td><td style="padding:0 0.5em 0 0.5em;text-align:left;border:none;">euribor3m</td></tr><tr style="background-color:hsl(120,100.00%,83.35%);border:none;"><td style="padding:0 1em 0 0.5em;text-align:right;border:none;">0.0087&plusmn;0.0009</td><td style="padding:0 0.5em 0 0.5em;text-align:left;border:none;">education_university.degree</td></tr><tr style="background-color:hsl(120,100.00%,87.70%);border:none;"><td style="padding:0 1em 0 0.5em;text-align:right;border:none;">0.0057&plusmn;0.0004</td><td style="padding:0 0.5em 0 0.5em;text-align:left;border:none;">marital_single</td></tr><tr style="background-color:hsl(120,100.00%,88.03%);border:none;"><td style="padding:0 1em 0 0.5em;text-align:right;border:none;">0.0054&plusmn;0.0009</td><td style="padding:0 0.5em 0 0.5em;text-align:left;border:none;">month_may</td></tr><tr style="background-color:hsl(120,100.00%,88.59%);border:none;"><td style="padding:0 1em 0 0.5em;text-align:right;border:none;">0.0051&plusmn;0.0004</td><td style="padding:0 0.5em 0 0.5em;text-align:left;border:none;">poutcome_nonexistent</td></tr><tr style="background-color:hsl(120,100.00%,89.45%);border:none;"><td style="padding:0 1em 0 0.5em;text-align:right;border:none;">0.0045&plusmn;0.0008</td><td style="padding:0 0.5em 0 0.5em;text-align:left;border:none;">month_oct</td></tr><tr style="background-color:hsl(120,100.00%,91.37%);border:none;"><td style="padding:0 1em 0 0.5em;text-align:right;border:none;">0.0034&plusmn;0.0005</td><td style="padding:0 0.5em 0 0.5em;text-align:left;border:none;">job_retired</td></tr><tr style="background-color:hsl(120,100.00%,92.44%);border:none;"><td style="padding:0 1em 0 0.5em;text-align:right;border:none;">0.0028&plusmn;0.0004</td><td style="padding:0 0.5em 0 0.5em;text-align:left;border:none;">previous</td></tr><tr style="background-color:hsl(120,100.00%,92.69%);border:none;"><td style="padding:0 1em 0 0.5em;text-align:right;border:none;">0.0027&plusmn;0.0004</td><td style="padding:0 0.5em 0 0.5em;text-align:left;border:none;">cons.price.idx</td></tr><tr style="background-color:hsl(120,100.00%,93.41%);border:none;"><td style="padding:0 1em 0 0.5em;text-align:right;border:none;">0.0023&plusmn;0.0006</td><td style="padding:0 0.5em 0 0.5em;text-align:left;border:none;">month_apr</td></tr><tr style="background-color:hsl(120,100.00%,94.01%);border:none;"><td style="padding:0 1em 0 0.5em;text-align:right;border:none;">0.0020&plusmn;0.0002</td><td style="padding:0 0.5em 0 0.5em;text-align:left;border:none;">job_blue-collar</td></tr><tr style="background-color:hsl(120,100.00%,94.38%);border:none;"><td style="padding:0 1em 0 0.5em;text-align:right;border:none;">0.0018&plusmn;0.0002</td><td style="padding:0 0.5em 0 0.5em;text-align:left;border:none;">month_sep</td></tr><tr style="background-color:hsl(120,100.00%,94.92%);border:none;"><td style="padding:0 1em 0 0.5em;text-align:right;border:none;">0.0016&plusmn;0.0002</td><td style="padding:0 0.5em 0 0.5em;text-align:left;border:none;">month_mar</td></tr><tr style="background-color:hsl(120,100.00%,95.75%);border:none;"><td style="padding:0 1em 0 0.5em;text-align:right;border:none;">0.0012&plusmn;0.0004</td><td style="padding:0 0.5em 0 0.5em;text-align:left;border:none;">job_student</td></tr><tr style="background-color:hsl(120,100.00%,97.95%);border:none;"><td style="padding:0 1em 0 0.5em;text-align:right;border:none;">0.0004&plusmn;0.0001</td><td style="padding:0 0.5em 0 0.5em;text-align:left;border:none;">month_dec</td></tr><tr style="background-color:hsl(120,100.00%,98.50%);border:none;"><td style="padding:0 1em 0 0.5em;text-align:right;border:none;">0.0003&plusmn;0.0003</td><td style="padding:0 0.5em 0 0.5em;text-align:left;border:none;">contact_cellular</td></tr><tr style="background-color:hsl(120,100.00%,98.89%);border:none;"><td style="padding:0 1em 0 0.5em;text-align:right;border:none;">0.0002&plusmn;0.0001</td><td style="padding:0 0.5em 0 0.5em;text-align:left;border:none;">nr.employed</td></tr><tr style="background-color:hsl(120,100.00%,99.28%);border:none;"><td style="padding:0 1em 0 0.5em;text-align:right;border:none;">0.0001&plusmn;0.0005</td><td style="padding:0 0.5em 0 0.5em;text-align:left;border:none;">contact_telephone</td></tr><tr style="background-color:hsl(120,100.00%,99.73%);border:none;"><td style="padding:0 1em 0 0.5em;text-align:right;border:none;">0.0000&plusmn;0.0002</td><td style="padding:0 0.5em 0 0.5em;text-align:left;border:none;">emp.var.rate</td></tr></tbody></table>
    

    


    

    

    

    

    

    



