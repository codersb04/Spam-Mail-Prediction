# Spam-Mail-Prediction
## Task
Build a Machine Learning Model to detect whether the incoming mail is spam.</br>
This comes under **Binary Classification Supervised Learning** as the outcome is categorised as 2 options spam or ham.</br>
Used **Logistic Regression** algorithm to implement the model.</br>
Tool Used - Jupyter Notebook, Python 

## Dataset
Dataset is taken from Kaggel's "Spam Mails Dataset". Dataset contains 5572 records of different mails and Category defineing whether spam or ham. The features are:
- Category: Spam or Ham
- Message: Mail in text format

Source: https://www.kaggle.com/datasets/venky73/spam-mails-dataset

## Steps Involved
### Import Dependencies
The libraries needed for the implementation are:

- NumPy
- Pandas
- train_test_split from sklearn.model_selection
- TfidVectorizer from sklearn.feature_extraction.text
- LogisticRegression from linear_model
- Accuracy Score from sklearn.metrics

 ### Data Collection and Analysis
 - Load the dataset to pandas dataframe using read_csv method
 - Use function such as describe, shape, head and value counts to know more about data

### Data Preprocessing
- Label Encoding: Replace the categorical data of Category column with the numeric data, Spam:0 and Ham: 1
- Handling the Unbalanced Data: The Above Data is unbalanced as we have 70%(4825) data in ham category and just 20%(747) in Spam, This will affect the model Performance. Under Sampling method is used to build a sample dataset containing similar distribution of Spam and Ham. Randomly choosing 747 data from 4825 Ham category in order to make the spam and ham dataset equal for further analysis. 
- Separate the the target and feature.
  1. Target: Category
  2. Feature: Message
### Model Building
- Splitting the data into training and test data. With 20% data for testing and rest 80% for the training
- Trasform the text data of Message Column to feature vector using TFidVectorizer library
- Train the model using Logistic Regression Algorithm
### Model Evaluation
Model evaluation is performed on both training and test data, in order to avoid "Overfitting" of the model.
- Accuarcy Score of Trained Data:  0.9841004184100418
- Accuarcy Score of Test Data:  0.9565217391304348
### Build a Predictive Syste,
- Take a random data from the dataset
- Convert it's text data to feature vector
- Feed the data to model and Predict the output
- Print the predicted output</br></br></br>

Reference: Project 17. Spam Mail Prediction using Machine Learning with Python | Machine Learning Projects, Siddhardhan, https://www.youtube.com/watch?v=rxkGItX5gGE&list=PLfFghEzKVmjvuSA67LszN1dZ-Dd_pkus6&index=18
