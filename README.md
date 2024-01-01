# deep-learning-challenge

## Background
Alphabet Soup, a non-profit foundation, aims to develop an algorithm for predicting the success of funding applicants. Leveraging your expertise in machine learning and neural networks, you will utilize the features within the given dataset to construct a binary classifier. This classifier will be designed to forecast the likelihood of success for applicants receiving funding from Alphabet Soup.

Provided by Alphabet Soup's business team, a CSV file encompasses information on over 34,000 organizations that have been recipients of funding from Alphabet Soup throughout the years. This dataset includes various columns containing metadata about each organization, detailing aspects such as:

* EIN and NAME—Identification columns
* APPLICATION_TYPE—Alphabet Soup application type
* AFFILIATION—Affiliated sector of industry
* CLASSIFICATION—Government organization classification
* USE_CASE—Use case for funding
* ORGANIZATION—Organization type
* STATUS—Active status
* INCOME_AMT—Income classification
* SPECIAL_CONSIDERATIONS—Special consideration for application
* ASK_AMT—Funding amount requested
* IS_SUCCESSFUL—Was the money used effectively

## Instructions
-------------------------------------------------------------------------
## Step 1: Preprocess the data


Leveraging your proficiency in Pandas and Scikit-Learn's StandardScaler(), it is necessary to preprocess the dataset to facilitate the compilation, training, and evaluation of the neural network model in the subsequent Step 2.

Following the guidance provided in the starter code, execute the preprocessing steps as instructed:

* Read the charity_data.csv into a Pandas DataFrame, ensuring to identify the target variable(s) and feature variable(s) in your model.
* Drop the EIN and NAME columns.
* Determine the count of unique values for each column.
* For columns with more than 10 unique values, ascertain the count of data points for each unique value.
* Utilize the count of data points for each unique value to establish a cutoff point for binning "rare" categorical variables into a new value, "Other," and subsequently validate the success of this binning.
* Apply pd.get_dummies() to encode categorical variables.
## Step 2: Compile, Train, and Evaluate the Model

Leveraging your understanding of TensorFlow, you will architect a neural network, specifically a deep learning model, to establish a binary classification system capable of predicting the success of an organization funded by Alphabet Soup, relying on the features within the dataset. An essential consideration involves determining the appropriate number of inputs, which subsequently guides the determination of the number of neurons and layers in your model. Following the completion of this crucial step, you will proceed to compile, train, and evaluate your binary classification model, ultimately computing the model's loss and accuracy.
* Continue using the jupter notebook where you’ve already performed the preprocessing steps from Step 1.
* Create a neural network model by assigning the number of input features and nodes for each layer using Tensorflow Keras.
* Create the first hidden layer and choose an appropriate activation function.
* If necessary, add a second hidden layer with an appropriate activation function.
* Create an output layer with an appropriate activation function.
* Check the structure of the model.
* Compile and train the model.
* Create a callback that saves the model's weights every 5 epochs.
* Evaluate the model using the test data to determine the loss and accuracy.
* Save and export your results to an HDF5 file, and name it      AlphabetSoupCharity.h5.
## Step 3: Optimize the Model

Using your knowledge of TensorFlow, optimize your model in order to achieve a target predictive accuracy higher than 75%. If you can't achieve an accuracy higher than 75%, you'll need to make at least three attempts to do so.

* Optimize your model in order to achieve a target predictive accuracy higher than 75% by using any or all of the following:
Adjusting the input data to ensure that there are no variables or outliers that are causing confusion in the model, such as:
Dropping more or fewer columns.
Creating more bins for rare occurrences in columns.
Increasing or decreasing the number of values for each bin.
* Adding more neurons to a hidden layer.
* Adding more hidden layers.
* Using different activation functions for the hidden layers.
* Adding or reducing the number of epochs to the training regimen.
NOTE: You will not lose points if your model does not achieve target performance, as long as you make three attempts at optimizing the model in your jupyter notebook.
* Create a new Jupyter Notebook file and name it AlphabetSoupCharity_Optimzation.ipynb.
* Import your dependencies, and read in the charity_data.csv to a Pandas DataFrame.
* Preprocess the dataset like you did in Step 1, taking into account any modifications to optimize the model.
* Design a neural network model, taking into account any modifications that will optimize the model to achieve higher than 75% accuracy.
* Save and export your results to an HDF5 file, and name it AlphabetSoupCharity_Optimization.h5.
-----------------------------------------------------------------------------

# Final Analysis and Report on the Neural Network Model

Presented here is my conclusive report and analysis of the Neural Network Model, which includes responses to the questions outlined in the assignment:

## Overview
The objective of the model was to develop an algorithm to assist Alphabet Soup in forecasting the success of funding applicants. It functioned as a binary classifier, demonstrating a relatively high accuracy in predicting the success or failure of funding.

## Results
------------------------------------------------------------------------------
Using bulleted lists and images to support your answers, address the following questions.

### Data Preprocessing

Q: What variable(s) are considered the target(s) for your model?

* A: 
The variable for the Target was identified as the column IS_SUCCESSFUL.

Q: What variable(s) are considered to be the features for your model?

* A: 
The following columns were considered as features for the model:
* NAME
* APPLICATION_TYPE
* AFFILIATION
* CLASSIFICATION
* USE_CASE
* ORGANIZATION
* STATUS
* INCOME_AMT
* SPECIAL_CONSIDERATIONS
* ASK_AMT

Q: What variable(s) are neither targets nor features, and should be removed from the input data?

* A: 

The column or variable that can be excluded is EIN, as it serves as an identifier for the applicant organization and does not influence the model's behavior.

### Compiling, Training, and Evaluating the Model

Q:How many neurons, layers, and activation functions did you select for your neural network model, and why?

* A:
In the improved iteration of the model, I incorporated three hidden layers, each with multiple neurons, resulting in an enhanced accuracy of 78%. In the initial model, there were only two layers. Despite keeping the number of epochs consistent between the initial and optimized models, introducing a third layer contributed to the increased accuracy of the model.

Q: Were you able to achieve the target model performance?

* A:
Certainly, through model optimization, I successfully elevated the accuracy from approximately 72% to slightly over 78%.

Q: What steps did you take to try and increase model performance?

* A: 
 The following steps were taken to optimize and increase the performance of the model:
 
Rather than eliminating both the EIN and Name columns, I opted to exclude only the EIN column. Additionally, I considered only the names that appeared more than five times.

In order to enhance the accuracy to over 75%, I introduced an additional activation layer to the model in the following sequence:

1st Layer - relu

2nd Layer - tanh

3rd Layer - sigmoid


It was noticed that by using tanh for the 2nd layer and sigmoid for the 3rd layer, instead of having both as sigmoid, the performance was elevated to over 78%.

## Summary:

Q:Summarize the overall results of the deep learning model. Include a recommendation for how a different model could solve this classification problem, and explain your recommendation.

* A: 
In summary, through model optimization, we achieved an accuracy exceeding 78%. This implies that we can accurately classify each point in the test data approximately 78% of the time. Essentially, an applicant has nearly an 80% chance of success if they meet the following criteria:

The NAME of the applicant appears more than 5 times (indicating they have applied more than 5 times).

The APPLICATION type is one of the following: T3, T4, T5, T6, and T19.

The application has one of the following values for CLASSIFICATION: C1000, C1200, C2000, C2100, and C3000.