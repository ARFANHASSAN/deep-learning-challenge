

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