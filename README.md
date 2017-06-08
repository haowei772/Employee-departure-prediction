# Employee-departure-prediction

## High level description of project:  
Develop a model to predict the departure of employees from a company based on HR data.

## Specific aims of the project:  
1. Develop a predictive model to determine which employees will leave the company based on historical data.  
2. Try to understand the major reasons that lead to departure of senior/experienced employees.

## Phase I: Develop a classification model with satisfactory sensitivity

### EDA:  
  Data is obtained from Kaggle Human-Resources-Analytics dataset.   
  The data contains 10 features: 'satisfaction_level', 'last_evaluation', 'number_project',  'average_montly_hours', 'time_spend_company', 'Work_accident', 'left', 'promotion_last_5years', 'sales', 'salary'.  

  The percentage of the departed employees is 23%, indicating that the class is relatively balanced, and there is no need for resampling.

  Through inspection of histograms of all numeric features, I noticed that the distribution of 'satisfaction_level’ has an interesting pattern with a separated peak of score equals 0.1, indicating a high proportion of unsatisfied employees.

  ![alt text](https://github.com/haowei772/Employee-departure-prediction/blob/master/figures/satisfaction_level.png)

  Through inspection of the scatter_matrix, I did not find numeric features with high collinearity

  ![alt text](https://github.com/haowei772/Employee-departure-prediction/blob/master/figures/scatter_matrix.png)

### Modeling and evaluation:
Select evaluation metrics. Use accuracy and recall as the evaluation metrics because accuracy provides a general view of the model performance and recall focuses on the employees who has departed.  

Split the dataset into training and testing dataset with 80% for training and 20% for testing.  

Started with the logistic regression model as a simple baseline. Then experimented with a random forest model and a k-nearst neighbors model.

*Table 1. Model evaluation.*  
| Models                             |    Accuracy   |     Recall    |
| ---------------------------------- |:-------------:|:-------------:|
| Logistic regression model          |      0.792    |     0.365     |
| Random forest model                |      0.994    |     0.981     |
| k-nearest neighbors model          |      0.945    |     0.943     |

The 'satisfaction_level' is the most important feature affecting employees’ departure.
![alt text](https://github.com/haowei772/Employee-departure-prediction/blob/master/figures/feature_importance.png)

Further investigation on ‘satisfaction_level’ of departed employees reveals three groups of departed employees:  

![alt text](https://github.com/haowei772/Employee-departure-prediction/blob/master/figures/high_risk_employees.png)

 *Group A* consists of employees with high ‘last_evaluation’, high ‘number_project’, and very low ‘satisfaction_level’. It is noticeable that employees with number of projects beyond 5 have very low satisfaction scores.These are the senior/experienced employees that carry out high number of projects (6 to 7) but have very low satisfaction, and tend to leave the company. This group should be the target for the company to retain with highest priority.  

 *Group B* consists of employees with low ‘last_evaluation’, low ‘number_project’ (mostly 2 projects), and medium ‘satisfaction_level’. These employees have relative low performance with medium level of satisfaction and the company should target this group to increase their performance and satisfaction.  

 *Group C* consists of employees with high ‘last_evaluation’, medium ‘number_project’ (4 to 5 projects), and high ‘satisfaction_level’.  These are highly performing and satisfied employees, but they leave the company for unidentified reasons.There is relatively little that the company can do to retain them.

## Phase II: Migrate the random forest model to Databricks Spark platform.  
(Project has been done, and the summary is to be finished)
