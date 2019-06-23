# Admission-Prediction-of-Students-applying-for-Masters
A ML based approach to predict chance of admission of students applying for master's degree in foreign university



Project Objective

1.	Application for master’s degree and that too in abroad is a very expensive and intensive.
2.	There needs to be certain reliable and precise sources to help students shortlist best possible universities in which they can study according to their academic profiles.
3.	The objective of this project is to develop a machine learning model which can predict chance of admission for aspiring candidates to different universities for Master’s degree in respective fields.
In this way, they can ascertain their chances of getting admitted in foreign universities and apply accordingly. If the probability is low they can prepare to retake the entrance exam and get  better grades.
 





Project Scope

To apply for a master's degree is a very expensive and  intensive work. With this model, students will guess their capacities and they will decide whether to apply for a master's degree or not.

In this project different machine learning algorithms are used, such as Classification and Regression to predict the probability of getting admission. The models are developed using Python.

Data Description
 
Abstract: The dataset contains student records and their respective result.
With this model, students can guess their capacities and they can decide whether to apply for a master's degree or not.
Dataset Characteristics:	Multivariate	Number of Instances:	500
Attribute Characteristics:	Real	Number of Attributes:	8
Associated Tasks:	Classification, Regression	Missing Values?	N/A




•	There are 9 columns: Serial No., GRE Score, TOEFL Score, University Rating, SOP, LOR , CGPA, Research, Chance of Admit.
•	There are no null records.
•	There are 500 samples in total.



Features in the dataset:

•	GRE Scores (290 to 340)
•	TOEFL Scores (92 to 120)
•	University Rating (1 to 5)
•	Statement of Purpose (1 to 5)
•	Letter of Recommendation Strength (1 to 5)
•	Undergraduate CGPA (6.8 to 9.92)
•	Research Experience (0 or 1)
•	Chance of Admit (0.34 to 0.97)
 
Attribute Information:


Input Features:	Range	Type
GRE Score	290 to 340	Continuous
TOEFL Scores	92 to 120	Continuous
University Rating	1 to 5	Categorical
Statement of Purpose	1 to 5	Categorical
Letter of Recommendation Strength	1 to 5	Categorical
Undergraduate CGPA	6.8 to 9.92	Continuous
Research Experience	0 or 1	Categorical
Output Features:		
Chance of Admit	0.34 to 0.97	Continuous

Conclusion:
1.	Linear Regression is the most appropriate Regression model for this project.

RMSE: 0.060865880415783113
Adjusted R2 method: 0.8188432567829629

2.	Logistic regression is the most appropriate Classification model for this project.

Accuracy Score: 0.86
Precision Score: 0.8360655737704918
Recall Score: 0.9272727272727272
F1 Score: 0.8793103448275862
AUC Score: 0.8525252525252526

