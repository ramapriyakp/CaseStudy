# Case Study  projects

# Retail Analysis using SQL
## Skills: SQL, EDA, Data Visualization, Presentation    
https://github.com/ramapriyakp/CaseStudy/blob/main/01-retail.pdf   
Analysed operations data of a prominent global retailer  to gain insights into the retailers  operations.    
Investigated factors influencing operations by analyzing correlations between variables like order status, price, payment and freight performance, customer location, product attributes, and customer reviews.   
Analysed sales data by order volumes,revenue and region to  improve revenue.   
Recommended focusing on features like  average time to delivery, identify peak sales season and time adequate stocking and flexible payement options for increased customer satisfaction.   


# Media Company Business Case    
## Skills: Python, EDA, Data Visualization, Presentation    
https://github.com/ramapriyakp/CaseStudy/blob/main/02_media.ipynb    
Analysed movies and TV shows of a popular media company to decide type of shows/movies to produce and  grow business in different countries.     
Investigated the factors influencing content popularity and help grow the business.    
Leveraged popularity scores and actor ratings  to identify shows which can give business growth.    
Recommended strategies  like focusing on popular categories of shows, and geographical distribution  for more revenue.    


# Fitness Company Business Case    
## Skills: Python, EDA, Data Visualization, Presentation    
https://github.com/ramapriyakp/CaseStudy/blob/main/03_fitness.ipynb    
Analyzed customer data af Multinational fitness company to provide better product recommendations to customers.    
Create profile for each threadmill product  by profiling customer on marital status, usage, income, fitness, miles    
For each treadmill product, constructed two-way contingency tables and computed all conditional and marginal probabilities along with their insights/impact on the business.    
Profiled data on product,customer and gender to provide business insights    
Recommended product based on income and age.    


# Retail Confidence Interval and CLT    
## Skills: Python, EDA, Data Visualization, Presentation    
https://github.com/ramapriyakp/CaseStudy/blob/main/04_Stores.ipynb    
Analysed customer purchase at MNC Retail Store to understand spending habits of men and women.    
Used Central limit theorem to compute confidence intervals of average male and female spends and leverage this to make improvements    
There is no overlap of purchase( mean) CI for males and females.So buying patterns is different.    
Recommeded products to be priced wihin the CI range  since this is a good population parameter.    
Analysed CI with respect to gender and age to provide business insights    


# Mobility service provider - Hypothesis Testing      
## Skills: Python, EDA, Data Visualization, Hypothesis testing    
https://github.com/ramapriyakp/CaseStudy/blob/main/05_mobility.ipynb    
Anlysed customer data of leading  mobility service provider  to understand factors affecting  demand for shared electric cycles.    
Perform hypothesis tests to understand relationship between factors    
used 2 sample t-test to check effect of working day on no of cycles rented    
used ANOVA test to check  if No. of cycles rented varies with weather and season    
used Chi-square test to check if Weather is dependent on the season    
Recommended focusing on features like weather and season to increase ridership.    
Major demand is when natural factors are favourable for riding cycles    



# Logistics - Feature Engineering     
## Skills: Python, EDA, Data Visualization, Hypothesis testing    
https://github.com/ramapriyakp/CaseStudy/blob/main/06_Logistics.ipynb    
Anlysed operational data of leading logistics company to build a forecasting model      
manipulate data create useful features out of raw fields.    
Perform hypothesis tests to understand relationship between factors    
used data aggration technique to obtain higher level features for further analysis     
hypothesis test indicated that actual and osrm times differ, they should be minimised for better efficiency.    
Recommended focusing on Metro hubs for futher growing business    


# Hospitals - Hypothesis Testing      
# Skills: Python, EDA, Data Visualization, Hypothesis testing    
https://github.com/ramapriyakp/CaseStudy/blob/main/07_hospital.ipynb    
Analysed patient data of leading hospital  to know significant variables in predicting  hospitalization.    
Investigated factors influencing hospitalization like viral load,smoking and severity level    
used  t-test to conclude with 95% confidence that hospitalization charges  of smoker is higher  and 
mean "viral load" for 'male' and 'female' are equal.    
used ANOVA test to conclude mean "viral load" for 'female' are equal for severity level 0,1,2.    
Recommended people to avoid smoking to lessen hospitalisation    
patients of low severity level 0-2 form 85% of patients so more prvisions sohuld be made for these type of patients     
Since patients are uniformly distributed across regions. Similar policies should be implemented across regions to maintain this balance    


# Education - Linear Regression    
## Skills: Python, EDA, Data Visualization, Hypothesis testing    
https://github.com/ramapriyakp/CaseStudy/blob/main/08_Education.ipynb    
Help an Education Institute  build a model which can predict probability of getting into IVY league college.    
Analyzed student data to understand  important factors in graduate admissions and how these factors are interrelated among themselves    
used Linear Regression model to predict chance of admission.    
Recommended better student scores to improve chances of admission as they are correlated.    
Removing multi collinearity  features with high VIF can improve model accuracy.CGPA has the highest importance. University Rating , Research experienc, SOP strength have no statistical importance in predicting the chance of admission.    



# loan approval Logistic Regression     
## Skills: Python, EDA, Data Visualization, classfification metrics    
https://github.com/ramapriyakp/CaseStudy/blob/main/09_loan.ipynb    
Help an Education Institute  build a model which can predict probability of getting into IVY league Analysed loan data at online platform to predict probability if a credit line should be extended to customer.     
Investigated factors incfluencing loan approval to customer and analyzing correlations between variables     
used Logistic regression model for doing prediction Since this a classification problem     
Used feature importance to determine the the most useful features    
Recommended choosing models with high precision to keep false positives low.    


# Delivery Company : Neural Networks Regression    
## Skills: Python, EDA, Data Visualization, neural network design,tensorflow     
https://github.com/ramapriyakp/CaseStudy/blob/main/10_delivery.ipynb    
Analysed delivery data of leading  Intra-City Logistics  train a regression model for delivery time
estimation.    
Investigated factors influencing delivery time, created  new features like hour of day and day of the week for deeper understanding    
used Neural Networks Regression modelwith relu activation & Adam optimizer for doing prediction      
Neural Network regressor can capture capture non-linear relationship between input and output
it automatically learn features  and  generalize well to unseen data    


# Supply Chain company : CV Classification    
## Skills: Python, EDA,keras, CNN,Transfer learning,Computer vision    
https://github.com/ramapriyakp/CaseStudy/blob/main/11_supply.ipynb    
Processed data of fresh produce supply chain company to develop a vegetable image classifier.    
Used CNN for image classification task.    
Used callbacks like EarlyStopping to handle overfitting    
used ModelCheckpoint and  TensorBoard callback to fine tune and monitor model training.    
Leveraged pre-trained for better accuracy using tranfer learnig technique.    
Best Model has Test Accuracy: 93.73%    
using Batch Normalization,Dropout regulariztion mentods to boost performance    

 

# News Categorisation: business case    
## Skills: Python, EDA, NLP,text processing    
https://github.com/ramapriyakp/CaseStudy/blob/main/12_News.ipynb    
Analyzed news data in database of News company and categorised them into different categories based on their content.    
used NLP techniques like Bag of Words and TF-IDF for vectorizing text data.    
TF-IDF is more efficient since it gives importance to rare words and ignores common words.    
processed text data by removing stopwords, word tokenization and lemmatization.    
Compared model performance using Confusion Matrix and Classification Report.    
The best model is Naive Bayes with F1 score = 0.9699    


# Taxi Driver Churn : Ensemble Learning    
## Skills: Python, EDA, ensemble modelling    
https://github.com/ramapriyakp/CaseStudy/blob/main/13_taxi.ipynb    
Analyzed customer data at Taxi company to predict and prevent attrition, increasing customer satisfaction.    
used KNN Imputation of Missing numeric Values to Work with an imbalanced dataset    
another method is to use SMOTE for data imbalance    
created aggregate features  for better problem modelling.    
used ensemble models like random forest, AdaBoostClassifier for better performance    
Compared model performance using Confusion Matrix and Classification Report.    


# Tech-versity : Clustering in Learner Profiling    
## Skills:  K-means and Hierarchical Clustering, Unsupervised     
https://github.com/ramapriyakp/CaseStudy/blob/main/13_taxi.ipynb    
Analyzed learner dataset to profile the best companies and job positions from tech-versity database    
clustered  data at tech-versity to group learners with similar profiles, aiding in delivering a
more personalized learning journey.    
used K-means and Hierarchical Clustering to reveal hidden patterns within data.    
indentified top 3 roles and companies using clustering analysis.    
recommended  tailored content and provide specialized mentorship to customizing learning experience, increasing retention and satisfaction.    
 Skills: Python, EDA, feature engineering, data pre-processing, and unsupervised learning.    

