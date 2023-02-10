![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)

# customer-segmentation
The aim of this customer segmentation project is to categorize customers into different segments (A,B,C,D) based on features like Gender, Age, Profession and etc. By doing so, it will help group the customers based on their prefferences which would aid in coming up with marketing strategies got those segments.

This project uses sklearn to develop a machine learning model using Random Forest Classifier. **The model accuracy is 96%**. Below is a brief explanation on the steps included in developing the model.
 
 
 ## Data Loading
 Uses Pandas to read the csv file in the dataset folder.
 
 ## EDA
 ### Features grouped by segments
- **Gender By Segmentation**<br>
 ![img](/images/gender_by_segment.png "Gender By Segmentation")<br>
 Although there's no major difference between genders in each segments, we can see that all for segments are favored by male
* **Profession By Segments**<br>
 ![img](/images/profession_by_segment.png)<br>
 We can see that segments A, B and C are favored by Artists. However for Segment D, majority of the consumers are from Healthcare
+ **Ever_Married By Segments**<br>
 ![img](/images/evermarried_by_segment.png)<br>
 Similar to the previous graph, consumers for segments A, B and C are mostly married except for segment D<br>

From these graphs we can see that segments A, B and C are mostly Artists, Male and Married. But Segment D have consumers that are from the Healthcare industry, also male but mostly are not married.

### Data Distribution
#### Age 
![img](/images/age_dist.png)<br>
Most of the consumers are within the mid 20s and mid 40s

## Data Cleaning
1. Uses LabelEncoding from sklearn to encode values for categorical data
2. Uses KNN Imputation to handle null values

## Feature Selection & Data Preprocessing
Uses PCA to perform feature selection to 2 components<br>
Here's a scatter plot of the 4 segments within the 2 components<br>
![img](/images/kmeans_cust.png)

## Model Development
Model used: Random Forest Classifier<br>
Accuracy: 96%


