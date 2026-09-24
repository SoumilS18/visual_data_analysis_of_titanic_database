import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

titanic_dataset=pd.read_csv("train.csv")

# Data Cleaning

missing_values_per_column=titanic_dataset.isnull().sum() # seeing the number of missing values per column
print("total missing value per column are : \n",missing_values_per_column,"\n")
total_missing_values=missing_values_per_column.sum() # total missing values across the whole dataset
print("total missing values are : ",total_missing_values,"\n")
total_values=np.prod(titanic_dataset.shape)
percentage_of_missing_data_across_dataset=(total_missing_values/total_values)*100 # Percentage of missing values across dataset
print("the percenatge of missing values are: ",percentage_of_missing_data_across_dataset,"\n")
titanic_dataset.drop(columns="Cabin",inplace=True) # Removing Cabin Column as the majority of it is empty
titanic_dataset.Age=titanic_dataset.Age.fillna(titanic_dataset["Age"].median()) # filling the missing values in Age column from the Age median
titanic_dataset.Embarked=titanic_dataset.Embarked.ffill()# filling the missing values in Embarked column
print("the dataset after removal and filling of missing values:\n",titanic_dataset,"\n") # Dataset after removal and filling of missing values
print("total missing values after filling and removal are : ",titanic_dataset.isnull().sum().sum(),"\n")
print(titanic_dataset.duplicated(),"\n") # to mark duplicated rows
titanic_dataset.drop_duplicates(inplace=True) # Removing Duplicate Rows
print("the database after removal of duplicates: \n",titanic_dataset,"\n") # Checking the database after the removal of the duplicate rows, here there were no duplicate rows hence no change in dataset
print("the shape of database after removal of duplicates: \n",titanic_dataset.shape,"\n") # seeing if any changes were made or not after removal of rows , no changes were made here
print("different data types of columns of the dataset are :\n",titanic_dataset.dtypes,"\n") # checking if the data types for any column requires a change or not, here from this output it can be verified there is no need to change any data time
print("unique values in \"Sex\" column are: \n",titanic_dataset["Sex"].unique(),"\n") # to verify if there are no inconsistencies across the "Sex" column -> No such inconsistencies
print("unique values in \"Embarked\" column are: \n",titanic_dataset["Embarked"].unique(),"\n") # to verify if there are no inconsistencies across the "Embarked" column -> No such inconsistencies

# Data Analysis

# Survival by gender 
sns.countplot(data=titanic_dataset, x="Sex", hue="Survived")
plt.show()

# Survival by class 
sns.countplot(x="Pclass",hue="Survived",data=titanic_dataset)
plt.show()

# Age Distribution by survival
survivedtaset=titanic_dataset[titanic_dataset["Age"]==1]
sns.histplot(x="Age",hue="Survived",kde=True,data=titanic_dataset)
plt.show()

# Age V/s Fare (Male and Female)
sns.scatterplot(x="Age",y="Fare",hue="Sex",data=titanic_dataset)
plt.show()

# Corelation Heatmap
numeric_data = titanic_dataset.select_dtypes(include="number")

sns.heatmap(
    numeric_data.corr(),
    annot=True
)
plt.show()

# Embarked and Survived Heatmap
xyz=titanic_dataset.pivot_table(index="Embarked",columns="Survived",values="PassengerId",aggfunc="count")
sns.heatmap(xyz)
plt.show()

# Emabarked Survival Analysis :
sns.countplot(x="Embarked",hue="Survived",data=titanic_dataset)
plt.show()

# Survvial according to age group :
conditions=[(titanic_dataset["Age"]>=60),(titanic_dataset["Age"]>=18),(titanic_dataset["Age"]>=0)]
choices=["Senior Citizen","Adult","Children"]
titanic_dataset["Agegroup"]=np.select(conditions,choices,default="Unknown")
sns.countplot(x="Agegroup",hue="Survived",data=titanic_dataset)
plt.show()

# Survival according to family size : 
titanic_dataset["FamilySize"]=titanic_dataset["SibSp"]+titanic_dataset["Parch"]+1
sns.boxplot(x="Survived", y="FamilySize", data=titanic_dataset)
plt.show()


# Analysis of passenegers age : 
sns.distplot(x=titanic_dataset["Age"],rug=True)
plt.show()

# all numericl database co-relation :
sns.pairplot(titanic_dataset)
plt.show()

# Age V/s Fare jointplot
sns.jointplot(x="Age",y="Fare",kind="hex",data=titanic_dataset)
plt.show()

# Embarked V/s Fare 
sns.catplot(x="Embarked",y="Fare",hue="Sex",kind="violin",data=titanic_dataset)
plt.show()

# Insights From the data analysis
print("\n\nInsights from the data analysis:\n")
print("A total of 891 passenegers were traveling in titanic")
print("559 passengers died while only 332 survived")
print("Male Passengers were more than Female passenegers")
print("Females survived more than Males")
print("Higher Passeneger class had more chances of survival")
print("passengers embarked on Port C had the highest chances of survival")
print("People traveling Alone had the lesser chances of survival than people travelling in family")
print("People of family of size 4 had the highest chances of survival")
print("Children had the higher chances of survival than the adults and Senior citizens")
print("Cabin Information was missing for the majority of cases which resulted in the deletion of the column")
print("Average Fare was: $32.20")