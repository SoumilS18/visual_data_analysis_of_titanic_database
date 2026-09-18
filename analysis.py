import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

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

# Data Visualization of the dataset

# No. of passengers in each class
classes=[3,1,2] # we set it according to the order of passengers_per_class.index
passengers_per_class=titanic_dataset["Pclass"].value_counts()
plt.bar(passengers_per_class.index,passengers_per_class.values)
plt.title("No. of passengers in each class")
plt.xlabel("Class")
plt.ylabel("No. of passengers")
plt.xticks(ticks=passengers_per_class.index,labels=classes)
plt.tight_layout()
plt.show()

# No of Passeneger of each gender
Sex=["Male","Female"] 
passengers_per_gender=titanic_dataset["Sex"].value_counts()
plt.barh(passengers_per_gender.index,passengers_per_gender.values)
plt.title("No. of passengers of each gender")
plt.xlabel("No. of passengers")
plt.ylabel("Gender")
plt.yticks(ticks=passengers_per_gender.index,labels=Sex) # using yticks as class shifted to y-axis
# also switch the x and y labels 
plt.tight_layout()
plt.show()      

# Age Distribution
bins=[0,10,20,30,40,50,60,70,80]
medianage=np.median(titanic_dataset["Age"])
plt.hist(titanic_dataset["Age"],bins=bins,edgecolor="Black",log=True)
plt.axvline(medianage,color="Red",label="Median Age")
plt.legend()
plt.xlabel("Ages")
plt.title("Age Distribution of the Passengers")
plt.ylabel("No of People")

plt.show()

# Fare Distribution
medianfare=np.median(titanic_dataset["Fare"])
bins=[0,50,100,150,200,250,300,350,400,450,500,550,600]
plt.hist(titanic_dataset["Fare"],bins=bins,color="Red",edgecolor="Black",log=True)
plt.axvline(medianfare,color="Blue",label="Median Fare")
plt.legend()
plt.xlabel("Fare")
plt.title("Fare Distribution of the Passengers")
plt.ylabel("No of People")

plt.show()

# Survival Count
s_d=["Dead","Survived"]
survived_and_dead_count=titanic_dataset["Survived"].value_counts()
plt.bar(survived_and_dead_count.index,survived_and_dead_count.values,color="Green")
plt.title("No. of passengers which survived and which didn't")
plt.xlabel("No. of passengers")
plt.ylabel("Survival/Death")
plt.xticks(ticks=survived_and_dead_count.index,labels=s_d)
plt.show()

# Survival by gender
data_by_gender=titanic_dataset.groupby("Sex")
surviaval_by_gender=data_by_gender["Survived"].value_counts().unstack()
surviaval_by_gender.plot(kind='bar')

plt.xlabel('Gender')
plt.ylabel('Number of Passengers')
plt.title('Survival by Gender')
plt.xticks(rotation=0)
plt.legend(['Died', 'Survived'])
plt.show()

# Survival by class
data_by_class=titanic_dataset.groupby("Pclass")
survival_by_class=data_by_class["Survived"].value_counts().unstack()
survival_by_class.plot(kind="bar")
plt.xlabel('Class')
plt.ylabel('Number of Passengers')
plt.title('Survival by Class')
plt.xticks(rotation=0)
plt.legend(['Died', 'Survived'])
plt.show()

# Age vs Fare
plt.scatter(titanic_dataset["Age"],titanic_dataset["Fare"],color="Red",edgecolors="Black",alpha=0.5)
plt.xlabel("Age")
plt.ylabel("Fare")
plt.title("Age V/s Fare")
plt.tight_layout()
plt.legend()
plt.show()

# Fare distribution by passenger class
class1 = titanic_dataset[titanic_dataset['Pclass'] == 1]['Fare']
class2 = titanic_dataset[titanic_dataset['Pclass'] == 2]['Fare']
class3 = titanic_dataset[titanic_dataset['Pclass'] == 3]['Fare']

plt.boxplot([class1, class2, class3])

plt.xlabel('Passenger Class')
plt.ylabel('Fare')
plt.title('Fare Distribution by Passenger Class')
plt.xticks([1, 2, 3], ['Pclass 1', 'Pclass 2', 'Pclass 3'])

plt.show()

# Family size
titanic_dataset["FamilySize"]=titanic_dataset["SibSp"]+titanic_dataset["Parch"]+1 # total no of Passenegrs either in family or alone in numerical way
titanic_dataset["Alone/Family"] = np.where(titanic_dataset["FamilySize"] > 1, "Family", "Alone") # marking if alone or with family\
average_family_size=titanic_dataset["FamilySize"].mean() # Average Family Size
family_size_count=titanic_dataset["FamilySize"].value_counts()
plt.bar(family_size_count.index,family_size_count.values,color="Blue")
plt.axvline(average_family_size)
plt.xlabel("Family Size")
plt.ylabel("No of people belonging to such families")
plt.title("Family Size")
plt.tight_layout()
plt.show()


# Family size vs survival

data_grouby_familysize=titanic_dataset.groupby("FamilySize") # Groupby Family Size
people_by_familysize = titanic_dataset["FamilySize"].value_counts() # Total no of people in a particular family size
survive_by_familysize = data_grouby_familysize["Survived"].value_counts().unstack(fill_value=0) 
survive_by_familysize.plot(kind='bar')

plt.xlabel("Family Size")
plt.ylabel("Number of People")
plt.title("Family Size vs Survival")
plt.xticks(rotation=0)
plt.legend(["Died", "Survived"])
plt.tight_layout()
plt.show()

# Age distribution of survivors vs non-survivors

died = titanic_dataset[titanic_dataset["Survived"] == 0]["Age"].dropna()
survived = titanic_dataset[titanic_dataset["Survived"] == 1]["Age"].dropna()

plt.hist(died, bins=20, alpha=0.5, label="Died")
plt.hist(survived, bins=20, alpha=0.5, label="Survived")

plt.title("Age Distribution of Survivors vs Non-Survivors")
plt.xlabel("Age")
plt.ylabel("No. of People")
plt.legend()
plt.tight_layout()
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