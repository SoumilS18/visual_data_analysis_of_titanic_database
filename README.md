# Titanic Data Visualization using Matplotlib

## 📌 Project Overview

This project performs data analysis and visualization on the **Titanic dataset** using **Pandas, NumPy, and Matplotlib**.

The main objective is to understand patterns in passenger demographics, fares, passenger classes, family sizes, and survival rates through meaningful visualizations.

This project is part of my learning journey in **Data Analysis and Machine Learning**, following my study of Python, NumPy, Pandas, and Matplotlib.

---

## 🎯 Objectives

The project aims to:

- Explore the Titanic dataset.
- Understand the distribution of passenger attributes.
- Analyze passenger survival patterns.
- Study relationships between different variables.
- Create meaningful visualizations using Matplotlib.
- Extract and communicate insights from visualizations.

---

## 🛠️ Technologies Used

- **Python**
- **NumPy** - Numerical operations
- **Pandas** - Data loading, manipulation, and analysis
- **Matplotlib** - Data visualization
- **Jupyter Notebook**

---

## 📂 Dataset

The project uses the **Titanic dataset**, containing information about passengers aboard the RMS Titanic.

Important columns include:

| Column | Description |
|---|---|
| `PassengerId` | Unique passenger identifier |
| `Survived` | Survival status |
| `Pclass` | Passenger class |
| `Name` | Passenger name |
| `Sex` | Passenger gender |
| `Age` | Passenger age |
| `SibSp` | Number of siblings/spouses aboard |
| `Parch` | Number of parents/children aboard |
| `Fare` | Passenger fare |
| `Cabin` | Cabin information |
| `Embarked` | Port of embarkation |

---

## 📊 Analysis and Visualizations

The project explores the following areas:

### 1. Passenger Age Distribution

A histogram is used to understand the distribution of passenger ages.

### 2. Fare Distribution

A histogram is used to examine the distribution of passenger fares.

### 3. Gender Distribution

A bar chart compares the number of male and female passengers.

### 4. Passenger Class Distribution

A bar chart shows the number of passengers belonging to each passenger class.

### 5. Overall Survival

A visualization compares the number of passengers who survived and did not survive.

### 6. Survival by Gender

A grouped visualization compares survival outcomes between male and female passengers.

### 7. Survival by Passenger Class

The project examines the relationship between passenger class and survival.

### 8. Age vs Fare

A scatter plot is used to investigate the relationship between passenger age and fare.

### 9. Fare Distribution by Passenger Class

A box plot compares fare distributions across different passenger classes.

### 10. Family Size Analysis

Family size is calculated using:

```python
FamilySize = SibSp + Parch + 1
