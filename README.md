# EX-01 Implementation of Data Preprocessing and Data Analysis
## Aim:
To implement Data analysis and data preprocessing using a dataset.&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;
<br>

## Algorithm:
- Step 1: Import the dataset necessary.
- Step 2: Perform Data Cleaning process by analyzing sum of Null values in each column a dataset.
- Step 3: Perform Categorical data analysis.
- Step 4: Use Sklearn tool from python to perform data preprocessing such as encoding and scaling.
- Step 5: Implement Quantile transfomer to make the column value more normalized.
- Step 6: Analyzing the dataset using visualizing tools form matplot library or seaborn.
<br>

## Program:
### Importing required python libraries
```Python
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
```
<br>

### Importing and Analyzing the Dataset
```Python
df=pd.read_csv("Toyota.csv")
df.head(10)
df.isnull().sum()
df.info()
```
<img width=50% valign=top height=17% src="https://github.com/user-attachments/assets/7f78c58b-9245-452a-919e-2ac4f44ce581">

![Screenshot 2024-09-13 091100](https://github.com/user-attachments/assets/404d7cbc-9c37-435f-956b-534f7d2b64bb)

![Screenshot 2024-09-13 090830](https://github.com/user-attachments/assets/abc9ecd0-c620-49e3-9f18-8fa509b14fd7)

### Preprocessing the Data
```Python
df=df.drop(df[df['KM'] == '??'].index)
df=df.drop(df[df['HP']=='????'].index)
df=df.drop('Unnamed: 0',axis=1)
df['Doors']=df['Doors'].replace({'three':3,'four':4,'five':5}).astype(int)
df[['FuelType','MetColor']]=df[['FuelType','MetColor']].fillna(method='ffill')
df[['Age']]=df[['Age']].fillna(df['Age'].mean()).astype(int)
df[['KM','HP']]=df[['KM','HP']].astype(int)
```
<br>

![Screenshot 2024-09-13 091107](https://github.com/user-attachments/assets/ab27fc29-5f90-4864-ad20-c2fa73ede099)

### Detecting and Removing Outliers:
```Python
numeric= ['Price','Age','KM','HP','CC','Automatic','Weight']
plt.figure(figsize=(8, 3 * ((len(numeric)) // 3)))
for i, column in enumerate(numeric, 1):
    plt.subplot((len(numeric)+3 ) // 3, 3, i)
    sns.boxplot(x=df[column])
    plt.title(f'{column}')
plt.tight_layout()
plt.show()
for column in numeric:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
plt.figure(figsize=(8, 2 * ((len(numeric)+2) // 3)))  
for i, column in enumerate(numeric, 1):
    plt.subplot(((len(numeric) + 2) // 3), 3, i)  
    sns.boxplot(x=df[column])
    plt.title(f'{column}')
plt.tight_layout()
plt.show()
```
![Screenshot 2024-09-13 091121](https://github.com/user-attachments/assets/fd4d9a95-1c24-4b81-92fc-90bfc0182363)


**After Removing Outliers** 
![image](https://github.com/user-attachments/assets/ddf15859-0e34-40d7-8e0a-ddee4516d79a)

### Identifying Categorical data and performing Categorical analysis

```Python
category=df.select_dtypes(include=['object'])
count=category.value_counts()
count.plot(kind='bar', color='yellow')
plt.title('Count of Fuel Types')
plt.xlabel('Fuel Type')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.grid(axis='x', linestyle='solid', alpha=0.7)
plt.show()
```

![Screenshot 2024-09-13 091142](https://github.com/user-attachments/assets/95aa9f70-d73a-41f7-996d-affa22a7d1ff)

### Performing Bivariate Analysis
```Python
sns.lineplot(x=df['Age'], y=df['Price'])
plt.title('Bivariate Analysis: Price vs Age')
plt.xlabel('Age')
plt.ylabel('Price')
plt.grid(True)
plt.show()
```

![Screenshot 2024-09-13 091150](https://github.com/user-attachments/assets/7a3fa459-7dd5-4d45-a32f-0c4e5a4db233)

### Performing Multivariate Analysis
```Python
sns.countplot(x='HP', hue='MetColor', data=df)
plt.title('Count Plot: HorsePower and MetalColor')
plt.grid(True)
plt.show()

### Data Encoding
```Python
le=LabelEncoder()
df['FuelType']=le.fit_transform(df['FuelType'])
```

![Screenshot 2024-09-13 091158](https://github.com/user-attachments/assets/29812030-5840-4f2f-b868-da00a00903b7)

### Data Scaling
```Python
scl=MinMaxScaler()
df[['Age']]=scl.fit_transform(df[['Age']])
```

### Data Visualization:
```Python
data = df.drop(columns=['Automatic'])  # Drop the 'Automatic' column
plt.figure(figsize=(10, 6))
sns.heatmap(data.corr(), annot=True, fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

plt.figure(figsize=(8, 6))
sns.barplot(x='FuelType', y='Price', data=df, estimator='mean')  # Bar plot for average price
plt.title('Average Car Price by Fuel Type')
plt.xlabel('Fuel Type')
plt.ylabel('Average Price')
plt.grid(True)
plt.show()
```

#### Heat Map
![Screenshot 2024-09-13 091212](https://github.com/user-attachments/assets/20a3f022-97dd-4d56-ba21-64a551f017ef)

#### Line Plot

![Screenshot 2024-09-13 091220](https://github.com/user-attachments/assets/c6a7bb1e-2a42-4945-a74b-abf9834069b4)


## Result:
Thus Data analysis and Data preprocessing implemeted using a dataset.
