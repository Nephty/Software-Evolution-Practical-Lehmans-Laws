import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit

# First, import the dataset that contains the total lines of code, total blank lines, total comment lines, total number of files, and release date for each tag.

dataset = pd.read_csv('data/LawIncreasingGrowthDataset.csv')

# Filter out non-official releases such as alpha, beta, release candidates, milestones, branches, and unnumbered tags.
# Additionally, version numbers typically follow a Major.Minor.Patch format, but exceptions exist (e.g., 1.4, 3.3.0.1, 4.3.0.1, or 4.3.1.1).
# Versions with more than three components are disregarded, and two-component versions are treated as implicit three-component versions (e.g., 1.4 is interpreted as 1.4.0).  
# Write a regular expression statement to identify tags following semantic versioning (major.minor.patch) guidelines. You can use the pattern: '^v?[0-9]+\.[0-9]+(\.[0-9]+)?$'  
# Once you've applied this regex, print out the selected major.minor.patch dataset.

regex_pattern = r'^v?[0-9]+\.[0-9]+(\.[0-9]+)?$'
datasetA = dataset[dataset['tag'].str.match(regex_pattern)]

print(datasetA)

print("======")

# Starting from the previous data (**Dataset A**), for each Major.Minor.Patch tag group,
# select the earliest version (e.g., for a set of 4.4.0, 4.4.1, and 4.4.2, consider 4.4.0 and rename it to 4.4).

datasetA['Major_Minor'] = datasetA['tag'].apply(lambda x: '.'.join(x.split('.')[:2]))
datasetB = datasetA.groupby('Major_Minor', as_index=False).first()
datasetB.drop(columns=['Major_Minor'], inplace=True)
datasetB['tag'] = datasetB['tag'].str.extract(r'^(\d+.\d+)')

print(datasetB)

# Visualize four plots representing the evolution of a software project's metrics, including the number of lines of code, blank lines, comment lines, and the number of files, as follows:  

# 1. Plot the metrics against software version numbers:  
#    1.1) For version tags of the format major.minor.patch.  
#    1.2) For version tags of the format major.minor.  
#    Place these plots at positions (0,0) and (0,1) respectively.  
  
# 2. Plot the metrics against dates:  
#    2.1) For version tags of the format major.minor.patch.  
#    2.2) For version tags of the format major.minor.  
#    Place these plots right below the corresponding version-based plots, at positions (1,0) and (1,1).  

# Hint: Ensure proper datetime formatting for the dates in the datasets instead of storing them as strings.
# For example, if you're working with a DataFrame:
# fig, axes = plt.subplots(2, 2, figsize=(18, 10))  
# dataset_a[['code', 'blank', 'comment', 'nFiles', 'tag']].plot(x='tag', ax=axes[0, 0], legend=True)  
# .....  
# axes[0, 0].set_xlabel('Major.Minor.Patch versions')  
# .....  
# plt.tight_layout()  
# plt.show()

# Convert 'release_date' column to datetime
datasetB['release_date'] = pd.to_datetime(datasetB['release_date'])

# Visualize the data
fig, axes = plt.subplots(2, 2, figsize=(18, 10))

# Plot metrics against version numbers (major.minor.patch)
axes[0, 0].plot(datasetA['tag'], datasetA['code'], marker='o', linestyle='-', label='Lines of Code')
axes[0, 0].plot(datasetA['tag'], datasetA['blank'], marker='o', linestyle='-', label='Blank Lines')
axes[0, 0].plot(datasetA['tag'], datasetA['comment'], marker='o', linestyle='-', label='Comment Lines')
axes[0, 0].plot(datasetA['tag'], datasetA['nFiles'], marker='o', linestyle='-', label='Number of Files')
axes[0, 0].set_xlabel('Major.Minor.Patch versions')
axes[0, 0].set_ylabel('Metrics')
axes[0, 0].legend()

# Plot metrics against version numbers (major.minor)
axes[0, 1].plot(datasetB['tag'], datasetB['code'], marker='o', linestyle='-', label='Lines of Code')
axes[0, 1].plot(datasetB['tag'], datasetB['blank'], marker='o', linestyle='-', label='Blank Lines')
axes[0, 1].plot(datasetB['tag'], datasetB['comment'], marker='o', linestyle='-', label='Comment Lines')
axes[0, 1].plot(datasetB['tag'], datasetB['nFiles'], marker='o', linestyle='-', label='Number of Files')
axes[0, 1].set_xlabel('Major.Minor versions')
axes[0, 1].legend()

# Plot metrics against dates (major.minor.patch)
axes[1, 0].plot(datasetA['release_date'], datasetA['code'], marker='o', linestyle='-', label='Lines of Code')
axes[1, 0].plot(datasetA['release_date'], datasetA['blank'], marker='o', linestyle='-', label='Blank Lines')
axes[1, 0].plot(datasetA['release_date'], datasetA['comment'], marker='o', linestyle='-', label='Comment Lines')
axes[1, 0].plot(datasetA['release_date'], datasetA['nFiles'], marker='o', linestyle='-', label='Number of Files')
axes[1, 0].set_xlabel('Release Date')
axes[1, 0].set_ylabel('Metrics')
axes[1, 0].legend()

# Plot metrics against dates (major.minor)
axes[1, 1].plot(datasetB['release_date'], datasetB['code'], marker='o', linestyle='-', label='Lines of Code')
axes[1, 1].plot(datasetB['release_date'], datasetB['blank'], marker='o', linestyle='-', label='Blank Lines')
axes[1, 1].plot(datasetB['release_date'], datasetB['comment'], marker='o', linestyle='-', label='Comment Lines')
axes[1, 1].plot(datasetB['release_date'], datasetB['nFiles'], marker='o', linestyle='-', label='Number of Files')
axes[1, 1].set_xlabel('Release Date')
axes[1, 1].legend()

# plt.tight_layout()
# plt.savefig('software_metrics_evolution.png')
# plt.show()

# Do you find any difference between the plot that is having date in the x-axis and the plot 
# that is having tag in the x-axis? If yes, then what is the difference? If no, then why is it same?

# The difference is that tags are plotted at regular intervals even if the time span between 
# two tags is not always identical.

# 2. Which type of plot is preferable for software evolution analysis?   
#     a) date in x-axis  
#     b) tag in x-axis  
# Why?

# b) because tags have a meaning in term of changes in the code (new tag = new feature, new change...)
# whereas the date simply mean that time passed and doesn't tell anything about code updates.

# Correlation is generally used to analyse the relationship between variables. Here, analyse the
# relationship between the number of lines of code and the number of files using Spearman correlation
# and Pearson correlation by considering **Dataset A**. Report the correlation upto 3 decimal places.

spearman_corr = datasetA[['code', 'nFiles']].corr(method='spearman').iloc[0, 1]
pearson_corr = datasetA[['code', 'nFiles']].corr().iloc[0, 1]
print("Spearman correlation:", round(spearman_corr, 3)) # 0.986
print("Pearson correlation:", round(pearson_corr, 3)) # 0.998

# 3. Do you find any difference in correlation values between Pearson and Spearman? 
# Which correlation measure is preferable for this use case? why?

# Not much. Spearman because it does not assume a specific data distribution and is less sensible to outliers.

# 4. Based on the above correlation value, please give your opinion on the relation between
# the number of lines and the number of files? Which of both size metrics do you propose to
# use in the remainder of your analysis?

# The correlation index is very close to 1, indicating that the number of lines of code and the
# number of files is very correlated. When one increases, the other increases similarly.
# We will use the number of lines of code.

# Visualize a linear regression analysis of the relationship between the release date and the total number of files by considering **Dataset B**.  
# Hint:
# 1. Prepare the data: Convert the release date to integers and set number of files as the target variable.  
# 2. Apply linear regression analysis to understand the relationship between the release date and the total number of files.  
# 3. Calculate Mean Relative Error and ajusted R-squared metrics.    
# 4. Generate a scatter plot showing the release dates against the total number of files. Then, overlay the linear regression line on the plot.

from sklearn.metrics import r2_score

datasetB['release_date'] = pd.to_datetime(datasetB['release_date']).astype(int)

x = datasetB[['release_date']]
y = datasetB['nFiles']
model = LinearRegression()
model.fit(x, y)
y_pred = model.predict(x)
mean_relative_error = np.mean(np.abs((y - y_pred) / y)) * 100
n = len(y)
p = x.shape[1]
r2 = r2_score(y, y_pred)
adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue', label='Actual Data')
plt.plot(x, y_pred, color='red', label='Linear Regression')
plt.xlabel('Release Date')
plt.ylabel('Total Number of Files')
plt.title('Linear Regression Analysis')
plt.legend()

plt.annotate(f'Mean Relative Error: {mean_relative_error:.2f}%', xy=(0.05, 0.9), xycoords='axes fraction', fontsize=10)
plt.annotate(f'Adjusted R-squared: {adjusted_r2:.3f}', xy=(0.05, 0.85), xycoords='axes fraction', fontsize=10)
plt.grid(True)
plt.tight_layout()
plt.show()

# 5. Choose an option regarding the growth of the software. Motivate your choice using the 2D regression plot.  
#     a) Linear  
#     b) Sub-linear  
#     c) Super-linear  

# a), we can clearly see that it follows a straight line.

# 6. Report the MRE and ajusted R-squared values.

# MRE : 6.49%
# ARS : 0.97