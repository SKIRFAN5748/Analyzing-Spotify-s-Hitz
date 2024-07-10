
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset

data = pd.read_csv('https://github.com/SKIRFAN5748/Analyzing-Spotify-s-Hitz/blob/main/spotify2023.csv', encoding = 'unicode_escape', engine ='python')

df=data.copy()
df.describe()

# Display basic information about the dataset
print(df.info())

df=pd.DataFrame(df)
df

#searching for Null values
print("Null values count :")
print(df.isnull().sum()) #sum of Null values in a colum

sns.heatmap(df.isnull(),yticklabels=False,cbar=True,cmap='viridis')

#filling null values
print (df)
df.fillna(0, inplace = True)    # filling with 0 value
df

#searching for Null values
print("Null values count :")
print(df.isnull().sum()) #sum of Null values in a colum

sns.heatmap(df.isnull(),yticklabels=False,cbar=True,cmap='viridis')

df[df.duplicated()]

"""**Music Analysis  Explore Audio Features**"""

# Explore audio feature distributions
audio_features = ['bpm', 'danceability_%', 'valence_%', 'energy_%', 'acousticness_%']
sns.pairplot(df[audio_features])
plt.show()

df.plot.hist()

df.released_year.plot.hist()

"""Platform Comparison"""

# Compare song popularity across different platforms
platform_metrics = ['streams', 'in_spotify_charts', 'in_apple_charts', 'in_deezer_charts']
platform_comparison = df[platform_metrics]

# You can use visualizations or statistical tests to compare these metrics across platforms
# For example, a bar chart or boxplot to visualize differences
sns.boxplot(data=platform_comparison )
plt.show()

"""Artist Impact"""

# Step 4: Artist Impact

# Check data types of relevant columns
print(df[['artist_count', 'streams']].dtypes)

# Convert 'streams' column to numeric, handling errors
df['streams'] = pd.to_numeric(df['streams'], errors='coerce')

# Drop rows with missing values in 'streams' column
df = df.dropna(subset=['streams'])

# Check data types again
print(df[['artist_count', 'streams']].dtypes)

# Now, perform linear regression
from sklearn.linear_model import LinearRegression

# Prepare data
X = df[['artist_count']]
y = df['streams']

# Create and fit the model
model = LinearRegression()
model.fit(X, y)

# Print the coefficients
print("Coefficient:", model.coef_)

"""Temporal Trends"""

# Identify temporal trends
temporal_metrics = df[['released_year', 'released_month', 'streams']]

# Group by year and month and calculate the average streams
temporal_trends = temporal_metrics.groupby(['released_year', 'released_month']).mean()

# Plot the trends
temporal_trends.plot(kind='line', y='streams',figsize=(10, 8))

"""**Cross-Platform Presence**"""

# Investigate how songs perform across different streaming services
platform_comparison = df[['in_spotify_charts', 'in_apple_charts', 'in_deezer_charts', 'in_shazam_charts']]

# Plot a bar chart or use descriptive statistics to compare
platform_comparison.mean().plot(kind='bar')

import seaborn as sns
import matplotlib.pyplot as plt

# Assuming 'streams' is the column you want to analyze
streams_data = df['streams']

# Calculate the interquartile range (IQR)
Q1 = streams_data.quantile(0.25)
Q3 = streams_data.quantile(0.75)
IQR = Q3 - Q1

# Define the lower and upper bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identify outliers
outliers = (streams_data < lower_bound) | (streams_data > upper_bound)

# Visualize outliers using a scatter plot
plt.figure(figsize=(10, 10))
plt.scatter(df.index, streams_data, label='Streams')
plt.scatter(df.index[outliers], streams_data[outliers], color='red', label='Outliers')

# Add labels and title
plt.xlabel('Songs')
plt.ylabel('Streams')
plt.title('Identification of Outliers in Streams')

# Show the plot
plt.legend()
plt.show()

# Assuming 'streams' is the target variable and 'bpm', 'danceability_%', 'valence_%', 'energy_%' are the features
X = df[['bpm', 'danceability_%', 'valence_%', 'energy_%']]
y = df['streams']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the Gradient Boosting Regressor model
model = GradientBoostingRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Visualize the predicted vs. actual values
plt.figure(figsize=(10, 8))
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Streams')
plt.ylabel('Predicted Streams')
plt.title('Actual vs. Predicted Streams (Gradient Boosting Regressor)')
plt.show()

# Assuming you have columns like 'in_spotify_charts', 'in_apple_charts', 'in_deezer_charts', 'in_shazam_charts'
platform_metrics = df[['in_spotify_charts', 'in_apple_charts', 'in_deezer_charts', 'in_shazam_charts']]

# Count the number of songs in each platform's charts
platform_counts = platform_metrics.sum()

# Create a bar plot
plt.figure(figsize=(10, 6))
sns.barplot(x=platform_counts.index, y=platform_counts.values, palette='viridis')
plt.xlabel('Streaming Platforms')
plt.ylabel('Number of Songs')
plt.title('Number of Songs in Charts/Playlists Across Platforms')
plt.show()

# Assuming you have columns like 'streams', 'released_year', and 'released_month'
temporal_metrics = df[['streams', 'released_year', 'released_month']]

# Group by year and month and calculate the average streams
monthly_trends = temporal_metrics.groupby(['released_year', 'released_month']).mean().reset_index()

# Create a line plot
plt.figure(figsize=(12, 6))
sns.lineplot(x=monthly_trends.index, y=monthly_trends['streams'], marker='o')
plt.xlabel('Month')
plt.ylabel('Average Streams')
plt.title('Temporal Trends: Average Streams per Month')
plt.show()

# Assuming you have a column like 'danceability_%' representing the danceability percentage
danceability_data = df['danceability_%']

# Create a histogram
plt.figure(figsize=(10, 6))
sns.histplot(danceability_data, bins=20, kde=True, color='skyblue')
plt.xlabel('Danceability Percentage')
plt.ylabel('Frequency')
plt.title('Distribution of Danceability Across Songs')
plt.show()

# Assuming you have columns for various audio features
audio_features = df[['bpm', 'danceability_%', 'valence_%', 'energy_%', 'acousticness_%', 'instrumentalness_%']]

# Calculate the cross-correlation matrix
correlation_matrix = audio_features.corr()

# Create a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5)
plt.title('Cross-Correlation Between Audio Features')
plt.show()

# Assuming you have a column like 'key' representing the musical key of each song
key_counts = df['key'].value_counts()

# Create a bar plot
plt.figure(figsize=(10, 6))
sns.barplot(x=key_counts.index, y=key_counts.values, palette='viridis')
plt.xlabel('Musical Key')
plt.ylabel('Number of Songs')
plt.title('Distribution of Songs Across Different Keys')
plt.show()

# Assuming you have columns like 'mode' and 'bpm' in your DataFrame
plt.figure(figsize=(10, 6))
sns.boxplot(x='mode', y='bpm', data=df, palette='pastel')
plt.xlabel('Mode')
plt.ylabel('BPM (Beats Per Minute)')
plt.title('Distribution of Song Tempo Across Major and Minor Modes')
plt.show()

# Assuming you have columns like 'acousticness_%' and 'streams' in your DataFrame
plt.figure(figsize=(10, 6))
sns.scatterplot(x='acousticness_%', y='streams', data=df, alpha=0.6, color='skyblue')
plt.xlabel('Acousticness Percentage')
plt.ylabel('Number of Streams')
plt.title('Relationship Between Acousticness and Popularity')
plt.show()

# Assuming you have columns like 'valence_%' and 'danceability_%' in your DataFrame
plt.figure(figsize=(10, 6))
sns.jointplot(x='valence_%', y='danceability_%', data=df, kind='hex', cmap='viridis')
plt.xlabel('Valence Percentage')
plt.ylabel('Danceability Percentage')
plt.title('Relationship Between Valence and Danceability')
plt.show()

# Assuming you have columns like 'energy_%' and 'valence_%' in your DataFrame
plt.figure(figsize=(10, 6))
sns.kdeplot(x='energy_%', y='valence_%', data=df, cmap='Blues', fill=True)
plt.xlabel('Energy Percentage')
plt.ylabel('Valence Percentage')
plt.title('Joint Distribution of Energy and Valence')
plt.show()

# Assuming you have columns like 'key' and 'liveness_%' in your DataFrame
plt.figure(figsize=(16, 12))
sns.violinplot(x='key', y='liveness_%', data=df, palette='muted')
plt.xlabel('Musical Key')
plt.ylabel('Liveness Percentage')
plt.title('Distribution of Liveness Across Different Keys')
plt.show()

# Assuming you have columns like 'energy_%' and 'in_spotify_charts' in your DataFrame
plt.figure(figsize=(22, 12))
sns.boxplot(x='in_spotify_charts', y='energy_%', data=df, palette='Set2')
plt.xlabel('In Spotify Charts')
plt.ylabel('Energy Percentage')
plt.title('Comparison of Energy Distribution for Songs in Spotify Charts')
plt.show()

fig_dims = (24,8)
fig, ax = plt.subplots(figsize=fig_dims)
fig = sns.barplot(x = 'released_year', y = 'streams', ax = ax,data=df, errwidth = False).set(title='Years vs streams')

df.released_year.plot.box()



