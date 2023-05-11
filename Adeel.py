# Import necessary libraries
import pandas as pd
import numpy as np
import sklearn.cluster as cluster
import sklearn.metrics as skmet
import matplotlib.pyplot as plt
import seaborn as sns
from pmdarima.arima import auto_arima
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error
from math import sqrt

# Import custom module
import cluster_tools as ct

#Starting of Clusters Analysis
# Set the option to display all columns

pd.set_option('display.max_columns', None)

# Read the dataset from a CSV file

data = pd.read_csv("dataset.csv")


"""This code is calling the `head()` function on a Pandas DataFrame named `data`.
The `head()` function returns the first n rows of the DataFrame, where n is 5 by default.
The resulting DataFrame is then printed to the console."""

data.head()


# This code prints a summary of the data using the describe() method.

print(data.describe())


# Selecting columns from dataframe
data2 = data[['1990', '2000', '2010', '2020']]


# This line of code prints the summary statistics of the data2 dataframe.
print(data2.describe())


# Compute correlation matrix
corr = data2.corr()

# Print correlation matrix
print(corr)

# Plot correlation matrix using ct.map_corr() function
ct.map_corr(data2)

# Save heatmap image to file
plt.savefig("heatmap.png", dpi=300)

# Display heatmap
plt.show()


# Plot a scatter matrix of data2
pd.plotting.scatter_matrix(data2, figsize=(12, 12), s=5, alpha=0.8)

# Save the scatter matrix plot as an image file
plt.savefig("scattermatrix.png", dpi=300)

# Display the scatter matrix plot
plt.show()


# Selecting '2000' and '2020' columns from 'data2' DataFrame
df_ex = data2[['2000', '2020']]

# Dropping rows with null values
df_ex = df_ex.dropna()

# Resetting index
df_ex = df_ex.reset_index()

# Printing first 15 rows of the DataFrame
print(df_ex.iloc[0:15])

# Dropping 'index' column
df_ex = df_ex.drop('index', axis=1)

# Printing first 15 rows of the DataFrame
print(df_ex.iloc[0:15])


# Scale the dataframe
df_norm, df_min, df_max = ct.scaler(df_ex)


print()

print('n  value')

#Applying the for loop

for ncluster in range(2, 10):
    # setup the  cluster with the number of expected clusters
    kmeans = cluster.KMeans(n_clusters=ncluster)
    #fitting the dataset 
    kmeans.fit(df_norm)
    labels = kmeans.labels_
    
    cen = kmeans.cluster_centers_
    
    print(ncluster, skmet.silhouette_score(df_ex, labels))




# Set number of clusters
ncluster = 7

# Perform KMeans clustering
kmeans = cluster.KMeans(n_clusters=ncluster)
kmeans.fit(df_norm)
labels = kmeans.labels_
cen = kmeans.cluster_centers_

# Extract x and y coordinates of cluster centers
cen = np.array(cen)
xcen = cen[:, 0]
ycen = cen[:, 1]

# Create scatter plot with labeled points and cluster centers

plt.figure(figsize=(8.0, 8.0))
cm = plt.cm.get_cmap('tab10')

# plotting the graph

plt.scatter(df_norm['2000'], df_norm['2020'], 10, labels, marker='o', cmap=cm)
plt.scatter(xcen, ycen, 45, 'k', marker='d')

#Setting labels and title

plt.title("Seven Clusters")
plt.xlabel("%Electric consumption(2000)")
plt.ylabel("%Electric consumption(2020)")

#Saving the figure

plt.savefig("sevenclusters.png", dpi=300)
plt.show()


# Printing the Center first then applying backscale

print(cen)

# Applying the backscale function to convert the cluster centre

scen = ct.backscale(cen, df_min, df_max)

print()

print(scen)

xcen = scen[:, 0]
ycen = scen[:, 1]

# cluster by cluster
plt.figure(figsize=(8.0, 8.0))
cm = plt.cm.get_cmap('tab10')

# plotting the graph

plt.scatter(df_ex["2000"], df_ex["2020"], 10, labels, marker="o", cmap=cm)
plt.scatter(xcen, ycen, 45, "k", marker="d")

#Setting labels and title

plt.title("Seven Clusters Center")
plt.xlabel("%Electric consumption(2000)")
plt.ylabel("%Electric consumption(2020)")

#Saving the figure

plt.savefig("sevencenclusters.png",dpi=300)
plt.show()


# Setting the cluster number

ncluster = 2

kmeans = cluster.KMeans(n_clusters=ncluster)

kmeans.fit(df_norm)

labels = kmeans.labels_

# Applying the K means

cen = kmeans.cluster_centers_

cen = np.array(cen)

xcen = cen[:,0]
ycen = cen[:, 1]

#Setting the figure size

plt.figure(figsize=(8.0,8.0))


cm = plt.cm.get_cmap('tab10')

#Ploting the graph

plt.scatter(df_norm['2000'], df_norm['2020'], 10, labels, marker='o', cmap=cm)
plt.scatter(xcen, ycen, 45, 'k', marker='d')

# Putting the labels and titles

plt.xlabel("%Electric consumption(2000)")
plt.ylabel("%Electric consumption(2020)")
plt.title("Two Clusters")

# Saving the figure

plt.savefig("twoclusters.png",dpi=300)
plt.show()

print(cen)
# Applying the backscale function to convert the cluster centre
scen = ct.backscale(cen, df_min, df_max)
print()
print(scen)
xcen = scen[:, 0]
ycen = scen[:, 1]
# cluster by cluster
plt.figure(figsize=(8.0, 8.0))
cm = plt.cm.get_cmap('tab10')

#Ploting the graph

plt.scatter(df_ex["2000"], df_ex["2020"], 10, labels, marker="o", cmap=cm)
plt.scatter(xcen, ycen, 45, "k", marker="d")

# Putting the labels and titles

plt.title("Two Clusters Center")
plt.xlabel("%Electric consumption(2000)")
plt.ylabel("%Electric consumption(2020)")

# Saving the figure

plt.savefig("twocenclusters.png",dpi=300)
plt.show()


#Creating new clusters

ncluster = 3

kmeans = cluster.KMeans(n_clusters=ncluster)

kmeans.fit(df_norm)
labels = kmeans.labels_

cen = kmeans.cluster_centers_

cen = np.array(cen)

xcen = cen[:,0]
ycen = cen[:, 1]


plt.figure(figsize=(8.0,8.0))


cm = plt.cm.get_cmap('tab10')

#Ploting the graph

plt.scatter(df_norm['2000'], df_norm['2020'], 10, labels, marker='o', cmap=cm)
plt.scatter(xcen, ycen, 45, 'k', marker='d')

# Putting the labels and titles

plt.title("Three Clusters")
plt.xlabel("%Electric consumption(2000)")
plt.ylabel("%Electric consumption(2020)")

# Saving the figure

plt.savefig("threeclusters.png",dpi=300)
plt.show()

print(cen)

# Applying the backscale function to convert the cluster centre

scen = ct.backscale(cen, df_min, df_max)

print()

print(scen)

xcen = scen[:, 0]
ycen = scen[:, 1]

# cluster by cluster
plt.figure(figsize=(8.0, 8.0))
cm = plt.cm.get_cmap('tab10')

#Ploting the graph

plt.scatter(df_ex["2000"], df_ex["2020"], 10, labels, marker="o", cmap=cm)
plt.scatter(xcen, ycen, 45, "k", marker="d")

# Putting the labels and titles

plt.title("Three Clusters Center")
plt.xlabel("%Electric consumption(2000)")
plt.ylabel("%Electric consumption(2020)")

# Saving the figure

plt.savefig("threecenclusters.png",dpi=300)
plt.show()



# Starting Trend Calculation
# Read in the "AirPassengers.csv" file as a pandas dataframe
columnsName = ["Year", "Passengers"]
df = pd.read_csv("AirPassengers.csv",names = columnsName, header = 0, parse_dates = [0])

#storing in the array format to convert rows into columns

newarray = df.to_numpy()

# Transpose of dataset

transposed = newarray.T

#print Transpose

print(transposed)

# Display the first 5 rows of the DataFrame

df.head()

# This code displays the last five rows of a DataFrame

df.tail()

# This code generates descriptive statistics of a dataframe.

df.describe()

# Convert 'Month' column to datetime format

df['Year'] = pd.to_datetime(df['Year'], format='%Y-%m')

# Print the first few rows of the DataFrame

print(df.head())

# Set the index of the DataFrame to the 'Month' column

df.index = df['Year']

# Delete the 'Month' column from the DataFrameqq

del df['Year']

# Print the first five rows of the DataFrame

print(df.head())

# Plots a line graph of the given dataframe

sns.lineplot(data=df)

# Adds a label to the y-axis

plt.ylabel("Number of Passengers")

# Saves the plot as a png file

plt.savefig("timeseries.png")

# Calculate rolling mean and rolling standard deviation for a window of 7 days

rolling_mean = df.rolling(7).mean()
rolling_std = df.rolling(7).std()

# Plot the original passenger data, rolling mean, and rolling standard deviation

plt.plot(df, color="blue", label="Original Passenger Data")
plt.plot(rolling_mean, color="red", label="Rolling Mean Passenger Number")
plt.plot(rolling_std, color="black", label="Rolling Standard Deviation in Passenger Number")

# Add title and legend to the plot

plt.title("Passenger Time Series, Rolling Mean, Standard Deviation")
plt.legend(loc="best")

# Save the plot as a PNG file

plt.savefig("mixplot.png")

# Use the augmented Dickey-Fuller test to check for stationarity

adft = adfuller(df,autolag="AIC")

# Create a DataFrame with ADF test results

output_df = pd.DataFrame({
    "Values": [adft[0], adft[1], adft[2], adft[3], adft[4]['1%'], adft[4]['5%'], adft[4]['10%']],
    "Metric": ["Test Statistics", "p-value", "No. of lags used", "Number of observations used",               
               "critical value (1%)", "critical value (5%)", "critical value (10%)"]
})

# Print the DataFrame

print(output_df)

# Calculate autocorrelation at lag 1

autocorrelation_lag1 = df['Passengers'].autocorr(lag=1)

# Print the result

print("One Month Lag: ", autocorrelation_lag1)


# Calculate autocorrelation for lags of 3, 6, and 9 months

autocorrelation_lag3 = df['Passengers'].autocorr(lag=3)
autocorrelation_lag6 = df['Passengers'].autocorr(lag=6)
autocorrelation_lag9 = df['Passengers'].autocorr(lag=9)

# Print the results

print("Three Month Lag:", autocorrelation_lag3)
print("Six Month Lag:", autocorrelation_lag6)
print("Nine Month Lag:", autocorrelation_lag9)



# seasonal_decompose() is used for time series decomposition 
# Perform seasonal decomposition of time series data

decompose = seasonal_decompose(df['Passengers'], model='additive', period=7)

# Plot decomposition

decompose.plot()

# Save the plot as an image file

plt.savefig("Seasonal.png")

# Display the plot

plt.show()

# Add a comment to explain the purpose of this code block
# The following code creates a train/test split for passenger data

# Create a 'Date' column in the dataframe with the same values as the index

df['Date'] = df.index

# Select the training data before August 1960

train = df[df['Date'] < pd.to_datetime("1960-08", format='%Y-%m')]

# Create a 'train' column with the same values as the 'Passengers' column

train['train'] = train['Passengers']

# Remove unnecessary columns from the 'train' dataframe

del train['Date']
del train['Passengers']

# Select the testing data on or after August 1960

test = df[df['Date'] >= pd.to_datetime("1960-08", format='%Y-%m')]

# Remove the 'Date' and 'Passengers' columns from the 'test' dataframe

del test['Date']
test['test'] = test['Passengers']
del test['Passengers']

# Plot the training and testing data

plt.plot(train, color="black")
plt.plot(test, color="red")

# Add a title and axis labels to the plot

plt.title("Train/Test split for Passenger Data")
plt.ylabel("Passenger Number")
plt.xlabel('Year')

# Set the plot style using seaborn

sns.set()

# Save the plot to a file and display it

plt.savefig("traintest.png")
plt.show()

# Train the ARIMA model

model = auto_arima(train, trace=True, error_action='ignore', suppress_warnings=True)
model.fit(train)

# Generate predictions for the test set

forecast = model.predict(n_periods=len(test))
forecast = pd.DataFrame(forecast, index=test.index, columns=['Prediction'])

# Calculate root mean squared error

rms = sqrt(mean_squared_error(test, forecast))

# Print the result

print("RMSE: ", rms)
