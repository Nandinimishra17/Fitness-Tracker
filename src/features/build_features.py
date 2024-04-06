import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction
from FrequencyAbstraction import FourierTransformation
from sklearn.cluster import KMeans


# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

df = pd.read_pickle("../../data/interim/02_outliers_removed_chauvenets.pkl")

predictor_columns = list(df.columns[:6])

#Plot settings

plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"]

# --------------------------------------------------------------
# Dealing with missing values (imputation)
# --------------------------------------------------------------

for col in predictor_columns:
    df[col] = df[col].interpolate()
    
df.info()

# --------------------------------------------------------------
# Calculating set duration
# --------------------------------------------------------------
#df[df["set"]==15]["acc_y"].plot() #this is heavy type set which has 5 reps
df[df["set"]== 60]["acc_y"].plot()#medium set with 10 reps
df[df["set"]== 76]["acc_y"].plot()
#now with the help of low pass filter we want to accomplish smoother lines/curves of the graph - meaning the subtle noise is removed
#meaning we'll look at only the overall movement patterns and not necessarily the small tiny differences that are apparent between every repetition
#for ex adjustments of your hands and feets between reps- these are the small detailed tiny movements that we wanna filter out
#And thats why we need to know how long a repetition takes because in doing so we can later adjust the frequency settings- as in attune to a frequency higher- faster reps to filter out the noise 
duration = df[df["set"] == 1].index[-1] - df[df["set"] == 1].index[0]
duration.seconds

for s in df["set"].unique():
    start = df[df["set"] == s].index[0]
    stop = df[df["set"] == s].index[-1]
    
    duration = stop - start
    df.loc[(df["set"] == s), "duration"] = duration.seconds
    
duration_df = df.groupby(["category"])["duration"].mean()
duration_df.iloc[0]/5 #for heavy
duration_df.iloc[1]/10 # for medium


# --------------------------------------------------------------
# Butterworth lowpass filter
# --------------------------------------------------------------
#fs = step size between individual records of our data frame ie 200 ms as set previously => 1s/fs we set => 1s/200ms => 1000ms/200ms => 5
#A Butterworth low-pass filter is a type of filter that is used to remove high frequency noise from a dataset. It is most commonly used in machine learning in order to improve the accuracy of the model. The filter works by removing any data points above a certain threshold frequency, while still preserving the underlying pattern of the data. By doing so, it helps to reduce the effect of noise on the model, which can lead to better results.here that threshhold value is the cut off freq
df_lowpass = df.copy()
LowPass = LowPassFilter()

fs = 1000/200
cutoff = 1.3

df_lowpass = LowPass.low_pass_filter(df_lowpass, "acc_y", fs, cutoff, order = 5)

subset = df_lowpass[df_lowpass["set"]==45]

fig, ax = plt.subplots(nrows=2, sharex= True, figsize = (20, 20))
ax[0].plot(subset["acc_y"].reset_index(drop=True), label="raw data")
ax[1].plot(subset["acc_y_lowpass"].reset_index(drop=True), label="butterworth filter")
ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True)
ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True)

for col in predictor_columns: #overwriting all the columns with the new low passed values
    df_lowpass = LowPass.low_pass_filter(df_lowpass, col, fs, cutoff, order=5)
    df_lowpass[col] = df_lowpass[col + "_lowpass"]
    del df_lowpass[col + "_lowpass"]

# --------------------------------------------------------------
# Principal component analysis PCA
# --------------------------------------------------------------

df_pca = df.copy()

PCA = PrincipalComponentAnalysis()

pc_values = PCA.determine_pc_explained_variance(df_pca, predictor_columns)

plt.figure(figsize=(10,10))
plt.plot(range(1, len(predictor_columns)+1), pc_values)
plt.xlabel("principal component number")
plt.ylabel("explained variance")
plt.show()
#the abv graph plot to determine the optimal number of components to use when conducting a PCA using the ELBOW TECHNIQUE. 

df_pca = PCA.apply_pca(df_pca, predictor_columns, 3)

subset = df_pca[df_pca["set"] == 35]
subset[["pca_1", "pca_2", "pca_3"]].plot()

# --------------------------------------------------------------
# Sum of squares attributes
# --------------------------------------------------------------

df_squared = df_pca.copy()

acc_r = df_squared["acc_x"] ** 2 + df_squared["acc_y"] ** 2 + df_squared["acc_z"] ** 2
gyr_r = df_squared["gyr_x"] ** 2 + df_squared["gyr_y"] ** 2 + df_squared["gyr_z"] ** 2

df_squared["acc_r"] = np.sqrt(acc_r)
df_squared["gyr_r"] = np.sqrt(gyr_r)

subset = df_squared[df_squared["set"] == 14]
subset[["acc_r", "gyr_r"]].plot(subplots=True)
# --------------------------------------------------------------
# Temporal abstraction
# --------------------------------------------------------------
 
df_temporal = df_squared.copy()
NumAbs = NumericalAbstraction()

predictor_columns = predictor_columns + ["acc_r", "gyr_r"]
ws= int(1000/200)

df_temporal_list = []
for s in df_temporal["set"].unique():
    subset = df_temporal[df_temporal["set"] == s].copy()
    for col in predictor_columns:
        subset = NumAbs.abstract_numerical(subset, [col], ws, "mean")
        subset = NumAbs.abstract_numerical(subset, [col], ws, "std")
    df_temporal_list.append(subset)

df_temporal = pd.concat(df_temporal_list)

subset[["acc_y", "acc_y_temp_mean_ws_2", "acc_y_temp_std_ws_2"]].plot()
subset[["gyr_y", "gyr_y_temp_mean_ws_2", "gyr_y_temp_std_ws_2"]].plot()
subset[
    [
        "acc_y_temp_mean_ws_2",
        "acc_y_temp_mean_ws_4",
        "acc_y_temp_mean_ws_5",
    ]
].plot()
subset[
    [
        "acc_y_temp_std_ws_2",
        "acc_y_temp_std_ws_4",
        "acc_y_temp_std_ws_5",
    ]
].plot()
subset[["acc_y"]].plot()
# --------------------------------------------------------------
# Frequency features
# --------------------------------------------------------------

#using the Discrete Fourier Transformation as a means to represent data in terms of frequency components, allowing for more efficient analysis of the data. This provides a way to better understand and model complex data sets, as the frequency components produced by the DFT can provide insight into patterns and trends that would not otherwise be visible. Additionally, the DFT can be used to reduce noise, allowing for more accurate models.

# This class performs a Fourier transformation on the data to find frequencies that occur
# often and filter noise.
df_freq = df_temporal.copy().reset_index()
FreqAbs = FourierTransformation()

fs = int(1000/200) #sampling rate-> number of samples per second
ws = int(2800/200) # avg length of a repetition (2.8sec )

#appying DFT on any 1 col first
df_freq = FreqAbs.abstract_frequency(df_freq, ["acc_y"], ws , fs)
df_freq.info()

#Visualize results
subset = df_freq[df_freq["set"] == 15]
subset[["acc_y"]].plot()
subset[
    [
        "acc_y_max_freq",
        "acc_y_freq_weighted",
        "acc_y_pse",
        "acc_y_freq_1.429_Hz_ws_14",
        "acc_y_freq_2.5_Hz_ws_14",
    ]
].plot()

df_freq_list = []
for s in df_freq["set"].unique():
    print(f"Applying Fourier transformation to set {s}")
    subset = df_freq[df_freq["set"] == s].reset_index(drop = True).copy()
    subset = FreqAbs.abstract_frequency(subset, predictor_columns, ws, fs)
    df_freq_list.append(subset)
    
df_freq= pd.concat(df_freq_list).set_index("epoch (ms)", drop = True)
# --------------------------------------------------------------
# Dealing with overlapping windows
# --------------------------------------------------------------
#since we've added all the extra cols which are based on a rolling window that the values in all of the columns between the different rows are highly correlated
#and this can typically cause overfitting which we wanna avoid
#in order to tackle that we wanna allow for a certain percentage of overlap and remove the rest of the data 
#In feature engineering, overlapping features refer to variables that convey similar or redundant information. When two or more features are highly correlated or redundant, they might provide redundant information to the model during training, which can lead to overfitting, increased computational complexity, and reduced generalization performance.
#Correlation is a statistical measure that quantifies the degree to which two variables are linearly related. In feature engineering, correlation analysis is often used to identify relationships between features. High correlation between features suggests redundancy or overlapping information. It's important to note that correlation does not imply causation but merely indicates a linear relationship between variables.
#so we first drop off all the missing values from the rows
df_freq = df_freq.dropna() 
# now in order to deal with the overlapping windows, we are going to get rid of some part of the data, and there's an allowance of 50% is recommended-> getting rid of 50% of the data by skippig every row and this may result in a lot of data loss but it'll also make your model less prone to overfitting
#Thus we'll use the 50% window as we have enough amount of data
#so we skip every second row-> this will reduce the correlation between the records and thus potential overfitting 
df_freq= df_freq.iloc[::2]
# --------------------------------------------------------------
# Clustering
# --------------------------------------------------------------

#K-MEANS CLUSTERING-> PREDICTIVE MODELING
df_cluster = df_freq.copy()

cluster_columns = ["acc_x", "acc_y", "acc_z"]
k_values = range(2,10)
inertias = []

for k in k_values:
    subset = df_cluster[cluster_columns]
    kmeans= KMeans(n_clusters=k, n_init=20, random_state=0)
    cluster_labels = kmeans.fit_predict(subset)
    inertias.append(kmeans.inertia_)
  
plt.figure(figsize=(10, 10))
plt.plot(k_values, inertias)
plt.xlabel("k")
plt.ylabel("Sum of squared distances")
plt.show()  

kmeans = KMeans(n_clusters=5, n_init=20, random_state=0)
subset = df_cluster[cluster_columns]
df_cluster["cluster"] = kmeans.fit_predict(subset)

#Plot clusters
fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(projection="3d")
for c in df_cluster["cluster"].unique():
    subset = df_cluster[df_cluster["cluster"] == c]
    ax.scatter(subset["acc_x"], subset ["acc_y"], subset["acc_z"], label=c)       
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
plt.legend()
plt.show()

#Plot acccelerometer data to compare-> by unique labels
fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(projection="3d")
for l in df_cluster["label"].unique():
    subset = df_cluster[df_cluster["label"] == l]
    ax.scatter(subset["acc_x"], subset ["acc_y"], subset["acc_z"], label=l)       
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
plt.legend()
plt.show()
# so doing this we can see where they belong using the 3 dimensional plot-> we can see that the bench press and the overhead press are very close to each other
#which is expected as the movements look a lot like each other
#in green and grey we have the deadlift and the row which also are very similar
#the other spread out few datapoints are of rest which is true bacause they are random- not restricted by any movements


# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------
df_cluster.to_pickle("../../data/interim/03_data_features.pkl")