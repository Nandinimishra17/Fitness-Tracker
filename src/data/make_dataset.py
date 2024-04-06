import pandas as pd
from glob import glob


# --------------------------------------------------------------
# Read single CSV file
# --------------------------------------------------------------
single_file_acc = pd.read_csv("../../data/raw/MetaMotion/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv")
single_file_gyr = pd.read_csv("../../data/raw/MetaMotion/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Gyroscope_25.000Hz_1.4.4.csv")

# --------------------------------------------------------------
# List all data in data/raw/MetaMotion
# --------------------------------------------------------------

files=glob("../../data/raw/MetaMotion/MetaMotion/*.csv")
len(files)

# --------------------------------------------------------------
# Extract features from filename
# --------------------------------------------------------------

#extraxt(by splitting the 3 variables types) mainly three variables from a file name/path - participant, label, category
#and then appending i
# t to the dataframe for operations on it
data_path = "../../data/raw/MetaMotion/MetaMotion"
f = files[1]

participant = f.split("-")[0].replace(data_path, "")
participant= participant[1:]
label = f.split("-")[1]
category = f.split("-")[2].rstrip("123").rstrip("_MetaWear_2019")

df =  pd.read_csv(f)

#creation of new columns in dataframe
df["participant"] = participant
df["label"] = label
df["category"] = category



#participant
#label
#category
# --------------------------------------------------------------
# Read all files
# --------------------------------------------------------------

acc_df = pd.DataFrame() #empty dataframes
gyr_df = pd.DataFrame()

acc_set = 1 #set numbers counter- unique set counters to present as unique identifiers for each set
gyr_set = 1
#these set counters will be used in for loop ie for all data files 
#now we do the above processing of reading and splitting file paths into various categories and appending them to the two dataframes- for all the files in our data set


for f in files:
    participant = f.split("-")[0].replace(data_path, "")
    participant = participant[1:]
    label = f.split("-")[1]
    category = f.split("-")[2].rstrip("123").rstrip("_MetaWear_2019")
    
    df = pd.read_csv(f)
    
    df["participant"] = participant
    df["label"] = label
    df["category"] = category
    
    if "Accelerometer" in f:
        df["set"] = acc_set
        acc_set+=1
        acc_df = pd.concat([acc_df, df])
    
    if "Gyroscope" in f:
        df["set"] =  gyr_set
        gyr_set+=1
        gyr_df = pd.concat([gyr_df, df])
        
len(acc_df[acc_df["set"]==1])
#acc_df.to_csv(r"C:\Users\KIIT\Downloads\df.csv")
#the abv syntax is basically for you to export the dataframe made(here accelerometer dataset) into an excell file so you can understand the dataset
#here, basically the guy has done in total 94 sets of various diff workouts- like heavy deadlift is done 4times and similarly other workouts are done 3-4times but each time has been marked 
#as a seperate set number


#here we split the entire dataset of accelerometer and gyroscope data into two different dataframes- one for accelerometer and the other for gyroscrope 
#each based on the participant, label, category



# --------------------------------------------------------------
# Working with datetimes
# --------------------------------------------------------------
#pandas doesnt know that these epochs and time variables are some 'times' units, if you check acc_df.info()- it'll show you the data types of all these variables
#time ka object and epoch ka int64 so we need to convert these data types/time variables into a standard time type so that we can perform functions on 'time' data
acc_df.info()

pd.to_datetime(df["epoch (ms)"], unit = "ms")
pd.to_datetime(df["time (01:00)"])
pd.to_datetime(df["time (01:00)"]).dt.month

#NOW we'll set this modeified datetime type as the index column of the acc_df rather than just 1,2,3,4...etc as the index
acc_df.index = pd.to_datetime(acc_df["epoch (ms)"], unit = "ms")
gyr_df.index = pd.to_datetime(gyr_df["epoch (ms)"], unit = "ms")
#since we've gotten one particular type of reference for time variable therefore we'll get rid of the other time var- epoch(unic time), time ms etc

del acc_df["epoch (ms)"]
del acc_df["time (01:00)"]
del acc_df["elapsed (s)"]
del gyr_df["epoch (ms)"]
del gyr_df["time (01:00)"]
del gyr_df["elapsed (s)"]
# --------------------------------------------------------------
# Turn into function- TURNING THE ENTIRE ONE LINERS ABOVE INTO A FUNCTION 
# --------------------------------------------------------------

files=glob("../../data/raw/MetaMotion/MetaMotion/*.csv")

def read_data_from_files(files):
    acc_df = pd.DataFrame() #empty dataframes
    gyr_df = pd.DataFrame()

    acc_set = 1 
    gyr_set = 1

    for f in files:
        participant = f.split("-")[0].replace(data_path, "")
        participant = participant[1:]
        label = f.split("-")[1]
        category = f.split("-")[2].rstrip("123").rstrip("_MetaWear_2019")
        
        df = pd.read_csv(f)
        
        df["participant"] = participant
        df["label"] = label
        df["category"] = category
        
        if "Accelerometer" in f:
            df["set"] = acc_set
            acc_set+=1
            acc_df = pd.concat([acc_df, df])
        
        if "Gyroscope" in f:
            df["set"] =  gyr_set
            gyr_set+=1
            gyr_df = pd.concat([gyr_df, df])
        
    acc_df.index = pd.to_datetime(acc_df["epoch (ms)"], unit = "ms")
    gyr_df.index = pd.to_datetime(gyr_df["epoch (ms)"], unit = "ms")


    del acc_df["epoch (ms)"]
    del acc_df["time (01:00)"]
    del acc_df["elapsed (s)"]
    del gyr_df["epoch (ms)"]
    del gyr_df["time (01:00)"]
    del gyr_df["elapsed (s)"]
    
    return acc_df, gyr_df

acc_df, gyr_df = read_data_from_files(files)
        
    

# --------------------------------------------------------------
# Merging datasets
#Bringing the accelerometer and gyroscope dataframe together into one using the pd.concat method on the basis of axis=1 which means column wise, axis = 0 means row wise
#Now this also gives out redundant datas like x,y,z axis and label, category, participant in that concatenated dataframe twice- which thus we slice out
#The syntax of the iloc function in Python is as follows:
#df.iloc[row_start:row_end, column_start:column_end]
# --------------------------------------------------------------

data_merged = pd.concat([acc_df.iloc[:, :3], gyr_df], axis=1)
#now we know that acc_df and gyr_df have datasets being taken at different/unique time intervals so when we concatenate/merge their datasets, different time intervals/units are merged into one
#and that merged data set takes that time as the index reference point. With time units being ideally avl for only and only accelerometer and the other time being only and only for gyroscope- this will give out null values for the others.
#therefore we use dropna method to drop NaN values- data_merged.dropna()
#SO THERE ARE TWO DIFFERENT SENSORS MEASURING WITHIN THE DEVICE AND THE CHANCE THAT THE ACCELEROMETER MEASUREMENT IS EXACTLY TO THE MILISECONDS AS THE GYROSCOPE IS VERY SMALL, Thus the number of 69k merged values containing null drops to 1k
#And the chances of this happening is coincidence- the frequencies of both of the measurements exactly lined up
#Now we got too less a dataset to work on so we reduce the frequency between the gaps where first its just an accelerometer data filled w gyro's nan values and then its gyro's data filled with acc's nan values- thus we want to reduce the space between the measurements to a point where we have DATA FOR EVERY ROW

#Renaming the columns
data_merged.columns =[
    "acc_x",
    "acc_y",
    "acc_z",
    "gyr_x",
    "gyr_y",
    "gyr_z",
    "participant",
    "label",
    "category",
    "set",
]

# --------------------------------------------------------------
# Resample data (frequency conversion)
# --------------------------------------------------------------

# Accelerometer:    12.500HZ- was measuring at every 0.8 seconds
# Gyroscope:        25.000Hz- measures at higher frequency, more measurements per second- measuring at every 0.4 seconds, thus gyroscope measures twice as fast
#Resamplig means- bringing the frequency measurements we have at a certain higher or lower frequency. So we want to make sure that we keep as much data/details as possible while keeping data for every row thus syncing all the data from both gyroscope

sampling = {
    "acc_x": "mean",
    "acc_y": "mean",
    "acc_z": "mean",
    "gyr_x": "mean",
    "gyr_y": "mean",
    "gyr_z": "mean",
    "participant": "last",
    "label" : "last",
    "category": "last",
    "set":"last",
}


#data_merged[:1000].resample(rule="200ms").mean()
data_merged[:1000].resample(rule="200ms").apply(sampling) #now this data for a week of 200ms intervals is too large so we split into days of data
# Split by day

days = [g for n, g in data_merged.groupby(pd.Grouper(freq="D"))]
data_resampled = pd.concat([df.resample(rule="200ms").apply(sampling).dropna() for df in days])

data_resampled["set"] = data_resampled["set"].astype("int")
data_resampled.info()
# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------

data_resampled.to_pickle("../../data/interim/01_data_processed.pkl")
data_resampled.to_csv("../../data/interim/01_data_processed.csv")