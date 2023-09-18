#!/usr/bin/env python
# coding: utf-8

# # Fault identification of the machines under surveillance using autoencoder and its explainability

# In[1]:


#import all required libraries

from os import listdir
from os.path import join
import numpy as np
from numpy import savetxt
import pandas as pd
from scipy import stats
import csv


import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import make_interp_spline, BSpline


from datetime import timedelta

import statistics

from sklearn.model_selection import train_test_split


# Step1: Pre-processing of the data from the available raw vibration data
# 
# #Preprocessing involves
# 1. Sorting data according to the device id
# 2. Remove NaN values if any
# 3. Remove and leave only first row if there are duplicate rows
# 4. Consider the data only when the device is 'ON'

# In[2]:


#Read the downloaded excel sheet with all device ids data

raw_data = pd.read_csv('raw_data_all.csv')

raw_data


# In[3]:


# Sort the data using the device id 

id_1 = raw_data.loc[raw_data['Device ID'] == 1501]   #PN-3	Vedanta-Dariba	1501


filename = '1501.csv'
savetxt(filename,id_1,delimiter = ',', header="Timestamp,Device ID,Temperature (°C),X-Axis RMS Velocity (mm/s),Y-Axis RMS Velocity (mm/s),Z-Axis RMS Velocity (mm/s),X-Axis RMS Acceleration (g),Y-Axis RMS Acceleration (g),Z-Axis RMS Acceleration (g),X-Axis CrestFactor,Y-Axis CrestFactor,Z-Axis CrestFactor,X-Axis Kurtosis,Y-Axis Kurtosis,Z-Axis Kurtosis,X-Axis Skewness,Y-Axis Skewness,Z-Axis Skewness,X-Axis Deviation,Y-Axis Deviation,Z-Axis Deviation,X-Axis Peak-to-Peak Displacement,Y-Axis Peak-to-Peak Displacement,Z-Axis Peak-to-Peak Displacement", fmt='%s')

#id_1


# In[4]:


#Read the raw vibration data acquired using the wireless sensor (corresponding device id raw data)

raw_data_1 = pd.read_csv('1501.csv',encoding= 'unicode_escape')

raw_data_1    


# In[5]:


# Pre-processing --- Remove if any NaN values present in the raw data (since wireless acquisition, sometimes signal data can be missing)

raw_data_1 = raw_data_1.dropna()                             # there are various NaN values in the dataframe, remove all those rows
raw_data_1 = raw_data_1.reset_index(drop=True)               # and reset the index values to increase linearly

raw_data_1   


# In[6]:


# Pre-processing ---- Remove duplicate rows if any

raw_data_1.drop_duplicates(subset=['Temperature (°C)','X-Axis RMS Velocity (mm/s)','Y-Axis RMS Velocity (mm/s)','Z-Axis RMS Velocity (mm/s)','X-Axis RMS Acceleration (g)','Y-Axis RMS Acceleration (g)','Z-Axis RMS Acceleration (g)','X-Axis CrestFactor','Y-Axis CrestFactor','Z-Axis CrestFactor','X-Axis Kurtosis','Y-Axis Kurtosis','Z-Axis Kurtosis','X-Axis Skewness','Y-Axis Skewness','Z-Axis Skewness','X-Axis Deviation','Y-Axis Deviation','Z-Axis Deviation','X-Axis Peak-to-Peak Displacement','Y-Axis Peak-to-Peak Displacement','Z-Axis Peak-to-Peak Displacement'],keep='first', inplace=True)

raw_data_1


# In[7]:


#Pre-processing (to consider data only when the device is "ON")

index_names = raw_data_1[raw_data_1['X-Axis RMS Acceleration (g)'] <= 0.01].index
  
# drop these row indexes from dataFrame
raw_data_1.drop(index_names, inplace = True)

#raw_data_1 = raw_data_1.reset_index(inplace=True)
 
raw_data_1


# In[8]:


raw_data_1.reset_index(inplace=True,drop=True)
raw_data_1


# In[9]:


# Modify the original data by spliting the Timestamp column which now will comprise of only date 
# split the time stamp (to separate the data and time from the time stamp)


timestamp_split = raw_data_1['# Timestamp']                    # split the timestamp into date and time

dated = timestamp_split.str.split(" ",2).str[0]        # date
timed = timestamp_split.str.split(" ",2).str[1]        # time

#dated
houred = timed.str.split(":",2).str[0]                 # split the time into hour
houred

################################## add new column with date and time values in the original data #############################

idx_1 = 2                                               # location to insert new column (date) into the original data
idx_2 = 3                                               # location to insert new column (time) into the original data
idx_3 = 4                                               # location to insert new column (date-hour) into the original data

raw_data_1.insert(loc=idx_1, column='Date', value=dated)      # location = 1st column, new column name = "date"
raw_data_1.insert(loc=idx_2, column='Time', value=houred)
a = (raw_data_1['Date'].str.cat(raw_data_1['Time'], sep ="-"))
raw_data_1.insert(loc=idx_3, column='D_T', value= a)


#data['D_T'] = data['Date'].str.cat(data['Time'], sep ="_")
raw_data_1                                               # modified data



filename = '1501_processed.csv'
savetxt(filename,raw_data_1,delimiter = ',', header="Timestamp,Device ID,Date,Time,D_T,Temperature (°C),X-Axis RMS Velocity (mm/s),Y-Axis RMS Velocity (mm/s),Z-Axis RMS Velocity (mm/s),X-Axis RMS Acceleration (g),Y-Axis RMS Acceleration (g),Z-Axis RMS Acceleration (g),X-Axis CrestFactor,Y-Axis CrestFactor,Z-Axis CrestFactor,X-Axis Kurtosis,Y-Axis Kurtosis,Z-Axis Kurtosis,X-Axis Skewness,Y-Axis Skewness,Z-Axis Skewness,X-Axis Deviation,Y-Axis Deviation,Z-Axis Deviation,X-Axis Peak-to-Peak Displacement,Y-Axis Peak-to-Peak Displacement,Z-Axis Peak-to-Peak Displacement", fmt='%s')


# In[10]:


print(type(raw_data_1.Date[0]))                             # type of date column

raw_data_1['Date'] = pd.to_datetime(raw_data_1['Date'])     # convert date (str) into timestamp
# verify datatype
#print(type(raw_data.D_T[0]))                               # verify the type of date column
print(type(raw_data_1.Date[0]))

#print(data['D_T'].iloc[0])
#data['D_T']


# In[95]:


raw_data_1


# Step2: Once the Pre-processing done, plot the time series data to visualially observe the change in data

# In[126]:


#print(dated)
sns.set(style="whitegrid")                                          # make background white colored 

tickfont = {'family' : 'Times New Roman', 'size'   : 44}
labelfont = {'family' : 'Times New Roman', 'size'   : 46}

fig = plt.figure(figsize = (6,4), dpi=200)                          # t figure size and resolution
ax = fig.add_subplot(111)

# plt.subplot(3, 3, 1)
# plt.plot(raw_data_1['# Timestamp'],raw_data_1['Temperature (°C)'],linestyle='-',color='orange', linewidth = 1.5)
# plt.title('Temperature', **tickfont)
# #plt.xticks(np.arange(0, len(raw_data_1['# Timestamp']), 300), **tickfont, rotation = '90')
# plt.xticks([])
# #plt.xticks(np.arange(0, len(raw_data_1['# Timestamp']), 300),**tickfont, rotation='90')
# plt.yticks(**tickfont)
# #plt.locator_params(axis='y', nbins=8)


#plt.subplot(3, 3, 2)
plt.plot(raw_data_1['D_T'],raw_data_1['X-Axis CrestFactor'],linestyle='-',color='magenta', linewidth = 1.5)
#plt.plot(raw_data_1['# Timestamp'],raw_data_1['Y-Axis RMS Velocity (mm/s)'],linestyle='-',color='mediumblue', linewidth = 1.5, label = 'y')
#plt.plot(raw_data_1['# Timestamp'],raw_data_1['Z-Axis RMS Velocity (mm/s)'],linestyle='-',color='green', linewidth = 1.5, label = 'z')
#plt.xticks([])
plt.xticks(**tickfont)
plt.xticks(np.arange(0, len(raw_data_1['D_T']), 900),['30 Dec','','','','5 Jun',
                                                       '','','1 Oct'])

#plt.xticks(np.arange(0, len(raw_data_1['D_T']), 900),rotation='90')

# plt.xticks(np.arange(1, len(x1)+1, 1), ['', '', '', '10 Nov', '','', 
#                                         '', '17 Nov','','','','21 Nov','','','','25 Nov'])
#plt.xticks()
plt.locator_params(axis='y', nbins=5)
#plt.xticks(positions,labels)



#plt.xticks(np.arange(0, len(raw_data_1['# Timestamp']), 300),**tickfont, rotation='90')
plt.yticks(**tickfont)


#plt.title('RMS Velocity', **tickfont)
#plt.legend(loc='upper left',prop={'size': 6})


# plt.subplot(3, 3, 3)
# plt.plot(raw_data_1['# Timestamp'],raw_data_1['X-Axis RMS Acceleration (g)'],linestyle='-',color='red', linewidth = 1.5, label = 'x')
# plt.plot(raw_data_1['# Timestamp'],raw_data_1['Y-Axis RMS Acceleration (g)'],linestyle='-',color='mediumblue', linewidth = 1.5, label = 'y')
# plt.plot(raw_data_1['# Timestamp'],raw_data_1['Z-Axis RMS Acceleration (g)'],linestyle='-',color='green', linewidth = 1.5, label = 'z')
# plt.xticks([])
# #plt.xticks(np.arange(0, len(raw_data_1['# Timestamp']), 300),**tickfont, rotation='90')
# plt.yticks(**tickfont)
# plt.title('RMS Acceleration',**tickfont)
# plt.legend(loc='upper left',prop={'size': 6})


# plt.subplot(3, 3, 4)
# plt.plot(raw_data_1['# Timestamp'],raw_data_1['X-Axis CrestFactor'],linestyle='-',color='red', linewidth = 1.5, label = 'x')
# plt.plot(raw_data_1['# Timestamp'],raw_data_1['Y-Axis CrestFactor'],linestyle='-',color='mediumblue', linewidth = 1.5, label = 'y')
# plt.plot(raw_data_1['# Timestamp'],raw_data_1['Z-Axis CrestFactor'],linestyle='-',color='green', linewidth = 1.5, label = 'z')
# plt.xticks([])
# #plt.xticks(np.arange(0, len(raw_data_1['# Timestamp']), 300),**tickfont, rotation='90')
# plt.yticks(**tickfont)
# plt.title('Crest Factor', **tickfont)
# plt.legend(loc='upper left',prop={'size': 6})


# plt.subplot(3, 3, 5)
# plt.plot(raw_data_1['# Timestamp'],raw_data_1['X-Axis Kurtosis'],linestyle='-',color='red', linewidth = 1.5, label = 'x')
# plt.plot(raw_data_1['# Timestamp'],raw_data_1['Y-Axis Kurtosis'],linestyle='-',color='mediumblue', linewidth = 1.5, label = 'y')
# plt.plot(raw_data_1['# Timestamp'],raw_data_1['Z-Axis Kurtosis'],linestyle='-',color='green', linewidth = 1.5, label = 'z')
# plt.xticks([])
# #plt.xticks(np.arange(0, len(raw_data_1['# Timestamp']), 300),**tickfont, rotation='90')
# plt.yticks(**tickfont)
# plt.title('Kurtosis', **tickfont)
# plt.legend(loc='upper right',prop={'size': 6})


# plt.subplot(3, 3, 6)
# plt.plot(raw_data_1['# Timestamp'],raw_data_1['X-Axis Skewness'],linestyle='-',color='mediumblue', linewidth = 1.5, label = 'x')
# plt.plot(raw_data_1['# Timestamp'],raw_data_1['Y-Axis Skewness'],linestyle='-',color='red', linewidth = 1.5, label = 'y')
# plt.plot(raw_data_1['# Timestamp'],raw_data_1['Z-Axis Skewness'],linestyle='-',color='green', linewidth = 1.5, label = 'z')
# plt.xticks([])
# plt.xticks(np.arange(0, len(raw_data_1['# Timestamp']), 200),**tickfont, rotation='90')
# plt.yticks(**tickfont)
# plt.title('Skew', **tickfont)
# plt.legend(loc='upper right',prop={'size': 6})



# plt.subplot(3, 3, 7)
# plt.plot(raw_data_1['# Timestamp'],raw_data_1['X-Axis Deviation'],linestyle='-',color='red', linewidth = 1.5, label = 'x')
# plt.plot(raw_data_1['# Timestamp'],raw_data_1['Y-Axis Deviation'],linestyle='-',color='mediumblue', linewidth = 1.5, label = 'y')
# plt.plot(raw_data_1['# Timestamp'],raw_data_1['Z-Axis Deviation'],linestyle='-',color='green', linewidth = 1.5, label = 'z')
# plt.xticks([])
# plt.xticks(np.arange(0, len(raw_data_1['# Timestamp']), 200),**tickfont, rotation='90')
# plt.yticks(**tickfont)
# plt.title('Standard Deviation', **tickfont)
# plt.legend(loc='upper left',prop={'size': 6})


# plt.subplot(3, 3, 8)
# plt.plot(raw_data_1['# Timestamp'],raw_data_1['X-Axis Peak-to-Peak Displacement'],linestyle='-',color='red', linewidth = 1.5, label = 'x')
# plt.plot(raw_data_1['# Timestamp'],raw_data_1['Y-Axis Peak-to-Peak Displacement'],linestyle='-',color='mediumblue', linewidth = 1.5, label = 'y')
# plt.plot(raw_data_1['# Timestamp'],raw_data_1['Z-Axis Peak-to-Peak Displacement'],linestyle='-',color='green', linewidth = 1.5, label = 'z')
# plt.title('Peak-Peak Displacement',**tickfont)
# plt.xticks([])
# plt.xticks(np.arange(0, len(raw_data_1['# Timestamp']), 200),**tickfont, rotation='90')
# plt.legend(loc='upper left',prop={'size': 6})


#plt.xlabel('# Timestamp', **labelfont)
#plt.ylabel("rms acceleration", **labelfont)

#plt.xticks([])

plt.yticks(**tickfont)

#ax.yaxis.grid(color='gray', linestyle='dashed')

plt.grid(b=True, which='major', color='darkgray', linestyle='-',alpha=0.2)
plt.grid(b=True, which='minor', color='darkgray', linestyle='-', alpha=0.2)

ax.spines['bottom'].set_color('k')
ax.spines['top'].set_color('k') 
ax.spines['right'].set_color('k')
ax.spines['left'].set_color('k')
   
plt.show()  

#fig.savefig('1501_feature_display.png')


# Step 3: Use of Machine learning algorithm for fault identification
# 
# Model used: Autoencoder (Unsupervised)
# 
# first 10 days data: training (Healthy)
# next 4 days data: testing  (Healthy), set the threshold based on this healthy test data
# 

# In[12]:


# First 10 days data for training and validating the algorithm 

delta_1 = timedelta(days=10)                                   # 10 days data for training algo

delta_2 = timedelta(days=1)                                    # 4 days data for testing algo
#print(delta_2)

y = len(raw_data_1)-1                                         # total number of samples

start_date_train = raw_data_1['Date'].iloc[0]                 # start date for training the algorithm         
print(start_date_train)

end_date_train = start_date_train + delta_1                 # end date, 2 week after the start date
print(end_date_train)

###################################################################################################################

data_train = [] 

while start_date_train <= end_date_train:
    #print(start_date_1.strftime("%Y-%m-%d"))
    z1 = raw_data_1.loc[raw_data_1['Date'] == start_date_train]
    #print(df)

    data_train.append(z1)
    
    start_date_train += delta_2                            # increment by 1 day till end date
    
#print(data_train)

a_file = open("1501_train_data.csv", "w")                   # open a csv file to store the seggragated data
for row in data_train:
    np.savetxt(a_file,row, delimiter=',',fmt='%s')
    #np.savetxt(a_file,row, delimiter=',',header='date,time,d_t,temperature,x_axis_rms_velocity,y_axis_rms_velocity,z_axis_rms_velocity,x_axis_rms_acceleration,y_axis_rms_acceleration,z_axis_rms_acceleration,x_axis_crest_factor,y_axis_crest_factor,z_axis_crest_factor,x_axis_kurtosis,y_axis_kurtosis,z_axis_kurtosis,x_axis_skew,y_axis_skew,z_axis_skew,x_axis_deviation,y_axis_deviation,z_axis_deviation,x_axis_peak_to_peak_displacement,y_axis_peak_to_peak_displacement,z_axis_peak_to_peak_displacement',fmt='%s')

    #plt.plot(data_new['temperature'],linestyle='-',color='red', linewidth = 1.5)
a_file.close()


# In[13]:


# 4 days data for testing the algorithm 

delta_3 = timedelta(days=4)                                   # 4 days data for testing algo
#delta_1 = timedelta(days=7)                

start_date_test_H = end_date_train + delta_2                  # start date for training the algorithm         
print(start_date_test_H)

end_date_test_H = start_date_test_H + delta_3                 # end date, 2 week after the start date
print(end_date_test_H)

###################################################################################################################

data_test_H = [] 

while start_date_test_H <= end_date_test_H:
    #print(start_date_1.strftime("%Y-%m-%d"))
    z2 = raw_data_1.loc[raw_data_1['Date'] == start_date_test_H]
    #print(df)

    data_test_H.append(z2)
    
    start_date_test_H += delta_2                            # increment by 1 day till end date
    
#print(data_train)

a_file1 = open("1501_test_data_H.csv", "w")                   # open a csv file to store the seggragated data
for row in data_test_H:
    np.savetxt(a_file1,row, delimiter=',',fmt='%s')
    #np.savetxt(a_file,row, delimiter=',',header='date,time,d_t,temperature,x_axis_rms_velocity,y_axis_rms_velocity,z_axis_rms_velocity,x_axis_rms_acceleration,y_axis_rms_acceleration,z_axis_rms_acceleration,x_axis_crest_factor,y_axis_crest_factor,z_axis_crest_factor,x_axis_kurtosis,y_axis_kurtosis,z_axis_kurtosis,x_axis_skew,y_axis_skew,z_axis_skew,x_axis_deviation,y_axis_deviation,z_axis_deviation,x_axis_peak_to_peak_displacement,y_axis_peak_to_peak_displacement,z_axis_peak_to_peak_displacement',fmt='%s')

    #plt.plot(data_new['temperature'],linestyle='-',color='red', linewidth = 1.5)
a_file1.close()


# In[14]:


# 1. Read the training data (corresponding device id)
data_train_1 = pd.read_csv('1501_train_data.csv', names=['Timestamp','Device ID','Date','Time','D_T','Temperature (°C)','X-Axis RMS Velocity (mm/s)','Y-Axis RMS Velocity (mm/s)','Z-Axis RMS Velocity (mm/s)','X-Axis RMS Acceleration (g)','Y-Axis RMS Acceleration (g)','Z-Axis RMS Acceleration (g)','X-Axis CrestFactor','Y-Axis CrestFactor','Z-Axis CrestFactor','X-Axis Kurtosis','Y-Axis Kurtosis','Z-Axis Kurtosis','X-Axis Skewness','Y-Axis Skewness','Z-Axis Skewness','X-Axis Deviation','Y-Axis Deviation','Z-Axis Deviation','X-Axis Peak-to-Peak Displacement','Y-Axis Peak-to-Peak Displacement','Z-Axis Peak-to-Peak Displacement'])
#data_train_1

data_train_1 = data_train_1.iloc[:, 5:27]
data_train_1


#2. Normalize the data (Feature scaling)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
data_train_scaled = sc.fit_transform(data_train_1)  #numpy array


# convert pre-processed numpy array to dataframe
data_train_scaled_updated = pd.DataFrame(data_train_scaled, columns = ['Temperature (°C)','X-Axis RMS Velocity (mm/s)','Y-Axis RMS Velocity (mm/s)','Z-Axis RMS Velocity (mm/s)','X-Axis RMS Acceleration (g)','Y-Axis RMS Acceleration (g)','Z-Axis RMS Acceleration (g)','X-Axis CrestFactor','Y-Axis CrestFactor','Z-Axis CrestFactor','X-Axis Kurtosis','Y-Axis Kurtosis','Z-Axis Kurtosis','X-Axis Skewness','Y-Axis Skewness','Z-Axis Skewness','X-Axis Deviation','Y-Axis Deviation','Z-Axis Deviation','X-Axis Peak-to-Peak Displacement','Y-Axis Peak-to-Peak Displacement','Z-Axis Peak-to-Peak Displacement'])
#print("\nPandas DataFrame: ")
print(data_train_scaled_updated.shape)


#3. Split the updated data into training, validation and test  (first 2 weeks data for training and validation 80:20, 2 days data for testing)

train_data, val_data = train_test_split(data_train_scaled_updated, test_size=0.20)

#train_data
print(len(train_data))
print(len(val_data))


# In[15]:


# 1. Read the testing data used to train the algorithm
data_test_H_1 = pd.read_csv('1501_test_data_H.csv', names=['Timestamp','Device ID','Date','Time','D_T','Temperature (°C)','X-Axis RMS Velocity (mm/s)','Y-Axis RMS Velocity (mm/s)','Z-Axis RMS Velocity (mm/s)','X-Axis RMS Acceleration (g)','Y-Axis RMS Acceleration (g)','Z-Axis RMS Acceleration (g)','X-Axis CrestFactor','Y-Axis CrestFactor','Z-Axis CrestFactor','X-Axis Kurtosis','Y-Axis Kurtosis','Z-Axis Kurtosis','X-Axis Skewness','Y-Axis Skewness','Z-Axis Skewness','X-Axis Deviation','Y-Axis Deviation','Z-Axis Deviation','X-Axis Peak-to-Peak Displacement','Y-Axis Peak-to-Peak Displacement','Z-Axis Peak-to-Peak Displacement'])
#data_test_H_1
data_test_H_1 = data_test_H_1.iloc[:, 5:27]
#data_test_H_1


#2. Normalize the test data_H (Feature scaling)
data_test_H_1_scaled = sc.transform(data_test_H_1)  #numpy array


#3. convert pre-processed numpy array to dataframe
test_data_H = pd.DataFrame(data_test_H_1_scaled, columns = ['Temperature (°C)','X-Axis RMS Velocity (mm/s)','Y-Axis RMS Velocity (mm/s)','Z-Axis RMS Velocity (mm/s)','X-Axis RMS Acceleration (g)','Y-Axis RMS Acceleration (g)','Z-Axis RMS Acceleration (g)','X-Axis CrestFactor','Y-Axis CrestFactor','Z-Axis CrestFactor','X-Axis Kurtosis','Y-Axis Kurtosis','Z-Axis Kurtosis','X-Axis Skewness','Y-Axis Skewness','Z-Axis Skewness','X-Axis Deviation','Y-Axis Deviation','Z-Axis Deviation','X-Axis Peak-to-Peak Displacement','Y-Axis Peak-to-Peak Displacement','Z-Axis Peak-to-Peak Displacement'])
#print("\nPandas DataFrame: ")
print(test_data_H.shape)

print(len(test_data_H))


# Training and validation data is available, its time to train the model

# In[16]:


# Model building

import keras,os
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPool1D , Flatten,InputLayer,UpSampling1D,MaxPooling1D,Reshape
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

model=keras.models.Sequential()

model.add(InputLayer(input_shape=(22)))
model.add(Reshape((22,1)))
model.add(Conv1D(16,kernel_size=3,activation="relu",padding="same"))
model.add(MaxPooling1D(pool_size=2,padding ="same"))
#model.add(Dropout(0.2))
model.add(Flatten())
model.add (Dense(88,activation="relu"))

model.add (Dense(176,activation="relu"))
model.add(Reshape((11,16)))
model.add(UpSampling1D(size=2))

model.add(Conv1D(1,kernel_size=3,padding="same"))
model.add(Flatten())

model.summary()


# In[17]:


# Add training parameters to model
opt = keras.optimizers.Adam(learning_rate=1e-3)
#opt= keras.optimizers.SGD(learning_rate=1e-2)
model.compile(optimizer=opt,loss='mse')


# In[18]:


autoencoder = model.fit(train_data,train_data,epochs=500,batch_size=128,validation_data=(val_data, val_data))


# In[19]:


loss = autoencoder.history['loss']
val_loss = autoencoder.history['val_loss']

epochs = range(1, len(loss) + 1)


tickfont = {'family' : 'serif', 'size'   : 16}
labelfont = {'family' : 'serif', 'size'   : 14}
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xticks(**tickfont)
plt.yticks(**tickfont)
plt.title('Training and validation loss',**labelfont)
plt.xlabel('Epoch',**labelfont)
plt.ylabel('Loss',**labelfont)
plt.legend()


plt.show()


# In[20]:


model.save("autoencoder_1501.h5")              # save the unsupervised autoencoder model


# Step4: Process the incoming new test dataset and check for its anomality
# 

# In[21]:


# Test the model

# get the test data
delta_4 = timedelta(days=10) 

start_date_test = end_date_test_H + delta_2           # start date of test data     
print(start_date_test)

end_date_test = raw_data_1['Date'].iloc[y]             # end date of test data         
#end_date_test = start_date_test + delta_4
print(end_date_test)


###################################################################################################

error_final = []
date_z = []

while start_date_test <= end_date_test:
    try: 
        print(start_date_test.strftime("%Y-%m-%d"))
        z_f = raw_data_1.loc[raw_data_1['Date'] == start_date_test]
        #data_test_final.append(z_f)

        #read the test data
        data_test = z_f.iloc[:, 5:27]

        #normalize the test data (Feature scaling)
        data_test_scaled = sc.transform(data_test)

        #Convert the normalized numpy array to a dataframe
        data_test_scaled_updated = pd.DataFrame(data_test_scaled, columns = ['Temperature (°C)','X-Axis RMS Velocity (mm/s)','Y-Axis RMS Velocity (mm/s)','Z-Axis RMS Velocity (mm/s)','X-Axis RMS Acceleration (g)','Y-Axis RMS Acceleration (g)','Z-Axis RMS Acceleration (g)','X-Axis CrestFactor','Y-Axis CrestFactor','Z-Axis CrestFactor','X-Axis Kurtosis','Y-Axis Kurtosis','Z-Axis Kurtosis','X-Axis Skewness','Y-Axis Skewness','Z-Axis Skewness','X-Axis Deviation','Y-Axis Deviation','Z-Axis Deviation','X-Axis Peak-to-Peak Displacement','Y-Axis Peak-to-Peak Displacement','Z-Axis Peak-to-Peak Displacement'])    #print(data_test_scaled_updated.shape)

        
        data_test_scaled_updated=np.asarray(data_test_scaled_updated)
        
        #Predict using the autoencoder model
        prediction_test_data = model.predict(data_test_scaled_updated)
        #print(prediction_test_data.shape)

        MSE=keras.losses.MeanSquaredError()

        error=[]

        for i in range(len(data_test_scaled_updated)):
            error.append(MSE(data_test_scaled_updated[i],prediction_test_data[i]).numpy())
            
        error_final.append(error)
        date_z.append(start_date_test)

        start_date_test += delta_2
    
    except:
        
        print('warning', start_date_test)
        
        start_date_test += delta_2
    
arr = np.asarray(error_final)



print(arr.shape)
print(date_z)


# In[22]:


print(type(arr))


# Step5: Set threshold based on the test (healthy) dataset

# In[23]:


test_data_H = np.array(test_data_H)
prediction_test_data_H = model.predict(test_data_H)
#print(np.array(prediction_test_data_H).shape)
#print(type(test_data_H))
#print(type(prediction_test_data_H))
error_H=[]
#print(len(test_data_H))

for i in range(len(test_data_H)):
    MSE=keras.losses.MeanSquaredError()
    error_H.append(MSE(test_data_H[i],prediction_test_data_H[i]).numpy())

error_H =  np.array(error_H)
print(error_H.shape)


# In[24]:


fig, ax = plt.subplots(figsize=(6,6))
# for i in range(0, len(error_H)):
#ax.hist(error_H, bins=50, density=True, alpha=1, color='blue')
#sns.distplot(error_H)
sns.distplot(error_H, bins=50, color='blue',hist_kws={"edgecolor": 'black'})
 
# for i in range(0, len(arr)):
#     ax.hist(arr[i], bins=1, density=True, alpha=1, color='red')


plt.show()


# In[25]:


threshold=(3*np.std(error_H)) + np.mean(error_H)
#threshold= statistics.median(arr[0])
print(threshold)


# In[26]:


threshold_2=(6*np.std(error_H)) + np.mean(error_H)
#threshold= statistics.median(arr[0])
print(threshold_2)


# In[27]:


threshold_1 = 3*threshold

#threshold= statistics.median(arr[0])
print(threshold_1)


# Step6: Find the anomalous data and save it for explainability

# In[50]:



sns.set(style="whitegrid")                                          # make background white colored 

tickfont = {'family' : 'Times New Roman', 'size'   : 60}
labelfont = {'family' : 'Times New Roman', 'size'   : 62}

fig = plt.figure(figsize = (18,12), dpi=200)                          # t figure size and resolution
ax = fig.add_subplot(111)
#----------------------------------------------------------------------------------------------------------------------
#arr contains the error values of the incoming test dataset

u = []
v = len(arr)      #16
#print(v)
for i in range (0,v):    #0-15
    #print(i)
    u.append(arr[i])
    #print(u)    
#plt.boxplot(u)
a=plt.boxplot(u, boxprops= dict(linewidth=3),patch_artist = True, vert = 1,zorder=1)

colors = ['white', 'white', 'white','white','white','white', 'white','white','white','white',
          'white', 'white', 'white','white','white','white', 'white','white','white','white',
          'white', 'white', 'white','white','white','white', 'white','white','white','white',
          'white', 'white', 'white','white','white','white', 'white','white','white','white',
          'white', 'white', 'white','white','white','white', 'white','white','white','white',
          'white', 'white', 'white','white','white','white', 'white','white','white','white',
          'white', 'white', 'white','white','white','white', 'white','white','white','white',
          'white', 'white', 'white','white','white','white', 'white','white','white','white',
          'white', 'white', 'white','white','white','white', 'white','white','white','white',
          'white', 'white', 'white','white','white','white', 'white','white','white','white',
          'white', 'white', 'white','white','white','white', 'white','white','white','white',
          'white', 'white', 'white','white','white','white', 'white','white','white','white',
          'white', 'white', 'white','white','white','white', 'white','white','white','white',
          'white', 'white', 'white','white','white','white', 'white','white','white','white',
          'white', 'white', 'white','white','white','white', 'white','white','white','white',
          'white', 'white', 'white','white','white','white', 'white','white','white','white',
          'white', 'white', 'white','white','white','white', 'white','white','white','white',
          'white', 'white', 'white','white','white','white', 'white','white','white','white',
          'white', 'white', 'white','white','white','white', 'white','white','white','white',
          'white', 'white', 'white','white','white','white', 'white','white','white','white',
          'white', 'white', 'white','white','white','white', 'white','white','white','white',
          'white', 'white', 'white','white','white','white', 'white','white','white','white',
          'white', 'white', 'white','white','white','white', 'white','white','white','white',
          'white', 'white', 'white','white','white','white', 'white','white','white','white',
          'white', 'white', 'white','white','white','white', 'white','white','white','white',
          'white', 'white', 'white','white','white','white', 'white','white','white','white',
          'white', 'white', 'white','white','white','white', 'white','white']


    #colors = ['blue', 'lime', 'orange', 'magenta']
  
for patch, color in zip(a['boxes'], colors):
    patch.set_facecolor(color)
  
# changing color and linewidth of
# whiskers
for whisker in a['whiskers']:
    whisker.set(color ='#8B008B',
                linewidth = 3,
                linestyle =":")
  
# changing color and linewidth of
# caps
for cap in a['caps']:
    cap.set(color ='#8B008B',
            linewidth = 2)
  
# changing color and linewidth of
# medians
for median in a['medians']:
    median.set(color ='orange',
               linewidth = 5)
  
# changing style of fliers

for flier in a['fliers']:
    flier.set(marker ='D',
              color ='#e7298a',
              alpha = 0.5)
    
    
me = []                                 #median

for i in range (len(u)):
    me.append(statistics.median(u[i]))
#     filename = 'median_error_VD1504.csv'
#     savetxt(filename,me,delimiter = ',')
    
jj = pd.DataFrame({'d':date_z, 'median':me})
#jj.index += 1 
#jj
#print(len(jj)) 
labels = jj['d']

#print(labels)
x1 = np.arange(1,len(jj)+1,1)
#print(len(x1))
#print(x1)    
#plt.plot(x1, jj['median'], color='orange', marker='^',linestyle='None',markersize=15, label='median',zorder=2)
    
plt.ylabel('MSE Loss',**labelfont)
plt.xlabel('Test data',**labelfont)

plt.xticks(x1, labels,**tickfont)#rotation = '90')
plt.xticks(np.arange(1, len(x1)+1, 1), ['15 Jan', '', '','','','', '','','','',
                                        '', '', '','','','', '','','','',
                                        '', '', '','','','', '','','','',
                                        '', '', '','','','', '','','','',
                                        '', '', '','','','', '','','','',
                                        '', '', '','','','', '','','','',
                                        '', '', '','','','', '','','','',
                                        '', '', '','','','', '','','','',
                                        '6 Apr', '', '','','','', '','','','',
                                        '', '', '','','','', '','','','',
                                        '', '', '','','','', '','','','',
                                        '', '', '','','','', '','','','',
                                        '', '', '','','','', '','','','',
                                        '', '', '','','','', '','','','',
                                        '', '', '','','','', '','','','',
                                        '', '', '','','','', '','','','',
                                        '25 Jun', '', '','','','', '','','','',
                                        '', '', '','','','', '','','','',
                                        '', '', '','','','', '','','','',
                                        '', '', '','','','', '','','','',
                                        '', '', '','','','', '','','','',
                                        '', '', '','','','', '','','','',
                                        '', '', '','','','', '','','','',
                                        '', '', '','','','', '','','','',
                                        '14 Sep', '', '','','','', '','','','',
                                        '', '', '','','','', '','','','',
                                        '', '', '','','','', '',''])
plt.yticks(**tickfont)

plt.grid(color = 'grey', linestyle = '--', linewidth = 0.5)

ax.spines['bottom'].set_color('k')
ax.spines['top'].set_color('k') 
ax.spines['right'].set_color('k')
ax.spines['left'].set_color('k')

#plt.axhline(y=threshold, color= 'blue', linewidth= 1.5, linestyle='--',label ='(3*std) + mean')
plt.axhline(y=threshold_2, color= 'blue', linewidth=5, linestyle='--', label ='6\u03C3 + \u03BC')
#plt.axhline(y=threshold_1, color= 'red', linewidth= 1.5, linestyle='--', label ='3*((3*std) + mean)')
#plt.ylim([0,2.5])
#plt.show()
#plt.show(a)
# # ###########################  difference between healthy and anomalous signal #########################################################

anomaly = []
anomaly_ind = []

for i in range (0, len(jj['median'])):
    if jj['median'][i] >= threshold_2:              #greater than 6std+mean
        anomaly.append(jj['median'][i])
        anomaly_ind.append(jj['d'][i])
        
    elif jj['median'][i] < threshold_2:
        anomaly.append(-1)
        anomaly_ind.append(jj['d'][i])
        
#print(anomaly)
#print(anomaly_ind)

anomaly =  np.array(anomaly)   
anomaly_ind = np.array(anomaly_ind)
#print(anomaly)
#print(anomaly_ind)

############################# detect the rising trend among the anomalous signal #######################################################

fault_c = []
fault_ind_c = []


for i in range (0, len(anomaly)):
    # detect the rising trend in the incoming values
    #print(i)
    if all(anomaly[i] > x for x in anomaly[:i]):   #compare value with its previous values and save if greater than all previous values
        fault_c.append(anomaly[i])
        fault_ind_c.append(anomaly_ind[i])
        
    elif any(anomaly[i] <= x for x in anomaly[:i]):
        fault_c.append(-1)
        fault_ind_c.append(anomaly_ind[i])
        
#print(len(fault_c))
#print(len(fault_ind_c))
fault_c =  np.array(fault_c)   
fault_ind_c = np.array(fault_ind_c)

#print(fault_c)
#print(fault_ind_c)

#b = (np.nonzero(np.array(fault_c))[0][0])
b = [n for n,i in enumerate(fault_c) if i>0][0]
#print(fault_c[b])
#print(fault_c[b])


# ################################## detect fault from the load condition ########################################

alarm = []
alarm_ind = []
n = []

for i in range(0, len(fault_c)):
    if fault_c[i] > fault_c[b]:
        alarm.append(fault_c[i])
        n = n + ['fault']
        alarm_ind.append(fault_ind_c[i])
        
    elif fault_c[i] <= fault_c[b]:
        alarm.append(-1)
        n = n+ ['no fault']
        alarm_ind.append(fault_ind_c[i])
        
#print(len(alarm))
#print(len(alarm_ind))
#print(alarm)
#print(alarm_ind)


data_gen = pd.DataFrame({'date':alarm_ind, 'median>threshold':alarm, 'state':n})
data_gen.index += 1 
#print(data_gen)

data_gen.to_csv('Q:/Supriya_ML/Explainability_Autoencoder_Paperwork/Promethean/1501_fault_sheet.csv')


plt.plot(data_gen['median>threshold'], marker='X',color = 'red',linestyle='None', markersize=20, label = 'alarm', zorder=2)

#plt.ylim([0,30])

plt.ylim(4e-2,1e1)
plt.yscale("log")

#plt.legend(prop={'size': 46}, loc = 'upper left')

plt.show()

#fig.savefig('1501_reconstruction_error.png')


# Step7: Select data for explainability

# In[127]:


#Find data corresponding to the faulty dataset

data_processed = pd.read_csv('1501_processed.csv',encoding= 'unicode_escape')
#data_processed

date_exp = '2022-01-25'        #date that has raised the alarm
test_data_exp = data_processed.loc[data_processed['Date'] == date_exp]
#test_data_exp

test_data_exp.reset_index(inplace=True,drop=True)
#test_data_exp

test_data_exp = test_data_exp.iloc[:, 5:27]
#test_data_exp

#Normalize the test data_H (Feature scaling)
test_data_exp = sc.transform(test_data_exp)  #numpy array

test_data_exp = test_data_exp.mean(axis=0) #take mean of all columns (all data in a day)

#print(test_data_exp.shape)

test_data_exp = test_data_exp.reshape(1,22)  #reshape data

#print(test_data_exp.shape)
#3. convert pre-processed numpy array to dataframe
test_data_exp = pd.DataFrame(test_data_exp, columns = ['TEMP','RMSV_X','RMSV_Y','RMSV_Z','RMSA_X','RMSA_Y','RMSA_Z','CF_X','CF_Y','CF_Z','K_X','K_Y','K_Z','Skew_X','Skew_Y','Skew_Z','STD_X','STD_Y','STD_Z','DPP_X','DPP_Y','DPP_Z'])
test_data_exp
#print(test_data_exp.shape)

# test_data_exp_M = test_data_exp.mean(axis=0)
# test_data_exp_M = pd.DataFrame(test_data_exp_M,columns = ['Temperature (°C)','X-Axis RMS Velocity (mm/s)','Y-Axis RMS Velocity (mm/s)','Z-Axis RMS Velocity (mm/s)','X-Axis RMS Acceleration (g)','Y-Axis RMS Acceleration (g)','Z-Axis RMS Acceleration (g)','X-Axis CrestFactor','Y-Axis CrestFactor','Z-Axis CrestFactor','X-Axis Kurtosis','Y-Axis Kurtosis','Z-Axis Kurtosis','X-Axis Skewness','Y-Axis Skewness','Z-Axis Skewness','X-Axis Deviation','Y-Axis Deviation','Z-Axis Deviation','X-Axis Peak-to-Peak Displacement','Y-Axis Peak-to-Peak Displacement','Z-Axis Peak-to-Peak Displacement'])
# test_data_exp_M
#print(len(test_data_exp))


# In[128]:


#fresh start
import shap
from shap import KernelExplainer, initjs, force_plot

import warnings
warnings.filterwarnings("ignore")

test_data_arr = np.array(test_data_exp[:1])     #input test data, shape = number of rows * feature values , 1*22
print(test_data_arr)
#print(len(test_data_arr))

top_m_features = []
top_m_features_i = []

all_test_data = []
all_pred_data= []


for i in range(len(test_data_arr)):      #len: number of rows in test data
    #print(i)
    
#---------------------- make prediction on the test data-------------------------------------------------------------#
    test_data_modified = test_data_arr[i]                    # access one row at a time
    #print(test_data_modified.shape)
    test_data_modified_reshape = test_data_modified.reshape(1,22)       #reshape for defined model compatibility, shape 1*22
    #print(test_data_modified_reshape.shape)
    prediction_modified = model.predict(test_data_modified_reshape)     #shape 1*22
    #print(prediction_modified.shape)
    test_data_modified_reshape_f = test_data_modified_reshape.flatten()  #22 flatten to calculate the difference of each element required later
    prediction_modified_f = prediction_modified.flatten()                #22
    
    all_test_data.append(test_data_modified_reshape_f)
    all_pred_data.append(prediction_modified_f)
    
#---------------------calculate difference and sort in descending order:(test data - predicted data)----------------------#    
    difference = []
    for j in range(len(test_data_exp.columns)):                            # iterate over feature size (column values), 22
        difference.append(test_data_modified_reshape_f[j] - prediction_modified_f[j])  #22 for each row 
                                            
    difference_arr = np.array(difference)
    difference_sort = -np.sort(-difference_arr)  #contains feature values
    #print('test',test_data_modified_reshape_f,'pred',prediction_modified_f,'diff',difference, 'diff_sort',difference_sort)
        
#---------------------consider only top M features from sorted difference for explaination--------------------------------#
    feat_count = difference_sort[:22]      #user define, select how many top M feature values we want
    #print(len(feat_count))
    top_m_feature = []
    top_m_feature_i = []
    for i in range(len(feat_count)) :
        index = np.where(difference_arr== difference_sort[i])   #getting index of feature values from difference
        #print(index[0][0])                                     #convert to int
        top_m_feature.append(test_data_modified_reshape_f[index[0][0]]) #search the index value in the test data
        top_m_feature_i.append(index[0][0])
    top_m_features.append(top_m_feature) #
    top_m_features_i.append(top_m_feature_i)
    
print(top_m_features)
print(top_m_features_i) 

#Top M features for all test data (for all rows in the test data) is now saved in top_m_features and its indexes in top_m_features_i

#-----------------------calculate the shap values for the top M features (2D output for each row data)--------------------------------------------------#

top_m_features_arr = np.array(top_m_features)             #convert list to numpy array
top_m_features_i_arr = np.array(top_m_features_i)         #convert list to numpy array
#print(top_m_features_i_arr)                           # len = number of rows                   

shap_values_top_m_feature = []
all_shap_values_top_m_feature=[]

for i in range(len(top_m_features_arr)): 
    #print(i)
    explainer = shap.KernelExplainer(model.predict,train_data[:100])   
    shap_values_top_m_feature = explainer.shap_values(top_m_features_arr[i])  #required input 2D data, cal shap values for each row
#     print(np.array(shap_values_top_m_feature).shape)
    #print(explainer.expected_value)
    all_shap_values_top_m_feature.append(shap_values_top_m_feature)
#print(np.array(all_shap_values_top_m_feature).shape)                         #shape = 1*22*22, no.ofrows*topMfeature*topMfeature
    all_shap_values_top_m_feature_df = pd.DataFrame(shap_values_top_m_feature)
#---------------------------------------------------------------------------------------    
    #fig = plt.figure()
    fig = plt.figure(figsize = (18,12), dpi=200)                          # t figure size and resolution
    ax = fig.add_subplot(111)
#-----------------------------------------------------------------------------------------------
    shap.summary_plot(all_shap_values_top_m_feature_df.T.values,test_data_exp.columns, 
                      max_display = 5, plot_type = "bar", color='dodgerblue', show = False)

#-----------------------------------------------------------------------------------------------------------    
    tickfont = {'family' : 'Times New Roman', 'size'   : 28}
    labelfont = {'family' : 'Times New Roman', 'size'   : 30}
    
    plt.xlabel("mean(|SHAP value|)",**labelfont)
    
    #ax.tick_params(direction='in',axis='both', which='major', labelsize=30)
    plt.xticks(**tickfont)
    plt.yticks(**tickfont)
    
    plt.xlim([0, 0.4])
    
    plt.locator_params(axis='x', nbins=4)
    plt.grid(color = 'grey', linestyle = '--', linewidth = 0.5)

    plt.show()
  


# In[231]:


#shap.summary_plot(np.array(all_shap_values_top_m_feature), top_m_features_arr, plot_type='bar')

