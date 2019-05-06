# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.1
#   kernelspec:
#     display_name: Python 2
#     language: python
#     name: python2
# ---

from IPython.core.display import display, HTML
display(HTML("<style>.container { width:90% !important; }</style>"))

#  

# # Introduction
# ### This project is GPS tracking data from buses and heavy vehicles on the northern motorway and the northern expressway routes.
#
#
#
# * [1. Import Libraries](#Import-Libraries) 
#     
#
# * [2. Import Data](#Import-Data)
#     * [Weather APIs](#Import-Weather-API)
#     * [Google Geocode APIs](#Import-Google-Geocode-Api)
#     * [Data Wrangling And Data Visualisation](#Data-Wrangling-And-Data-Visualisation)
#     
#
# * [3. Data Analysis](#Data-Analysis)
#   
#   
# * [4. Supervised Learning](#Supervised-Learning)
#
#      * [Logistic Regression](#Logistic-Regression )
#      * [K Nearest Neighbors]( #K-Nearest-Neighbors)
#      * [Desicion Tree]( #Desicion-Tree)
#      * [Random Forest](#Random-Forest) 
#      
# * [5. Unupervised Learning](#Unsupervised-Learning) 
#
#      *[K Means Clustering](#K-Means-Clustering)
#      
# * [6. Result](#Result)

#  

# ### Import Libraries

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
from sklearn import neighbors
from datetime import datetime
from statsmodels.sandbox.regression.predstd import wls_prediction_std
# %matplotlib inline
import warnings
warnings.filterwarnings('ignore')
import glob
import os
from datetime import datetime
from pandas.io.json import json_normalize
import math
from math import sin, asin, cos, sqrt, atan2, radians
from plotly.graph_objs import *

# +

from pylab import rcParams

sns.set(style="ticks")
sns.set_style("whitegrid")
rcParams['figure.dpi'] = 350
rcParams['lines.linewidth'] = 2
rcParams['axes.facecolor'] = 'white'
rcParams['patch.edgecolor'] = 'white'
rcParams['font.family'] = 'StixGeneral'
rcParams['font.size'] = 20
# -

rcParams['figure.figsize'] = 13, 8
mpl.style.use( 'ggplot' )
sns.set_style( 'white' )

# ### Import Data

# +
import glob
interesting_files = glob.glob("*[!output].csv")
df_list=[]
for filename in sorted(interesting_files):
    df_list.append(pd.read_csv(filename))
    
full_df = pd.concat(df_list)

full_df.to_csv('output.csv', sep='\t',header=None, index=None)
# -

interesting_files = pd.DataFrame(interesting_files)

#Checking all our files tottally 1600 csv files!
interesting_files.tail()

dataframe = pd.read_csv('output.csv',header=None, index_col=None)

# The total data of in 1600 csv files! 
dataframe.tail()

dataframe['id'] = map(lambda x: x.split(';')[0], dataframe[0])
dataframe['event_timestamp'] = map(lambda x: x.split(';')[1], dataframe[0])
dataframe['course_over_ground'] = map(lambda x: x.split(';')[2], dataframe[0])
dataframe['machine_id'] = map(lambda x: x.split(';')[3], dataframe[0])
dataframe['vehicle_weight_type'] = map(lambda x: x.split(';')[4], dataframe[0])
dataframe['speed_gps_kph'] = map(lambda x: x.split(';')[5], dataframe[0])
dataframe['latitude'] = map(lambda x: x.split(';')[6], dataframe[0])
dataframe['longitude'] = map(lambda x: x.split(';')[7], dataframe[0])
del dataframe[0]


#REMOVE single quote arount vehicle type!
def remove_first_character(x):
    return (x.replace(x[:1], ''))


dataframe[dataframe.columns[4:5]] = dataframe[dataframe.columns[4:5]].applymap(remove_first_character)

dataframe.head(10)

#change type of date
dataframe['event_timestamp'] = pd.to_datetime(dataframe['event_timestamp'])


dataframe['speed_gps_kph'] = dataframe['speed_gps_kph'].astype(int)
#dataframe['id'] = dataframe['id'].astype('int64')
dataframe['course_over_ground'] = dataframe['course_over_ground'].astype(int)
dataframe['machine_id'] = dataframe['machine_id'].astype(int)
dataframe['latitude'] = dataframe['latitude'].astype(float)
dataframe['longitude'] = dataframe['longitude'].astype(float)
dataframe['vehicle_weight_type'] = dataframe['vehicle_weight_type'].astype(str)

dataframe.describe()

# Our data with descriptive statistical method mostly we need with vehicle speed data. It shows min and max number, we can see max speed is 255km/h that seems a record problem. 

#  

dataframe.info()

# +

data = Data([
    Scattermapbox(
        lat=dataframe['latitude'][:5000],
        lon=dataframe['longitude'][:5000],
        mode='markers',
        marker=Marker(
            size=4,
        ),
        
    )
])
layout = Layout(
    autosize=True,
    hovermode='closest',
    width=1500,
        margin=go.Margin(
        l=0,
        r=0,
        b=0,
        t=0
        ),
    height=700,
    mapbox=dict(
        accesstoken=mapbox_access_token,
        bearing=0,#
        center=dict(
            lat=-36.7832,
            lon=174.75
        ),
        pitch=0,
        zoom=11.3,
        style='dark',
          ),
)

fig = dict(data=data, layout=layout)
py.iplot(fig, filename='Multiple Mapbox')
# -



# Above maps shows us first 5000 data on the map. Showing all our data on the map most likely would crash our notebook but  all our GPS points between Auckland bridge and Oteha Valley intersection.

#  

#  

#  

# ### Import Weather API

# ###### The weather api is specific for our data which only store between 14-Now-2015 and 21-Now-2015 data

# +
# our data is from openweathermap.org. It is specific weather results between 14-11-2015 to 21-11-2015 Auckland weather which our cover our GPS tracking dates.
import urllib, json
url = "http://history.openweathermap.org//storage/debd7a72617dd61b0fc871a2c83fcabf.json"
response = urllib.urlopen(url)
data = json.loads(response.read())

from pandas.io.json import json_normalize    
df = json_normalize(data)
df1 = pd.DataFrame(df['weather'].str[0].values.tolist()).add_prefix('weather.')
print (df1.head())
 
df = pd.concat([df.drop('weather', 1), df1], axis=1)
# -

# In the df dataframe we have so many information temperature, wind speed, humidity information but only we need how the weather is so we can check how weather
#affevt traffic!
df.head()


weather = df[['dt_iso','weather.main']]

#we got date(dt_iso) and weather result(weather.main) also we can see data recorded every two hours.
weather.head()

weather.rename(columns={'dt_iso': 'time',
                     'weather.main':'dayweather'}, inplace = True)

# we have 4 different weather types between 5 days 
weather['dayweather'].unique()



#this function is removing each column last 10 characters 
# we cannot convert datetime with +0000 UTC data
def remove_first_three_chars(x):
    return (x.replace(x[-10:], ''))


weather[weather.columns[0:1]] = weather[weather.columns[0:1]].applymap(remove_first_three_chars)



weather['time'] = pd.to_datetime(weather['time'])

#changing name of time evet_timestamp for merge with other datafrane.
dataframe.rename(columns={'event_timestamp': 'time'}, inplace=True)

weather.info()

weather.head()

# +
# This function makes merge two data on time column on hours because weather data recorded every two hours but we have so many results between two in hours
# 

m = weather.set_index('time').resample('H')['dayweather'].first().ffill()
dataframe['dayweather'] = dataframe['time'].dt.floor('H').map(m)
# -

dataframe.head()

dataframe['dayweather'].unique()

#  

#  

# ## Import Google Geocode Api 

# ###### Google Geocode api is find details of your location data. The Google Api takes your location latitude and longitide data and gives you 'Country name', 'City name', 'District name','Formatted address' and so on. 

reverse_geocode_result = gmaps.reverse_geocode((-36.792033,174.755135))

reverse_geocode_result

# The Google geocode result all information above you can see. We will use Formeatted adress that will give a point of location.

reverse_geocode_result = gmaps.reverse_geocode((-36.752, 174.728))
df = json_normalize(reverse_geocode_result)
df.iloc[:,1][0]

# Picked a point in our data and then put in a piece of google api code then gave us a point. Our point on the Busway the district is Rosedale

dataframetrial = dataframe[:10]
dataframetrial['point'] = np.nan
dataframetrial

for row in dataframetrial.index:
    reverse_geocode_result = gmaps.reverse_geocode((dataframetrial['latitude'][row], dataframetrial['longitude'][row]))
    df = json_normalize(reverse_geocode_result)
    dataframetrial['point'][row] = df.iloc[:,1][0]

# Above code is for an example data that showing us how takes ingormation from Google Geocode Api.

dataframetrial

#  Dataframetrial is only a trial data of original data for showing work on Google geocode api example because pointing all data from Google geocode taking more than two days.

trial = dataframe[:50]

# +

data = Data([
    Scattermapbox(
        lat=trial['latitude'],
        lon=trial['longitude'],
        text = trial['speed_gps_kph'],
        mode='markers',
        marker=Marker(
            showscale=True,
                cmax=100,
                cmin=0,
            size=5,
            color =trial['speed_gps_kph'],
            colorscale= 'YlOrRd',
            #opacity=0.3,
            symbol = 'circle',
            
        ),
        
    )
])
layout = Layout(
    autosize=True,
    hovermode='closest',
    width=1450,
        margin=go.Margin(
        l=0,
        r=0,
        b=0,
        t=0
        ),
    height=650,
    mapbox=dict(
        accesstoken=mapbox_access_token,
        bearing=0,#
        center=dict(
            lat=-36.7832,
            lon=174.75
        ),
        pitch=0,
        zoom=11.3,
        style='dark',
        
        
    ),
)

fig = dict(data=data, layout=layout)
py.iplot(fig, filename='Multiple Mapbox')
# -



# Above graph shows us each point color that describe speed data points also you can see speed of vehicles when put mouse cursor on points.

#  

# ## Data Wrangling And Data Visualisation

# #### In this part we will have data transformation, merging, cleaning, and preparation for modelling. Also we will have Data Visualisation

#check unique values of vehicle types
list(dataframe['vehicle_weight_type'].unique())
#dataframe.vehicle_weight_type.value_counts()


# As we can see we have some null values in vehicle weight type and has double quote on values we are goint to remove all.
#

#convert null values
dataframe[dataframe.columns[4:5]] = dataframe[dataframe.columns[4:5]].replace(['NA'], np.nan)

dataframe.loc[pd.isnull(dataframe[['vehicle_weight_type']]).any(axis=1)].head()

#drop null values
dataframe.dropna(inplace=True,axis=0 )
list(dataframe['vehicle_weight_type'].unique())





# #### An imported point is we record all previous imports

newdataframe =pd.read_csv('secoutput.csv',index_col=False)

# After find all point from Google geocode api then stored in the "secoutput.csv" we will keep work on it.

newdataframe.head()

newdataframe['time'] = pd.to_datetime(newdataframe['time'])

# In our data point columns include city and country but all our data from auckland and we dont need it so we can drop city and country data.

newdataframe[['point', 'district', 'city','country']] = newdataframe['point'].str.split(',\s+', expand=True)
newdataframe.head()

newdataframe['point'].values

newdataframe = newdataframe.drop(['city','country'] , axis=1)

newdataframe.head(10)

# Now we have point value which describe our points exactly what street or road on it

newdataframe= newdataframe[newdataframe.speed_gps_kph != 255]

# In our data some record is 255 speed for vehicle we assume this all data are wrong so we will drop all 255km/h speed data.

newdataframe[newdataframe['speed_gps_kph']==255]



for x in newdataframe.columns: 
    if x[:] == 'dayweather':
        newdataframe[x] = newdataframe[x].replace('Drizzle', 'Rain')
        newdataframe[x] = newdataframe[x].replace('Clouds', 'Not-Rain')
        newdataframe[x] = newdataframe[x].replace('Clear', 'Not-Rain')

# We have 4 different weather data type but as we know drizzle is a type of rain and we suppose clear weather and cloud(If not rain) weather affect same traffic. So we are going to make only two type values Rain or Not-Rain

newdataframe[85:95]

# ##### Generally before and after work times have bad traffic on motorway so we will check on our data and what can we see on our data

#  

dataframe_monday_evening2 = newdataframe[(newdataframe['time']>='2015-11-16 17:00:00') & (newdataframe['time']<='2015-11-16 19:00:00')]

dataframe_monday_evening2.head()



dataframe_monday_evening = newdataframe[(newdataframe['time']>='2015-11-16 16:00:00') & (newdataframe['time']<='2015-11-16 19:00:00')]

dataframe_monday_evening.head()

# +

x1 = dataframe_monday_evening['time']
y1 =dataframe_monday_evening['speed_gps_kph']


# Create traces
trace0 = go.Scatter(
    x = x1,
    y = y1,
    mode = 'markers',
    name = 'Vehicle speed point'
)
layout= go.Layout(
    title= 'Vehicle Speed Data',
    hovermode= 'closest',
    xaxis= dict(
        title= 'Time',
        ticklen= 3,
        zeroline= False,
        gridwidth= 1.5,
    ),
    yaxis=dict(
        title= 'Speed',
        ticklen= 3,
        gridwidth= 1.5,
    ),
    showlegend= True
)
data =[trace0]
fig = dict(data=data, layout=layout)
py.iplot(fig, filename='styled-scatter')


# -



# Our grapgh is from 16 November 2015 Monday between 4 pm to 7 pm vehicle speed graph. A busy time for vehicle on motorway  as we can see we have well speed vehicle data as well because in the data included buses data which traffic is not affect them and the second thing is on the motorway if one side of rode has traffic most likely other side does not have heavy traffic. First we can remove bus data from motorway data.

motorwayframe = newdataframe[newdataframe['point'].str.contains("Motorway")]

motorwayframe.head()

buswayframe = newdataframe[newdataframe['point'].str.contains("Busway")]

buswayframe.head()



motorway_monday_morning = motorwayframe[(motorwayframe['time']>='2015-11-16 07:00:00') & (motorwayframe['time']<='2015-11-16 10:00:00')]

busway_monday_morning = buswayframe[(buswayframe['time']>='2015-11-16 06:00:00') & (buswayframe['time']<='2015-11-16 08:00:00')]

motorway_monday_evening = motorwayframe[(motorwayframe['time']>='2015-11-16 16:00:00') & (motorwayframe['time']<='2015-11-16 19:00:00')]

busway_monday_evening = buswayframe[(buswayframe['time']>='2015-11-16 16:00:00') & (buswayframe['time']<='2015-11-16 19:00:00')]

# +

x1 = motorway_monday_evening['time']
y1 =motorway_monday_evening['speed_gps_kph']
x2 = busway_monday_evening['time']
y2 =busway_monday_evening['speed_gps_kph']




# Create traces
trace0 = go.Scatter(
    x = x1,
    y = y1,
    mode = 'markers',
    name = 'Motorways vehicle'
)
trace1 = go.Scatter(
    x = x2,
    y = y2,
    mode = 'markers',
    name = 'Busway vehicle'
)
layout= go.Layout(
    title= 'Vehicle Speed Data',
    hovermode= 'closest',
    xaxis= dict(
        title= 'Time',
        ticklen= 3,
        zeroline= False,
        gridwidth= 1.5,
    ),
    yaxis=dict(
        title= 'Speed',
        ticklen= 3,
        gridwidth= 1.5,
    ),
    showlegend= True
)
data =[trace0, trace1]
fig = dict(data=data, layout=layout)
py.iplot(fig, filename='styled-scatter')


# -



# The Above graph shows us seperate each motorway abd busway data. As we can see busway data mostly 0 km/h data and well speed data but we can see still we have so many fast speed data and slow speed data together. It seems bots side of motorway vehicles has different speed so we should  find both side of motorway data.

#  

# +

data = Data([
    Scattermapbox(
        lat=dataframe_monday_evening2['latitude'],
        lon=dataframe_monday_evening2['longitude'],
        mode='markers',
        marker=Marker(
            showscale=True,
                cmax=100,
                cmin=0,
            size=5,
            color =dataframe_monday_evening2['speed_gps_kph'],
            colorscale= 'YlOrRd',
            #opacity=0.3,
            symbol = 'circle',
            
        ),
        
    )
])
layout = Layout(
    autosize=True,
    hovermode='closest',
    width=1580,
        margin=go.Margin(
        l=0,
        r=0,
        b=0,
        t=0
        ),
    height=700,
    mapbox=dict(
        accesstoken=mapbox_access_token,
        bearing=0,#
        center=dict(
            lat=-36.7549,
            lon=174.73
        ),
        pitch=0,
        zoom=15.2,
        style='dark',
        
        
    ),
)

fig = dict(data=data, layout=layout)
py.iplot(fig, filename='Multiple Mapbox')
# -



# Above map graph shows us trafic density for each side of motorway and busway. The color scale is next graph as you can see red points describe slow speed and yellow speeds are describe fast speed data. The scale between 0-100 km/h. Please zoom out and check each specific locations.

motorwayrain = motorwayframe[(motorwayframe['dayweather'] == "Rain")&((motorwayframe['time']>='2015-11-15 17:00:00') & (motorwayframe['time']<='2015-11-18 19:00:00'))]
motorwayrain.head(3)

motorwaynotrain = motorwayframe[(motorwayframe['dayweather'] == "Not-Rain")&((motorwayframe['time']>='2015-11-14 17:00:00') & (motorwayframe['time']<='2015-11-18 19:00:00'))]
motorwaynotrain.head(3)



motorwayrain['speed_gps_kph'].describe()

motorwaynotrain['speed_gps_kph'].describe()

# The rain affect speed of vehicle as we can see when motorwayrain day show us mean of speed 66 and standart deviation is 28.

newdataframe['time']= pd.to_datetime(newdataframe['time'])
newdataframe = newdataframe.sort_values(by=['machine_id', 'time'])

#  

# An external python files that store a function that name is Locationpoint
execfile('functions.py')

LocationPoint(newdataframe)
newdataframe.head()



newdataframe.reset_index(drop=True, inplace=True)

newdataframe.head()

pointstart = LocationPoint(newdataframe)
newdataframe['GPS_start'] = pointstart
pointlast = pointstart
pointlast.pop(0)
pointlast.append(True)
newdataframe['GPS_end'] = pointlast

newdataframe.head()

direction = []
for x in (newdataframe.index):
    if((newdataframe['GPS_end'][x] != True)):
        if((newdataframe['latitude'][x]) > (newdataframe['latitude'][x+1])):
            direction.append('To_City')
           # motorway_monday_evening['direction'][x] = 'To_city'
        else:
            direction.append('From_City')
            #motorway_monday_evening['direction'][x] = 'From_city'
    else:
        
        direction.append('NAN')
  

#  

# With above functions first find travel start and and points(calculate for each vehicle ID separately), then create 2 new columns and then calculate latitude for each point that is important to see movement on the motorway. Idea is between each point if latitude increase then vehicle move to From city if each point latitude values decrease means the vehicles moves to city

#  

direction = pd.DataFrame(direction)

newdataframe = pd.concat([newdataframe, direction], axis=1)

newdataframe.rename(columns={0: 'direction'}, inplace=True)
newdataframe.head()

newdataframe[newdataframe.columns[-1]] = newdataframe[newdataframe.columns[-1]].replace(['NAN'], np.nan)

newdataframe.fillna(method='ffill', inplace=True)

# +

newdataframe.drop(newdataframe.columns[[11,12]], axis=1, inplace=True)
newdataframe.head(10)
# -



motorway_monday_evening = newdataframe[(newdataframe['time']>='2015-11-16 16:00:00') & (newdataframe['time']<='2015-11-16 19:00:00')]

motorway_monday_evening_tocity = motorway_monday_evening[motorway_monday_evening['direction'] == 'To_City']

motorway_monday_evening_tocity.head()

# +
x1 = motorway_monday_evening_tocity['time']
y1 =motorway_monday_evening_tocity['speed_gps_kph']


# Create traces
trace0 = go.Scatter(
    x = x1,
    y = y1,
    mode = 'markers',
    name = 'Vehicle speed point'
)
layout= go.Layout(
    title= 'Vehicle Speed Data',
    hovermode= 'closest',
    width=1500,
    xaxis= dict(
        title= 'Time',
        ticklen= 3,
        zeroline= False,
        gridwidth= 1.5,
    ),
    yaxis=dict(
        title= 'Speed',
        ticklen= 3,
        gridwidth= 1.5,
    ),
    showlegend= True
)
data =[trace0]
fig = dict(data=data, layout=layout)
py.iplot(fig, filename='styled-scatter')



# -



# This result is more expected result. Also our graph shows us after 5 pm car speed strongly decreasing. Now our question is what can we make if we know exactly traffic time and location?

latlong = newdataframe.iloc[1:,[6,7]]
# create this statement idea is put lanitude/longitude 1 rows name is latitude2 and longitude2 then find distance between two points.

latlong.rename(columns={'latitude': 'latitude2','longitude':'longitude2'}, inplace=True)

latlong.reset_index(drop=True, inplace=True)

newdataframe =pd.concat([newdataframe, latlong], axis=1)

newdataframe.dropna(axis=0, inplace=True)

# +
#motorwayframe.reset_index(drop=True, inplace=True)
# -

newdataframe.head()

# Latitude2 and Longitude2 values each previous rows Latitude and Longitude values. It created for finding distance each previous point distance

# +

distance = []
for x in (newdataframe.index):
        
    dlon = newdataframe['longitude2'][x] - newdataframe['longitude'][x]
    dlat = newdataframe['latitude2'][x] - newdataframe['latitude'][x]
    a = (sin(dlat/2))**2 + cos(newdataframe['latitude'][x]) * cos(newdataframe['latitude2'][x]) * (sin(dlon/2))**2
    c = 2 * asin(sqrt(a)) 
    distance.append(6373.0 * c)

# This function will give us between two points distance as a miles  
# -

distance = pd.DataFrame(distance)

newdataframe =pd.concat([newdataframe, distance], axis=1)



newdataframe.rename(columns={0: 'distance'}, inplace=True)

# +

newdataframe.head()
# -

# Below code is running quite long time please be aware

# +
for x in (newdataframe.index):
    if((newdataframe['machine_id'][x]) != (newdataframe['machine_id'][x+1])):
        newdataframe['distance'][x] = np.nan
    elif((newdataframe.index[x+1]) == 566949):
        break
    else:
        pass
        
        
        
# Our idea is find distance points with same machine id so between two diffrent machine id data will be useless so we are going to drop with this function
# Also we should use manual break function because of using [x] != [x +1]
# -

newdataframe.dropna(axis=0,inplace=True)



for x in newdataframe.columns: 
    if x[:] == 'dayweather':
        newdataframe[x] = newdataframe[x].replace('Rain', 1)
        newdataframe[x] = newdataframe[x].replace('Not-Rain', 0)

for x in newdataframe.columns: 
    if x[:] == 'direction':
        newdataframe[x] = newdataframe[x].replace('To_City', 1)
        newdataframe[x] = newdataframe[x].replace('From_City', 0)

# +
#motorwayframe.head()
# -

#  

# # Data Analysis

# ## Supervised Learning

# #### Prediction of Vehicle Direction. 

# As we saw our data visiulation part shows us on the specific time and locations we can see different result on the graphs so now we can predict data in out Machine Learning algorithms. In this part we will choose some values and predict data.

#   

#  

motorway_monday_evening = newdataframe[(newdataframe['time']>='2015-11-16 17:00:00') & (newdataframe['time']<='2015-11-16 19:00:00')]

motorway_monday_evening.head()

# ## Logistic Regression

directionanalysis2 = motorway_monday_evening.iloc[:,([5,11,14])]
directionanalysis2.head()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import classification_report, confusion_matrix

directionanalysis2.head()

# +
X = directionanalysis2.iloc[:,[0,2]].values
y = directionanalysis2['direction']

train_test_split = X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
# -

logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)

predictions = logmodel.predict(X_test)

print (confusion_matrix(y_test, predictions))
print(classification_report(y_test,predictions))
print('\n')
print "Test accuracy for Logistic Regression:", logmodel.fit(X_train, y_train).score(X_test, y_test)

#  

# ### K Nearest Neighbors

from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# +
#directionanalysis2 = motorway_monday_evening.iloc[:,([5,11,14])]
#directionanalysis2.head()
# -

# Assign scaler to StandardScaler library
scaler = StandardScaler()
scaler.fit(directionanalysis2.drop('direction', axis=1))

scaled_feature = scaler.transform(directionanalysis2.drop('direction', axis=1))


df_feat  = pd.DataFrame(scaled_feature, columns=['speed','distance'])

# +
# Splitting the dataset into the Training set and Test set
X = df_feat
y = directionanalysis2['direction']

train_test_split = X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
# -

from sklearn.neighbors import KNeighborsClassifier

# Fitting K-NN to the Training set
# When first see result we will start k=1 then we choose best k value after see result
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
# Predicting the Test set results
pred = knn.predict(X_test)


# Making the Confusion Matrix
print (confusion_matrix(y_test, pred))
print (classification_report(y_test, pred))

# +
error_rate = []

for x in range(1,40):
    knn = KNeighborsClassifier(n_neighbors=x)
    knn.fit(X_train, y_train)
    pred_x = knn.predict(X_test)
    error_rate.append(np.mean(pred_x != y_test))


# +
plt.figure(figsize=(10,6))
plt.plot(range(1,40), error_rate, color = 'blue', linestyle = 'dashed', marker = 'o', markerfacecolor = 'red', markersize = 10)


# -

knn = KNeighborsClassifier(n_neighbors=17)
knn.fit(X_train, y_train)
pred = knn.predict(X_test)
print (confusion_matrix(y_test, pred))
print('\n')
print (classification_report(y_test, pred))
print('\n')
print "Test accuracy for KNN:", knn.fit(X_train, y_train).score(X_test, y_test)

#  

# ## Decision Tree

# +
# Splitting the dataset into the Training set and Test set
X = directionanalysis2.iloc[:,[0,2]]
y = directionanalysis2['direction']


# -

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)

predictions = dtree.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
print('\n')
print "Test accuracy for Decision Tree:", dtree.fit(X_train, y_train).score(X_test, y_test)

#  

# ## Random Forest

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train)

rfc_pred = rfc.predict(X_test)
print(confusion_matrix(y_test,rfc_pred))
print(classification_report(y_test,rfc_pred))
print('\n')
print "Test accuracy for Random Forest:", rfc.fit(X_train, y_train).score(X_test, y_test)

#
#  

#  

# ####  We tried with our sample data 4 different Machine learning Supervised algorithm then we can see the best result is K Nearest Neighbors algorith. The result of Test accuracy for KNN is : 0.747019551741

#  Unfortuanely a few last minute changing and therefore loosing data our new result is 0.763471626133. previous result was 0.847456725171.

#  

# # Unsupervised Learning

# ## K Means Clustering

scaler = StandardScaler()
scaler.fit(directionanalysis2.drop('direction', axis=1))

scaled_feature = scaler.transform(directionanalysis2.drop('direction', axis=1))
df_feat  = pd.DataFrame(scaled_feature, columns=['speed','distance'])

# +

X = df_feat.values
# y = dataset.iloc[:, 3].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# -

from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# The elbow method shows us the number of clusters is 3

kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)

y_kmeans[125:140]

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0,1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'grey', label = 'Cluster 3')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'black', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Distance')
plt.ylabel('Speeed')
plt.legend()
plt.show()

# Mostly 0 speed but we have distance so red data more distance and more 0 value we can say that might be the bus data blue data is small distance looks like traffic records. Grey data might be not busy side of motorway vehicle data.

# ## Result

#   ##### In this project we had a occasion to chech the Northern Motorway GPS data. we used Data Wrangling And Data Visualisation techniques also add the weather ang  find different features. 
#  ##### Weather data shows us how weather can affect speed of vehicles and Google api data shows us some location is very importatnt for traffic.
#   ##### We used machine learning algorithms to predict some data. We used supervised learning algorithms that best result came from K Nearest Neighbors. Also we used unsupervised Learning algorithm method that name is K Means Clustering.
#  


