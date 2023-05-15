import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

#Import all of my Strava data
allData = pd.read_csv('activities.csv')

#Keep only the run data
runData=allData[allData["Activity Type"]=="Run"]

#Convert distances from km to miles
distMilesNP=np.array(runData["Distance"]*0.621371)
#Convert time from seconds to minutes
timeMinutesNP=np.array(runData["Moving Time"]/60)

#Calculate average pace for each run
avePace=timeMinutesNP/distMilesNP
#Insert the average pace back into the data frame
avePaceDF=pd.DataFrame(avePace,runData.index)
runData.insert(5,"Pace (Min/Mile)",avePaceDF,False)

#Remove outliers from run data
runDataClean=runData[runData["Pace (Min/Mile)"]<8]

#Convert activity dates to datetime objects
actDates=pd.to_datetime(runDataClean["Activity Date"])

#Add a column of activity dates to the runDataClean dataframe in standardized format
runDataClean.insert(2,"Date",actDates.apply(lambda x: x.date()),False)

#Import weather data from Atlanta
weatherData = pd.read_csv('weatherData.csv')

weatherDates=pd.to_datetime(weatherData["DATE_OLD"])

#Make new column with formatted dates
weatherData.insert(2,"Date",weatherDates.apply(lambda x: x.date()),False)

#Inner join the run and weather data on the Dates column
mergedData = runDataClean.merge(weatherData, on="Date", how="inner")

#Find regression line
x=np.array(mergedData["TMAX"])
y=np.array(mergedData["Pace (Min/Mile)"])
#linear fit
m, d = np.polyfit(x,y,1)
#quadratic fit
a, b, c = np.polyfit(x,y,2)

#--------------- Plot Pace over time -------------%


# plot
fig, ax1 = plt.subplots()

ax2 = ax1.twinx()

color = 'tab:red'
ax2.set_ylabel('Daily Temperature Range', color = color)
ax2.fill_between(mergedData["Date"],mergedData["TMIN"],mergedData["TMAX"],color=color,alpha=0.4)
#ax2.plot(mergedData["Date"],mergedData["TMAX"], color = color)
#ax2.plot(mergedData["Date"],mergedData["TMIN"], color = color)
ax2.tick_params(axis ='y', labelcolor = color)

color = 'tab:blue'
ax1.set_ylabel('Average Running Pace (Min/Mile)', color = color)
ax1.plot(mergedData["Date"],mergedData["Pace (Min/Mile)"],'o', linewidth=2, markersize=8, color=color)
ax1.tick_params(axis ='y', labelcolor = color)

# Set the locator and format
locator = mdates.MonthLocator(interval=3)  # every 3 months
fmt = mdates.DateFormatter("%m-%y") # MM-YY format
X = plt.gca().xaxis
X.set_major_locator(locator)
X.set_major_formatter(fmt)
for tick in ax1.get_xticklabels():
    tick.set_rotation(45)
ax1.set_xlabel('Date')
plt.autoscale(enable=True, axis='both', tight=None)
plt.show()

#------------ Plot Pace vs Temp -------------_#


plt.figure()
plt.plot(mergedData["TMAX"],mergedData["Pace (Min/Mile)"],'o')
plt.plot(x,m*x+d,label ='Linear fit')
plt.plot(x,a*x*x+b*x+c,label ='Quadratic fit')
plt.xlabel("Daily Max Temp (F)")
plt.ylabel("Average Running Pace (Min/Mile)")
plt.legend()
plt.show()
