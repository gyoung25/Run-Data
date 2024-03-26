#Prepare activity data from Strava by only keeping runs, converting units, adding new columns, removing outliers, and putting the dates into standard form
def prep_run_data(activityDataCSV):
    
    import pandas as pd
    import numpy as np
    import matplotlib.dates as mdates
    
    #Load activity data
    runData = pd.read_csv(activityDataCSV)
    
    #Keep only the run data
    runData = runData[runData["Activity Type"]=="Run"]
    
    #Convert distance from kilometers to miles
    runData.loc[:,"Distance"] = runData["Distance"]*0.621371
    #Convert elevation gain from meters to feet
    runData.loc[:,"Elevation Gain"] = runData["Elevation Gain"]*3.2808
    #Convert time from seconds to minutes
    runData.loc[:,"Moving Time"] = runData["Moving Time"]/60

    
    #Calculate average pace for each run
    avePace = runData["Moving Time"].to_numpy()/runData["Distance"].to_numpy()
    #Insert the average pace back into the data frame
    runData.insert(5, "Pace (Min/Mile)", pd.Series(avePace, runData.index), False)
    
    #Calculate average elevation gain for each run
    aveElevGainNP = runData["Elevation Gain"].to_numpy()/runData["Distance"].to_numpy()
    #Insert the average elevation gain back into the data frame
    runData.insert(2, "Ave Elev Gain (Ft/Mi)", pd.Series(aveElevGainNP, runData.index), False)
    
    #Remove pace outliers from run data
    #Remove paces slower than 8 min/mile (probably running with another person) or faster than 6 min/mile (probably racing)
    runData = runData[(runData["Pace (Min/Mile)"]<8) & (runData["Pace (Min/Mile)"]>=6)]
    #Remove runs less than 2 miles from the data
    runData = runData[runData["Distance"] > 2]
    
    #Convert activity dates to datetime objects
    actDates = pd.to_datetime(runData["Activity Date"])
    
    #Add a column of activity dates to the runDataClean dataframe in standardized format
    runData.insert(2, "Date", actDates.apply(lambda x: x.date()), False)
    
    
    
    return runData[["Date", "Pace (Min/Mile)", "Distance", "Elevation Gain", "Ave Elev Gain (Ft/Mi)"]]

#Prepare weather data by converting the dates into standard form and removing uncessary columns
def prep_weather_data(weatherDataCSV):

    import pandas as pd
    import numpy as np
    import matplotlib.dates as mdates
    
    #Import weather data from Atlanta
    weatherData = pd.read_csv(weatherDataCSV)
    
    weatherDates = pd.to_datetime(weatherData["DATE_OLD"])
    
    #Make new column with formatted dates
    weatherData.insert(2, "Date", weatherDates.apply(lambda x: x.date()), False)
    
    return weatherData[["Date", "AWND", "PRCP", "TAVG", "TMAX", "TMIN"]]

#Inner join the run data and weather data on the date
def merge_data(activityDataCSV, weatherDataCSV):

    runData = prep_run_data(activityDataCSV)
    weatherData = prep_weather_data(weatherDataCSV)
    
    #Inner join the run and weather data on the Dates column
    mergedData = runData.merge(weatherData, on = "Date", how = "inner")
    
    return mergedData

#Plot my pace over time AND the temperature range each day
def plot_pace_over_time(mergedData):
    
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    
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
    
    return None

#Fit a regression curve to temperature and pace data
def regression_fit(mergedData, degree = 1):

    import numpy as np
    
    #Find regression line
    x=np.array(mergedData["TMAX"])
    y=np.array(mergedData["Pace (Min/Mile)"])
    if degree == 1:
        #linear fit
        m, d = np.polyfit(x,y,1)
        return m, d
    elif degree == 2:
        #quadratic fit
        a, b, c = np.polyfit(x,y,2)
        return a, b, c
    else:
        raise Exception("Degree must be 1 or 2.")
        return None

#Plot pace as a function of daily max temperature
def plot_pace_over_temp(mergedData):
    import matplotlib.pyplot as plt
    
    paceTemp = mergedData[["Pace (Min/Mile)","TMAX"]].copy(True)
    tempBins = {}
    for x in range(30, 91, 10):
        tempBins["{0}-{1}".format(x,x+10)] = paceTemp[(paceTemp["TMAX"]>=x) & (paceTemp["TMAX"]<x+10)]    
        paceOverTemp={}
        
    for x in tempBins:
        paceOverTemp[x] = tempBins[x]["Pace (Min/Mile)"].to_numpy()
        
    plt.figure()
    plt.boxplot(paceOverTemp.values())
    plt.xticks(range(1, len(paceOverTemp.keys()) + 1), paceOverTemp.keys())
    plt.xlabel("Daily Max Temp (F)")
    plt.ylabel("Average Running Pace (Min/Mile)")
    plt.show()
    
    return None

#Returns a dictionary of lists. Each list in the dictionary contains average run paces on runs over a range of average elevation gains
#Almost all of my runs fall within 30 ft/mi to 120 ft/mi elevation gain. We'll therefore divide the interval [30,120] into numInts intervals
def stratify_pace_by_elevation(numInts = 3):
    
    mergedData = merge_data('activities_Dec2023.csv', 'weatherData_Dec2023.csv')
    stratPaceByElev={}
    #Define a list dividing [30, 120] into numInts equal steps (excluding the right end point)
    elevStrats=[30 + x*90/numInts for x in range(numInts)]
    
    for x in elevStrats:
        stratPaceByElev[x]=mergedData[(mergedData["Ave Elev Gain (Ft/Mi)"] >= x) & (mergedData["Ave Elev Gain (Ft/Mi)"] < x+90/numInts)]
    return stratPaceByElev

#Plot run pace as a function of temperature for different ranges of average elevation gain
def plot_pace_by_elevation(numInts = 3):
    import numpy as np
    import matplotlib.pyplot as plt
    from numpy.polynomial import polynomial as P
    
    stratPaceByElev = stratify_pace_by_elevation(numInts)
    
    fig, axs = plt.subplots(2,2,figsize = (8,6), sharey = True, sharex = True)
    fig.supylabel("Average Running Pace (Min/Mile)", fontsize = 18)
    fig.supxlabel("Daily Max Temp (F)", fontsize = 18)
    color = ['tab:red', 'tab:blue', 'tab:green']
    axis_dict = {0: 0, 1: 0, 2: 1}
    for i, j in enumerate(stratPaceByElev):
        ax = plt.subplot(2, 2, i+1)
        
        #compute regression line
        x = np.array(stratPaceByElev[j]["TMAX"])
        y = np.array(stratPaceByElev[j]["Pace (Min/Mile)"])
        c, stats = P.polyfit(x,y,1,full=True)
        
        #plot data and regression line on the current axis of the subplot
        ax.plot(x,y,"o", alpha = 0.5, color = color[i])
        ax.plot(x, c[1]*x+c[0], linewidth = 3, color = color[i])
        ax.title.set_text("{} to {} ft/mi".format(int(j),int(j+30)))
        
        #create a mask to hide some of the data for the final plot
        ran = (np.random.rand(len(y)) < 0.5)*1
        y_mask = np.ma.masked_where(ran == 1,y)
        axs[1,1].plot(x,y_mask,"o", alpha = 0.4, color = color[i])
        axs[1,1].plot(x, c[1]*x+c[0], linewidth = 3, color = color[i], label="[{},{}]".format(int(j),int(j+30)))
        ax.set(xlim=(30, 105), ylim=(6, 8))
        #hide axis labels of interior axes
        if i < 2:
            plt.tick_params(bottom = False)
        if i == 1:
            plt.tick_params(left = False)
    ax = plt.subplot(2, 2, 4)
    plt.tick_params(left = False)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    #axs[1,1].legend()
    plt.show()
    return None

#Calculates the linear regression slope as a function of temperature for each elevation band
#and returns an array of regression line slopes with length equal to the number of elevation bands
def regression_slope_over_elevation(numInts):
    
    import numpy as np
    from numpy.polynomial import polynomial as P
    
    #Define a list dividing [30, 120] into numInts equal steps (excluding the right end point)
    elevStrats = [30 + x*90/numInts for x in range(numInts)]
    
    stratPaceByElev = stratify_pace_by_elevation(numInts)
    
    slopes = [None] * numInts
    
    for i, j in enumerate(stratPaceByElev):
        #compute regression line
        t = np.array(stratPaceByElev[j]["TMAX"])
        y = np.array(stratPaceByElev[j]["Pace (Min/Mile)"])
        c, stats = P.polyfit(t,y,1,full=True)
        slopes[i] = c[1]
        
    return slopes, elevStrats

#TO ADD: function that plots the output of regression_slope_over_elevation
def plot_regression_slopes(numInts):
    
    import matplotlib.pyplot as plt
    
    slopes, elevStrats = regression_slope_over_elevation(numInts)
    
    stratPaceByElev = stratify_pace_by_elevation(numInts)
    
    avePaceByElev = [None]*len(stratPaceByElev)
    for i, x in enumerate(stratPaceByElev):
        avePaceByElev[i] = stratPaceByElev[x]['Pace (Min/Mile)'].mean()
    
    #slopes = [60*slopes[i]/avePaceByElev[i] for i in range(len(slopes))] #uncomment for slopes relative to pace
    slopes = [60*slopes[i] for i in range(len(slopes))]

    plt.figure()
    plt.plot(elevStrats, slopes)
    plt.xlabel("Elevation gain (feet)")
    plt.ylabel("Pace change (seconds per $^\circ$F)")
    plt.show()
