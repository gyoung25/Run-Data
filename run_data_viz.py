from runDataFunctions import *

mergedData = merge_data('activities_Dec2023.csv', 'weatherData_Dec2023.csv')

plot_pace_over_time(mergedData)
plot_pace_over_temp(mergedData)
plot_pace_by_elevation(mergedData)
plot_regression_slopes(mergedData, 5)