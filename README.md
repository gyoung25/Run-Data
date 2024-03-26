# Run-Data
Clean, visualize, and analyze running data from my Strava account.

Several functions found in runDataFunctions for visualizing my run data, mostly as a function of temperature and elevation.

- plot_pace_over_time() plots my running pace over time along with the temperature range for each day I ran.
- plot_pace_over_temp() plots my pace against daily max temperature
- plot_pace_by_elevation() plots my pace against daily max temperature in separate subplots corresponding to different elevation-gain ranges
- plot_regression_slopes() is intended to visualized second-order effects of elevation gain. It plots the linear regression slope as a function of temperature for each elevation band. This illustrates how much slower i get for each additional degree F across a range of elevation gains.
