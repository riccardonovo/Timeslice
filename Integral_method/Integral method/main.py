# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 14:14:00 2021

@author: Riccardo Novo
"""

from IPython import get_ipython
get_ipython().magic('reset -sf')

import pandas as pd
# import matplotlib as plt
import numpy as np
import os
import re
import natsort


# Paths and files
parent_folder = os.path.dirname(os.getcwd())
data_folder = os.path.join(parent_folder,'Data')
file_input = os.path.join(data_folder, 'Input.xlsx')
file_param_output = os.path.join(data_folder, 'Output_parameters.xlsx') 


# %% Reading inputs and opening output file

# Reading seasons set
df_season = pd.read_excel(file_input, 'Sets', header=0, usecols='A:C', keep_default_na='True', dtype='str')
df_season.dropna(inplace=True)

# Reading daytype set
df_daytype = pd.read_excel(file_input, 'Sets', header=0, usecols='E:F', keep_default_na='True', dtype='str')
df_daytype.dropna(inplace=True)

# Reading dailytimebracket set
df_dailytimebracket = pd.read_excel(file_input, 'Sets', header=0, usecols='H:J', keep_default_na='True', dtype='str')
df_dailytimebracket.dropna(inplace=True)

# Reading timeslice set
df_timeslice = pd.read_excel(file_input, 'Sets', header=0, usecols='L', keep_default_na='True', dtype='str')
df_timeslice.dropna(inplace=True)

# Reading the timeseries summary table
df_timeseries = pd.read_excel(file_input, 'Timeseries', header=0, usecols='A:H', keep_default_na='True', dtype={
    'A':str, 'B':str, 'C':str, 'D':'int64'})
df_timeseries.dropna(inplace=True)

# Reading the whole excel file for obtaining the timeseries
xl = pd.read_excel(file_input, sheet_name=None)


# %% Resampling all time series

# For cycle on the list of timeseries
for index, row in df_timeseries.iterrows():
    
    ## Assigning the last value of the previous year and the first value of the following year
    
    # Defining a new variable containing the timeseries name
    ts_name = row['Timeseries name']
    
    # Defining the row containing data for the last time instant of the previous year and the first time instant of the following year
    first_row = pd.DataFrame({'Timestamp':xl[ts_name]['Timestamp'].iloc[-1] - pd.to_timedelta(365, unit='days'), 'Value':xl[ts_name]['Value'].iloc[-1]}, index=[-1])
    last_row = pd.DataFrame({'Timestamp':xl[ts_name]['Timestamp'].iloc[0] + pd.to_timedelta(365, unit='days'), 'Value':xl[ts_name]['Value'].iloc[0]}, index=[0])
    
    # Adding the last time instant of the previous year
    xl[ts_name] = xl[ts_name].append(first_row, ignore_index=False)
    
    # Shifting the last time instant of the previous year at the first place
    xl[ts_name] = xl[ts_name].sort_index().reset_index(drop=True)
    
    # Adding the first time instant of the following year
    xl[ts_name] = xl[ts_name].append(last_row, ignore_index=True)


    ## Resampling all time series
    
    # CASE A: Instantaneous value time serie
    if row['Timeseries type'] == 'Average value':
        
        # Resample the time serie with a 1 minute time interval calculating the mean value, linearly interpolating missing values; resample with a 1 hour time interval
        xl[ts_name] = xl[ts_name].set_index('Timestamp').resample('1T').mean().interpolate('linear').resample('1H').first()
    
    # CASE B: Integral value time serie   
    elif row['Timeseries type'] == 'Integral value':
        
        # Resample the time serie with a 1 minute time interval
        xl[ts_name] = xl[ts_name].set_index('Timestamp').resample('1T').first()
        
        # Group the time serie based on the NaN values following each not-NaN value and assign the average value to each of them 
        xl[ts_name] = xl[ts_name].groupby(xl[ts_name]['Value'].notna().cumsum()).apply(lambda x: x/len(x.index)).ffill()
        
        # Resample the time serie with a 1 hour time interval
        xl[ts_name] = xl[ts_name].resample('1H').sum()
    
    # Dropping values which are not in the reference year
    xl[ts_name] = xl[ts_name][xl[ts_name].index.year == row['Reference year']]
    
    # Apply units conversion if the output parameter needed is SpecifiedAnnualDemand
    if row['Output parameter'] == 'SpecifiedDemandProfile':
        # Units conversion
        xl[ts_name] = xl[ts_name]*row['Conversion factor']



# %% Assigning a time slice to every hour in the year

# Year of each timestamp
t_year = xl[df_timeseries['Timeseries name'].iloc[0]].index.year

# Month of each timestamp
t_month = xl[df_timeseries['Timeseries name'].iloc[0]].index.month

# Day of each timestamp
t_day = xl[df_timeseries['Timeseries name'].iloc[0]].index.day

# Hour of each timestamp
t_hour = xl[df_timeseries['Timeseries name'].iloc[0]].index.hour

# Day of the year for each timestamp
t_dayofyear = xl[df_timeseries['Timeseries name'].iloc[0]].index.dayofyear

# Day of the week for each timestamp
t_dayofweek = xl[df_timeseries['Timeseries name'].iloc[0]].index.weekday

# Define a new dataframe for hosting the association between timestamps and time-slices
xl['Timeslices'] = pd.DataFrame(columns={'Timestamp','TIMESLICE'})
xl['Timeslices']['Timestamp'] = xl[df_timeseries['Timeseries name'].iloc[0]].index


## Define the association timestamp - time-slice

# For loop over all the hours of the year
for ii in np.arange(0, len(t_year)):
    
    # Find the SEASON in the current timestamp and assign it to variable sss
    for index, row in df_season.iterrows():
        if t_dayofyear[ii] <= int(df_season['End day'].iloc[index]):
            sss = row['SEASON'];
            break        
    # Find the DAYTYPE in the current timestamp and assign it to variable ddd
    if df_daytype['Type'].iloc[0] == 'Weekday' or df_daytype['Type'].iloc[0] == 'Weekend':
        if t_dayofweek[ii] <= 4:
            ddd = df_daytype.loc[df_daytype['Type'] == 'Weekday']['DAYTYPE'].iloc[0]
        else:
            ddd = df_daytype.loc[df_daytype['Type'] == 'Weekend']['DAYTYPE'].iloc[0]
    elif df_daytype['Type'].iloc[0] == 'All days' or df_daytype['Type'].iloc[0] == 'Single day':
        ddd = df_daytype['DAYTYPE'].iloc[0] 
    
    # Find the DAILYTIMEBRACKET in the current timestamp and assign it to variable ttt
    for index, row in df_dailytimebracket.iterrows():
        if t_hour[ii] < int(df_dailytimebracket['End hour'].iloc[index]):
            ttt = row['DAILYTIMEBRACKET'];
            break
    
    # Concatenate strings to assign the right timeslice to the timestamp 
    xl['Timeslices'].loc[ii, 'TIMESLICE'] = 'S' + sss + 'D' + ddd + 'T' + ttt
    
# Set the timestamp as index of the dataframe
xl['Timeslices'] = xl['Timeslices'].set_index('Timestamp')

# Create a new dataframe for every timeserie to be elaborated and insert the ordered timeslice in correspondance of every timestamp
for index, row in df_timeseries.iterrows():
    ts_name = row['Timeseries name']
    xl[ts_name]['TIMESLICE'] = xl['Timeslices'].values


# %% Computing time slice parameters
xs ={}


## Yearsplit

# Define a new dataframe for YearSplit parameter
xs['YearSplit'] = pd.DataFrame(columns={'TIMESLICE', 'YearSplit'})

# Assign the values of timeslices to the TIMESLICE column
xs['YearSplit']['TIMESLICE'] = df_timeslice['TIMESLICE']

# Loop for every timeslice
for tl in xs['YearSplit']['TIMESLICE']:
    
    # Define the number of hourly timestamps in that timeslice for every year
    n_hours_tl_year = len(xl['Timeslices'][xl['Timeslices']['TIMESLICE'] == tl])
    
    # Calculate the YearSplit parameter and assign the value
    xs['YearSplit']['YearSplit'][xs['YearSplit']['TIMESLICE']==tl] = n_hours_tl_year/8760

# Set TIMESLICE column as index of the dataframe
xs['YearSplit'] = xs['YearSplit'].set_index('TIMESLICE')


## DaySplit

# Define a new dataframe for DaySplit parameter
xs['DaySplit'] = pd.DataFrame(columns={'DAILYTIMEBRACKET', 'DaySplit'})

# Assign the values of dailytimebrackets to the DAILYTIMEBRACKET column
xs['DaySplit']['DAILYTIMEBRACKET'] = df_dailytimebracket['DAILYTIMEBRACKET']

# For loop over all the dailytimebrackets
for index, row in xs['DaySplit'].iterrows():
    
    # Define the starting hour of that dailytimebracket
    h_start = int(df_dailytimebracket['Start hour'][df_dailytimebracket['DAILYTIMEBRACKET'] == row['DAILYTIMEBRACKET']])
    
    # Define the end hour of that dailytimebracket   
    h_end = int(df_dailytimebracket['End hour'][df_dailytimebracket['DAILYTIMEBRACKET'] == row['DAILYTIMEBRACKET']])  
    
    # Define the number of hours in that day as the difference between the end hour and the starting hour
    n_hours_tl_day =  h_end - h_start
    
    # Calculate YearSplit parameter and assign the value
    xs['DaySplit']['DaySplit'][xs['DaySplit']['DAILYTIMEBRACKET']==row['DAILYTIMEBRACKET']] = n_hours_tl_day/(24*365)

# Set DAILYTIMEBRACKET colunmn as index of the dataframe    
xs['DaySplit'] = xs['DaySplit'].set_index('DAILYTIMEBRACKET')


## DaysInDayType

# Define a new parameter for DaysInDayType parameter
xs['DaysInDayType'] = pd.DataFrame(columns={'DAYTYPE', 'DaysInDayType'})

# Assign the values of daytypes to the DAYTYPE column
xs['DaysInDayType']['DAYTYPE'] = df_daytype['DAYTYPE']

# For loop over all the daytypes
for index, row in xs['DaysInDayType'].iterrows():
    
    # Define the type of day of the week 
    dtype = df_daytype[df_daytype['DAYTYPE'] == row['DAYTYPE']]['Type'].iloc[0]
    
    # If the type of day is a day of the week, the length of the days is 5
    if dtype == 'Weekday':
        xs['DaysInDayType']['DaysInDayType'][index] = 5
    
    # If the type of day is a weekday, the length of the days is 2
    elif dtype == 'Weekend':
        xs['DaysInDayType']['DaysInDayType'][index] = 2

    # If the type of day is every day of the week, the length of the days is 7
    elif dtype == 'All days':
        xs['DaysInDayType']['DaysInDayType'][index] = 7
        
    # If the type of the day is a single day of the week
    elif dtype == 'Single day':
        xs['DaysInDayType']['DaysInDayType'][index] = 1

# Set DAYTPE column as index of the dataframe        
xs['DaysInDayType'] = xs['DaysInDayType'].set_index('DAYTYPE')


# %% Computing timeslice values of timeseries and generating output parameter

# Define an additional column in df_dailytimebracket containing the length of each dailytimebracket
df_dailytimebracket['Length'] = df_dailytimebracket['End hour'].astype(int) - df_dailytimebracket['Start hour'].astype(int)

# Define new columns in df_timeslice for the dailytimebracket of each timeslice and its length
df_timeslice['Dailytimebracket'] = ''
df_timeslice['Dailytimebracket_length'] = ''

# For loop over all timeslices
for index, row in df_timeslice.iterrows():
    
    # Define the dailytimebracket and the length in terms of hours of every timeslice
    df_timeslice['Dailytimebracket'][index] = re.findall('\d+', df_timeslice['TIMESLICE'][index])[2]
    df_timeslice['Dailytimebracket_length'][index] = df_dailytimebracket['Length'][int(df_timeslice['Dailytimebracket'][index])-1]

# For loop over all timeseries to be elaborated
for index, row in df_timeseries.iterrows():
    
    # Define a variable with the timeserie name
    ts = row['Timeseries name']
    
    # Define a new dataframe
    xs[ts] = pd.DataFrame(columns = {'TIMESLICE', 'Value'})
    
    # Assign the timeslices values to the TIMESLICE column
    xs[ts]['TIMESLICE'] = df_timeslice['TIMESLICE']
    
    # If the needed output parameter is a SpecifiedDemandProfile
    if row['Output parameter'] == 'SpecifiedDemandProfile':
        
        # Group the timeserie values by timeslice and sum them
        xs[ts] = xl[ts]['Value'].groupby(xl[ts]['TIMESLICE']).sum().to_frame()
        
        # Sort the timeslice names
        xs[ts] = xs[ts].iloc[natsort.index_humansorted(xs[ts].index)]
        
        # Normalize the values
        xs[ts]['Value'] = xs[ts]['Value'] / xs[ts]['Value'].sum()
        
    # If instead the needed output parameter is a CapacityFactor
    elif row['Output parameter'] == 'CapacityFactor':
        
        # Group the timeserie values by timeslice and compute the mean
        xs[ts] = xl[ts]['Value'].groupby(xl[ts]['TIMESLICE']).mean().to_frame()

        # Sort the timeslice names
        xs[ts] = xs[ts].iloc[natsort.index_humansorted(xs[ts].index)]


# %% Define a summary dataframe with key informations on the elaborated timeseries and the output parameters

# Define the new dataframe
xs['Summary'] = pd.DataFrame(columns={'Name','Parameter','SpecifiedAnnualDemand','Unit'})

# For loop over all the timeseries
for index, row in df_timeseries.iterrows():
    
    # Define a variable with the timeserie name
    ts = row['Timeseries name']
    
    # Assign the value to the Name column
    xs['Summary'].loc[index,'Name'] = ts
    
    # Assign the value to the Output parameter column
    xs['Summary'].loc[index,'Parameter'] = row['Output parameter']
    
    # If the needed output parameter is a SpecifiedDemandProfile
    if row['Output parameter'] == 'SpecifiedDemandProfile':
        
        # Assign the value of SpecifiedAnnualDemand
        xs['Summary'].loc[index,'SpecifiedAnnualDemand'] = sum(xl[ts]['Value'])
        
        # Assign the units of the SpecifiedAnnualDemand
        xs['Summary'].loc[index,'Unit'] = row['Output unit']
        
        # Rename the Value column of the xs[ts] dataframe with SpecifiedDemandProfile
        xs[ts].rename(columns={'Value' : 'SpecifiedDemandProfile'}, inplace=True)
    
    # If instead the needed output parameter is a CapacityFactor
    elif row['Output parameter'] == 'CapacityFactor':
        
        # Neglect the SpecifiedAnnualDemand value
        xs['Summary'].loc[index,'SpecifiedAnnualDemand'] = '-'
        
        # Neglect the SpecifiedAnnualDemand unit
        xs['Summary'].loc[index,'Unit'] = '-'
        
        # Rename the Value column of the xs[ts] dataframe with CapacityFactor
        xs[ts].rename(columns={'Value' : 'CapacityFactor'}, inplace=True)

# Reorder columns
xs['Summary'] = xs['Summary'][['Name','Parameter','SpecifiedAnnualDemand','Unit']]


# %% Writing the output file

# Open the file
writer = pd.ExcelWriter(file_param_output, engine = 'xlsxwriter')

# Write the YearSplit sheet, define columns length and freeze first sheet row
xs['YearSplit'].to_excel(writer, sheet_name = 'YearSplit', index=True)
worksheet = writer.sheets["YearSplit"]
worksheet.set_column('A:B',20)
worksheet.freeze_panes(1,0)

# Write the DaySplit sheet, define columns length and freeze first sheet row
xs['DaySplit'].to_excel(writer, sheet_name = 'DaySplit', index=True)
worksheet = writer.sheets["DaySplit"]
worksheet.set_column('A:B',20)
worksheet.freeze_panes(1,0)

# Write the DaysInDayType sheet, define columns length and freeze first sheet row
xs['DaysInDayType'].to_excel(writer, sheet_name = 'DaysInDayType', index=True)
worksheet = writer.sheets["DaysInDayType"]
worksheet.set_column('A:B',20)
worksheet.freeze_panes(1,0)

# Write the Summary sheet, define columns length and freeze first sheet row
xs['Summary'].to_excel(writer, sheet_name = 'Summary', index=False)
worksheet = writer.sheets["Summary"]
worksheet.set_column('A:A',20)
worksheet.set_column('B:C',25)
worksheet.set_column('D:D',20)
worksheet.freeze_panes(1,0)

# For every timeserie that has been elaborated
for index, row in df_timeseries.iterrows():
    
    # Write the parameter sheet, define columns length and freeze first sheet row   
    ts = row['Timeseries name']
    xs[ts].to_excel(writer, sheet_name = ts, index=True)
    worksheet = writer.sheets[ts]
    worksheet.set_column('B:B',25)

# Close the file
writer.close()


