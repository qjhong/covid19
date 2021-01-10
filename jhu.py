import pandas as pd
import numpy as np
from pandas import *
from datetime import datetime
from datetime import timedelta

os.getcwd()

jhu_df = pd.read_csv('../COVID-19/csse_covid_19_data/csse_covid_19_daily_reports_us/07-29-2020.csv')

jhu_df.head()

a = jhu_df.loc[jhu_df['Province_State']=='Florida']


jhu_df = pd.read_csv('../COVID-19/csse_covid_19_data/csse_covid_19_daily_reports_us/07-28-2020.csv')

b = jhu_df.loc[jhu_df['Province_State']=='Florida']

a
b
record = a.append(b)

record.iloc[:,[2,6]]

jhu_df['Deaths'].sum()

a = jhu_df.loc[jhu_df['Province_State']=='Florida']['Deaths']
 int(a) + 1
a

datetime.today()
date =datetime(2020,4,12)
while date < datetime.today():#-timedelta(days=1):
    day = date.day;month=date.month
    if day < 10: day = '0'+str(day)
    day = str(day)
    if month < 10: month = '0' + str(month)
    month = str(month)
    if Flagg:
        jhu_df = pd.read_csv("../COVID-19/csse_covid_19_data/csse_covid_19_daily_reports_us/"+month+"-"+day+"-2020.csv")
    else:
        jhu_df.append(pd.read_csv("../COVID-19/csse_covid_19_data/csse_covid_19_daily_reports_us/"+month+"-"+day+"-2020.csv")
    date = date + timedelta(days = 1)
jhu_df.loc[jhu_df['Province_State']=='Florida']
len(jhu_df)
