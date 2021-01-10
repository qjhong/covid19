import json
import requests
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
import matplotlib.dates as mdates
from datetime import datetime
from datetime import timedelta
import numpy
import math
import sys
import os
import pandas as pd
tmp = sys.argv
#state = "US"

def get_DSC(state):
    if state == "US":
        response = requests.get("https://covidtracking.com/api/v1/us/daily.json")
    else:
        response = requests.get("https://covidtracking.com/api/v1/states/daily.json")
    todos = json.loads(response.text)

    counter,nbreak = 0,85
    dates=[];pos=[];tot=[];dth=[];
    for todo in todos:
        if (state == "US" or todo["state"]==state):# and todo["date"]<20200707:
            counter = counter + 1
            dates.append(todo["date"])
            pos.append(todo["positive"])
            dth.append(todo["death"])
            if 'negative' in todo and todo["negative"] is not None:
                tot.append(todo["negative"]);#print todo["negative"]
            else:
                tot.append(0)
            if todo["date"]==20200317: break # get at most 70 data points

    date0 = dates[0]
    for i in range(len(dates)):
        date = dates[i]
        dates[i] = datetime(date//10000,date//100%100,date%100)

    for i in range(len(pos)-1):
        pos[i] = pos[i] - pos[i+1]
        tot[i] = tot[i] - tot[i+1] + pos[i]
#        dth[i] = dth[i] - dth[i+1]
    pos.pop(-1);tot.pop(-1);dth.pop(-1);dates.pop(-1)

    return dates,pos,tot,dth


# get pos, tot, date
enddate = datetime(2020,8,13);
[dates,pos,tot,dth] = get_DSC("GA")
dth_GA = [0]*(len(dth)-1)
for i in range(len(dth)-1): dth_GA[i] = dth[i]-dth[i+1]
[dates,pos,tot,dth] = get_DSC("SC")
dth_SC = [0]*(len(dth)-1)
for i in range(len(dth)-1): dth_SC[i] = dth[i]-dth[i+1]
[dates,pos,tot,dth] = get_DSC("AL")
dth_AL = [0]*(len(dth)-1)
for i in range(len(dth)-1): dth_AL[i] = dth[i]-dth[i+1]
[dates,pos,tot,dth] = get_DSC("TN")
for i in range(5): dth.pop(-1)
dth_TN = [0]*(len(dth)-1)
for i in range(len(dth)-1): dth_TN[i] = dth[i]-dth[i+1]
l = min(len(dth_GA),len(dth_SC),len(dth_AL),len(dth_TN))
dth_sum = []
for i in range(l): dth_sum.append(dth_GA[i]+dth_SC[i]+dth_AL[i]+dth_TN[i])
dth_sum_avg = []
for i in range(l-7): dth_sum_avg.append(numpy.mean(dth_sum[i:i+7]))
if True:
    plt.plot(dates[:len(dth_sum_avg)],dth_sum_avg,'r')
    #plt.semilogy(dates[:len(dth_sum_avg)],dth_sum_avg,'r')
    plt.xticks([datetime(2020,4,1),datetime(2020,5,1),datetime(2020,6,1),datetime(2020,7,1)],['2020/4/1','2020/5/1','2020/6/1','2020/7/1'])
    plt.xlabel('Date');plt.ylabel('Daily Deaths');plt.grid(which='both')
    #plt.xlim([datetime(2020,04,20),datesJHU[-1]+timedelta(days = 1)])
    #plt.ylim([0,3500])
    plt.legend(['GA+SC+AL+TN'])
    plt.tight_layout()
    plt.savefig('US_Death_4states_next',dpi=150)
plt.close()


datesJHU = [];dth_FL=[];dth_TX=[];dth_CA=[];dth_AZ=[];
date =datetime(2020,4,12)
while date < dates[0]:
    day = date.day;month=date.month
    if day < 10: day = '0'+str(day)
    day = str(day)
    if month < 10: month = '0' + str(month)
    month = str(month)
    jhu_df = pd.read_csv("../COVID-19/csse_covid_19_data/csse_covid_19_daily_reports_us/"+month+"-"+day+"-2020.csv")
    #os.system("curl -s https://github.com/CSSEGISandData/COVID-19/blob/master/csse_covid_19_data/csse_covid_19_daily_reports_us/"+month+"-"+day+"-2020.csv > tmp");
    #os.system("grep -A7 data-line-number tmp | grep -A6 'Flor' | tail -1 | cut -d'>' -f2 | cut -d'<' -f1 > tmp1")
    death = jhu_df.loc[jhu_df['Province_State']=='Georgia']['Deaths']
    #F = open("tmp1","r")
    #for line in F: death= int(line)
    datesJHU.append(date)
    dth_FL.append(int(death))
    death = jhu_df.loc[jhu_df['Province_State']=='South Carolina']['Deaths']
    #os.system("grep -A7 data-line-number tmp | grep -A6 'Texas' | tail -1 | cut -d'>' -f2 | cut -d'<' -f1 > tmp1")
    #F = open("tmp1","r")
    #for line in F: death= int(line)
    dth_TX.append(int(death))
    death = jhu_df.loc[jhu_df['Province_State']=='Alabama']['Deaths']
    #os.system("grep -A7 data-line-number tmp | grep -A6 'Califor' | tail -1 | cut -d'>' -f2 | cut -d'<' -f1 > tmp1")
    #F = open("tmp1","r")
    #for line in F: death= int(line)
    dth_CA.append(int(death))
    death = jhu_df.loc[jhu_df['Province_State']=='Tennessee']['Deaths']
    #os.system("grep -A7 data-line-number tmp | grep -A6 'Arizona' | tail -1 | cut -d'>' -f2 | cut -d'<' -f1 > tmp1")
    #F = open("tmp1","r")
    #for line in F: death= int(line)
    dth_AZ.append(int(death))
    date = date + timedelta(days = 1)

for i in range(len(dth_FL)-1): dth_FL[i] = dth_FL[i+1]-dth_FL[i]
for i in range(len(dth_TX)-1): dth_TX[i] = dth_TX[i+1]-dth_TX[i]
for i in range(len(dth_CA)-1): dth_CA[i] = dth_CA[i+1]-dth_CA[i]
for i in range(len(dth_AZ)-1): dth_AZ[i] = dth_AZ[i+1]-dth_AZ[i]
l = min(len(dth_FL),len(dth_TX),len(dth_AZ),len(dth_CA))-1
dth_sum = []
for i in range(l): dth_sum.append(dth_FL[i]+dth_TX[i]+dth_AZ[i]+dth_CA[i])
dth_sum_avg = []
for i in range(l): dth_sum_avg.append(numpy.mean(dth_sum[max(0,i-7):i]))
if True:
    plt.plot(datesJHU[:len(dth_sum_avg)],dth_sum_avg,'r')
    plt.semilogy(datesJHU[:len(dth_sum_avg)],dth_sum_avg,'r')
    #plt.semilogy(dates[:len(dth_sum_avg)],dth_sum_avg,'r')
    plt.xticks([datetime(2020,4,1),datetime(2020,5,1),datetime(2020,6,1),datetime(2020,7,1)],['2020/4/1','2020/5/1','2020/6/1','2020/7/1'])
    plt.xlabel('Date');plt.ylabel('Daily Deaths');plt.grid(which='both')
    #plt.xlim([datetime(2020,04,20),datesJHU[-1]+timedelta(days = 1)])
    #plt.ylim([0,3500])
    plt.legend(['GA+SC+AL+TN'])
    plt.tight_layout()
    plt.savefig('US_Death_4states_next_JHU',dpi=150)
plt.close()
