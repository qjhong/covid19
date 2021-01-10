import json
import requests
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
import matplotlib.dates as mdates
from datetime import datetime
from datetime import timedelta
import numpy
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
import math
import sys
from scipy.signal import savgol_filter
import os

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
            if todo["date"]==20200331: break # get at most 70 data points

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

states = ['AK', 'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA', 'HI', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', 'ME', 'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM', 'NV', 'NY', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VA', 'VT', 'WA', 'WI', 'WV', 'WY'] 
for i in range(len(states)):
    [dates,pos,tot,dth] = get_DSC(states[i])
    dth_tmp = [0]*(len(dth)-1)
    pos_tmp = [0]*(len(dth)-1)
    for j in range(len(dth)-1): 
        if dates[j]==datetime(2020,6,25) and states[i]=='NJ': dth[j] -= 1800
        dth_tmp[j] = dth[j]-dth[j+1]
        pos_tmp[j] = pos[j]
    dth_tmp_avg = []
    pos_tmp_avg = []
    for j in range(len(dth)-7): dth_tmp_avg.append(numpy.mean(dth_tmp[j:j+7]))
    for j in range(len(dth)-7): pos_tmp_avg.append(numpy.mean(pos_tmp[j:j+7]))
    if i==0:
        dth_increase = [0]*len(dth_tmp)
        dth_decrease = [0]*len(dth_tmp)
        dth_increase_avg = [0]*len(dth_tmp_avg)
        dth_decrease_avg = [0]*len(dth_tmp_avg)
        pos_increase = [0]*len(pos_tmp)
        pos_decrease = [0]*len(pos_tmp)
        pos_increase_avg = [0]*len(pos_tmp_avg)
        pos_decrease_avg = [0]*len(pos_tmp_avg)
    if dth_tmp_avg[0] > dth_tmp_avg[30]:
    #if pos_tmp_avg[0] > pos_tmp_avg[30]*1.0:
       for j in range(len(dth_tmp_avg)): dth_increase_avg[j] += dth_tmp_avg[j]
       for j in range(len(dth_tmp)): dth_increase[j] += dth_tmp[j]
       for j in range(len(pos_tmp_avg)): pos_increase_avg[j] += pos_tmp_avg[j]
       for j in range(len(pos_tmp)): pos_increase[j] += pos_tmp[j]
    else:
       for j in range(len(dth_tmp_avg)): dth_decrease_avg[j] += dth_tmp_avg[j]
       for j in range(len(dth_tmp)): dth_decrease[j] += dth_tmp[j]
       for j in range(len(pos_tmp_avg)): pos_decrease_avg[j] += pos_tmp_avg[j]
       for j in range(len(pos_tmp)): pos_decrease[j] += pos_tmp[j]

    
dth_total_avg = [0]*len(dth_tmp_avg)
dth_total = [0]*len(dth_tmp)
for i in range(len(dth_tmp_avg)): dth_total_avg[i] = dth_increase_avg[i] + dth_decrease_avg[i]
for i in range(len(dth_tmp)): dth_total[i] = dth_increase[i] + dth_decrease[i]
pos_total_avg = [0]*len(pos_tmp_avg)
pos_total = [0]*len(pos_tmp)
for i in range(len(pos_tmp_avg)): pos_total_avg[i] = pos_increase_avg[i] + pos_decrease_avg[i]
for i in range(len(pos_tmp)): pos_total[i] = pos_increase[i] + pos_decrease[i]

#plt.plot(dates[:len(dth_total)],dth_total,'k+')
#plt.plot(dates[:len(dth_total_avg)],dth_total_avg,'k')
plt.fill_between(dates[:len(dth_total_avg)],dth_total_avg,color=[0,0,0.4])
#plt.plot(dates[:len(dth_total)],dth_increase,'+r')
#plt.plot(dates[:len(dth_total_avg)],dth_increase_avg,'r')
plt.fill_between(dates[:len(dth_total_avg)],dth_increase_avg,color=[0.6,0,0])
plt.xticks([datetime(2020,4,1),datetime(2020,5,1),datetime(2020,6,1),datetime(2020,7,1)],['2020/4/1','2020/5/1','2020/6/1','2020/7/1'])
plt.xlabel('Date');plt.ylabel('Daily Deaths');plt.grid(which='both')
#plt.xlim([datetime(2020,04,20),datesJHU[-1]+timedelta(days = 1)])
#plt.ylim([0,3500])
plt.legend(['States where deaths are decreasing','States where deaths are increasing'])
plt.tight_layout()
plt.savefig('US_Death_Analysis',dpi=150)
plt.close()

#plt.plot(dates[:len(pos_total)],pos_total,'k+')
#plt.plot(dates[:len(pos_total_avg)],pos_total_avg,'k')
plt.fill_between(dates[:len(pos_total_avg)],pos_total_avg,color=[0,0,0.4])
#plt.plot(dates[:len(pos_total)],pos_increase,'+r')
#plt.plot(dates[:len(pos_total_avg)],pos_increase_avg,'r')
plt.fill_between(dates[:len(pos_total_avg)],pos_increase_avg,color=[0.6,0,0])
plt.xticks([datetime(2020,4,1),datetime(2020,5,1),datetime(2020,6,1),datetime(2020,7,1)],['2020/4/1','2020/5/1','2020/6/1','2020/7/1'])
plt.xlabel('Date');plt.ylabel('Daily New Cases');plt.grid(which='both')
#plt.xlim([datetime(2020,04,20),datesJHU[-1]+timedelta(days = 1)])
#plt.ylim([0,3500])
plt.legend(['States where cases are decreasing','States where cases are increasing'],loc=9)
plt.tight_layout()
plt.savefig('US_Death_Analysis_pos',dpi=150)
plt.close()
