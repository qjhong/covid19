#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 15:23:07 2020

@author: qijunhong
"""

import json
import requests
from datetime import datetime
from datetime import timedelta
import numpy
import matplotlib.pyplot as plt
import math
import pandas as pd
from scipy.signal import savgol_filter
import os

def get_DNC(state):
    if state == "US":
        response = requests.get("https://covidtracking.com/api/v1/us/daily.json")
    else:
        response = requests.get("https://covidtracking.com/api/v1/states/daily.json")
    todos = json.loads(response.text)

    counter = 0
    dates=[];pos=[];tot=[];dth=[];
    for todo in todos:
        if (state == "US" or todo["state"]==state) and todo["date"]<20210321:
            #print(todo)
            counter = counter + 1
            dates.append(todo["date"])
            pos.append(todo["positive"])
            dth.append(todo["death"])
            #if state == 'OR' and todo["totalTestResults"] is not None:
            #    tot.append(todo["totalTestResults"]);#print todo["negative"]
            #elif 'negative' in todo and todo["negative"] is not None:
            #    tot.append(todo["negative"]);#print todo["negative"]
            #else:
            #    tot.append(0)
            if state == 'OR' and todo["totalTestsViral"] is not None:
                tot.append(todo["totalTestsViral"]);#print todo["negative"]
            elif 'totalTestsViral' in todo and todo["totalTestsViral"] is not None:
                tot.append(todo["totalTestsViral"]);#print todo["negative"]
            elif 'totalTestsPeopleViral' in todo and todo["totalTestsPeopleViral"] is not None:
                tot.append(todo["totalTestsPeopleViral"]);#print todo["negative"]
            elif 'totalTestResults' in todo and todo["totalTestResults"] is not None:
                tot.append(todo["totalTestResults"]);#print todo["negative"]
            else:
                tot.append(0)
            if todo["date"]==20200316: break # get at most 70 data points
            #print( todo["negative"] )

    for i in range(len(dates)):
        date = dates[i]
        dates[i] = datetime(date//10000,date//100%100,date%100)

    #print(pos)
    #print(tot)
    for i in range(len(pos)-2):
        if state == 'MA' and dates[i] == datetime(2020,9,2): pos[i]=0
        pos[i] = pos[i] - pos[i+1]
        tot[i] = tot[i] - tot[i+1] #+ pos[i]
#        dth[i] = dth[i] - dth[i+1]
        if state == 'MA' and dates[i] == datetime(2020,9,2): pos[i]=0
    pos.pop(-1);tot.pop(-1);dth.pop(-1);dates.pop(-1)
    #print(dates)
    #print(pos)
    #print(tot)

    return dates,pos,tot,dth

def get_DNC_JHU(state):
    dict = {"AL": "Alabama",
    "AK": "Alaska",
    "AZ": "Arizona",
    "AR": "Arkansas",
    "CA": "California",
    "CO": "Colorado",
    "CT": "Connecticut",
    "DE": "Delaware",
    "DC": "District of Columbia",
    "FL": "Florida",
    "GA": "Georgia",
    "HI": "Hawaii",
    "ID": "Idaho",
    "IL": "Illinois",
    "IN": "Indiana",
    "IA": "Iowa",
    "KS": "Kansas",
    "KY": "Kentucky",
    "LA": "Louisiana",
    "ME": "Maine",
    "MD": "Maryland",
    "MA": "Massachusetts",
    "MI": "Michigan",
    "MN": "Minnesota",
    "MS": "Mississippi",
    "MO": "Missouri",
    "MT": "Montana",
    "NE": "Nebraska",
    "NV": "Nevada",
    "NH": "New Hampshire",
    "NJ": "New Jersey",
    "NM": "New Mexico",
    "NY": "New York",
    "NC": "North Carolina",
    "ND": "North Dakota",
    "OH": "Ohio",
    "OK": "Oklahoma",
    "OR": "Oregon",
    "PA": "Pennsylvania",
    "RI": "Rhode Island",
    "SC": "South Carolina",
    "SD": "South Dakota",
    "TN": "Tennessee",
    "TX": "Texas",
    "UT": "Utah",
    "VT": "Vermont",
    "VA": "Virginia",
    "WA": "Washington",
    "WV": "West Virginia",
    "WI": "Wisconsin",
    "US": "US",
    "WY": "Wyoming" }
    state = dict[state]
    print(state)
    #if state == "US":
    #    response = requests.get("https://covidtracking.com/api/v1/us/daily.json")
    #else:
    #    response = requests.get("https://covidtracking.com/api/v1/states/daily.json")
    #todos = json.loads(response.text)


    #counter = 0
    #for todo in todos:
    #    if (state == "US" or todo["state"]==state) and todo["date"]<20210321:
    #        #print(todo)
    #        counter = counter + 1
    #        dates.append(todo["date"])
    #        pos.append(todo["positive"])
    #        dth.append(todo["death"])
    #        if state == 'OR' and todo["totalTestResults"] is not None:
    #            tot.append(todo["totalTestResults"]);#print todo["negative"]
    #        elif 'negative' in todo and todo["negative"] is not None:
    #            tot.append(todo["negative"]);#print todo["negative"]
    #        else:
    #            tot.append(0)
    #        if todo["date"]==20200316: break # get at most 70 data points
    #        #print( todo["negative"] )

    dates=[];pos=[];tot=[];dth=[];
    date =datetime(2021,3,8)
    dates_end = date.today()
    if state == 'US':
        while date < dates_end - timedelta(days = 1):
            day = date.day;month=date.month; year = date.year
            if day < 10: day = '0'+str(day)
            day = str(day)
            if month < 10: month = '0' + str(month)
            month = str(month)
            year = str(year)
            #F = open("tmp1","r")
            #for line in F: death= int(line)
            jhu_df = pd.read_csv("../COVID-19/csse_covid_19_data/csse_covid_19_daily_reports_us/"+month+"-"+day+"-"+year+".csv")
            death=jhu_df['Deaths'].sum()
            positive=jhu_df['Confirmed'].sum()
            total=jhu_df['Total_Test_Results'].sum()
            dates.append(date)
            dth.append(death)
            pos.append(positive)
            tot.append(total)
            date = date + timedelta(days = 1)
    else:
        while date < dates_end - timedelta(days = 1):
            day = date.day;month=date.month; year = date.year
            if day < 10: day = '0'+str(day)
            day = str(day)
            if month < 10: month = '0' + str(month)
            month = str(month)
            year = str(year)
            #F = open("tmp1","r")
            #for line in F: death= int(line)
            jhu_df = pd.read_csv("../COVID-19/csse_covid_19_data/csse_covid_19_daily_reports_us/"+month+"-"+day+"-"+year+".csv")
            death=jhu_df.query("Province_State == '" + state + "'")['Deaths'].values[0]
            positive=jhu_df.query("Province_State == '" + state + "'")['Confirmed'].values[0]
            total=jhu_df.query("Province_State == '" + state + "'")['Total_Test_Results'].values[0]
            dates.append(date)
            dth.append(death)
            pos.append(positive)
            tot.append(total)
            date = date + timedelta(days = 1)

    #for i in range(len(dates)):
    #    date = dates[i]
    #    dates[i] = datetime(date//10000,date//100%100,date%100)

    #print(pos)
    for i in range(len(pos)-1):
        if state == 'MA' and dates[i] == datetime(2020,9,2): pos[i]=0
        pos[i] = pos[i+1] - pos[i]
        tot[i] = tot[i+1] - tot[i]
        #if tot[i] < 0: tot[i] = tot[i+1]
        #if math.isnan(tot[i]): tot[i] = tot[i+1]
#        dth[i] = dth[i+1] - dth[i]
        if state == 'MA' and dates[i] == datetime(2020,9,2): pos[i]=0
    for i in range(len(pos)-2,0,-1):
        if pos[i] < 0: pos[i] = pos[i+1]
        if tot[i] < 0: tot[i] = tot[i+1]
        if math.isnan(tot[i]): tot[i] = tot[i+1]
    pos.pop(-1);tot.pop(-1);dth.pop(-1);dates.pop(-1)
    pos.reverse()
    tot.reverse()
    dates.reverse()
    #print(dates)
    #print(pos)

    return dates,pos,tot,dth

def read_states():
    F = open("proj/US_proj_states","r")
    data = [ line.split() for line in F]
    date=[];pos=[];pos_l=[];pos_h=[];
    for i in range(len(data)):
        tmp = data[i]
        date.append(datetime(int(tmp[0]),int(tmp[1]),int(tmp[2])))
        pos.append(float(tmp[3]))
        pos_l.append(float(tmp[4]))
        pos_h.append(float(tmp[5]))
    return date,pos,pos_l,pos_h

def get_SD(state):
    F = open("data/US_AM","r")
    data = [ line.split() for line in F]
    date=[];AM=[]
    for i in range(len(data[0])):
        tmp=data[0][i].split(',')
        date.append(float(tmp[0]))
        ytmp=float(tmp[1])
        AM.append(2-((370-ytmp)*0.1+(ytmp-100)*1.9)/(370-100))
    F = open("data/US_NE","r")
    data = [ line.split() for line in F]
    NE=[]
    for i in range(len(data[0])):
        tmp=data[0][i].split(',')
        ytmp=float(tmp[1])
        NE.append(2-((370-ytmp)*0.1+(ytmp-100)*1.9)/(370-100))
    F = open("data/"+state+"_ED","r")
    data = [ line.split() for line in F]
    F = open("data/"+state+"_ED_coef","r")
    data2 = [ line.split() for line in F];#print(data2)
    ED=[]
    for i in range(len(data[0])):
        tmp=data[0][i].split(',')
        ytmp=float(tmp[1])
        coef = float(data2[0][0])
        ED.append(coef+1-((370-ytmp)*0.0+(ytmp-100)*(1.0+coef))/(370-100))

    return AM,NE,ED

def plot_DNC(dates,pos,tot,state):
    logpos = [0]*len(pos);logtot = [0]*len(pos)
    for i in range(len(pos)-1):
        logpos[i] = numpy.log(pos[i])
        if tot[i]<=0: tot[i] = tot[i-1]
        #print(tot[i])
        logtot[i] = numpy.log(tot[i])

    lfit = 16;
    datefit = [i for i in range(len(pos))]
    
    #print(pos)
    #print(tot)
    cpos=numpy.polyfit(datefit[0:lfit],logpos[0:lfit],1)
    ctot=numpy.polyfit(datefit[0:lfit],logtot[0:lfit],1)
    posfit = [0]*len(pos);totfit = [0]*len(pos)
    for i in range(len(datefit)):
        posfit[i]= numpy.exp(cpos[0]*datefit[i] + cpos[1])
        totfit[i]= numpy.exp(ctot[0]*datefit[i] + ctot[1])
    plt.semilogy(dates[:-2],pos[:-2],'r');plt.semilogy(dates[:-2],tot[:-2],'k')
    plt.semilogy(dates[0:lfit],totfit[0:lfit],':k');plt.semilogy(dates[0:lfit],posfit[0:lfit],':r')
    plt.text(dates[68],min(pos[0:10])*2.0,"Trend: "+str(format(-cpos[0]*100, '.2f'))+"% per day")
    plt.text(dates[68],min(tot[0:10])/1.99,"Trend: "+str(format(-ctot[0]*100, '.2f'))+"% per day")
    plt.xlim([datetime(2020,3,20),dates[0]]);plt.ylim([min(pos[0:70])*0.9,max(tot)*1.1])
    plt.text(dates[45],max(tot)/2,state,fontsize=20)
    plt.xticks([datetime(2020,3,1),datetime(2020,5,1),datetime(2020,7,1),datetime(2020,9,1),datetime(2020,11,1),datetime(2021,1,1),datetime(2021,3,1),datetime(2021,5,1),datetime(2021,7,1)],['2020/3/1','5/1','7/1','9/1','11/1','2021/1/1','3/1','5/1','7/1'])
    plt.xlabel('Date');plt.ylabel('Daily New Cases');plt.grid(which='both')
    plt.legend(['Daily New Cases','Daily Total Tests'])
    plt.tight_layout()
    plt.savefig(state+'_DailyNewCases',dpi=150)

def plot_PR(dates,pos,tot,state):
    pr = [0]*len(pos)
    for i in range(len(pos)-1):
        pr[i] = pos[i]/tot[i]
    plt.plot(dates[:-2],pr[:-2],'r');
    plt.xticks([datetime(2020,4,1),datetime(2020,5,1),datetime(2020,6,1),datetime(2020,7,1),datetime(2020,8,1),datetime(2020,9,1),datetime(2020,10,1),datetime(2020,11,1),datetime(2020,12,1)],['2020/4/1','5/1','6/1','7/1','8/1','9/1','10/1','11/1','12/1'])
    plt.xlabel('Date');plt.ylabel('Positivity Ratio');plt.grid(which='both')
    plt.tight_layout()
    plt.ylim([0,0.3])
    plt.savefig(state+'_PositiveRatio',dpi=150)

def plot_proj_US(dates,pos,posavg,dates_proj,enddate,pos_l,pos_h,date_past,pos_past,pos_l_past,pos_h_past,date_past2,pos_past2,pos_l_past2,pos_h_past2,date_past3,pos_past3,pos_l_past3,pos_h_past3,date_past4,pos_past4,pos_l_past4,pos_h_past4,state):
    for i in range(len(posavg),len(pos)):
        if dates[i]<=dates_proj[-1]: idx = i
    plt.plot(dates[len(posavg):idx],pos[len(posavg):idx],'.r');
    plt.plot(dates[0:len(posavg)],pos[0:len(posavg)],'+k');
    plt.plot(dates[0:len(posavg)],posavg,'k');
    plt.plot(dates[len(posavg):idx],pos_l[len(posavg):idx],'r');
    #plt.plot(date_past,pos_past,'--',color=(1,0.7,0.7));
    #plt.plot(date_past,pos_l_past,'--',color=(1,0.7,0.7));
    plt.plot(dates[idx+1:len(pos)-1],pos[idx+1:len(pos)-1],'.r');
    plt.plot(dates[idx+1:len(pos)-1],pos_l[idx+1:len(pos)-1],'r');
    plt.plot(dates[len(posavg):idx],pos_h[len(posavg):idx],'r');
    plt.plot(dates[idx+1:len(pos)-1],pos_h[idx+1:len(pos)-1],'r');
    #plt.plot(date_past,pos_past,color=(1,0.7,0.7));
    #plt.plot(date_past,pos_h_past,'--',color=(1,0.7,0.7));
    #plt.plot(date_past2,pos_past2,color=(1,0.7,0.7));
    #plt.plot(date_past2,pos_l_past2,'--',color=(1,0.7,0.7));
    #plt.plot(date_past2,pos_h_past2,'--',color=(1,0.7,0.7));
    #plt.plot(date_past3,pos_past3,color=(1,0.7,0.7));
    #plt.plot(date_past3,pos_l_past3,'--',color=(1,0.7,0.7));
    #plt.plot(date_past3,pos_h_past3,'--',color=(1,0.7,0.7));
    #plt.plot(date_past4,pos_past4,color=(1,0.7,0.7));
    #plt.plot(date_past4,pos_l_past4,'--',color=(1,0.7,0.7));
    #plt.plot(date_past4,pos_h_past4,'--',color=(1,0.7,0.7));
    #plt.xticks([datetime(2020,5,1),datetime(2020,6,1),datetime(2020,7,1),datetime(2020,8,1)],['2020/5/1','2020/6/1','2020/7/1','2020/8/1'])
    plt.xticks([datetime(2020,3,1),datetime(2020,5,1),datetime(2020,7,1),datetime(2020,9,1),datetime(2020,11,1),datetime(2021,1,1),datetime(2021,3,1),datetime(2021,5,1),datetime(2021,7,1),datetime(2021,9,1)],['2020/3/1','5/1','7/1','9/1','11/1','2021/1/1','3/1','5/1','7/1','9/1'])
    #plt.xticks([datetime(2020,3,1),datetime(2020,4,1),datetime(2020,5,1),datetime(2020,6,1),datetime(2020,7,1),datetime(2020,8,1)],['2020/3/1','2020/4/1','2020/5/1','2020/6/1','2020/7/1','2020/8/1'])
    plt.xlabel('Date');plt.ylabel('Daily New Cases');plt.grid(which='both')
    #plt.legend(['Projection','Daily New Cases','7-day Average','95% Confidence Interval','Apply latest model to Aug07,','Jul07, Jun07 and May12'])
    plt.legend(['Projection','Daily New Cases','7-day Average','95% Confidence Interval'])#,'Apply latest model to Aug07,','Jul07, Jun07 and May12'])
    #plt.legend(['Projection','Projection (extended)','Daily New Cases','7-day Average','95% Confidence Interval','95% Confidence Interval (incl. Method Error)','Projection Made 2 Weeks Ago'])
    #plt.legend(['Projection','Daily New Cases','7-day Average','95% Confidence Interval (Statistical)','95% Confidence Interval (incl. Method Error)','Projection Made 2 Weeks Ago'])
    plt.text(dates[40],max(pos[:-1])/1.1,state,fontsize=20)
    plt.ylim([0,500001])
    plt.tight_layout()
    plt.savefig(state+'_Projection',dpi=150)

def plot_proj(dates,pos,posavg,dates_proj,enddate,pos_l,pos_h,date_past,pos_past,pos_l_past,pos_h_past,state):
    for i in range(len(posavg),len(pos)):
        if dates[i]<=dates_proj[-1]: idx = i
    plt.plot(dates[len(posavg):idx+1],pos[len(posavg):idx+1],'.r');
    plt.plot(dates[0:len(posavg)],pos[0:len(posavg)],'+k');
    plt.plot(dates[0:len(posavg)],posavg,'k');
#    if state=='US':
#        for i in range(len(posavg),len(pos)): pos_l2[i] = pos[i] - (pos[i]-pos_l[i])
#        for i in range(len(posavg),len(pos)): pos_h2[i] = pos[i] + (pos_h[i]-pos[i])
#    plt.plot(dates[len(posavg):idx+1],pos_l2[len(posavg):idx+1],':r');
    plt.plot(dates[idx+1:len(pos)],pos[idx+1:len(pos)],'.r');
    plt.plot(dates[len(posavg):idx+1],pos_l[len(posavg):idx+1],'r');
    plt.plot(date_past,pos_past,'--',color=(1,0.7,0.7));
    plt.plot(dates[idx+1:len(pos)],pos_l[idx+1:len(pos)],'r');
#    plt.plot(dates[idx+1:len(pos)],pos_l2[idx+1:len(pos)],':m');
    plt.plot(dates[len(posavg):idx+1],pos_h[len(posavg):idx+1],'r');
#    plt.plot(dates[len(posavg):idx+1],pos_h2[len(posavg):idx+1],':r');
    plt.plot(dates[idx+1:len(pos)],pos_h[idx+1:len(pos)],'r');
#    plt.plot(dates[idx+1:len(pos)],pos_h2[idx+1:len(pos)],':m');
    plt.plot(date_past,pos_past,'--',color=(1,0.7,0.7));
    plt.plot(date_past,pos_l_past,'--',color=(1,0.7,0.7));
    plt.plot(date_past,pos_h_past,'--',color=(1,0.7,0.7));
#    for i in range(len(posavg),idx+1):
#        plt.plot([dates[i],dates[i]],[pos_l[i],pos_h[i]],'.r');
#    for i in range(idx+1,len(pos)):
#        plt.plot([dates[i],dates[i]],[pos_l[i],pos_h[i]],'.m');
    plt.xticks([datetime(2020,3,1),datetime(2020,5,1),datetime(2020,7,1),datetime(2020,9,1),datetime(2020,11,1),datetime(2021,1,1),datetime(2021,3,1),datetime(2021,5,1),datetime(2021,7,1),datetime(2021,9,1)],['2020/3/1','5/1','7/1','9/1','11/1','2021/1/1','3/1','5/1','7/1','9/1'])
    #plt.xticks([datetime(2020,4,1),datetime(2020,5,1),datetime(2020,6,1),datetime(2020,7,1),datetime(2020,8,1),datetime(2020,9,1),datetime(2020,10,1),datetime(2020,11,1),datetime(2020,12,1),datetime(2021,1,1),datetime(2021,2,1),datetime(2021,3,1),datetime(2021,4,1)],['2020/4/1','5/1','6/1','7/1','8/1','9/1','10/1','11/1','12/1','2021/1/1','2/1','3/1','4/1'])
    plt.xlabel('Date');plt.ylabel('Daily New Cases');plt.grid(which='both')
    plt.legend(['Projection','Daily New Cases','7-day Average','95% Confidence Interval (Statistical)','Projection Made on June 7'])
    plt.text(dates[15],max(pos)/1.1,state,fontsize=20)
    plt.ylim([0,max(pos)*1.2])
    plt.tight_layout()
    plt.savefig(state+'_Projection',dpi=150)

def plot_R_D(SLP_SD,std,dates,state):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    for i in range(len(SLP_SD)):
        ax2.plot(SLP_SD[i][4],SLP_SD[i][3],'.k')
        #ax2.xticks([datetime(2020,3,1),datetime(2020,4,1),datetime(2020,5,1),datetime(2020,6,1)],['2020/3/1','2020/4/1','2020/5/1','2020/6/1'])
        if SLP_SD[i][4] in dates:
            #print(i,SLP_SD[i][4])
            ax1.plot(SLP_SD[i][4],SLP_SD[i][0]+1,'.r');
            ax1.plot([SLP_SD[i][4],SLP_SD[i][4]],[SLP_SD[i][0]+1+std[i],SLP_SD[i][0]+1-std[i]],'r',linewidth=0.5)#plt.grid(which='both');
            #ax1.xticks([datetime(2020,3,1),datetime(2020,4,1),datetime(2020,5,1),datetime(2020,6,1)],['2020/3/1','2020/4/1','2020/5/1','2020/6/1'])
        #else:
            #ax2.plot(SLP_SD[i][4],SLP_SD[i][0],'+k');#plt.grid(which='both');plt.xticks([datetime(2020,3,1),datetime(2020,4,1),datetime(2020,5,1),datetime(2020,6,1)],['2020/3/1','2020/4/1','2020/5/1','2020/6/1'])
    #plt.xticks([datetime(2020,3,1),datetime(2020,4,1),datetime(2020,5,1),datetime(2020,6,1),datetime(2020,7,1),datetime(2020,8,1),datetime(2020,9,1),datetime(2020,10,1)],['2020/3/1','2020/4/1','2020/5/1','2020/6/1','2020/7/1','2020/8/1','2020/9/1','2020/10/1'])
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Reproductive number R_d, Daily', color='r')
    ax1.tick_params(axis='y', labelcolor='r')
    ax2.set_ylabel('Encounter Density, Adjusted', color='k')
    ax2.tick_params(axis='y', labelcolor='k')
    #ax1.tight_layout()
    fig.savefig(state+'_R_D'+'_0',dpi=150)

def plot_R_D_proj(SLP_SD,predict,dates_pred,nshift,std,dates,state):
    #plt.rcParams['lines.linewidth'] = 0.5
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ymax = 0
    std0 = numpy.mean(std[:10])
    for i in range(len(SLP_SD)):
        ax2.plot(SLP_SD[i][4],SLP_SD[i][3],'.k')
        if SLP_SD[i][3]>ymax: ymax = SLP_SD[i][3]
        if SLP_SD[i][4] in dates:
            ax1.plot(SLP_SD[i][4],SLP_SD[i][0]+1,'.r');#plt.grid(which='both');plt.xticks([datetime(2020,3,1),datetime(2020,4,1),datetime(2020,5,1),datetime(2020,6,1)],['2020/3/1','2020/4/1','2020/5/1','2020/6/1'])
            ax1.plot([SLP_SD[i][4],SLP_SD[i][4]],[SLP_SD[i][0]+1+std[i],SLP_SD[i][0]+1-std[i]],'r',linewidth=0.5)#plt.grid(which='both');plt.xticks([datetime(2020,3,1),datetime(2020,4,1),datetime(2020,5,1),datetime(2020,6,1)],['2020/3/1','2020/4/1','2020/5/1','2020/6/1'])
        else:
            d = dates_pred.index(SLP_SD[i][4])
            ax1.plot(SLP_SD[i][4],predict[d]+1,'+r');
            ax1.plot([SLP_SD[i][4],SLP_SD[i][4]],[predict[d]+1+std0,predict[d]+1-std0],'r',linewidth=0.5)#plt.grid(which='both');plt.xticks([datetime(2020,3,1),datetime(2020,4,1),datetime(2020,5,1),datetime(2020,6,1)],['2020/3/1','2020/4/1','2020/5/1','2020/6/1'])
    #plt.xticks([datetime(2020,3,1),datetime(2020,4,1),datetime(2020,5,1),datetime(2020,6,1),datetime(2020,7,1),datetime(2020,8,1),datetime(2020,9,1),datetime(2020,10,1)],['2020/3/1','2020/4/1','2020/5/1','2020/6/1','2020/7/1','2020/8/1','2020/9/1','2020/10/1'])
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Reproductive Number R_d, Daily', color='r')
    ax1.tick_params(axis='y', labelcolor='r')
    ax2.set_ylabel('Encounter Density, Adjusted', color='k')
    ax2.tick_params(axis='y', labelcolor='k')
    #ax1.tight_layout()
    ax2.text(dates[5],ymax/1.2,state,fontsize=20)
    ax1.legend(['R'],loc=9)
    ax2.legend(['D, shifted '+str(nshift)+' days'],loc=1)
    fig.savefig(state+'_R_D'+'_1',dpi=150)

def plot_regression(x_train,y_train,x_pred,predict,state):
    for i in range(len(x_pred)):  plt.plot(x_pred[i],predict[i]+1,'+r')
    for i in range(len(x_train)): plt.plot(x_train[i],y_train[i]+1,'.k');
    plt.xlabel('Encounter Density, Adjusted')
    plt.ylabel('Reproductive Number R_d, Daily')
    plt.legend(['Regression'])
    plt.savefig(state+'_Regression',dpi=150)
    
def prepare_data(state,nshift,coef_tot,enddate):
    print("Working on the state of "+state)
    [dates0,pos0,tot0,dth0] = get_DNC(state)
    [dates1,pos1,tot1,dth1] = get_DNC_JHU(state)
    dates = dates1 + dates0
    #print(dates)
    pos = pos1 + pos0
    tot = tot1 + tot0
    dth = dth1 + dth0
    #print(pos)
    #print(pos[0])
    # generate pos_fit, tot_fit, remove excessive oscillation
    flagstop = False;navg=0;
    epsilon = 1.e-8
    while not flagstop:
        tot_tmp=[0]*len(tot);flagstop=True
        for i in range(len(tot)): tot_tmp[i] = numpy.mean(tot[max(i-navg,0):min(i+navg+1,len(tot))])
        for i in range(60):
            r2 = tot_tmp[i] / (tot_tmp[i+1] + epsilon)
            if r2>1.4 or r2<0.6:
            #if r2>1.1 or r2<0.9:
                flagstop=False;navg=navg+1;break
    print(navg,r2)
    flagstop = False;navg=0;
    while not flagstop:
        pos_tmp=[0]*len(pos);flagstop=True
        for i in range(len(pos)): pos_tmp[i] = numpy.mean(pos[max(i-navg,0):min(i+navg+1,len(pos))])
        for i in range(60):
            r1 = pos_tmp[i] / ( pos_tmp[i+1] + epsilon )
            if r1>1.5 or r1<0.6:
            #if r1>1.1 or r1<0.9:
                flagstop=False;navg=navg+1;break
    print(navg,r1)
    #print(pos_tmp[0])
    pos_fit=[0]*len(pos);tot_fit=[0]*len(pos);
    for i in range(len(pos)-1): pos_fit[i] = pos_tmp[i];tot_fit[i] = tot_tmp[i];
    #change pos,tot to 3-day average
    for i in range(len(pos)-1):
        if state != 'US': tot_tmp[i] = numpy.mean(tot[max(i-1,0):min(i+2,len(tot))])
    for i in range(len(pos)-1):
        if state != 'US': pos_tmp[i] = numpy.mean(pos[max(i-1,0):min(i+2,len(tot))])
    for i in range(len(pos)-1): pos[i] = pos_tmp[i];tot[i] = tot_tmp[i];
    #print(pos[0])
    
    plot_DNC(dates,pos,tot,state)
    plt.close()
    plot_DNC(dates,pos_fit,tot_fit,state)
    plt.close()
    plot_PR(dates,pos,tot,state)
    plt.close()
    
    posavg = [0]*len(pos);totavg = [0]*len(pos);logpos = [0]*len(pos);logtot = [0]*len(pos);
    datefit = [i for i in range(len(pos))];n_avg = 7; ndays=min(len(pos)-2,450);
    for i in range(len(pos)-1):
        #posavg[i] = numpy.average(pos_fit[max(0,i-n_avg//2):min(i+n_avg//2+1,len(pos))])
        #totavg[i] = numpy.average(tot_fit[max(0,i-n_avg//2):min(i+n_avg//2+1,len(pos))])
        posavg[i] = numpy.average(pos_fit[max(0,i):min(i+n_avg+1,len(pos))])
        totavg[i] = numpy.average(tot_fit[max(0,i):min(i+n_avg+1,len(pos))])
        if posavg[i] < 0: posavg[i] = posavg[i-1]
        if totavg[i] < 0: totavg[i] = totavg[i-1]
        logpos[i] = numpy.log(posavg[i]);logtot[i] = numpy.log(totavg[i]);
    #print(posavg)
    #print(dates)
    slope = [];slope0 = [];slope1 = [];lfit = 10;
    for i in range(ndays):
        cpos=numpy.polyfit(datefit[i:lfit+i],logpos[i:lfit+i],1);
        ctot=numpy.polyfit(datefit[i:lfit+i],logtot[i:lfit+i],1)
        slope.append(-cpos[0]*1+ctot[0]*coef_tot)
        slope0.append(-cpos[0]*1);slope1.append(-ctot[0]*1)
        if i == 0: pos0 = numpy.exp(cpos[1])
    
    slope_avg = [0]*len(slope);n_avg=1
    std = [0]*len(slope);n_std=14;
    for i in range(len(slope)):
        slope_avg[i] = numpy.average(slope[i:min(i+n_avg,len(slope))])
        std[i] = numpy.std(slope[i:min(i+n_std,len(slope))])
    slope_avghat = savgol_filter(slope_avg,61,3,mode='mirror')
    #slope_avghat = slope_avg
    #plt.plot(dates[0:ndays],slope_avghat[0:ndays]+1.,'b');plt.plot(dates[0:ndays],slope_avg[0:ndays]+1.,'m');plt.plot(dates[0:ndays],slope0[0:ndays]+1.,'r');plt.plot(dates[0:ndays],slope1[0:ndays]+1.,'k');plt.grid(which='both');
    #plt.plot(dates[0:ndays],slope_avghat[0:ndays]+1.,'b');plt.plot(dates[0:ndays],slope0[0:ndays]+1.,'r');plt.plot(dates[0:ndays],slope1[0:ndays]+1.,'k');plt.grid(which='both');
    plt.plot(dates[0:ndays],numpy.asarray(slope_avghat[0:ndays])+1.,'r');plt.plot(dates[0:ndays],numpy.asarray(slope0[0:ndays])+1.,'b');plt.plot(dates[0:ndays],numpy.asarray(slope1[0:ndays])+1.,'k');plt.grid(which='both');
    plt.xticks([datetime(2020,3,1),datetime(2020,5,1),datetime(2020,7,1),datetime(2020,9,1),datetime(2020,11,1),datetime(2021,1,1),datetime(2021,3,1),datetime(2021,5,1),datetime(2021,7,1),datetime(2021,9,1)],['2020/3/1','5/1','7/1','9/1','11/1','2021/1/1','3/1','5/1','7/1','9/1'])
    slope_avg = slope_avghat
    plt.legend(['Daily reproductive number (smoothed)', 'Daily reproductive number', 'Testing increase rate'])
    plt.xlabel('Date')
    plt.ylabel('Daily reproductive number')
    plt.savefig(state+'_slope',dpi=150)
    plt.close()
    
    os.chdir("/Users/qijunhong/Work/GitHub/covid19/")
    [AM,NE,ED0] = get_SD(state)# len(dates) == len(pos) # len(dates_SD) == len(AM)
    dn = datetime(2020,10,1);day = timedelta(days = 1);dates_SD = [];l=len(ED0)
    for i in range(l): dates_SD.append(dn-day*(l-i-1))
    AM_avg=[0]*l;NE_avg=[0]*l;ED_avg=[0]*l;ED0_avg=[0]*l;ED=[0]*l;n_avg = 7;
    for i in range(len(ED0)): #ED[i] = ED0[i]
        if dates_SD[i] == datetime(2020,4,3): iref = i
        #if dates_SD[i] == datetime(2020,5,1): iref = i
        if dates_SD[i] > datetime(2020,4,3): ED[i] = ED0[i] * min( max( 1 - (i-iref)*0.015, 0.66 ), 1 )
        #elif dates_SD[i] > datetime(2020,5,1): ED[i] = ED0[i] * min( max( 0.5 + (i-iref)*0.015, 0.5), 0.9 )
        else: ED[i] = ED0[i]
    for i in range(len(ED0)):
        #AM_avg[i] = numpy.mean(AM[max(0,i-n_avg//2):min(i+n_avg//2+1,len(ED))])
        #NE_avg[i] = numpy.mean(NE[max(0,i-n_avg//2):min(i+n_avg//2+1,len(ED))])
        ED0_avg[i] = numpy.mean(ED0[max(0,i-n_avg//2):min(i+n_avg//2+1,len(ED))])
        ED_avg[i] = numpy.mean(ED[max(0,i-n_avg//2):min(i+n_avg//2+1,len(ED))])
    
    plt.plot(dates_SD,AM_avg,'k');plt.plot(dates_SD,NE_avg,'b');plt.plot(dates_SD,ED_avg,'g');plt.plot(dates_SD,ED0_avg,'r');plt.grid(which='both');plt.xticks([datetime(2020,3,1),datetime(2020,4,1),datetime(2020,5,1),datetime(2020,6,1),datetime(2020,7,1),datetime(2020,8,1)],['2020/3/1','2020/4/1','2020/5/1','2020/6/1','2020/7/1','2020/8/1']);
    plt.close()
    plt.plot(dates_SD,ED0,'k');plt.plot(dates_SD,ED0_avg,'r');plt.grid(which='both');plt.xticks([datetime(2020,3,1),datetime(2020,4,1),datetime(2020,5,1),datetime(2020,6,1),datetime(2020,7,1),datetime(2020,8,1)],['2020/3/1','2020/4/1','2020/5/1','2020/6/1','2020/7/1','2020/8/1']);plt.xlabel('Date');plt.ylabel('Encounter Density');plt.legend(['Encounter Density','7-day Average'])
    if state == 'US': plt.savefig(state+'_ED',dpi=150)
    
    #%%
    data0 = pd.read_csv('data/2020_US_Region_Mobility_Report.csv')
    data1 = pd.read_csv('data/2021_US_Region_Mobility_Report.csv')
    data = pd.concat([data0,data1],axis=0).reset_index()
    if state != 'US':
        data_dates = data.loc[data['iso_3166_2_code']=='US-'+state]['date']
        data_mobil = data.loc[data['iso_3166_2_code']=='US-'+state]['residential_percent_change_from_baseline']
    else:
        data_tmp = data.loc[data['iso_3166_2_code']=='US-AL'].reset_index()
        len_data = data.loc[data['iso_3166_2_code']=='US-AL']['date'].shape[0]
        #print(len_data)
        data_dates = data_tmp.iloc[0:len_data]['date']
        data_mobil = data_tmp.iloc[0:len_data]['residential_percent_change_from_baseline']
    mobil_history = [0.]*7
    mobil_mean_7day = 0.
    mobil_min = -100;
    nshift = 25;
    if state=="AZ": nshift=23;
    if state=="CA": nshift=25;
    #print(data_dates)
    for dday in range(nshift,nshift+1):
        SLP_SD = [];p1=len(slope_avg)-1;x=[];y=[]#dday = 22;
        #print(p1)
        #while dates[p1]-day*dday not in dates_SD: p1 = p1+1
        #p2 = dates_SD.index(dates[p1]-day*dday)
        ht_date = {}
        idx = 0
        while p1 >= 0:
            date = dates[p1]-day*dday
            year = str(date.year)
            month = '0'+str(date.month) if date.month<10 else str(date.month)
            dayday = '0'+str(date.day) if date.day<10 else str(date.day)
            #print p1,p2
            #SLP_SD.append([slope_avg[p1], AM[p2],NE[p2],ED[p2], dates[p1]])
            #SLP_SD.append([slope_avg[p1], AM_avg[p2],NE_avg[p2],ED_avg[p2], dates[p1]])
            idx_data = data_dates.loc[data_dates==year+'-'+month+'-'+dayday].index
            #print(data_dates)
            #print(idx_data)
            #print(year,month,dayday)
            #print(data_mobil[idx_data].values[0])
            mobil_mean_7day = ( mobil_mean_7day*7. + data_mobil[idx_data].values[0] - mobil_history[0] ) / 7.
            mobil_min = max(mobil_min,mobil_mean_7day)
            mobil_history.pop(0)
            mobil_history.append(data_mobil[idx_data].values[0])
            #print(year,month,dayday,idx_data,data_mobil[idx_data].values[0],mobil_mean_7day,mobil_min,mobil_history)
            if date > datetime(2020,4,15):
                mobil_adj = 100. - mobil_min + (mobil_min-mobil_mean_7day)*0.5
            else:
                mobil_adj = 100. - mobil_mean_7day
            if p1+dday < len(posavg):
                SLP_SD.append([slope_avg[p1], mobil_adj, mobil_adj, mobil_adj, dates[p1], posavg[p1+dday] ])
            else:
                SLP_SD.append([slope_avg[p1], mobil_adj, mobil_adj, mobil_adj, dates[p1], posavg[-1] ])
            ht_date[dates[p1]] = idx
            idx += 1
            #print dates[p1],dates_SD[p2],slope_avg[p1],ED_avg[p2]
            if math.isnan(slope_avg[p1]): break
            x.append(slope_avg[p1])
            #y.append(ED_avg[p2])
            y.append(mobil_mean_7day)
            #print(slope_avg[p1],ED_avg[p2],p2,dates[p1])
            p1 = p1 - 1;#p2 = p2 - 1
            #if p2 == 0 or dates[p1]==datetime(2020,3,5): break
            if dates[p1]==datetime(2020,3,5): break
        for i in range(len(data_dates)):
            date = data_dates.iloc[i]
            year = int(date[0:4]);month=int(date[5:7]);dayday=int(date[8:10])
            date = datetime(year,month,dayday)+day*dday
            if date not in dates and date > dates[-1]:
                date0 = date
                date = date-day*dday
                year = str(date.year)
                month = '0'+str(date.month) if date.month<10 else str(date.month)
                dayday = '0'+str(date.day) if date.day<10 else str(date.day)
                #print p1,p2
                #SLP_SD.append([slope_avg[p1], AM[p2],NE[p2],ED[p2], dates[p1]])
                #SLP_SD.append([slope_avg[p1], AM_avg[p2],NE_avg[p2],ED_avg[p2], dates[p1]])
                #idx_data = data_dates.loc[data_dates==year+'-'+month+'-'+dayday].index
                mobil_mean_7day = ( mobil_mean_7day*7. + data_mobil.iloc[i] - mobil_history[0] ) / 7. 
                #print(data_mobil.iloc[i] , mobil_history[0], mobil_mean_7day)
                mobil_history.pop(0)
                mobil_history.append(data_mobil.iloc[i])
                if date > datetime(2020,5,10):
                    mobil_adj = 100. - mobil_min + (mobil_min-mobil_mean_7day)*0.5
                else:
                    mobil_adj = 100. - mobil_mean_7day
                SLP_SD.append([0., mobil_adj, mobil_adj, mobil_adj, date0, 0.])
                #print(year,month,dayday,idx_data,data_mobil.iloc[i],mobil_mean_7day,mobil_min,mobil_history)
        date = date0
        while date <= enddate:
            date += day
            SLP_SD.append([0, mobil_adj, mobil_adj, mobil_adj, date])
        print(dday,numpy.corrcoef(x[-20:-1],y[-20:-1]))
    #print(SLP_SD)
    x_train = [];y_train = []; weight = []; x_pred = [];dates_pred = []
    val_unknown = -100000.;
    awareness = 1.;
    slope0 = 1.;
    flag_weather = 0.;
    for i in range(len(SLP_SD)):
        if SLP_SD[i][4] in dates:
            date0 = SLP_SD[i][4]
            if not math.isnan(SLP_SD[i][0]):
                x_tmp = [SLP_SD[i][3]]
                date1 = date0 - day*60
                if date1 in ht_date:
                    x_tmp.append(SLP_SD[ht_date[date1]][0])
                else:
                    #x_tmp.append(val_unknown)
                    x_tmp.append(SLP_SD[0][0])
                date1 = date0 - day*30
                if date1 in ht_date:
                    x_tmp.append(SLP_SD[ht_date[date1]][0])
                else:
                    #x_tmp.append(val_unknown)
                    x_tmp.append(SLP_SD[0][0])
                date1 = date0 - day*90
                if date1 in ht_date:
                    x_tmp.append(SLP_SD[ht_date[date1]][0])
                else:
                    #x_tmp.append(val_unknown)
                    x_tmp.append(SLP_SD[0][0])
                if i-nshift>=0:
                    if slope0 < 0. and y_train[i-nshift] > 0.:
                        awareness = 1.
                    if slope0 > 0. and y_train[i-nshift] < 0.:
                        counter = 14
                    slope0 = y_train[i-nshift]
                    if y_train[i-nshift] > 0.:
                        awareness *= 1+y_train[i-nshift]+0.001
                    elif y_train[i-nshift] < 0. and counter > 0:
                        awareness *= 1+y_train[i-nshift]+0.001
                        counter -= 1
                    else:
                        awareness = 1.
                else:
                    awareness = 1.
                x_tmp.append(SLP_SD[i][5])
                x_tmp.append(min(100., awareness))
                #x_tmp.append( abs( (date0-datetime(2020,9,15)) / timedelta(days=1) ) * 0.)
                if date0 > datetime(2020,8,25):
                    x_tmp.append( numpy.sin( (date0-datetime(2020,10,25)) / timedelta(days=1) / 60. * 3.14 / 2 ) * flag_weather)
                    #x_tmp.append( ( (date0-datetime(2020,10,25)) / timedelta(days=1) / 60. / 2. ) * 1.)
                else:
                    x_tmp.append( -numpy.sin( (date0-datetime(2020,4,25)) / timedelta(days=1) / 120. * 3.14 / 2 ) * flag_weather)
                    #x_tmp.append( -( (date0-datetime(2020,4,25)) / timedelta(days=1) / 120. / 2. ) * 1.)
    #           date1 = date0 - day*1
    #           if date1 in ht_date:
    #               x_tmp.append(SLP_SD[ht_date[date1]][0])
    #           else:
    #               x_tmp.append(val_unknown)
    #           date2 = date0 - day*2
    #           if date1 in ht_date and date2 in ht_date:
    #               x_tmp.append(SLP_SD[ht_date[date1]][0]-SLP_SD[ht_date[date2]][0])
    #           else:
    #               x_tmp.append(val_unknown)    plt.plot(dates_SD,ED0,'k');plt.plot(dates_SD,ED0_avg,'r');plt.grid(which='both');plt.xticks([datetime(2020,3,1),datetime(2020,4,1),datetime(2020,5,1),datetime(2020,6,1),datetime(2020,7,1),datetime(2020,8,1)],['2020/3/1','2020/4/1','2020/5/1','2020/6/1','2020/7/1','2020/8/1']);plt.xlabel('Date');plt.ylabel('Encounter Density');plt.legend(['Encounter Density','7-day Average'])

                if ( date0 < datetime(2021,1,1) ):
                    vac_effect = 0
                else:
                    vac_effect = ( date0 - datetime(2021,1,1) ) / timedelta(days=1)
                x_tmp.append(vac_effect)
                x_tmp.append(state)
                x_train.append(x_tmp)
                y_train.append(SLP_SD[i][0])
                weight.append(numpy.power(1.03,i))
                x_pred_est = SLP_SD[i][5]
        else:
            dates_pred.append(SLP_SD[i][4])
            x_tmp = [SLP_SD[i][3]]
            date0 = SLP_SD[i][4]
            date1 = date0 - day*60
            if date1 in ht_date:
                x_tmp.append(SLP_SD[ht_date[date1]][0])
            else:
                x_tmp.append(x_pred[-1][1])
            date1 = date0 - day*30
            if date1 in ht_date:
                x_tmp.append(SLP_SD[ht_date[date1]][0])
            else:
                x_tmp.append(x_pred[-1][2])
            date1 = date0 - day*90
            if date1 in ht_date:
                x_tmp.append(SLP_SD[ht_date[date1]][0])
            else:
                x_tmp.append(x_pred[-1][3])
            if i-nshift>=0 and i-nshift < len(y_train):
                if y_train[i-nshift] > -0.01:
                    awareness *= 1+y_train[i-nshift]+0.001
                    x_pred_est *= 1+y_train[i-nshift]+0.001
                else:
                    awareness = 1.
            elif i-nshift >= len(y_train):
                if y_train[-1] > -0.01:
                    #awareness *= 1+max(y_train[-1],0.005)
                    awareness *= 1+y_train[-1]+0.001
                    x_pred_est *= 1+y_train[-1]+0.001
                else:
                    awareness = 1.                
            else:
                awareness = 1.
            x_tmp.append(x_pred_est)
            x_tmp.append(awareness)
            #x_tmp.append( abs( (date0-datetime(2020,9,15)) / timedelta(days=1) ) * 0.)
            if date0 > datetime(2020,8,25):
                x_tmp.append( numpy.sin( (date0-datetime(2020,10,25)) / timedelta(days=1) / 60. * 3.14 / 2 ) * flag_weather)
                #x_tmp.append( ( (date0-datetime(2020,10,25)) / timedelta(days=1) / 60. / 2. ) * 1.)
            else:
                x_tmp.append( -numpy.sin( (date0-datetime(2020,4,25)) / timedelta(days=1) / 120. * 3.14 / 2 ) * flag_weather)
                #x_tmp.append( -( (date0-datetime(2020,4,25)) / timedelta(days=1) / 120. / 2. ) * 1.)
    #       date1 = date0 - day*1
    #       if date1 in ht_date:
    #           x_tmp.append(SLP_SD[ht_date[date1]][0])
    #       else:
    #           x_tmp.append(val_unknown)
    #       date2 = date0 - day*2
    #       if date1 in ht_date and date2 in ht_date:
    #           x_tmp.append(SLP_SD[ht_date[date1]][0]-SLP_SD[ht_date[date2]][0])
    #       else:
    #           x_tmp.append(val_unknown)
            if ( date0 < datetime(2021,1,1) ):
                vac_effect = 0
            else:
                vac_effect = ( date0 - datetime(2021,1,1) ) / timedelta(days=1)
            x_tmp.append(vac_effect)
            x_tmp.append(state)
            x_pred.append(x_tmp)
    plot_R_D(SLP_SD,std[::-1],dates,state)
    plt.close()
    plt.close()
    
    for i in range(len(x_train)):
        print(x_train[i],y_train[i])
    for x in x_pred:
        print(x)
    return SLP_SD, x_train, y_train, weight, std, x_pred, dates, dates_pred, slope1, pos, pos0, posavg
