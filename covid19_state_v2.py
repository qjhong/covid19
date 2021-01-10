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
import pandas as pd

tmp = sys.argv
state = tmp[1]
#state = "US"

def get_DNC(state):
    if state == "US":
        response = requests.get("https://covidtracking.com/api/v1/us/daily.json")
    else:
        response = requests.get("https://covidtracking.com/api/v1/states/daily.json")
    todos = json.loads(response.text)

    counter,nbreak = 0,125
    dates=[];pos=[];tot=[];dth=[];
    for todo in todos:
        if (state == "US" or todo["state"]==state) and todo["date"]<20200918:
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
    data2 = [ line.split() for line in F];print(data2)
    ED=[]
    for i in range(len(data[0])):
        tmp=data[0][i].split(',')
        ytmp=float(tmp[1])
        coef = float(data2[0][0])
        ED.append(coef+1-((370-ytmp)*0.0+(ytmp-100)*(1.0+coef))/(370-100))

    return AM,NE,ED

def plot_DNC(dates,pos,tot):
    logpos = [0]*len(pos);logtot = [0]*len(pos)
    for i in range(len(pos)-1):
        logpos[i] = numpy.log(pos[i])
        if tot[i]<0: tot[i] = tot[i-1]
        logtot[i] = numpy.log(tot[i])

    lfit = 16;
    datefit = [i for i in range(len(pos))]
    cpos=numpy.polyfit(datefit[0:lfit],logpos[0:lfit],1)
    ctot=numpy.polyfit(datefit[0:lfit],logtot[0:lfit],1)
    posfit = [0]*len(pos);totfit = [0]*len(pos)
    for i in range(len(datefit)):
        posfit[i]= numpy.exp(cpos[0]*datefit[i] + cpos[1])
        totfit[i]= numpy.exp(ctot[0]*datefit[i] + ctot[1])
    plt.semilogy(dates[:-2],pos[:-2],'r');plt.semilogy(dates[:-2],tot[:-2],'k')
    plt.semilogy(dates[0:lfit],totfit[0:lfit],':k');plt.semilogy(dates[0:lfit],posfit[0:lfit],':r')
    plt.text(dates[28],min(pos[0:10])*1.2,"Trend: "+str(format(-cpos[0]*100, '.2f'))+"% per day")
    plt.text(dates[28],min(tot[0:10])/1.8,"Trend: "+str(format(-ctot[0]*100, '.2f'))+"% per day")
    plt.xlim([datetime(2020,3,20),dates[0]]);plt.ylim([min(pos[0:70])*0.9,max(tot)*1.1])
    plt.text(dates[45],max(tot)/2,state,fontsize=20)
    plt.xticks([datetime(2020,4,1),datetime(2020,5,1),datetime(2020,6,1),datetime(2020,7,1),datetime(2020,8,1),datetime(2020,9,1)],['2020/4/1','2020/5/1','2020/6/1','2020/7/1','2020/8/1','2020/9/1'])
    plt.xlabel('Date');plt.ylabel('Daily New Cases');plt.grid(which='both')
    plt.legend(['Daily New Cases','Daily Total Tests'])
    plt.tight_layout()
    plt.savefig(state+'_DailyNewCases',dpi=150)

def plot_PR(dates,pos,tot):
    pr = [0]*len(pos)
    for i in range(len(pos)-1):
        pr[i] = pos[i]/tot[i]
    plt.plot(dates[:-2],pr[:-2],'r');
    plt.xticks([datetime(2020,4,1),datetime(2020,5,1),datetime(2020,6,1),datetime(2020,7,1),datetime(2020,8,1),datetime(2020,9,1)],['2020/4/1','2020/5/1','2020/6/1','2020/7/1','2020/8/1','2020/9/1'])
    plt.xlabel('Date');plt.ylabel('Positivity Ratio');plt.grid(which='both')
    plt.tight_layout()
    plt.ylim([0,0.3])
    plt.savefig(state+'_PositiveRatio',dpi=150)

def plot_proj_US(dates,pos,posavg,dates_proj,enddate,pos_l,pos_h,date_past,pos_past,pos_l_past,pos_h_past,date_past2,pos_past2,pos_l_past2,pos_h_past2,date_past3,pos_past3,pos_l_past3,pos_h_past3,date_past4,pos_past4,pos_l_past4,pos_h_past4,state):
    pos_l2 = [pos_l[i] for i in range(len(pos_l))]
    pos_h2 = [pos_l[i] for i in range(len(pos_h))]
    for i in range(len(posavg),len(pos)):
        if dates[i]<=dates_proj[-1]: idx = i
    plt.plot(dates[len(posavg):idx+1],pos[len(posavg):idx+1],'.r');
    plt.plot(dates[idx+1:len(pos)],pos[idx+1:len(pos)],'.m');
    plt.plot(dates[0:len(posavg)],pos[0:len(posavg)],'+k');
    plt.plot(dates[0:len(posavg)],posavg,'k');
    plt.plot(dates[len(posavg):idx+1],pos_l[len(posavg):idx+1],'r');
    plt.plot(date_past,pos_past,'--',color=(1,0.7,0.7));
    plt.plot(date_past,pos_l_past,'--',color=(1,0.7,0.7));
    plt.plot(dates[idx+1:len(pos)],pos_l[idx+1:len(pos)],'m');
    plt.plot(dates[len(posavg):idx+1],pos_h[len(posavg):idx+1],'r');
    plt.plot(dates[idx+1:len(pos)],pos_h[idx+1:len(pos)],'m');
    plt.plot(date_past,pos_past,color=(1,0.7,0.7));
    plt.plot(date_past,pos_h_past,'--',color=(1,0.7,0.7));
    plt.plot(date_past2,pos_past2,color=(1,0.7,0.7));
    plt.plot(date_past2,pos_l_past2,'--',color=(1,0.7,0.7));
    plt.plot(date_past2,pos_h_past2,'--',color=(1,0.7,0.7));
    plt.plot(date_past3,pos_past3,color=(1,0.7,0.7));
    plt.plot(date_past3,pos_l_past3,'--',color=(1,0.7,0.7));
    plt.plot(date_past3,pos_h_past3,'--',color=(1,0.7,0.7));
    plt.plot(date_past4,pos_past4,color=(1,0.7,0.7));
    plt.plot(date_past4,pos_l_past4,'--',color=(1,0.7,0.7));
    plt.plot(date_past4,pos_h_past4,'--',color=(1,0.7,0.7));
    #plt.xticks([datetime(2020,5,1),datetime(2020,6,1),datetime(2020,7,1),datetime(2020,8,1)],['2020/5/1','2020/6/1','2020/7/1','2020/8/1'])
    plt.xticks([datetime(2020,3,1),datetime(2020,4,1),datetime(2020,5,1),datetime(2020,6,1),datetime(2020,7,1),datetime(2020,8,1),datetime(2020,9,1),datetime(2020,10,1)],['2020/3/1','2020/4/1','2020/5/1','2020/6/1','2020/7/1','2020/8/1','2020/9/1','2020/10/1'])
    #plt.xticks([datetime(2020,3,1),datetime(2020,4,1),datetime(2020,5,1),datetime(2020,6,1),datetime(2020,7,1),datetime(2020,8,1)],['2020/3/1','2020/4/1','2020/5/1','2020/6/1','2020/7/1','2020/8/1'])
    plt.xlabel('Date');plt.ylabel('Daily New Cases');plt.grid(which='both')
    plt.legend(['Projection','Projection (extended)','Daily New Cases','7-day Average','95% Confidence Interval (Statistical)','Apply latest model to Aug07, Jul07,',' Jun07 and May12'])
    #plt.legend(['Projection','Projection (extended)','Daily New Cases','7-day Average','95% Confidence Interval (Statistical)','95% Confidence Interval (incl. Method Error)','Projection Made 2 Weeks Ago'])
    #plt.legend(['Projection','Daily New Cases','7-day Average','95% Confidence Interval (Statistical)','95% Confidence Interval (incl. Method Error)','Projection Made 2 Weeks Ago'])
    plt.text(dates[40],max(pos)/1.1,state,fontsize=20)
    plt.ylim([0,90001])
    plt.tight_layout()
    plt.savefig(state+'_Projection',dpi=150)

def plot_proj(dates,pos,posavg,dates_proj,enddate,pos_l,pos_h,date_past,pos_past,pos_l_past,pos_h_past,state):
    pos_l2 = [pos_l[i] for i in range(len(pos_l))]
    pos_h2 = [pos_l[i] for i in range(len(pos_h))]
    for i in range(len(posavg),len(pos)):
        if dates[i]<=dates_proj[-1]: idx = i
    plt.plot(dates[len(posavg):idx+1],pos[len(posavg):idx+1],'.r');
    plt.plot(dates[idx+1:len(pos)],pos[idx+1:len(pos)],'.m');
    plt.plot(dates[0:len(posavg)],pos[0:len(posavg)],'+k');
    plt.plot(dates[0:len(posavg)],posavg,'k');
#    if state=='US':
#        for i in range(len(posavg),len(pos)): pos_l2[i] = pos[i] - (pos[i]-pos_l[i])
#        for i in range(len(posavg),len(pos)): pos_h2[i] = pos[i] + (pos_h[i]-pos[i])
#    plt.plot(dates[len(posavg):idx+1],pos_l2[len(posavg):idx+1],':r');
    plt.plot(dates[len(posavg):idx+1],pos_l[len(posavg):idx+1],'r');
    plt.plot(date_past,pos_past,'--',color=(1,0.7,0.7));
    plt.plot(dates[idx+1:len(pos)],pos_l[idx+1:len(pos)],'m');
#    plt.plot(dates[idx+1:len(pos)],pos_l2[idx+1:len(pos)],':m');
    plt.plot(dates[len(posavg):idx+1],pos_h[len(posavg):idx+1],'r');
#    plt.plot(dates[len(posavg):idx+1],pos_h2[len(posavg):idx+1],':r');
    plt.plot(dates[idx+1:len(pos)],pos_h[idx+1:len(pos)],'m');
#    plt.plot(dates[idx+1:len(pos)],pos_h2[idx+1:len(pos)],':m');
    plt.plot(date_past,pos_past,'--',color=(1,0.7,0.7));
    plt.plot(date_past,pos_l_past,'--',color=(1,0.7,0.7));
    plt.plot(date_past,pos_h_past,'--',color=(1,0.7,0.7));
#    for i in range(len(posavg),idx+1):
#        plt.plot([dates[i],dates[i]],[pos_l[i],pos_h[i]],'.r');
#    for i in range(idx+1,len(pos)):
#        plt.plot([dates[i],dates[i]],[pos_l[i],pos_h[i]],'.m');
    plt.xticks([datetime(2020,3,1),datetime(2020,4,1),datetime(2020,5,1),datetime(2020,6,1),datetime(2020,7,1),datetime(2020,8,1),datetime(2020,9,1),datetime(2020,10,1)],['2020/3/1','2020/4/1','2020/5/1','2020/6/1','2020/7/1','2020/8/1','2020/9/1','2020/10/1'])
    #plt.xticks([datetime(2020,3,1),datetime(2020,4,1),datetime(2020,5,1),datetime(2020,6,1),datetime(2020,7,1)],['2020/3/1','2020/4/1','2020/5/1','2020/6/1','2020/7/1'])
    plt.xlabel('Date');plt.ylabel('Daily New Cases');plt.grid(which='both')
    plt.legend(['Projection','Projection (extended)','Daily New Cases','7-day Average','95% Confidence Interval (Statistical)','Projection Made on June 7'])
    #plt.legend(['Projection','Projection (extended)','Daily New Cases','7-day Average','95% Confidence Interval (Statistical)','95% Confidence Interval (incl. Method Error)','Projection Made 2 Weeks Ago'])
    #plt.legend(['Projection','Daily New Cases','7-day Average','95% Confidence Interval (Statistical)','95% Confidence Interval (incl. Method Error)','Projection Made 2 Weeks Ago'])
    plt.text(dates[15],max(pos)/1.1,state,fontsize=20)
    plt.ylim([0,max(pos)*1.2])
    plt.tight_layout()
    plt.savefig(state+'_Projection',dpi=150)

def plot_R_D(SLP_SD,std):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    for i in range(len(SLP_SD)):
        ax2.plot(SLP_SD[i][4],SLP_SD[i][3],'.k')
        #ax2.xticks([datetime(2020,3,1),datetime(2020,4,1),datetime(2020,5,1),datetime(2020,6,1)],['2020/3/1','2020/4/1','2020/5/1','2020/6/1'])
        if SLP_SD[i][4] in dates:
            ax1.plot(SLP_SD[i][4],SLP_SD[i][0]+1,'.r');
            ax1.plot([SLP_SD[i][4],SLP_SD[i][4]],[SLP_SD[i][0]+1+std[i],SLP_SD[i][0]+1-std[i]],'r')#plt.grid(which='both');
            #ax1.xticks([datetime(2020,3,1),datetime(2020,4,1),datetime(2020,5,1),datetime(2020,6,1)],['2020/3/1','2020/4/1','2020/5/1','2020/6/1'])
        #else:
            #ax2.plot(SLP_SD[i][4],SLP_SD[i][0],'+k');#plt.grid(which='both');plt.xticks([datetime(2020,3,1),datetime(2020,4,1),datetime(2020,5,1),datetime(2020,6,1)],['2020/3/1','2020/4/1','2020/5/1','2020/6/1'])
    plt.xticks([datetime(2020,3,1),datetime(2020,4,1),datetime(2020,5,1),datetime(2020,6,1),datetime(2020,7,1),datetime(2020,8,1),datetime(2020,9,1),datetime(2020,10,1)],['2020/3/1','2020/4/1','2020/5/1','2020/6/1','2020/7/1','2020/8/1','2020/9/1','2020/10/1'])
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Reproductive number R_d, Daily', color='r')
    ax1.tick_params(axis='y', labelcolor='r')
    ax2.set_ylabel('Encounter Density, Adjusted', color='k')
    ax2.tick_params(axis='y', labelcolor='k')
    #ax1.tight_layout()
    fig.savefig(state+'_R_D'+'_0',dpi=150)

def plot_R_D_proj(SLP_SD,predict,dates_pred,nshift,std):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ymax = 0
    std0 = numpy.mean(std[:10])
    for i in range(len(SLP_SD)):
        ax2.plot(SLP_SD[i][4],SLP_SD[i][3],'.k')
        if SLP_SD[i][3]>ymax: ymax = SLP_SD[i][3]
        if SLP_SD[i][4] in dates:
            ax1.plot(SLP_SD[i][4],SLP_SD[i][0]+1,'.r');#plt.grid(which='both');plt.xticks([datetime(2020,3,1),datetime(2020,4,1),datetime(2020,5,1),datetime(2020,6,1)],['2020/3/1','2020/4/1','2020/5/1','2020/6/1'])
            #ax1.plot([SLP_SD[i][4],SLP_SD[i][4]],[SLP_SD[i][0]+1+std[i],SLP_SD[i][0]+1-std[i]],'r')#plt.grid(which='both');plt.xticks([datetime(2020,3,1),datetime(2020,4,1),datetime(2020,5,1),datetime(2020,6,1)],['2020/3/1','2020/4/1','2020/5/1','2020/6/1'])
        else:
            d = dates_pred.index(SLP_SD[i][4])
            ax1.plot(SLP_SD[i][4],predict[d]+1,'+r');
            ax1.plot([SLP_SD[i][4],SLP_SD[i][4]],[predict[d]+1+std0,predict[d]+1-std0],'r')#plt.grid(which='both');plt.xticks([datetime(2020,3,1),datetime(2020,4,1),datetime(2020,5,1),datetime(2020,6,1)],['2020/3/1','2020/4/1','2020/5/1','2020/6/1'])
    plt.xticks([datetime(2020,3,1),datetime(2020,4,1),datetime(2020,5,1),datetime(2020,6,1),datetime(2020,7,1),datetime(2020,8,1),datetime(2020,9,1),datetime(2020,10,1)],['2020/3/1','2020/4/1','2020/5/1','2020/6/1','2020/7/1','2020/8/1','2020/9/1','2020/10/1'])
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

def plot_regression(x_train,y_train,x_pred,predict):
    for i in range(len(x_pred)):  plt.plot(x_pred[i],predict[i]+1,'+r')
    for i in range(len(x_train)): plt.plot(x_train[i],y_train[i]+1,'.k');
    plt.xlabel('Encounter Density, Adjusted')
    plt.ylabel('Reproductive Number R_d, Daily')
    plt.legend(['Regression'])
    plt.savefig(state+'_Regression',dpi=150)

# get pos, tot, date
enddate = datetime(2020,10,15);
[dates,pos,tot,dth] = get_DNC(state)
# generate pos_fit, tot_fit, remove excessive oscillation
flagstop = False;navg=0;
while not flagstop:
    tot_tmp=[0]*len(tot);flagstop=True
    for i in range(len(tot)): tot_tmp[i] = numpy.mean(tot[max(i-navg,0):min(i+navg+1,len(tot))])
    for i in range(60):
        r2=tot_tmp[i]/tot_tmp[i+1]
        if r2>1.4 or r2<0.6:
        #if r2>1.1 or r2<0.9:
            flagstop=False;navg=navg+1;break
print(navg,r2)
flagstop = False;navg=0;
while not flagstop:
    pos_tmp=[0]*len(pos);flagstop=True
    for i in range(len(pos)): pos_tmp[i] = numpy.mean(pos[max(i-navg,0):min(i+navg+1,len(pos))])
    for i in range(60):
        r1=pos_tmp[i]/pos_tmp[i+1]
        if r1>1.5 or r1<0.6:
        #if r1>1.1 or r1<0.9:
            flagstop=False;navg=navg+1;break
print(navg,r1)
pos_fit=[0]*len(pos);tot_fit=[0]*len(pos);
for i in range(len(pos)-1): pos_fit[i] = pos_tmp[i];tot_fit[i] = tot_tmp[i];
#change pos,tot to 3-day average
for i in range(len(pos)-1):
    if state != 'US': tot_tmp[i] = numpy.mean(tot[max(i-1,0):min(i+2,len(tot))])
for i in range(len(pos)-1):
    if state != 'US': pos_tmp[i] = numpy.mean(pos[max(i-1,0):min(i+2,len(tot))])
for i in range(len(pos)-1): pos[i] = pos_tmp[i];tot[i] = tot_tmp[i];

plot_DNC(dates,pos,tot)
plt.close()
plot_DNC(dates,pos_fit,tot_fit)
plt.close()
plot_PR(dates,pos,tot)
#plt.savefig('US_positivity_ratio',dpi=150)


posavg = [0]*len(pos);totavg = [0]*len(pos);logpos = [0]*len(pos);logtot = [0]*len(pos);
datefit = [i for i in range(len(pos))];n_avg = 7; ndays=min(len(pos)-2,125);
for i in range(len(pos)):
    posavg[i] = numpy.average(pos_fit[max(0,i-n_avg//2):min(i+n_avg//2+1,len(pos))])
    totavg[i] = numpy.average(tot_fit[max(0,i-n_avg//2):min(i+n_avg//2+1,len(pos))])
    logpos[i] = numpy.log(posavg[i]);logtot[i] = numpy.log(totavg[i]);
slope = [];slope0 = [];slope1 = [];lfit = 10;coef_tot = 0.25;
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
slope_avghat = savgol_filter(slope_avg,31,3,mode='mirror')
#slope_avghat = savgol_filter(slope_avg,31,3)
plt.plot(dates[0:ndays],slope_avghat[0:ndays],'b');plt.plot(dates[0:ndays],slope_avg[0:ndays],'m');plt.plot(dates[0:ndays],slope0[0:ndays],'r');plt.plot(dates[0:ndays],slope1[0:ndays],'k');plt.grid(which='both');plt.xticks([datetime(2020,3,1),datetime(2020,4,1),datetime(2020,5,1),datetime(2020,6,1),datetime(2020,7,1),datetime(2020,8,1)],['2020/3/1','2020/4/1','2020/5/1','2020/6/1','2020/7/1','2020/8/1']);
slope_avg = slope_avghat
plt.close()
#plt.plot(dates[0:ndays],std[0:ndays],'b')

os. chdir("/Users/qijunhong/Work/GitHub/covid19/")
[AM,NE,ED0] = get_SD(state)# len(dates) == len(pos) # len(dates_SD) == len(AM)
dn = datetime(2020,9,2);day = timedelta(days = 1);dates_SD = [];l=len(ED0)
for i in range(l): dates_SD.append(dn-day*(l-i-1))
AM_avg=[0]*l;NE_avg=[0]*l;ED_avg=[0]*l;ED0_avg=[0]*l;ED=[0]*l;n_avg = 7;
for i in range(len(ED0)): #ED[i] = ED0[i]
    if dates_SD[i] == datetime(2020,4,3): iref = i
    #if dates_SD[i] == datetime(2020,5,1): iref = i
    if dates_SD[i] > datetime(2020,4,3): ED[i] = ED0[i] * min( max( 1 - (i-iref)*0.015, 0.66 ), 1 )
    #elif dates_SD[i] > datetime(2020,5,1): ED[i] = ED0[i] * min( max( 0.5 + (i-iref)*0.015, 0.5), 0.9 )
    else: ED[i] = ED0[i]
for i in range(len(ED0)):
    AM_avg[i] = numpy.average(AM[max(0,i-n_avg//2):min(i+n_avg//2+1,len(ED))])
    NE_avg[i] = numpy.average(NE[max(0,i-n_avg//2):min(i+n_avg//2+1,len(ED))])
    ED0_avg[i] = numpy.average(ED0[max(0,i-n_avg//2):min(i+n_avg//2+1,len(ED))])
    ED_avg[i] = numpy.average(ED[max(0,i-n_avg//2):min(i+n_avg//2+1,len(ED))])
#while dates_SD[-1] < enddate:
#    dates_SD.append(dates_SD[-1]+day)
#    AM_avg.append(max(AM_avg[-1]*0.9,0.1*AM_avg[0]))
#    NE_avg.append(max(NE_avg[-1]*0.9,0.1*NE_avg[0]))
#    ED0_avg.append(max(ED0_avg[-1]*0.9,0.1*ED0_avg[0]))
#    ED_avg.append(max(ED_avg[-1]*0.9,0.1*ED_avg[0]))

plt.plot(dates_SD,AM_avg,'k');plt.plot(dates_SD,NE_avg,'b');plt.plot(dates_SD,ED_avg,'g');plt.plot(dates_SD,ED0_avg,'r');plt.grid(which='both');plt.xticks([datetime(2020,3,1),datetime(2020,4,1),datetime(2020,5,1),datetime(2020,6,1),datetime(2020,7,1),datetime(2020,8,1)],['2020/3/1','2020/4/1','2020/5/1','2020/6/1','2020/7/1','2020/8/1']);
plt.close()
plt.plot(dates_SD,ED0,'k');plt.plot(dates_SD,ED0_avg,'r');plt.grid(which='both');plt.xticks([datetime(2020,3,1),datetime(2020,4,1),datetime(2020,5,1),datetime(2020,6,1),datetime(2020,7,1),datetime(2020,8,1)],['2020/3/1','2020/4/1','2020/5/1','2020/6/1','2020/7/1','2020/8/1']);plt.xlabel('Date');plt.ylabel('Encounter Density');plt.legend(['Encounter Density','7-day Average'])
if state == 'US': plt.savefig(state+'_ED',dpi=150)

nshift = 25;
#if state=="LA" or state=="OK": nshift=20;
#if state=="NY" or state=="MI": nshift=21;
#if state=="UT" or state=="US": nshift=22;
if state=="AZ": nshift=23;
if state=="CA": nshift=25;
for dday in range(nshift,nshift+1):
    SLP_SD = [];p1=0;x=[];y=[]#dday = 22;
    while dates[p1]-day*dday not in dates_SD: p1 = p1+1
    p2 = dates_SD.index(dates[p1]-day*dday)
    print(dates[p1])
    while p1 < len(slope_avg):
        #print p1,p2
        #SLP_SD.append([slope_avg[p1], AM[p2],NE[p2],ED[p2], dates[p1]])
        SLP_SD.append([slope_avg[p1], AM_avg[p2],NE_avg[p2],ED_avg[p2], dates[p1]])
        #print dates[p1],dates_SD[p2],slope_avg[p1],ED_avg[p2]
        if math.isnan(slope_avg[p1]): break
        x.append(slope_avg[p1])
        y.append(ED_avg[p2])
        #print(slope_avg[p1],ED_avg[p2],p2,dates[p1])
        p1 = p1 + 1;p2 = p2 - 1
        if p2 == 0 or dates[p1]==datetime(2020,4,5): break
    for i in range(len(ED0_avg)):
        if dates_SD[i]+day*dday not in dates:
            SLP_SD.append([0, AM_avg[i],NE_avg[i],ED_avg[i],dates_SD[i]+day*dday])
    print(dday,numpy.corrcoef(x[-20:-1],y[-20:-1]))
x_train = [];y_train = []; weight = []; x_pred = [];dates_pred = []
for i in range(len(SLP_SD)):
    if SLP_SD[i][4] in dates:
        if not math.isnan(SLP_SD[i][0]):
            x_train.append([SLP_SD[i][3]])
            y_train.append(SLP_SD[i][0])
            weight.append(numpy.power(0.995,i))
    else:
        dates_pred.append(SLP_SD[i][4])
        x_pred.append([SLP_SD[i][3]])
plot_R_D(SLP_SD,std)
plt.close()
plt.close()

bdt = LinearRegression()
bdt.fit(x_train, y_train, weight);
y_pred = bdt.predict(x_train)
err = [y_pred[i]-y_train[i] for i in range(len(y_train))]
#err = [y_pred[i]-y_train[i] for i in range(0,30)]
print(bdt.score(x_train,y_train))
predict = bdt.predict(x_pred)
if state == 'AZ':
    for i in range(len(predict)):
        predict[i] = (predict[i]+0.01)/2.0

plot_regression(x_train,y_train,x_pred,predict)
plt.close()

std0=numpy.mean(std[:10])
stderr = numpy.linalg.norm(err)/numpy.sqrt(len(y_train))
print(stderr,std0,numpy.mean(err),numpy.linalg.norm(err)/numpy.sqrt(len(y_train)))
plot_R_D_proj(SLP_SD,predict,dates_pred,nshift,std)
plt.close()

err2 = min(0,y_pred[0] - y_train[0])
print(y_train[0],y_pred[0])
print(stderr,std0,err2)
std0 = numpy.sqrt(stderr*stderr+std0*std0+err2*err2)
print(std0)
#adj_tot = numpy.average(slope1[0:7])*coef_tot - max(0,y_pred[0] - y_train[0]) - max(0,(y_train[14]-y_train[0])/4)
adj_tot = numpy.average(slope1[0:7])*coef_tot - max(0,(y_pred[0] - y_train[0])*0.99) - max(0,(y_train[14]-y_train[0])/4)
if y_train[0] > 0: adj_tot -= max(0,(y_train[14]-y_train[0])/4)
#adj_tot = numpy.average(slope1[0:7])*coef_tot
pos_l = [pos[i] for i in range(len(pos))]
pos_h = [pos[i] for i in range(len(pos))]
print(adj_tot)
flag_first = True;f = 1.0; f=2.0
for i in range(len(dates_pred)):
    if dates_pred[i] not in dates:
        if flag_first:
            flag_first = False
            pos_now = pos0 * (predict[i]+adj_tot+1)
            pos_now_l = pos0 * (predict[i]+adj_tot+1-std0*f)
            pos_now_h = pos0 * (predict[i]+adj_tot+1+std0*f)
        else:
            pos_now = pos[dates.index(dates_pred[i]-day)] * (predict[i]+adj_tot+1)
            pos_now_l = pos_l[dates.index(dates_pred[i]-day)] * (predict[i]+adj_tot+1-std0*f)
            pos_now_h = pos_h[dates.index(dates_pred[i]-day)] * (predict[i]+adj_tot+1+std0*f)
        dates.append(dates_pred[i])
        pos.append(pos_now)
        pos_l.append(pos_now_l)
        pos_h.append(pos_now_h)

dates_add = dates_pred[-1] + day
while dates_add <= enddate:
    dates.append(dates_add)
    pos_now = pos[-1] * (predict[-1]+adj_tot+1)
    pos_now_l = pos_l[-1] * (predict[-1]+adj_tot+1-std0*f)
    pos_now_h = pos_h[-1] * (predict[-1]+adj_tot+1+std0*f)
#   if dates_add > datetime(2020,7,16):
#       pos_now = pos[-1] * 0.97
#       pos_now_l = pos_l[-1] * 0.97
#       pos_now_h = pos_h[-1] * 0.97
    pos.append(pos_now)
    pos_l.append(pos_now_l)
    pos_h.append(pos_now_h)
    dates_add = dates_add + day

len(dates)
len(pos)
F = open("proj_Jun07/"+state+"_proj","r")
data = [ line.split() for line in F]
date_past=[];pos_past=[];pos_l_past=[];pos_h_past=[];
for i in range(len(data)):
    tmp = data[i]
    date_past.append(datetime(int(tmp[0]),int(tmp[1]),int(tmp[2])))
    pos_past.append(float(tmp[3]))
    pos_l_past.append(float(tmp[4]))
    pos_h_past.append(float(tmp[5]))
    if state == "US":
        pos_l_past[-1] = pos_past[-1] - pos_l_past[-1]
        pos_h_past[-1] = pos_past[-1] + pos_h_past[-1]
plot_proj(dates,pos,posavg,dates_pred,enddate,pos_l,pos_h,date_past,pos_past,pos_l_past,pos_h_past,state)
if state == 'US': plt.savefig(state+'_Projection0',dpi=150)
plt.close()
F = open(state+"_proj","w")
for i in range(len(posavg),len(pos)):
    F.write(str(dates[i].year)+' '+str(dates[i].month)+' '+str(dates[i].day)+' ');
    F.write(str(pos[i])+' '+str(pos_l[i])+' '+str(pos_h[i])+'\n')

if state=="US":
    F = open("proj_Aug07/"+state+"_proj","r")
    data = [ line.split() for line in F]
    date_past4=[];pos_past4=[];pos_l_past4=[];pos_h_past4=[];
    for i in range(len(data)):
        tmp = data[i]
        date_past4.append(datetime(int(tmp[0]),int(tmp[1]),int(tmp[2])))
        pos_past4.append(float(tmp[3]))
        pos_l_past4.append(float(tmp[4]))
        pos_h_past4.append(float(tmp[5]))
        if state == "US":
            pos_l_past4[-1] = pos_past4[-1] - pos_l_past4[-1]
            pos_h_past4[-1] = pos_past4[-1] + pos_h_past4[-1]
    F = open("proj_Jul06/"+state+"_proj","r")
    data = [ line.split() for line in F]
    date_past=[];pos_past=[];pos_l_past=[];pos_h_past=[];
    for i in range(len(data)):
        tmp = data[i]
        date_past.append(datetime(int(tmp[0]),int(tmp[1]),int(tmp[2])))
        pos_past.append(float(tmp[3]))
        pos_l_past.append(float(tmp[4]))
        pos_h_past.append(float(tmp[5]))
        if state == "US":
            pos_l_past[-1] = pos_past[-1] - pos_l_past[-1]
            pos_h_past[-1] = pos_past[-1] + pos_h_past[-1]
    F = open("proj_Jun07/"+state+"_proj","r")
    data = [ line.split() for line in F]
    date_past2=[];pos_past2=[];pos_l_past2=[];pos_h_past2=[];
    for i in range(len(data)):
        tmp = data[i]
        date_past2.append(datetime(int(tmp[0]),int(tmp[1]),int(tmp[2])))
        pos_past2.append(float(tmp[3]))
        pos_l_past2.append(float(tmp[4]))
        pos_h_past2.append(float(tmp[5]))
        if state == "US":
            pos_l_past2[-1] = pos_past2[-1] - pos_l_past2[-1]
            pos_h_past2[-1] = pos_past2[-1] + pos_h_past2[-1]
    F = open("proj_May12/"+state+"_proj","r")
    data = [ line.split() for line in F]
    date_past3=[];pos_past3=[];pos_l_past3=[];pos_h_past3=[];
    for i in range(len(data)):
        tmp = data[i]
        date_past3.append(datetime(int(tmp[0]),int(tmp[1]),int(tmp[2])))
        pos_past3.append(float(tmp[3]))
        pos_l_past3.append(float(tmp[4]))
        pos_h_past3.append(float(tmp[5]))
        if state == "US":
            pos_l_past3[-1] = pos_past3[-1] - pos_l_past3[-1]
            pos_h_past3[-1] = pos_past3[-1] + pos_h_past3[-1]
    [dates_state,pos_state,pos_state_l,pos_state_h] = read_states()
    for p1 in range(len(dates_state)):
        for i in range(len(dates)):
            if dates[i]==dates_state[p1]: break
        pos[i]=pos_state[p1]
        #if p1==0:
        #    pos[i]=pos0*0.99
        #else:
        #    pos[i]=pos[i-1]*0.99
        pos_l[i]=pos_state[p1]-pos_state_l[p1]
        pos_h[i]=pos_state[p1]+pos_state_h[p1]
    plot_proj(dates,pos,posavg,dates_pred,enddate,pos_l,pos_h,date_past,pos_past,pos_l_past,pos_h_past,state)
    plt.close()
    plot_proj_US(dates,pos,posavg,dates_pred,enddate,pos_l,pos_h,date_past,pos_past,pos_l_past,pos_h_past,date_past2,pos_past2,pos_l_past2,pos_h_past2,date_past3,pos_past3,pos_l_past3,pos_h_past3,date_past4,pos_past4,pos_l_past4,pos_h_past4,state)
    plt.close()

    datesJHU = [];dth=[];dth_l=[];dth_h=[]
    date =datetime(2020,4,12)
    while date <= dates[0]:
        day = date.day;month=date.month
        if day < 10: day = '0'+str(day)
        day = str(day)
        if month < 10: month = '0' + str(month)
        month = str(month)
        #os.system("curl -s https://github.com/CSSEGISandData/COVID-19/blob/master/csse_covid_19_data/csse_covid_19_daily_reports_us/"+month+"-"+day+"-2020.csv > tmp");
        #os.system("grep -A7 data-line-number tmp | grep -B1 '\-\-' | grep -v '\-\-' | grep -v Confirm | cut -d'>' -f2 | cut -d'<' -f1 | awk '{s+=$1} END {print s}' > tmp1")
        #F = open("tmp1","r")
        #for line in F: death= int(line)
        jhu_df = pd.read_csv("../COVID-19/csse_covid_19_data/csse_covid_19_daily_reports_us/"+month+"-"+day+"-2020.csv")
        death=jhu_df['Deaths'].sum()
        datesJHU.append(date)
        dth.append(death)
        dth_l.append(death)
        dth_h.append(death)
        date = date + timedelta(days = 1)
    for i in range(len(datesJHU)):
        if datesJHU[i] > datetime(2020,6,24): dth[i] -= 1800
        if datesJHU[i] > datetime(2020,6,29): dth[i] -= 600
        if datesJHU[i] > datetime(2020,7,26): dth[i] -= 675
        if datesJHU[i] > datetime(2020,8,5): dth[i] -= 500

    print(datesJHU[-1],dth[-1])
    pos_poisson=[]
    for i in range(len(datesJHU)):
        j=0;sum_weight=0;sum_pos=0;decay=-0.075;
        while datesJHU[i]-j*timedelta(days=1) in dates:
            idx=dates.index(datesJHU[i]-j*timedelta(days=1))
            sum_pos += numpy.exp(decay*j)*pos[idx]
            sum_weight += numpy.exp(decay*j)
            j += 1
        #pos_poisson.append(sum_pos/sum_weight)
        pos_poisson.append(sum_pos)
    plt.plot(datesJHU,pos_poisson,'.r');plt.grid(which='both');plt.xticks([datetime(2020,3,1),datetime(2020,4,1),datetime(2020,5,1),datetime(2020,6,1),datetime(2020,7,1),datetime(2020,8,1)],['2020/3/1','2020/4/1','2020/5/1','2020/6/1','2020/7/1','2020/8/1']);
    plt.savefig('US_Death_accum_poisson',dpi=150)
    pos_poisson_long=[]
    for i in range(len(dates)):
        j=0;sum_weight=0;sum_pos=0;
        while dates[i]-j*timedelta(days=1) in dates:
            idx=dates.index(dates[i]-j*timedelta(days=1))
            sum_pos += numpy.exp(decay*j)*pos[idx]
            sum_weight += numpy.exp(decay*j)
            j += 1
        if sum_weight>0:
            #pos_poisson_long.append(sum_pos/sum_weight)
            pos_poisson_long.append(sum_pos)
        else:
            pos_poisson_long.append(pos_poisson_long[-1])
    plt.plot(dates,pos_poisson_long,'.r');plt.grid(which='both');plt.xticks([datetime(2020,3,1),datetime(2020,4,1),datetime(2020,5,1),datetime(2020,6,1),datetime(2020,7,1),datetime(2020,8,1)],['2020/3/1','2020/4/1','2020/5/1','2020/6/1','2020/7/1','2020/8/1']);
    plt.close()

    len(datesJHU)
    len(dates)
    len(pos_poisson)
    len(pos_poisson_long)
    fatality_rate_poisson=[]
    x_fit_poisson=[];y_fit_poisson=[];dates_fit_poisson=[];day_fit=42;nday=12;
    if True:
        for i in range(nday,len(datesJHU)):
            #print datesJHU[i],(dth[i]-dth[i-7]),pos_poisson[i-nday]
            fat_rate_tmp = (dth[i]-dth[i-7])/pos_poisson[i-nday]/7/(1-numpy.exp(decay))
            plt.semilogy(datesJHU[i],fat_rate_tmp,'.r');plt.xlabel('Date');plt.ylabel('Fatality Rate');
            fatality_rate_poisson.append(fat_rate_tmp)
            x_fit_poisson.append(i);y_fit_poisson.append(numpy.log(fat_rate_tmp));dates_fit_poisson.append(datesJHU[i])
        plt.grid(which='both')
        coef_poisson=numpy.polyfit(x_fit_poisson[-day_fit:],y_fit_poisson[-day_fit:],1);
        for i in range(-day_fit,0): y_fit_poisson[i]=numpy.exp(coef_poisson[0]*x_fit_poisson[i]+coef_poisson[1])
        plt.plot(dates_fit_poisson[-day_fit:],y_fit_poisson[-day_fit:],'b');
        plt.savefig('US_Death_ratio_poisson',dpi=150)

    plt.close()
    nday=21;
    x_fit=[];y_fit=[];dates_fit=[]
    if True:
        for i in range(nday,len(datesJHU)):
            #if datesJHU[i] > datetime(2020,6,24): dth[i] -= 1800
            #if datesJHU[i] > datetime(2020,6,29): dth[i] -= 600
            if datesJHU[i]-nday*timedelta(days=1) in dates:
                idx=dates.index(datesJHU[i]-nday*timedelta(days=1))
                plt.semilogy(datesJHU[i],(dth[i]-dth[i-7])/numpy.mean(pos[idx:idx+7])/7,'.r')
                x_fit.append(i);y_fit.append(numpy.log((dth[i]-dth[i-7])/numpy.mean(pos[idx:idx+7])/7));dates_fit.append(datesJHU[i])
                #print datesJHU[i],(dth[i]-dth[i-7])/7,numpy.mean(pos[idx:idx+7])
        plt.legend(['Daily Deaths / Daily Cases (Shifted 21 Days)'])
        coef=numpy.polyfit(x_fit,y_fit,1);
        for i in range(len(x_fit)): y_fit[i]=numpy.exp(coef[0]*x_fit[i]+coef[1])
        plt.plot(dates_fit,y_fit,'b');
        plt.xlabel('Date');plt.ylabel('Fatality Rate');plt.grid(which='both')
    plt.savefig('US_Death_ratio',dpi=150)
    plt.close()

    logdth = [0]*len(dth);dthavg = [0]*len(dth)
    datefit = [i for i in range(len(dth))];n_avg = 7;
    for i in range(7,len(dth)):
        #dthavg[i] = numpy.average(dth[max(0,i-n_avg/2):min(i+n_avg/2+1,len(dth))])
        logdth[i] = numpy.log((dth[i]-dth[i-7])/7)
    slope = [];lfit = 14;
    for i in range(lfit,len(dth)):
        cdth=numpy.polyfit(datefit[i:lfit+i],logdth[i:lfit+i],1);
        slope.append(cdth[0])
    plt.plot(datesJHU[lfit:len(dth)],slope,'b');
    plt.savefig('US_Death_slope',dpi=150)
    plt.close()

    for i in range(len(datesJHU)):
        if datesJHU[i] > datetime(2020,6,24): dth[i] += 1800
        if datesJHU[i] > datetime(2020,6,29): dth[i] += 600
        if datesJHU[i] > datetime(2020,7,26): dth[i] += 675
        if datesJHU[i] > datetime(2020,8,5): dth[i] += 500
    nday=21;count=0
    nday=12
    #while date < enddate+timedelta(days = 1)*(nday-4):
    while date < enddate:
        datesJHU.append(date)
        if date-nday*timedelta(days = 1) in dates:
            count += 1
            #avg_pos=[];avg_pos_l=[];avg_pos_h=[];
            #for i in range(7):
            #    idx = dates.index(date-(nday+3-i)*timedelta(days = 1))
            #    avg_pos.append(pos[idx]);avg_pos_l.append(pos_l[idx]);avg_pos_h.append(pos_h[idx]);
            #dth_ratio = numpy.exp(coef[0]*(x_fit[-1]+count)+coef[1])
            #print date,numpy.mean(avg_pos),dth_ratio
            #death = dth[-1]+ numpy.mean(avg_pos)*dth_ratio
            #death_l = dth_l[-1]+ numpy.mean(avg_pos_l)*dth_ratio*0.9
            #death_h = dth_h[-1]+ numpy.mean(avg_pos_h)*dth_ratio*1.1
            dth_ratio_poisson = numpy.exp(coef_poisson[0]*(x_fit_poisson[-1]+count)+coef_poisson[1])
            print(dth_ratio_poisson)
            idx = dates.index(date-nday*timedelta(days = 1))
            death = dth[-1] + pos_poisson_long[idx]*dth_ratio_poisson*(1-numpy.exp(decay))
            death_l = dth_l[-1] + pos_poisson_long[idx]*dth_ratio_poisson*(1-numpy.exp(decay))*0.8
            death_h = dth_h[-1] + pos_poisson_long[idx]*dth_ratio_poisson*(1-numpy.exp(decay))*1.2
        else:
            death = dth[-1]+ pos[-1]*0.020
            death_l = dth_l[-1]+ pos_l[-1]*0.015
            death_h = dth_h[-1]+ pos_h[-1]*0.022
        dth.append(death)
        dth_l.append(death_l)
        dth_h.append(death_h)
        date = date + timedelta(days = 1)

    if True:
        idx = datesJHU.index(dates[0])
        plt.plot(datesJHU[0:idx+1],dth[0:idx+1],'k')
        plt.plot(datesJHU[idx+1:],dth[idx+1:],'r')
        if enddate > datetime(2020,6,30):
            plt.xticks([datetime(2020,5,1),datetime(2020,6,1),datetime(2020,7,1),datetime(2020,8,1),datetime(2020,9,1)],['2020/5/1','2020/6/1','2020/7/1','2020/8/1','2020/9/1'])
        else:
            plt.xticks([datetime(2020,3,1),datetime(2020,4,1),datetime(2020,5,1),datetime(2020,6,1),datetime(2020,7,1)],['2020/3/1','2020/4/1','2020/5/1','2020/6/1','2020/7/1'])
        plt.xlabel('Date');plt.ylabel('Total Deaths');plt.grid(which='both')
        plt.xlim([datetime(2020,4,20),datesJHU[-1]+timedelta(days = 1)])
        plt.tight_layout()
        plt.savefig('US_Death_Projection',dpi=150)
    plt.close()
    idx = datesJHU.index(dates[0])
    n_avg = 7; dthdaily=[0]*idx; dthavg=[0]*idx
    for i in range(idx):
        dthdaily[i] = dth[i+1]-dth[i]
    for i in range(idx):
        dthavg[i] = numpy.average(dthdaily[max(0,i-n_avg//2):min(i+n_avg//2+1,len(dth))])
        dthavg[i] = numpy.average(dthdaily[max(0,i-n_avg+1):min(i+1,len(dth))])
    if True:
        plt.plot(datesJHU[:idx],dthavg[:idx],'k')
        plt.plot(datesJHU[0],dth[0+1]-dth[0],'+k')
        plt.plot(datesJHU[-2],dth[-1]-dth[-2],'.r')
        #plt.plot(datesJHU[-2],(dth_l[-1]-dth_l[-2])*0.75,'.b')
        for i in range(0,idx):
            plt.plot(datesJHU[i],dth[i+1]-dth[i],'+k')
            #print datesJHU[i],dth[i+1]-dth[i]
        for i in range(idx,len(dth)-1):
            plt.plot(datesJHU[i],dth[i+1]-dth[i],'.r')
            plt.plot(datesJHU[i],dth_l[i+1]-dth_l[i],'.r',markersize=2)
            plt.plot(datesJHU[i],dth_h[i+1]-dth_h[i],'.r',markersize=2)
            #plt.plot(datesJHU[i],(dth_l[i+1]-dth_l[i])*0.75,'.b')
        if enddate > datetime(2020,6,30):
            plt.xticks([datetime(2020,5,1),datetime(2020,6,1),datetime(2020,7,1),datetime(2020,8,1),datetime(2020,9,1)],['2020/5/1','2020/6/1','2020/7/1','2020/8/1','2020/9/1'])
        else:
            plt.xticks([datetime(2020,3,1),datetime(2020,4,1),datetime(2020,5,1),datetime(2020,6,1),datetime(2020,7,1)],['2020/3/1','2020/4/1','2020/5/1','2020/6/1','2020/7/1'])
        plt.xlabel('Date');plt.ylabel('Daily Deaths');plt.grid(which='both')
        plt.xlim([datetime(2020,5,1),datesJHU[-1]+timedelta(days = 1)])
        plt.ylim([0,2500])
        plt.legend(['7-day Average','Daily Deaths (JHU)','Projection'])
        plt.tight_layout()
    plt.savefig('US_Death_Projection_daily',dpi=150)
    plt.close()

    F = open("US_death_proj","w")
    for i in range(len(dth)):
        F.write(str(datesJHU[i].year)+' '+str(datesJHU[i].month)+' '+str(datesJHU[i].day)+' ');
        F.write(str(dth[i])+'\n')

    today = dates[0]
    out = str(today.year)+'-'
    if today.month<10: out += '0'
    out += str(today.month)+'-'
    if today.day<10: out = out+'0'
    out += str(today.day)+','
    idx = datesJHU.index(today)
    F = open("QJHong-Encounter.csv","w")
    count = 0
    pos_out = 0
    pos_l_out = 0
    pos_h_out = 0
    dth_l0 = dth_l[idx]
    dth_h0 = dth_h[idx]
    dth0 = dth[idx]
    for i in range(idx-3,len(dth)):
        print(datesJHU[i],dth[i])
        if datesJHU[i] in dates:
            idx2 = dates.index(datesJHU[i])
            pos_out += pos[idx2]
            pos_l_out += pos_l[idx2]
            pos_h_out += pos_h[idx2]
        if datesJHU[i].month==9 and datesJHU[i].day%7 == 5 and datesJHU[i].day>12 or datesJHU[i].month==10 and datesJHU[i].day%7 == 3:
            count+=1
            out2 = out+str(count)+' wk ahead cum death,'
            date = str(datesJHU[i].year)+'-'
            if datesJHU[i].month<10: date += '0'
            date += str(datesJHU[i].month)+'-'
            if datesJHU[i].day<10: date = date+'0'
            date += str(datesJHU[i].day)
            out2 += date
            out2 += ',US,'
            out3 = out2+'quantile,0.025,'
            F.write(out3+str(dth_l[i])+'\n')
            out3 = out2+'quantile,0.25,'
            F.write(out3+str(dth_l[i]*0.33+dth[i]*0.67)+'\n')
            out3 = out2+'point,NA,'
            F.write(out3+str(dth[i])+'\n')
            out3 = out2+'quantile,0.75,'
            F.write(out3+str(dth_h[i]*0.33+dth[i]*0.67)+'\n')
            out3 = out2+'quantile,0.975,'
            F.write(out3+str(dth_h[i])+'\n')
            out2 = out+str(count)+' wk ahead inc death,'
            date = str(datesJHU[i].year)+'-'
            if datesJHU[i].month<10: date += '0'
            date += str(datesJHU[i].month)+'-'
            if datesJHU[i].day<10: date = date+'0'
            date += str(datesJHU[i].day)
            out2 += date
            out2 += ',US,'
            out3 = out2+'quantile,0.025,'
            F.write(out3+str(dth_l[i]-dth_l0)+'\n')
            out3 = out2+'quantile,0.25,'
            F.write(out3+str((dth_l[i]-dth_l0)*0.33+(dth[i]-dth0)*0.67)+'\n')
            out3 = out2+'point,NA,'
            F.write(out3+str(dth[i]-dth0)+'\n')
            out3 = out2+'quantile,0.75,'
            F.write(out3+str((dth_h[i]-dth_h0)*0.33+(dth[i]-dth0)*0.67)+'\n')
            out3 = out2+'quantile,0.975,'
            F.write(out3+str(dth_h[i]-dth_h0)+'\n')
            if count>0 and count<=5:
                out2 = out+str(count)+' wk ahead inc case,'
                date = str(datesJHU[i].year)+'-'
                if datesJHU[i].month<10: date += '0'
                date += str(datesJHU[i].month)+'-'
                if datesJHU[i].day<10: date = date+'0'
                date += str(datesJHU[i].day)
                out2 += date
                out2 += ',US,'
                out3 = out2+'quantile,0.10,'
                F.write(out3+str(pos_l_out)+'\n')
                out3 = out2+'point,NA,'
                F.write(out3+str(pos_out)+'\n')
                out3 = out2+'quantile,0.90,'
                F.write(out3+str(pos_h_out)+'\n')
            pos_out = 0
            pos_l_out = 0
            pos_h_out = 0
        if  datesJHU[i].month==8 and datesJHU[i].day%7 == 1 or datesJHU[i].month==9 and datesJHU[i].day%7 == 5 or datesJHU[i].month==10 and datesJHU[i].day%7 == 3:
            pos_out = 0
            pos_l_out = 0
            pos_h_out = 0
            dth_l0 = dth_l[i]
            dth_h0 = dth_h[i]
            dth0 = dth[i]
