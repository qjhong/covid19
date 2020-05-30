import json
import requests
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
import matplotlib.dates as mdates
from datetime import datetime
from datetime import timedelta
import numpy
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

state = 'US'

def get_DNC():
    response = requests.get("https://covidtracking.com/api/v1/us/daily.json")
    todos = json.loads(response.text)

    counter = 0
    dates=[];pos=[];tot=[]
    for todo in todos:
        counter = counter + 1
        dates.append(todo["date"])
        pos.append(todo["positive"])
        tot.append(todo["negative"])
        if counter > 83: break # get at most 70 data points

    date0 = dates[0]
    for i in range(len(dates)):
        date = dates[i]
        dates[i] = datetime(date/10000,date/100%100,date%100)

    for i in range(len(pos)-1):
        pos[i] = pos[i] - pos[i+1]
        tot[i] = tot[i] - tot[i+1] + pos[i]

    return dates,pos,tot

def get_SD():
    F = open("/Users/qijunhong/Documents/GitHub/covid19/data/US_AM","r")
    data = [ line.split() for line in F]
    date=[];AM=[]
    for i in range(len(data[0])):
        tmp=data[0][i].split(',')
        date.append(float(tmp[0]))
        ytmp=float(tmp[1])
        AM.append(2-((370-ytmp)*0.1+(ytmp-100)*1.9)/(370-100))
    F = open("/Users/qijunhong/Documents/GitHub/covid19/data/US_NE","r")
    data = [ line.split() for line in F]
    NE=[]
    for i in range(len(data[0])):
        tmp=data[0][i].split(',')
        ytmp=float(tmp[1])
        NE.append(2-((370-ytmp)*0.1+(ytmp-100)*1.9)/(370-100))
    F = open("/Users/qijunhong/Documents/GitHub/covid19/data/US_ED","r")
    data = [ line.split() for line in F]
    ED=[]
    for i in range(len(data[0])):
        tmp=data[0][i].split(',')
        ytmp=float(tmp[1])
        ED.append(2-((370-ytmp)*0.0+(ytmp-100)*2.0)/(370-100))

    return AM,NE,ED

def plot_DNC(dates,pos,tot):
    logpos = [0]*len(pos);logtot = [0]*len(pos)
    for i in range(len(pos)-1):
        logpos[i] = numpy.log(pos[i])
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
    plt.text(dates[18],posfit[0]/1.8,"Trend: "+str(format(-cpos[0]*100, '.2f'))+"% per day")
    plt.text(dates[18],totfit[0]/2.2,"Trend: "+str(format(-ctot[0]*100, '.2f'))+"% per day")
    plt.xlim([datetime(2020,03,20),dates[0]]);plt.ylim([5000,500000])
    plt.xticks([datetime(2020,3,16),datetime(2020,4,1),datetime(2020,4,16),datetime(2020,5,1),datetime(2020,5,16)],['2020/3/16','2020/4/1','2020/4/16','2020/5/1','2020/5/16'])
    plt.xlabel('Date');plt.ylabel('Daily New Cases');plt.grid(which='both')
    plt.legend(['Daily New Cases','Daily Total Tests'])
    plt.tight_layout()
    plt.savefig('DailyNewCases',dpi=150)
    plt.close()

def plot_proj(dates,pos,posavg):
    plt.plot(dates[len(posavg):len(pos)],pos[len(posavg):len(pos)],'.r');
    plt.plot(dates[0:len(posavg)],pos[0:len(posavg)],'+k');
    plt.plot(dates[0:len(posavg)],posavg,'k');
    plt.xticks([datetime(2020,3,1),datetime(2020,4,1),datetime(2020,5,1),datetime(2020,6,1)],['2020/3/1','2020/4/1','2020/5/1','2020/6/1'])
    plt.xlabel('Date');plt.ylabel('Daily New Cases');plt.grid(which='both')
    plt.legend(['Projection','Daily Total Cases','Past 7-day Average'])
    plt.tight_layout()
    plt.savefig('Projection',dpi=150)
    #plt.close()

def plot_R_D(SLP_SD):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    for i in range(len(SLP_SD)):
        ax2.plot(SLP_SD[i][4],SLP_SD[i][3],'.k')
        #ax2.set_xticks([datetime(2020,3,1),datetime(2020,4,1),datetime(2020,5,1),datetime(2020,6,1)],['2020/3/1','2020/4/1','2020/5/1','2020/6/1'])
        if SLP_SD[i][4] in dates:
            ax1.plot(SLP_SD[i][4],SLP_SD[i][0]+1,'.r');#plt.grid(which='both');plt.xticks([datetime(2020,3,1),datetime(2020,4,1),datetime(2020,5,1),datetime(2020,6,1)],['2020/3/1','2020/4/1','2020/5/1','2020/6/1'])
            #ax1.set_xticks([datetime(2020,3,1),datetime(2020,4,1),datetime(2020,5,1),datetime(2020,6,1)],['2020/3/1','2020/4/1','2020/5/1','2020/6/1'])
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Reproductive Number $R_d$, Daily', color='r')
    ax1.tick_params(axis='y', labelcolor='r')
    ax2.set_ylabel('Encounter Density $D_{adj}$, Adjusted', color='k')
    ax2.tick_params(axis='y', labelcolor='k')
    ax1.grid(axis='x')
    fig.tight_layout()
    fig.savefig('R_D',dpi=150)

[dates,pos,tot] = get_DNC()

plot_DNC(dates,pos,tot)

pos[0:3]
posavg = [0]*len(pos);totavg = [0]*len(pos);logpos = [0]*len(pos);logtot = [0]*len(pos);
datefit = [i for i in range(len(pos))];n_avg = 7; ndays=75;#n_avg = (n_avg-1)/2
for i in range(len(pos)): # average last 7 days to remove fluctuation
    posavg[i] = numpy.average(pos[max(i,0):min(i+n_avg,len(pos))])
    totavg[i] = numpy.average(tot[max(i,0):min(i+n_avg,len(pos))])
    logpos[i] = numpy.log(posavg[i]);logtot[i] = numpy.log(totavg[i]);#logtot[i] = numpy.log(totavg[i])-2
slope = [];slope0 = [];slope1 = [];lfit = 7;coef_tot = 0.33;
for i in range(ndays):
    cpos=numpy.polyfit(datefit[i:lfit+i],logpos[i:lfit+i],1);ctot=numpy.polyfit(datefit[i:lfit+i],logtot[i:lfit+i],1)
    slope.append(-cpos[0]*1+ctot[0]*coef_tot)
    slope0.append(-cpos[0]*1);slope1.append(-ctot[0]*1)
slope_avg = [0]*len(slope);n_avg=1
for i in range(len(slope)):
    slope_avg[i] = numpy.average(slope[i:min(i+n_avg,len(slope))])
plt.plot(dates[0:ndays],slope_avg[0:ndays]);plt.plot(dates[0:ndays],slope0[0:ndays]);plt.plot(dates[0:ndays],slope1[0:ndays]);plt.grid(which='both');plt.xticks([datetime(2020,3,1),datetime(2020,3,16),datetime(2020,4,1),datetime(2020,4,16),datetime(2020,5,1),datetime(2020,5,16)],['2020/3/1','2020/3/16','2020/4/1','2020/4/16','2020/5/1','2020/5/16']);

[AM,NE,ED0] = get_SD()# len(dates) == len(pos) # len(dates_SD) == len(AM)
dn = datetime(2020,5,21);day = timedelta(days = 1);dates_SD = [];l=len(AM)
for i in range(l): dates_SD.append(dn-day*(l-i-1))
AM_avg=[0]*l;NE_avg=[0]*l;ED_avg=[0]*l;ED0_avg=[0]*l;ED=[0]*l;n_avg = 7;
for i in range(len(AM)):
    if dates_SD[i] == datetime(2020,4,03): iref = i
    if dates_SD[i] == datetime(2020,4,23): iref = i
    ED[i] = ED0[i]
    if dates_SD[i] >= datetime(2020,4,03): ED[i] = ED0[i] * min( max( 1 - (i-iref)*0.015, 0.7 ), 1 )
    if dates_SD[i] >= datetime(2020,4,23): ED[i] = ED0[i] * min( max( 0.7 + (i-iref)*0.03, 0.7), 0.9 )
    AM_avg[i] = numpy.average(AM[max(0,i-n_avg):i])
    NE_avg[i] = numpy.average(NE[max(0,i-n_avg):i])
    ED0_avg[i] = numpy.average(ED0[max(0,i-n_avg):i])
    ED_avg[i] = numpy.average(ED[max(0,i-n_avg):i])
plt.plot(dates_SD,AM_avg);plt.plot(dates_SD,NE_avg);plt.plot(dates_SD,ED_avg);plt.plot(dates_SD,ED0_avg);plt.grid(which='both');plt.xticks([datetime(2020,3,1),datetime(2020,3,16),datetime(2020,4,1),datetime(2020,4,16),datetime(2020,5,1),datetime(2020,5,16)],['2020/3/1','2020/3/16','2020/4/1','2020/4/16','2020/5/1','2020/5/16']);

for dday in range(22,23):
    SLP_SD = [];p1=0;x=[];y=[]#dday = 22;
    while dates[p1]-day*dday not in dates_SD: p1 = p1+1
    p2 = dates_SD.index(dates[p1]-day*dday)
    dates[p1]
    while p1 < len(slope_avg):
        #print p1,p2
        #SLP_SD.append([slope_avg[p1], AM[p2],NE[p2],ED[p2], dates[p1]])
        SLP_SD.append([slope_avg[p1], AM_avg[p2],NE_avg[p2],ED_avg[p2], dates[p1]])
        x.append(slope_avg[p1])
        y.append(ED_avg[p2])
        p1 = p1 + 1;p2 = p2 - 1
        if p2 == 0: break
    for i in range(len(AM)):
        if dates_SD[i]+day*dday not in dates:
            SLP_SD.append([0, AM_avg[i],NE_avg[i],ED_avg[i],dates_SD[i]+day*dday])
    print dday,numpy.corrcoef(x,y)

x_train = [];y_train = []; x_pred = [];dates_pred = []
for i in range(len(SLP_SD)):
    if SLP_SD[i][4] in dates:
        x_train.append([SLP_SD[i][3]])
        y_train.append(SLP_SD[i][0])
    else:
        dates_pred.append(SLP_SD[i][4])
        x_pred.append([SLP_SD[i][3]])

plot_R_D(SLP_SD)

bdt = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),n_estimators=100)
bdt.fit(x_train, y_train)
print bdt.score(x_train,y_train)
print bdt.feature_importances_
predict = bdt.predict(x_pred)

plt.plot(x_train,y_train,'k');plt.plot(x_pred,predict,'r')
for i in range(len(SLP_SD)):
    #plt.plot(SLP_SD[i][3],SLP_SD[i][0],'.')
    plt.plot(SLP_SD[i][4],SLP_SD[i][3]*0.25-0.05,'.r')
    #plt.plot(SLP_SD[i][4],SLP_SD[i][2]*0.25,'.g')
    #plt.plot(SLP_SD[i][4],SLP_SD[i][1]*0.25,'.b')
#plt.xticks([datetime(2020,3,1),datetime(2020,3,16),datetime(2020,4,1),datetime(2020,4,16),datetime(2020,5,1),datetime(2020,5,16)],['2020/3/1','2020/3/16','2020/4/1','2020/4/16','2020/5/1','2020/5/16'])
#for i in range(len(SLP_SD)):
    if SLP_SD[i][4] in dates:
        plt.plot(SLP_SD[i][4],SLP_SD[i][0],'.k');plt.grid(which='both');plt.xticks([datetime(2020,3,1),datetime(2020,4,1),datetime(2020,5,1),datetime(2020,6,1)],['2020/3/1','2020/4/1','2020/5/1','2020/6/1'])
    else:
        d = dates_pred.index(SLP_SD[i][4])
        plt.plot(SLP_SD[i][4],predict[d],'+k');plt.grid(which='both');plt.xticks([datetime(2020,3,1),datetime(2020,4,1),datetime(2020,5,1),datetime(2020,6,1)],['2020/3/1','2020/4/1','2020/5/1','2020/6/1'])

adj_tot = numpy.average(slope1[0:7])*coef_tot
print dates_pred
flag_first = True
for i in range(len(dates_pred)):
    if dates_pred[i] not in dates:
        if flag_first:
            flag_first = False
            pos_now = posavg[dates.index(dates_pred[i]-day)] * (predict[i]+adj_tot+1)
        else:
            pos_now = pos[dates.index(dates_pred[i]-day)] * (predict[i]+adj_tot+1)
        dates.append(dates_pred[i])
        pos.append(pos_now)
        print predict[i]+adj_tot+1

len(pos)
len(posavg)
plt.close()
plt.close()
len(dates)
len(pos)
plot_proj(dates,pos,posavg)
