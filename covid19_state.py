import json
import requests
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
import matplotlib.dates as mdates
from datetime import datetime
from datetime import timedelta
import numpy
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import math
import sys

tmp = sys.argv
state = tmp[1]
#state = "NC"

def get_DNC(state):
    if state == "US":
        response = requests.get("https://covidtracking.com/api/v1/us/daily.json")
    else:
        response = requests.get("https://covidtracking.com/api/v1/states/daily.json")
    todos = json.loads(response.text)

    todos
    counter = 0
    dates=[];pos=[];tot=[]
    for todo in todos:
        if state == "US" or todo["state"]==state:
            counter = counter + 1
            dates.append(todo["date"])
            pos.append(todo["positive"])
            if 'negative' in todo and todo["negative"] is not None:
                tot.append(todo["negative"]);#print todo["negative"]
            else:
                tot.append(0)
            if counter > 83: break # get at most 70 data points

    date0 = dates[0]
    for i in range(len(dates)):
        date = dates[i]
        dates[i] = datetime(date/10000,date/100%100,date%100)

    for i in range(len(pos)-1):
        pos[i] = pos[i] - pos[i+1]
        tot[i] = tot[i] - tot[i+1] + pos[i]

    return dates,pos,tot

def get_SD(state):
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
    F = open("/Users/qijunhong/Documents/GitHub/covid19/data/"+state+"_ED","r")
    data = [ line.split() for line in F]
    F = open("/Users/qijunhong/Documents/GitHub/covid19/data/"+state+"_ED_coef","r")
    data2 = [ line.split() for line in F];print data2
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
    plt.text(dates[18],min(pos[0:10])/2.2,"Trend: "+str(format(-cpos[0]*100, '.2f'))+"% per day")
    plt.text(dates[18],min(tot[0:10])/2.2,"Trend: "+str(format(-ctot[0]*100, '.2f'))+"% per day")
    plt.xlim([datetime(2020,03,20),dates[0]]);plt.ylim([min(pos[0:70])*0.9,max(tot)*1.1])
    plt.text(dates[45],max(tot)/2,state,fontsize=20)
    plt.xticks([datetime(2020,3,16),datetime(2020,4,1),datetime(2020,4,16),datetime(2020,5,1),datetime(2020,5,16)],['2020/3/16','2020/4/1','2020/4/16','2020/5/1','2020/5/16'])
    plt.xlabel('Date');plt.ylabel('Daily New Cases');plt.grid(which='both')
    plt.legend(['Daily New Cases','Daily Total Tests'])
    plt.tight_layout()
    plt.savefig(state+'_DailyNewCases',dpi=150)

def plot_PR(dates,pos,tot):
    pr = [0]*len(pos)
    for i in range(len(pos)-1):
        pr[i] = pos[i]/tot[i]
    plt.plot(dates[:-2],pr[:-2],'r');
    plt.xticks([datetime(2020,3,16),datetime(2020,4,1),datetime(2020,4,16),datetime(2020,5,1),datetime(2020,5,16)],['2020/3/16','2020/4/1','2020/4/16','2020/5/1','2020/5/16'])
    plt.xlabel('Date');plt.ylabel('Positive Ratio');plt.grid(which='both')
    plt.tight_layout()
    #plt.savefig(state+'_PositiveRatio',dpi=150)

def plot_proj(dates,pos,posavg):
    plt.plot(dates[len(posavg):len(pos)],pos[len(posavg):len(pos)],'.r');
    plt.plot(dates[0:len(posavg)],pos[0:len(posavg)],'+k');
    plt.plot(dates[0:len(posavg)],posavg,'k');
    plt.xticks([datetime(2020,3,1),datetime(2020,4,1),datetime(2020,5,1),datetime(2020,6,1)],['2020/3/1','2020/4/1','2020/5/1','2020/6/1'])
    plt.xlabel('Date');plt.ylabel('Daily New Cases');plt.grid(which='both')
    plt.legend(['Projection','Daily New Cases','Past 7-day Average'])
    plt.text(dates[45],max(pos)/1.1,state,fontsize=20)
    plt.tight_layout()
    plt.savefig(state+'_Projection',dpi=150)
    #plt.close()
def plot_R_D(SLP_SD):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    for i in range(len(SLP_SD)):
        ax2.plot(SLP_SD[i][4],SLP_SD[i][3],'.k')
        if SLP_SD[i][4] in dates:
            ax1.plot(SLP_SD[i][4],SLP_SD[i][0]+1,'.r');#plt.grid(which='both');plt.xticks([datetime(2020,3,1),datetime(2020,4,1),datetime(2020,5,1),datetime(2020,6,1)],['2020/3/1','2020/4/1','2020/5/1','2020/6/1'])
        #else:
            #ax2.plot(SLP_SD[i][4],SLP_SD[i][0],'+k');#plt.grid(which='both');plt.xticks([datetime(2020,3,1),datetime(2020,4,1),datetime(2020,5,1),datetime(2020,6,1)],['2020/3/1','2020/4/1','2020/5/1','2020/6/1'])
    #ax1.xticks([datetime(2020,3,1),datetime(2020,4,1),datetime(2020,5,1),datetime(2020,6,1)],['2020/3/1','2020/4/1','2020/5/1','2020/6/1'])
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Reproductive number R_d, Daily', color='r')
    ax1.tick_params(axis='y', labelcolor='r')
    ax2.set_ylabel('Encounter Density, Adjusted', color='k')
    ax2.tick_params(axis='y', labelcolor='k')
    #ax1.tight_layout()
    fig.savefig(state+'_R_D'+'_0',dpi=150)

def plot_R_D_proj(SLP_SD,predict,dates_pred,nshift):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ymax = 0
    for i in range(len(SLP_SD)):
        ax2.plot(SLP_SD[i][4],SLP_SD[i][3],'.k')
        if SLP_SD[i][3]>ymax: ymax = SLP_SD[i][3]
        if SLP_SD[i][4] in dates:
            ax1.plot(SLP_SD[i][4],SLP_SD[i][0]+1,'.r');#plt.grid(which='both');plt.xticks([datetime(2020,3,1),datetime(2020,4,1),datetime(2020,5,1),datetime(2020,6,1)],['2020/3/1','2020/4/1','2020/5/1','2020/6/1'])
        else:
            d = dates_pred.index(SLP_SD[i][4])
            ax1.plot(SLP_SD[i][4],predict[d]+1,'+r');
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Reproductive number R_d, Daily', color='r')
    ax1.tick_params(axis='y', labelcolor='r')
    ax2.set_ylabel('Encounter Density, Adjusted', color='k')
    ax2.tick_params(axis='y', labelcolor='k')
    #ax1.tight_layout()
    ax2.text(dates[5],ymax/1.2,state,fontsize=20)
    ax1.legend(['R'],loc=9)
    ax2.legend(['D, shifted '+str(nshift)+' days'],loc=1)
    fig.savefig(state+'_R_D'+'_1',dpi=150)

[dates,pos,tot] = get_DNC(state)
flagstop = False;navg=0;
while not flagstop:
    tot_tmp=[0]*len(tot);flagstop=True
    for i in range(len(tot)): tot_tmp[i] = numpy.mean(tot[max(i-navg,0):min(i+navg+1,len(tot))])
    for i in range(50):
        r2=tot_tmp[i]/tot_tmp[i+1]
        if r2>1.5 or r2<0.6:
            flagstop=False;navg=navg+1;break
print navg,r2
flagstop = False;navg=0;
while not flagstop:
    pos_tmp=[0]*len(pos);flagstop=True
    for i in range(len(pos)): pos_tmp[i] = numpy.mean(pos[max(i-navg,0):min(i+navg+1,len(pos))])
    for i in range(50):
        r1=pos_tmp[i]/pos_tmp[i+1]
        if r1>1.5 or r1<0.6:
            flagstop=False;navg=navg+1;break
print navg,r1
pos_fit=[0]*len(pos);tot_fit=[0]*len(pos);
for i in range(len(pos)-1): pos_fit[i] = pos_tmp[i];tot_fit[i] = tot_tmp[i];
#for i in range(len(pos)-1):
#    if state=='MI' or state=='TX' or state=='MO' or state=='WA' or state=='MD' or state=='NC': tot_tmp[i] = numpy.mean(tot[max(i-6,0):min(i+7,len(tot))])
#    elif state=='DC' or state=='GA': tot_tmp[i] = numpy.mean(tot[max(i-8,0):min(i+9,len(tot))])
#    else: tot_tmp[i] = numpy.mean(tot[max(i-2,0):min(i+3,len(tot))])
#for i in range(len(pos)-1):
#    if state=='CT' or state=='GA' or state=='WA' or state=='MD' or state=='NC' or state=='MT': pos_tmp[i] = numpy.mean(pos[max(i-3,0):min(i+4,len(tot))])
#    elif state=='HI': pos_tmp[i] = numpy.mean(pos[max(i-5,0):min(i+6,len(tot))])
#    else: pos_tmp[i] = numpy.mean(pos[max(i-1,0):min(i+2,len(tot))])
for i in range(len(pos)-1):
    tot_tmp[i] = numpy.mean(tot[max(i-1,0):min(i+2,len(tot))])
for i in range(len(pos)-1):
    pos_tmp[i] = numpy.mean(pos[max(i-1,0):min(i+2,len(tot))])
for i in range(len(pos)-1): pos[i] = pos_tmp[i];tot[i] = tot_tmp[i];

plot_DNC(dates,pos,tot)
#print tot
#plt.close()
#plot_DNC(dates,pos_fit,tot_fit)
plt.close()
plot_PR(dates,pos,tot)

pos[0:3]
posavg = [0]*len(pos);totavg = [0]*len(pos);logpos = [0]*len(pos);logtot = [0]*len(pos);
datefit = [i for i in range(len(pos))];n_avg = 7; ndays=min(len(pos)-2,75);#n_avg = (n_avg-1)/2
for i in range(len(pos)): # average last 7 days to remove fluctuation
    posavg[i] = numpy.average(pos_fit[max(0,i-n_avg/2):min(i+n_avg/2+1,len(pos))])
    totavg[i] = numpy.average(tot_fit[max(0,i-n_avg/2):min(i+n_avg/2+1,len(pos))])
    logpos[i] = numpy.log(posavg[i]);logtot[i] = numpy.log(totavg[i]);#logtot[i] = numpy.log(totavg[i])-2
posavg0 = numpy.average(pos[0:14])
slope = [];slope0 = [];slope1 = [];lfit = 7;coef_tot = 0.5;
for i in range(ndays):
    cpos=numpy.polyfit(datefit[i:lfit+i],logpos[i:lfit+i],1);
    ctot=numpy.polyfit(datefit[i:lfit+i],logtot[i:lfit+i],1)
    slope.append(-cpos[0]*1+ctot[0]*coef_tot)
    slope0.append(-cpos[0]*1);slope1.append(-ctot[0]*1)

slope_avg = [0]*len(slope);n_avg=1
for i in range(len(slope)):
    slope_avg[i] = numpy.average(slope[i:min(i+n_avg,len(slope))])
plt.plot(dates[0:ndays],slope_avg[0:ndays],'m');plt.plot(dates[0:ndays],slope0[0:ndays],'r');plt.plot(dates[0:ndays],slope1[0:ndays],'k');plt.grid(which='both');plt.xticks([datetime(2020,3,1),datetime(2020,3,16),datetime(2020,4,1),datetime(2020,4,16),datetime(2020,5,1),datetime(2020,5,16)],['2020/3/1','2020/3/16','2020/4/1','2020/4/16','2020/5/1','2020/5/16']);
plt.close()

[AM,NE,ED0] = get_SD(state)# len(dates) == len(pos) # len(dates_SD) == len(AM)
dn = datetime(2020,5,26);day = timedelta(days = 1);dates_SD = [];l=len(AM)
for i in range(l): dates_SD.append(dn-day*(l-i-1))
AM_avg=[0]*l;NE_avg=[0]*l;ED_avg=[0]*l;ED0_avg=[0]*l;ED=[0]*l;n_avg = 7;
for i in range(len(AM)): #ED[i] = ED0[i]
    if dates_SD[i] == datetime(2020,4,03): iref = i
    #if dates_SD[i] == datetime(2020,5,1): iref = i
    if dates_SD[i] > datetime(2020,4,03): ED[i] = ED0[i] * min( max( 1 - (i-iref)*0.015, 0.75 ), 1 )
    #elif dates_SD[i] > datetime(2020,5,1): ED[i] = ED0[i] * min( max( 0.5 + (i-iref)*0.015, 0.5), 0.9 )
    else: ED[i] = ED0[i]
for i in range(len(AM)):
    AM_avg[i] = numpy.average(AM[max(0,i-n_avg/2):min(i+n_avg/2+1,len(ED))])
    NE_avg[i] = numpy.average(NE[max(0,i-n_avg/2):min(i+n_avg/2+1,len(ED))])
    ED0_avg[i] = numpy.average(ED0[max(0,i-n_avg/2):min(i+n_avg/2+1,len(ED))])
    ED_avg[i] = numpy.average(ED[max(0,i-n_avg/2):min(i+n_avg/2+1,len(ED))])
plt.plot(dates_SD,AM_avg,'k');plt.plot(dates_SD,NE_avg,'b');plt.plot(dates_SD,ED_avg,'g');plt.plot(dates_SD,ED0_avg,'r');plt.grid(which='both');plt.xticks([datetime(2020,3,1),datetime(2020,3,16),datetime(2020,4,1),datetime(2020,4,16),datetime(2020,5,1),datetime(2020,5,16)],['2020/3/1','2020/3/16','2020/4/1','2020/4/16','2020/5/1','2020/5/16']);

nshift = 18;
#if state=="CA": nshift=14;

if state=="GA" or state=="SC" or state=="TN" or state=="WV": nshift=16;
if state=="IA": nshift=17;
if state=="LA" or state=="OK": nshift=20;
if state=="NY" or state=="MI": nshift=21;
if state=="UT": nshift=22;
if state=="NE": nshift=23;
if state=="CT" or state=="MA": nshift=24;
#if state=="MD": nshift=26;
for dday in range(nshift,nshift+1):
    SLP_SD = [];p1=0;x=[];y=[]#dday = 22;
    while dates[p1]-day*dday not in dates_SD: p1 = p1+1
    p2 = dates_SD.index(dates[p1]-day*dday)
    dates[p1]
    while p1 < len(slope_avg):
        #print p1,p2
        #SLP_SD.append([slope_avg[p1], AM[p2],NE[p2],ED[p2], dates[p1]])
        SLP_SD.append([slope_avg[p1], AM_avg[p2],NE_avg[p2],ED_avg[p2], dates[p1]])
        if math.isnan(slope_avg[p1]): break
        x.append(slope_avg[p1])
        y.append(ED_avg[p2])
        p1 = p1 + 1;p2 = p2 - 1
        if p2 == 0: break
    for i in range(len(AM)):
        if dates_SD[i]+day*dday not in dates:
            SLP_SD.append([0, AM_avg[i],NE_avg[i],ED_avg[i],dates_SD[i]+day*dday])
    print dday,numpy.corrcoef(x,y)

x_train = [];y_train = []; weight = []; x_pred = [];dates_pred = []
for i in range(len(SLP_SD)):
    if SLP_SD[i][4] in dates:
        if not math.isnan(SLP_SD[i][0]):
            x_train.append([SLP_SD[i][3]])
            y_train.append(SLP_SD[i][0])
            weight.append(numpy.power(0.95,i))
    else:
        dates_pred.append(SLP_SD[i][4])
        x_pred.append([SLP_SD[i][3]])
plot_R_D(SLP_SD)
plt.close()

bdt = LinearRegression()
bdt.fit(x_train, y_train, weight)
print bdt.score(x_train,y_train)
predict = bdt.predict(x_pred)

plt.plot(x_train,y_train,'k');plt.plot(x_pred,predict,'r')
plt.close()

plot_R_D_proj(SLP_SD,predict,dates_pred,nshift)
plt.close()

adj_tot = numpy.average(slope1[0:7])*coef_tot
print adj_tot
print dates_pred
flag_first = True
for i in range(len(dates_pred)):
    if dates_pred[i] not in dates:
        if flag_first:
            flag_first = False
            #pos_now = posavg[dates.index(dates_pred[i]-day)] * (predict[i]+adj_tot+1)
            pos_now = posavg0 * (predict[i]+adj_tot+1)
        else:
            pos_now = pos[dates.index(dates_pred[i]-day)] * (predict[i]+adj_tot+1)
        #print (predict[i]+adj_tot+1)
        dates.append(dates_pred[i])
        pos.append(pos_now)

len(dates)
len(pos)
plot_proj(dates,pos,posavg)
F = open("/Users/qijunhong/Documents/GitHub/covid19/"+state+"_proj","w")
for i in range(len(posavg),len(pos)):
    F.write(str(dates[i].year)+' '+str(dates[i].month)+' '+str(dates[i].day)+' ');
    F.write(str(pos[i])+'\n')
