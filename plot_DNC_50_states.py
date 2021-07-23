import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
from datetime import datetime
from datetime import timedelta
import numpy
from support import get_DNC,get_DNC_JHU
import os

states = ['AK', 'WA', 'ID', 'MT', 'ND', 'MN', 'WI', 'VT', 'NH', 'ME', \
          'OR', 'WY', 'SD', 'IA', 'IL', 'MI', 'NY', 'CT', 'RI', 'MA', \
          'CA', 'NV', 'UT', 'CO', 'NE', 'MO', 'IN', 'OH', 'PA', 'NJ', \
          'AZ', 'NM', 'KS', 'OK', 'AR', 'KY', 'WV', 'VA', 'MD', 'DE', \
          'HI', 'TX', 'LA', 'MS', 'TN', 'AL', 'GA', 'SC', 'NC', 'FL' ] 
day = timedelta(days = 1)
nshift = 0
coef_tot = 0.
enddate = datetime(2021,12,31)
fs = 8

#for j in range(len(states)):
for j in range(50):
    state = states[j]
    [dates0,pos0,tot0,dth0] = get_DNC(state)
    [dates1,pos1,tot1,dth1] = get_DNC_JHU(state)
    dates = dates1 + dates0
    #print(dates)
    pos = pos1 + pos0
    tot = tot1 + tot0
    dth = dth1 + dth0

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
    pos_fit=[0]*len(pos);tot_fit=[0]*len(pos);
    for i in range(len(pos)-1): pos_fit[i] = pos_tmp[i];tot_fit[i] = tot_tmp[i];
    #change pos,tot to 3-day average
    for i in range(len(pos)-1):
        tot_tmp[i] = numpy.mean(tot[max(i-1,0):min(i+2,len(tot))])
    for i in range(len(pos)-1):
        pos_tmp[i] = numpy.mean(pos[max(i-1,0):min(i+2,len(tot))])
    for i in range(len(pos)-1): pos[i] = pos_tmp[i];tot[i] = tot_tmp[i];

    posavg = [0]*len(pos);totavg = [0]*len(pos);logpos = [0]*len(pos);logtot = [0]*len(pos);
    datefit = [i for i in range(len(pos))];n_avg = 7; ndays=min(len(pos)-2,215);
    for i in range(len(pos)-1):
        #posavg[i] = numpy.average(pos_fit[max(0,i-n_avg//2):min(i+n_avg//2+1,len(pos))])
        #totavg[i] = numpy.average(tot_fit[max(0,i-n_avg//2):min(i+n_avg//2+1,len(pos))])
        posavg[i] = numpy.average(pos_fit[max(0,i):min(i+n_avg+1,len(pos))])
        totavg[i] = numpy.average(tot_fit[max(0,i):min(i+n_avg+1,len(pos))])
    #print(posavg)
    #print(dates)
    #(ratio(l)/max(ratio) > 0.7 || ratio(l)/ratio_avg>1.5 ) && ratio(l)>100
    plt.subplot(5,10,j+1)
    max_pos = max(posavg)
    pos_local_min = numpy.mean(posavg[15:45])
    if ( posavg[0] / max_pos > 0.7 or posavg[0] / pos_local_min > 1.5 ) and posavg[0] > 100. :
        plt.title(states[j],color='r',fontsize=fs, pad = 1)
    else:
        plt.title(states[j],color='k',fontsize=fs, pad = 1)
    plt.plot(dates[:-2],posavg[:-2],'k')
    plt.xticks([],[])
    plt.yticks([],[])
    #os.chdir("/Users/qijunhong/Documents/MATLAB/Finance/covid19/state")
    f = open(states[j], 'w')
    for i in range(len(pos)-2):
        f.write("%i %i\n" % (sum(pos[i:]),sum(tot[i:])))
    f.close()

plt.suptitle('Daily New Cases (7-Day Avg) over the Past 200 Days in 50 US States')
plt.savefig('states_DNC',dpi=300)
