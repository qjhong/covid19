import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
from datetime import datetime
from datetime import timedelta
import numpy
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from support import get_DNC,read_states,get_SD,plot_DNC,plot_PR,plot_R_D,plot_regression,plot_R_D_proj,plot_proj,plot_proj_US,prepare_data
from os import path

tmp = sys.argv
state = tmp[1]
#state = "US"

if state == 'all':
    states = ['AK', 'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', \
              'GA', 'HI', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', \
              'MD', 'ME', 'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE', \
              'NH', 'NJ', 'NM', 'NV', 'NY', 'OH', 'OK', 'OR', 'PA', 'RI', \
              'SC', 'SD', 'TN', 'TX', 'UT', 'VA', 'VT', 'WA', 'WI', 'WV', \
              'WY', 'US'] 
    n_states = 52
else:
    states = [state]
    n_states = 1
#states = ['AK', 'AL']
enddate = datetime(2021,2,20);
nshift=25;
coef_tot = 0.25;
day = timedelta(days = 1)

X_train = []; Y_train = []; WEIGHT = []; X_pred = [];
for state in states:
    SLP_SD, x_train, y_train, weight, std, x_pred, dates, dates_pred, slope1, pos, pos0, posavg = prepare_data(state,nshift,coef_tot,enddate)
    #print(numpy.asarray(x_train)[:,4])
    #print(numpy.asarray(x_pred)[:,4])
    X_train += x_train
    Y_train += y_train
    WEIGHT += weight
    X_pred += x_pred
#%%
l_train = len(X_train)
l_pred = len(X_pred)
X_data = X_train + X_pred
X_df = pd.DataFrame(X_data)
X_df.columns = ['mobility','R 60d','R 30d','R 1d','DNC','awareness','weather','state']
US_aware = X_df.loc[X_df['state']=='US']['awareness']
US_aware.head()
US_aware_merge = pd.concat([US_aware]*n_states, ignore_index=True)
X_df = pd.concat([US_aware_merge, X_df], axis=1, sort=False)
print(X_df.head())
print(X_df.tail(500))
enc = OneHotEncoder().fit_transform(X_df.iloc[:,-1].values.reshape(-1,1)).toarray()
enc_df = pd.DataFrame(enc)
X_df_enc = pd.concat([X_df.iloc[:,:-1], enc_df], axis=1)

# 'US awareness','mobility','R 60d','R 30d','R 1d','DNC','state awareness','weather','state'
#X_enc_scale_part1 = pd.DataFrame( MinMaxScaler().fit_transform(X_df_enc.iloc[:,:4]) )
#X_enc_scale_part2 = pd.DataFrame( X_df_enc.iloc[:,4]*20 )
#X_enc_scale_part3 = pd.DataFrame( MinMaxScaler().fit_transform(X_df_enc.iloc[:,5:]) )
#X_enc_scale = pd.concat([X_enc_scale_part1, X_enc_scale_part2, X_enc_scale_part3], axis=1, sort=False)
X_enc_scale = MinMaxScaler().fit_transform(X_df_enc)
x_train = X_enc_scale[0:l_train]
x_pred = X_enc_scale[l_train:]
x_train0 = numpy.asarray(x_train)
y_train0 = Y_train.copy()
x_train, x_test, y_train, y_test = train_test_split(x_train0, y_train0,random_state=1)
#bdt = RandomForestRegressor()
bdt = XGBRegressor(reg_lambda=3)
len(x_train)
bdt.fit(x_train, y_train,);
y_test_fit = bdt.predict(x_test)
err = [y_test_fit[i]-y_test[i] for i in range(len(y_test))]
stderr = numpy.linalg.norm(err)/numpy.sqrt(len(y_test))
print('XGBoost training score: ', bdt.score(x_train,y_train))
print('XGBoost testing score: ', bdt.score(x_test,y_test))
print(bdt.feature_importances_)
#bdt.fit(x_train0, y_train0, WEIGHT);
bdt.fit(x_train0, y_train0);
print('XGBoost training score: ', bdt.score(x_train0,y_train0))
print(bdt.feature_importances_)
x_pred = numpy.asarray(x_pred)
Y_pred = []
#for i in range(len(x_pred)):
#    if x_pred[i,4] == -20:
#        x_pred[i,4] = Y_pred[-1]*20
#    print(x_pred[i,:])
#    Y_pred.append( bdt.predict(x_pred[i,:].reshape(1,8)) )
Y_pred = bdt.predict(x_pred)
Y_df = pd.concat([pd.DataFrame(Y_train), pd.DataFrame(Y_pred)], axis=0)
Y_df.reset_index(drop=True, inplace=True)
XY_df = pd.concat([X_df, Y_df], axis=1)
XY_df.columns = ['awareness US','mobil','rep_30_days_ago','rep_45_days_ago','rep_60_days_ago','DNC','awareness','weather','state','rep_actual_or_predicted']
#date0 = SLP_SD[0][4]
#for i in range(len(x_pred)):
#    x_pred_curr = x_pred[i,:].reshape(1,6)
#    date0 += day*1
#    if x_pred_curr[0,4] == val_unknown:
#        x_pred_curr[0,4] = y_pred[i-1]
#    if x_pred_curr[0,5] == val_unknown:
#        if i==1:
#            date2 = date0 - day*2
#            x_pred_curr[0,5] = y_pred[i-1] - SLP_SD[ht_date[date2]][0]
#        else:
#            x_pred_curr[0,5] = y_pred[i-1] - y_pred[i-2]
#    #print(x_pred_curr)
#    y_pred.append(bdt.predict(x_pred_curr))
#%%

for state in states:
    SLP_SD, x_train, y_train, weight, std, x_pred, dates, dates_pred, slope1, pos, pos0, posavg = prepare_data(state,nshift,coef_tot,enddate)
    l_pred = len(x_pred)
    l_XY_df = XY_df[XY_df['state']==state].shape[0]
    y_pred = XY_df[XY_df['state']==state].iloc[l_XY_df-l_pred:l_XY_df].iloc[:,-1].values
    x_plot = [ line[:-1] for line in x_train]
    x_plot = numpy.asarray(x_plot)
    x_plot_pred = [ line[:-1] for line in x_pred]
    x_plot_pred = numpy.asarray(x_plot_pred)
    
    plot_regression(x_plot[:,1],y_train,x_plot_pred[:,1],y_pred,state);plt.xlim([-.2,.2])
    plt.close()
    plot_regression(x_plot[:,2],y_train,x_plot_pred[:,2],y_pred,state);plt.xlim([-.2,.2])
    plt.close()
    plot_regression(x_plot[:,3],y_train,x_plot_pred[:,3],y_pred,state);plt.xlim([-.2,.2])
    plt.close()
    plot_regression(x_plot[:,0],y_train,x_plot_pred[:,0],y_pred,state)
    plt.close()
    
    std0=numpy.mean(std[:10])
    print(stderr,std0,numpy.mean(err),numpy.linalg.norm(err)/numpy.sqrt(len(y_train)))
    plot_R_D_proj(SLP_SD,y_pred,dates_pred,nshift,std[::-1],dates,state)
    plt.close()
    
    del y_train0
    y_train0 = y_train
    err2 = min(0,y_pred[0] - y_train0[-1])
    print(stderr,std0,err2)
    #std0 = numpy.sqrt(stderr*stderr+std0*std0+err2*err2)
    std0 = numpy.sqrt(stderr*stderr+std0*std0)#+err2*err2)
    print(std0)
    #adj_tot = numpy.average(slope1[0:7])*coef_tot - max(0,y_pred[0] - y_train[0]) - max(0,(y_train[14]-y_train[0])/4)
    adj_tot = min(0.001,numpy.average(slope1[0:7])*coef_tot) - max(0,(numpy.mean(y_pred[0:5]) - y_train0[-1]))*0.99 - min(0,(max( numpy.mean(y_pred[0:10]), numpy.mean(y_pred[0:20]) ) - y_train0[-1]))*0.00 - 0.*max(0,(y_train0[-14]-y_train0[-1])/4)
    print('Today\'s R (predicted) and yesterday\'s R (actual): ',y_pred[0]+1.,y_train0[-1]+1.)
    #if y_train0[0] > 0: adj_tot -= max(0,(y_train0[14]-y_train0[0])/4)
    #adj_tot = numpy.average(slope1[0:7])*coef_tot
    pos_l = [pos[i] for i in range(len(pos))]
    pos_h = [pos[i] for i in range(len(pos))]
    print('Adj_tot:',adj_tot)
    flag_first = True;f = 1.0; f=2.0
    print('y_pred:',y_pred)
    for i in range(len(dates_pred)):
        if dates_pred[i] not in dates:
            adj_tot *= 0.95
            if flag_first:
                flag_first = False
                pos_now = pos0 * (y_pred[i]+adj_tot+1)
                pos_now_l = pos0 * (y_pred[i]+adj_tot+1-std0*f)
                pos_now_h = pos0 * (y_pred[i]+adj_tot+1+std0*f)
            else:
                pos_now = pos[dates.index(dates_pred[i]-day)] * (y_pred[i]+adj_tot+1)
                pos_now_l = pos_l[dates.index(dates_pred[i]-day)] * (y_pred[i]+adj_tot+1-std0*f)
                pos_now_h = pos_h[dates.index(dates_pred[i]-day)] * (y_pred[i]+adj_tot+1+std0*f)
            dates.append(dates_pred[i])
            pos.append(pos_now)
            pos_l.append(pos_now_l)
            pos_h.append(pos_now_h)
    
    dates_add = dates_pred[-1] + day
    while dates_add <= enddate:
        dates.append(dates_add)
        pos_now = pos[-1] * (y_pred[-1]+adj_tot+1)
        pos_now_l = pos_l[-1] * (y_pred[-1]+adj_tot+1-std0*f)
        pos_now_h = pos_h[-1] * (y_pred[-1]+adj_tot+1+std0*f)
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
            day = date.day;month=date.month; year = date.year
            if day < 10: day = '0'+str(day)
            day = str(day)
            if month < 10: month = '0' + str(month)
            month = str(month)
            year = str(year)
            #os.system("curl -s https://github.com/CSSEGISandData/COVID-19/blob/master/csse_covid_19_data/csse_covid_19_daily_reports_us/"+month+"-"+day+"-"+year+".csv > tmp");
            #os.system("grep -A7 data-line-number tmp | grep -B1 '\-\-' | grep -v '\-\-' | grep -v Confirm | cut -d'>' -f2 | cut -d'<' -f1 | awk '{s+=$1} END {print s}' > tmp1")
            #F = open("tmp1","r")
            #for line in F: death= int(line)
            jhu_df = pd.read_csv("../COVID-19/csse_covid_19_data/csse_covid_19_daily_reports_us/"+month+"-"+day+"-"+year+".csv")
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
            #if datesJHU[i] > datetime(2020,11,10): dth[i] -= 550
    
        print(datesJHU[-1],dth[-1])
        pos_poisson=[]
        for i in range(len(datesJHU)):
            j=0;sum_weight=0;sum_pos=0;decay=-0.06;
            while datesJHU[i]-j*timedelta(days=1) in dates:
                idx=dates.index(datesJHU[i]-j*timedelta(days=1))
                sum_pos += numpy.exp(decay*j)*pos[idx]
                sum_weight += numpy.exp(decay*j)
                j += 1
            #pos_poisson.append(sum_pos/sum_weight)
            pos_poisson.append(sum_pos)
        plt.plot(datesJHU,pos_poisson,'.r');plt.grid(which='both');plt.xticks([datetime(2020,4,1),datetime(2020,5,1),datetime(2020,6,1),datetime(2020,7,1),datetime(2020,8,1),datetime(2020,9,1),datetime(2020,10,1),datetime(2020,11,1)],['2020/4/1','2020/5/1','2020/6/1','2020/7/1','2020/8/1','2020/9/1','2020/10/1','2020/11/1'])
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
        x_fit_poisson=[];y_fit_poisson=[];dates_fit_poisson=[];day_fit=185;nday=10;
        if True:
            for i in range(nday,len(datesJHU)):
                #print datesJHU[i],(dth[i]-dth[i-7]),pos_poisson[i-nday]
                fat_rate_tmp = (dth[i]-dth[i-7])/pos_poisson[i-nday]/7/(1-numpy.exp(decay))
                fatality_rate_poisson.append(fat_rate_tmp)
                x_fit_poisson.append(i);y_fit_poisson.append(numpy.log(fat_rate_tmp));dates_fit_poisson.append(datesJHU[i])
            plt.semilogy(dates_fit_poisson,numpy.exp(y_fit_poisson),'.r');plt.xlabel('Date');plt.ylabel('Fatality Rate');
            plt.grid(which='both')
            coef_poisson=numpy.polyfit(x_fit_poisson[-day_fit:],y_fit_poisson[-day_fit:],1);
            x_fit_poisson_proj = x_fit_poisson.copy(); y_fit_poisson_proj = y_fit_poisson.copy(); dates_fit_poisson_proj = dates_fit_poisson.copy()
            for i in range(60):
                x_fit_poisson_proj.append(x_fit_poisson_proj[-1]+1);dates_fit_poisson_proj.append(dates_fit_poisson_proj[-1]+timedelta(days=1))
            for i in range(-day_fit-60,0): y_fit_poisson_proj[i]=numpy.exp(coef_poisson[0]*x_fit_poisson_proj[i]+coef_poisson[1])
            plt.plot(dates_fit_poisson_proj[-day_fit-60:],y_fit_poisson_proj[-day_fit-60:],'b');
            plt.xticks([datetime(2020,5,1),datetime(2020,6,1),datetime(2020,7,1),datetime(2020,8,1),datetime(2020,9,1),datetime(2020,10,1),datetime(2020,11,1),datetime(2020,12,1),datetime(2021,1,1),datetime(2021,2,1),datetime(2021,3,1)],['5/1','6/1','7/1','8/1','9/1','10/1','11/1','12/1','2021/1/1','2/1','3/1'])
            plt.ylim([0.01,0.1])
            plt.plot([datetime(2020,5,31),datetime(2020,5,31)],[0.045,0.05],'k')
            plt.text(datetime(2020,5,21),0.051,'Memorial Day')
            plt.plot([datetime(2020,7,4),datetime(2020,7,4)],[0.023,0.025],'k')
            plt.text(datetime(2020,6,24),0.026,'Independence Day')
            plt.plot([datetime(2020,9,9),datetime(2020,9,9)],[0.019,0.021],'k')
            plt.text(datetime(2020,8,31),0.022,'Labor Day')
            plt.plot([datetime(2020,11,28),datetime(2020,11,28)],[0.018,0.020],'k')
            plt.text(datetime(2020,11,18),0.021,'Thanksgiving')
            plt.legend(['Actual','Projected'])
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
            #if datesJHU[i] > datetime(2020,11,10): dth[i] += 550
        nday=21;count=0
        nday=10
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
            if date <= datetime(2020,12,4):
                death += ( 267140.6 - 266047. ) / 5.
                death_l += ( 267140.6 - 266047. ) / 5. * 0.8
                death_h += ( 267140.6 - 266047. ) / 5. * 1.2
            dth.append(death)
            dth_l.append(death_l)
            dth_h.append(death_h)
            date = date + timedelta(days = 1)
    
        if True:
            idx = datesJHU.index(dates[0])
            plt.plot(datesJHU[0:idx+1],dth[0:idx+1],'k')
            plt.plot(datesJHU[idx+1:],dth[idx+1:],'r')
            if enddate > datetime(2020,6,30):
                plt.xticks([datetime(2020,4,1),datetime(2020,5,1),datetime(2020,6,1),datetime(2020,7,1),datetime(2020,8,1),datetime(2020,9,1),datetime(2020,10,1),datetime(2020,11,1),datetime(2020,12,1),datetime(2021,1,1),datetime(2021,2,1),datetime(2021,3,1)],['2020/4/1','5/1','6/1','7/1','8/1','9/1','10/1','11/1','12/1','2021/1/1','2/1','3/1'])
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
            date_proj = datetime(2020,7,20)
            while date_proj < datetime(2020,12,24):
                date_proj_tmp = date_proj
                while True:
                    date = str(date_proj_tmp.year)+'-'
                    if date_proj_tmp.month<10: date += '0'
                    date += str(date_proj_tmp.month)+'-'
                    if date_proj_tmp.day<10: date = date+'0'
                    date += str(date_proj_tmp.day)
                    filename = date + '-QJHong-Encounter.csv'
                    if path.exists('../covid19-forecast-hub/data-processed/QJHong-Encounter/'+filename):
                        #print('success')
                        print(filename)
                        break
                    else:
                        date_proj_tmp -=timedelta(days = 1)
                data = pd.read_csv('../covid19-forecast-hub/data-processed/QJHong-Encounter/'+filename)
                #print(data.loc[data['type']=='point']
                date_his, deaths_proj = [], []
                for i in range(1,5):
                    date_his.append(date_proj + timedelta(days = 1) * (2 + 7 * (i-1)))
                    line = data.loc[data['type']=='point'].loc[data['target']==str(i)+' wk ahead inc death']
                    deaths_proj.append(line['value'].values[0]/7.)
                    print(line['target_end_date'])
                x_fit = [4,11,18,25]
                coef=numpy.polyfit(x_fit,deaths_proj,2);
                y_fit = [0]*4
                for i in range(len(x_fit)): y_fit[i]=coef[0]*x_fit[i]**2+coef[1]*x_fit[i]+coef[2]
                line4, = plt.plot(date_his,y_fit,color=(1,0.7,0.7))
                date_proj += timedelta(days = 21)
            date_proj = datetime(2020,7,20)
            while date_proj < datetime(2020,10,8):
                date_proj_tmp = date_proj
                while date_proj_tmp > date_proj - timedelta(days = 5):
                    date = str(date_proj_tmp.year)+'-'
                    if date_proj_tmp.month<10: date += '0'
                    date += str(date_proj_tmp.month)+'-'
                    if date_proj_tmp.day<10: date = date+'0'
                    date += str(date_proj_tmp.day)
                    filename = date + '-YYG-ParamSearch.csv'
                    if path.exists('../covid19-forecast-hub/data-processed/YYG-ParamSearch/'+filename):
                        #print('success')
                        print(filename)
                        break
                    else:
                        date_proj_tmp -=timedelta(days = 1)
                data = pd.read_csv('../covid19-forecast-hub/data-processed/YYG-ParamSearch/'+filename)
                #print(data.loc[data['type']=='point']
                date_his, deaths_proj = [], []
                for i in range(1,5):
                    date_his.append(date_proj + timedelta(days = 1) * (2 + 7 * (i-1)))
                    line = data.loc[data['type']=='point'].loc[data['location']=='US'].loc[data['target']==str(i)+' wk ahead inc death']
                    deaths_proj.append(line['value'].values[0]/7.)
                    print(line['target_end_date'])
                x_fit = [4,11,18,25]
                coef=numpy.polyfit(x_fit,deaths_proj,2);
                y_fit = [0]*4
                for i in range(len(x_fit)): y_fit[i]=coef[0]*x_fit[i]**2+coef[1]*x_fit[i]+coef[2]
                line5, = plt.plot(date_his,y_fit,color=(0.7,0.7,1))
                date_proj += timedelta(days = 21)
            line1, = plt.plot(datesJHU[:idx],dthavg[:idx],'k')
            line2, = plt.plot(datesJHU[0],dth[0+1]-dth[0],'+k')
            plt.plot(datesJHU[-2],dth[-1]-dth[-2],'.r')
            #plt.plot(datesJHU[-2],(dth_l[-1]-dth_l[-2])*0.75,'.b')
            for i in range(0,idx):
                plt.plot(datesJHU[i],dth[i+1]-dth[i],'+k')
                #print datesJHU[i],dth[i+1]-dth[i]
            for i in range(idx,len(dth)-1):
                line3, = plt.plot(datesJHU[i],dth[i+1]-dth[i],'.r')
                plt.plot(datesJHU[i],dth_l[i+1]-dth_l[i],'.r',markersize=2)
                plt.plot(datesJHU[i],dth_h[i+1]-dth_h[i],'.r',markersize=2)
                #plt.plot(datesJHU[i],(dth_l[i+1]-dth_l[i])*0.75,'.b')
            if enddate > datetime(2020,6,30):
                plt.xticks([datetime(2020,5,1),datetime(2020,6,1),datetime(2020,7,1),datetime(2020,8,1),datetime(2020,9,1),datetime(2020,10,1),datetime(2020,11,1),datetime(2020,12,1),datetime(2021,1,1),datetime(2021,2,1),datetime(2021,3,1)],['5/1','6/1','7/1','8/1','9/1','10/1','11/1','12/1','2021/1/1','2/1','3/1'])
            else:
                plt.xticks([datetime(2020,3,1),datetime(2020,4,1),datetime(2020,5,1),datetime(2020,6,1),datetime(2020,7,1)],['2020/3/1','2020/4/1','2020/5/1','2020/6/1','2020/7/1'])
            plt.xlabel('Date');plt.ylabel('Daily Deaths');plt.grid(which='both')
            plt.xlim([datetime(2020,5,1),datesJHU[-1]+timedelta(days = 1)])
            plt.ylim([0,4000])
            plt.legend([line1, line2, line3, line4, line5], ['7-day Average','Daily Deaths (JHU)','Projection','Previous Projections (QJHong @ covid forecast hub)','Previous Projections (YYG @ covid forecast hub)'])
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
            if datesJHU[i].month==1 and datesJHU[i].day%7 == 2 and datesJHU[i].day > 16 or datesJHU[i].month==2 and datesJHU[i].day%7 == 6 or datesJHU[i].month==3 and datesJHU[i].day%7 == 6:
                count+=1
                out2 = out+str(count)+' wk ahead cum death,'
                date = str(datesJHU[i].year)+'-'
                if datesJHU[i].month<10: date += '0'
                date += str(datesJHU[i].month)+'-'
                if datesJHU[i].day<10: date = date+'0'
                date += str(datesJHU[i].day)
                out2 += date
                out2 += ',US,'
                out3 = out2+'point,NA,'
                F.write(out3+str(dth[i])+'\n')
                out3 = out2+'quantile,0.01,'
                F.write(out3+str(dth_l[i]*1.16-dth[i]*0.16)+'\n')
                out3 = out2+'quantile,0.025,'
                F.write(out3+str(dth_l[i])+'\n')
                out3 = out2+'quantile,0.05,'
                F.write(out3+str(dth_l[i]*0.82+dth[i]*0.18)+'\n')
                out3 = out2+'quantile,0.10,'
                F.write(out3+str(dth_l[i]*0.64+dth[i]*0.36)+'\n')
                out3 = out2+'quantile,0.15,'
                F.write(out3+str(dth_l[i]*0.52+dth[i]*0.48)+'\n')
                out3 = out2+'quantile,0.20,'
                F.write(out3+str(dth_l[i]*0.42+dth[i]*0.58)+'\n')
                out3 = out2+'quantile,0.25,'
                F.write(out3+str(dth_l[i]*0.33+dth[i]*0.67)+'\n')
                out3 = out2+'quantile,0.30,'
                F.write(out3+str(dth_l[i]*0.26+dth[i]*0.74)+'\n')
                out3 = out2+'quantile,0.35,'
                F.write(out3+str(dth_l[i]*0.19+dth[i]*0.81)+'\n')
                out3 = out2+'quantile,0.40,'
                F.write(out3+str(dth_l[i]*0.13+dth[i]*0.87)+'\n')
                out3 = out2+'quantile,0.45,'
                F.write(out3+str(dth_l[i]*0.06+dth[i]*0.94)+'\n')
                out3 = out2+'quantile,0.5,'
                F.write(out3+str(dth_l[i]*0.00+dth[i]*1.00)+'\n')
                out3 = out2+'quantile,0.55,'
                F.write(out3+str(dth_h[i]*0.06+dth[i]*0.94)+'\n')
                out3 = out2+'quantile,0.60,'
                F.write(out3+str(dth_h[i]*0.13+dth[i]*0.87)+'\n')
                out3 = out2+'quantile,0.65,'
                F.write(out3+str(dth_h[i]*0.19+dth[i]*0.81)+'\n')
                out3 = out2+'quantile,0.70,'
                F.write(out3+str(dth_h[i]*0.26+dth[i]*0.74)+'\n')
                out3 = out2+'quantile,0.75,'
                F.write(out3+str(dth_h[i]*0.33+dth[i]*0.67)+'\n')
                out3 = out2+'quantile,0.80,'
                F.write(out3+str(dth_h[i]*0.42+dth[i]*0.58)+'\n')
                out3 = out2+'quantile,0.85,'
                F.write(out3+str(dth_h[i]*0.52+dth[i]*0.48)+'\n')
                out3 = out2+'quantile,0.90,'
                F.write(out3+str(dth_h[i]*0.64+dth[i]*0.36)+'\n')
                out3 = out2+'quantile,0.95,'
                F.write(out3+str(dth_h[i]*0.82+dth[i]*0.18)+'\n')
                out3 = out2+'quantile,0.975,'
                F.write(out3+str(dth_h[i])+'\n')
                out3 = out2+'quantile,0.99,'
                F.write(out3+str(dth_h[i]*1.16-dth[i]*0.16)+'\n')
                out2 = out+str(count)+' wk ahead inc death,'
                date = str(datesJHU[i].year)+'-'
                if datesJHU[i].month<10: date += '0'
                date += str(datesJHU[i].month)+'-'
                if datesJHU[i].day<10: date = date+'0'
                date += str(datesJHU[i].day)
                out2 += date
                out2 += ',US,'
                out3 = out2+'point,NA,'
                F.write(out3+str((dth[i]-dth0))+'\n')
                out3 = out2+'quantile,0.01,'
                F.write(out3+str((dth_l[i]-dth_l0)*1.16-(dth[i]-dth0)*0.16)+'\n')
                out3 = out2+'quantile,0.025,'
                F.write(out3+str((dth_l[i]-dth_l0))+'\n')
                out3 = out2+'quantile,0.05,'
                F.write(out3+str((dth_l[i]-dth_l0)*0.82+(dth[i]-dth0)*0.18)+'\n')
                out3 = out2+'quantile,0.10,'
                F.write(out3+str((dth_l[i]-dth_l0)*0.64+(dth[i]-dth0)*0.36)+'\n')
                out3 = out2+'quantile,0.15,'
                F.write(out3+str((dth_l[i]-dth_l0)*0.52+(dth[i]-dth0)*0.48)+'\n')
                out3 = out2+'quantile,0.20,'
                F.write(out3+str((dth_l[i]-dth_l0)*0.42+(dth[i]-dth0)*0.58)+'\n')
                out3 = out2+'quantile,0.25,'
                F.write(out3+str((dth_l[i]-dth_l0)*0.33+(dth[i]-dth0)*0.67)+'\n')
                out3 = out2+'quantile,0.30,'
                F.write(out3+str((dth_l[i]-dth_l0)*0.26+(dth[i]-dth0)*0.74)+'\n')
                out3 = out2+'quantile,0.35,'
                F.write(out3+str((dth_l[i]-dth_l0)*0.19+(dth[i]-dth0)*0.81)+'\n')
                out3 = out2+'quantile,0.40,'
                F.write(out3+str((dth_l[i]-dth_l0)*0.13+(dth[i]-dth0)*0.87)+'\n')
                out3 = out2+'quantile,0.45,'
                F.write(out3+str((dth_l[i]-dth_l0)*0.06+(dth[i]-dth0)*0.94)+'\n')
                out3 = out2+'quantile,0.5,'
                F.write(out3+str((dth_l[i]-dth_l0)*0.00+(dth[i]-dth0)*1.00)+'\n')
                out3 = out2+'quantile,0.55,'
                F.write(out3+str((dth_h[i]-dth_h0)*0.06+(dth[i]-dth0)*0.94)+'\n')
                out3 = out2+'quantile,0.60,'
                F.write(out3+str((dth_h[i]-dth_h0)*0.13+(dth[i]-dth0)*0.87)+'\n')
                out3 = out2+'quantile,0.65,'
                F.write(out3+str((dth_h[i]-dth_h0)*0.19+(dth[i]-dth0)*0.81)+'\n')
                out3 = out2+'quantile,0.70,'
                F.write(out3+str((dth_h[i]-dth_h0)*0.26+(dth[i]-dth0)*0.74)+'\n')
                out3 = out2+'quantile,0.75,'
                F.write(out3+str((dth_h[i]-dth_h0)*0.33+(dth[i]-dth0)*0.67)+'\n')
                out3 = out2+'quantile,0.80,'
                F.write(out3+str((dth_h[i]-dth_h0)*0.42+(dth[i]-dth0)*0.58)+'\n')
                out3 = out2+'quantile,0.85,'
                F.write(out3+str((dth_h[i]-dth_h0)*0.52+(dth[i]-dth0)*0.48)+'\n')
                out3 = out2+'quantile,0.90,'
                F.write(out3+str((dth_h[i]-dth_h0)*0.64+(dth[i]-dth0)*0.36)+'\n')
                out3 = out2+'quantile,0.95,'
                F.write(out3+str((dth_h[i]-dth_h0)*0.82+(dth[i]-dth0)*0.18)+'\n')
                out3 = out2+'quantile,0.975,'
                F.write(out3+str((dth_h[i]-dth_h0))+'\n')
                out3 = out2+'quantile,0.99,'
                F.write(out3+str((dth_h[i]-dth_h0)*1.16-(dth[i]-dth0)*0.16)+'\n')
                if count>0 and count<=5:
                    out2 = out+str(count)+' wk ahead inc case,'
                    date = str(datesJHU[i].year)+'-'
                    if datesJHU[i].month<10: date += '0'
                    date += str(datesJHU[i].month)+'-'
                    if datesJHU[i].day<10: date = date+'0'
                    date += str(datesJHU[i].day)
                    out2 += date
                    out2 += ',US,'
                    out3 = out2+'point,NA,'
                    F.write(out3+str(pos_out)+'\n')
                    out3 = out2+'quantile,0.025,'
                    F.write(out3+str(pos_l_out)+'\n')
                    out3 = out2+'quantile,0.10,'
                    F.write(out3+str(pos_l_out*0.64+pos_out*0.36)+'\n')
                    out3 = out2+'quantile,0.25,'
                    F.write(out3+str(pos_l_out*0.33+pos_out*0.67)+'\n')
                    out3 = out2+'quantile,0.5,'
                    F.write(out3+str(pos_l_out*0.00+pos_out*1.00)+'\n')
                    out3 = out2+'quantile,0.75,'
                    F.write(out3+str(pos_h_out*0.33+pos_out*0.67)+'\n')
                    out3 = out2+'quantile,0.90,'
                    F.write(out3+str(pos_h_out*0.64+pos_out*0.36)+'\n')
                    out3 = out2+'quantile,0.975,'
                    F.write(out3+str(pos_h_out*1.00+pos_out*0.00)+'\n')
                pos_out = 0
                pos_l_out = 0
                pos_h_out = 0
            if  datesJHU[i].month==1 and datesJHU[i].day%7 == 2 or datesJHU[i].month==2 and datesJHU[i].day%7 == 6 or datesJHU[i].month==3 and datesJHU[i].day%7 == 6:
                pos_out = 0
                pos_l_out = 0
                pos_h_out = 0
                dth_l0 = dth_l[i]
                dth_h0 = dth_h[i]
                dth0 = dth[i]
