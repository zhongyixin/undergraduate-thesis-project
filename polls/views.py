from django.http import HttpResponse,HttpResponseRedirect
from django.shortcuts import render
from .models import crossinfo_8,crossinfo_9,crossinfo_13,crossinfo_14,crossinfo_15,crossinfo_19,crossinfo_20,crossinfo_22,crossinfo_27,crossinfo_68,crossinfo_70,crossinfo_71,crossinfo_79,crossinfo_106,crossinfo_157,crossinfo_194,crossinfo_195,crossinfo_199,crossinfo_201,crossinfo_263,crossinfo_319,crossinfo_320,crossinfo_321,crossinfo_336,crossinfo_405,crossinfo_471,crossinfo_472,crossinfo_474,crossinfo_475,crossinfo_564,crossinfo_565,crossinfo_647,crossinfo_754,crossinfo_755,crossinfo_920,crossinfo_968     
#因python版本问题，此处models之前必须加"."
from .models import pl_22,pl_8,pl_14,pl_19
import json
import MySQLdb
import collections
from .models import address_info
from django.db.models import Sum,Count,Max,Min,Avg,Q
from django.http import JsonResponse
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler,LabelEncoder,label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn import svm
#train_test_split:将数据集随机分成训练集和测试集的函数
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, MaxPooling3D, Flatten, Input, Conv2D, Conv3D, concatenate, Dropout, Reshape,Permute,merge
from keras.optimizers import RMSprop,sgd,Adam,Adagrad,Adadelta
from keras.layers.recurrent import LSTM,GRU
from keras.models import Model
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from keras.models import load_model
import random
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_hastie_10_2
from sklearn.feature_selection import SelectKBest,chi2,RFE
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
from sklearn.svm import  LinearSVC
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_absolute_error,mean_squared_error,median_absolute_error,r2_score
from keras.models import model_from_json

def index(request):
    return render(request, 'index.html')                            
def login_action(request):
    if request.method == 'POST':                                            #判断是否为post提交方式
        username = request.POST.get('username', '')                         #通过post.get()方法获取输入的用户名及密码
        password =request.POST.get('password', '')

        if username == 'zyx' and password == '123':                        #判断用户名及密码是否正确
            return HttpResponseRedirect('/map_page/')                    #如果正确，（这里调用另一个函数，实现登陆成功页面独立，使用HttpResponseRedirect()方法实现
        else:
            return render(request,'index.html',{'error':'用户名或密码错误！'})#不正确，通过render(request,"index.html")方法在error标签处显示错误提示

def mailbox(request):
    return render(request, 'mailbox.html')

def introduction(request):
    return render(request, 'introduction.html')


def dashboard_2(request): 
    years = range(2018, 2020)
    months = range(1, 13)
    days = range(1, 32)
    ddate = '11/01/2018'
    cross_lst = crossinfo_8.objects.filter(
        Q(detectorID='2'),
        Q(date='11/04/2018')
    )
    cross='8 延安路-体育场路'
    detector=2
    ds_lst = list(cross_lst.values_list('DS',flat=True))
    actualVolume_lst = list(cross_lst.values_list('actualVolume',flat=True))
    pl_lst = list(cross_lst.values_list('PL',flat=True))
    time_lst = list(cross_lst.values_list('time',flat=True))
    crosslst = crossinfo_8.objects.filter(detectorID=detector)
    dic0 = crosslst.aggregate(Max('DS'))
    dic1 = crosslst.aggregate(Max('actualVolume'))
    for k in dic0:
        M_MAX_DS=dic0[k]
    for k in dic1:
        M_MAX_actualVolume=dic1[k]
    dic2 = cross_lst.aggregate(Max('DS'))         
    for k in dic2:
        MAX_DS=dic2[k]
    dic3 = cross_lst.aggregate(Max('actualVolume'))   
    for k in dic3:
        MAX_actualVolume=dic3[k]      
    return render(request,"dashboard_2.html",{"years":years,"months":months,"days":days,"ds_lst":ds_lst,"pl_lst":pl_lst,
                                              "actualVolume_lst":actualVolume_lst,"time_lst":time_lst,
                                              "ddate":ddate,"cross":cross,"detector":detector,
                                              "M_MAX_DS":M_MAX_DS,"M_MAX_actualVolume":M_MAX_actualVolume,
                                              "MAX_DS":MAX_DS,"MAX_actualVolume":MAX_actualVolume})                               

def graph_flot(request): 
    return render(request,"graph_flot.html")

def predict(request): 
    jingdu = 120.187609
    weidu = 30.30000
    #jingdu = 120.178132
    #weidu = 30.263773
    pre = 0
    return render(request,"predict.html",{"jingdu":jingdu,"weidu":weidu,"pre":pre})

def predict_m(request): 
    jingdu1 = 120.187609
    weidu1 = 30.30000
    jingdu2 = 120.187609
    weidu2 = 30.30000
    pre1 = 0
    pre2 = 0
    return render(request,"predict_m.html",{"jingdu1":jingdu1,"weidu1":weidu1,"jingdu2":jingdu2,"weidu2":weidu2,"pre1":pre1,"pre2":pre2})

def map_page(request):
    return render(request,"map_page.html")
    
def test(request):
    years = range(2018, 2020)
    months = range(1, 13)
    days = range(1, 32)
        #cross_lst = crossinfo_8.objects.all()[0:6]
    cross_lst = crossinfo_8.objects.filter(
        Q(detectorID='2'),
        Q(date='11/01/2018')
    )
    ds_lst = list(crossinfo_8.objects.values_list('DS',flat=True)[0:7])
    actualVolume_lst = list(crossinfo_8.objects.values_list('actualVolume',flat=True)[0:7])
    time_lst = list(crossinfo_8.objects.values_list('time',flat=True)[0:7])
    return render(request,"test.html",{"years":years,"months":months,"days":days,
                                       "cross_lst":cross_lst,"ds_lst":ds_lst,"actualVolume_lst":actualVolume_lst,"time_lst":time_lst
                                       })
def search(request):
    year = str(request.POST.get('byear'))
    month = str(request.POST.get('bmonth'))
    day = str(request.POST.get('bday'))
    ddate = month+'/0'+day+'/'+year 
        #cross_lst = crossinfo_8.objects.all()[0:6]
    cross_lst = crossinfo_8.objects.filter(
        Q(detectorID = '2'),
        Q(date = '11/02/2018')
    )
    ds_lst =[]
    actualVolume_lst=[]
    time_lst=[]
    for i in cross_lst:
        ds_lst.append(i.DS)
        actualVolume_lst.append(i.actualVolume)
        time_lst.append(i.time)
    #ds_lst = list(cross_lst.values_list('DS',flat=True)[0:7])
    #actualVolume_lst = list(cross_lst.values_list('actualVolume',flat=True)[0:7])
    #time_lst = list(cross_lst.values_list('time',flat=True)[0:7])
    #ret = {'time_lst':time_lst}
    ret = {'ds_lst':ds_lst,'actualVolume_lst':actualVolume_lst,'time_lst':time_lst}
    #return JsonResponse(ret)
    return HttpResponse(json.dumps(ret))

def search_forms(request):
    years = range(2018, 2020)
    months = range(1, 13)
    days = range(1, 32)
    if request.method == 'POST': 
        ddate = request.POST.get('bdate','')
        cross = request.POST.get('bcross','')
        detector = request.POST.get('bdetector','')
        if cross == '8 延安路-体育场路':
            cross_lst = crossinfo_8.objects.filter(
                Q(detectorID = detector),
                Q(date = ddate)
            )
            crosslst = crossinfo_8.objects.filter(detectorID=detector)
        elif cross == '9 中山北路-体育场路':
            cross_lst = crossinfo_9.objects.filter(
                Q(detectorID = detector),
                Q(date = ddate)
            )
            crosslst = crossinfo_9.objects.filter(detectorID=detector)
        elif cross == '13 武林路-凤起路':
            cross_lst = crossinfo_13.objects.filter(
                Q(detectorID = detector),
                Q(date = ddate)
            )
            crosslst = crossinfo_13.objects.filter(detectorID=detector)
        elif cross == '14 延安路-凤起路':
            cross_lst = crossinfo_14.objects.filter(
                Q(detectorID = detector),
                Q(date = ddate)
            )
            crosslst = crossinfo_14.objects.filter(detectorID=detector)
        elif cross == '15 中山北路-凤起路':
            cross_lst = crossinfo_15.objects.filter(
                Q(detectorID = detector),
                Q(date = ddate)
            )
            crosslst = crossinfo_15.objects.filter(detectorID=detector)
        elif cross == '19 延安路-庆春路':
            cross_lst = crossinfo_19.objects.filter(
                Q(detectorID = detector),
                Q(date = ddate)
            )
            crosslst = crossinfo_19.objects.filter(detectorID=detector)
        elif cross == '20 浣纱路-庆春路':
            cross_lst = crossinfo_20.objects.filter(
                Q(detectorID = detector),
                Q(date = ddate)
            )
            crosslst = crossinfo_20.objects.filter(detectorID=detector)
        elif cross == '22 中河中路-庆春路':
            cross_lst = crossinfo_22.objects.filter(
                Q(detectorID = detector),
                Q(date = ddate)
            )
            crosslst = crossinfo_22.objects.filter(detectorID=detector)
        elif cross == '27 建国北路-凤起路':
            cross_lst = crossinfo_27.objects.filter(
                Q(detectorID = detector),
                Q(date = ddate)
            )
            crosslst = crossinfo_27.objects.filter(detectorID=detector)
        elif cross == '68 体育场路-环城东路':
            cross_lst = crossinfo_68.objects.filter(
                Q(detectorID = detector),
                Q(date = ddate)
            )
            crosslst = crossinfo_68.objects.filter(detectorID=detector)
        elif cross == '70 建国北路-朝晖路':
            cross_lst = crossinfo_70.objects.filter(
                Q(detectorID = detector),
                Q(date = ddate)
            )
            crosslst = crossinfo_70.objects.filter(detectorID=detector)
        elif cross == '71 文晖路-河东路':
            cross_lst = crossinfo_71.objects.filter(
                Q(detectorID = detector),
                Q(date = ddate)
            )
            crosslst = crossinfo_71.objects.filter(detectorID=detector)
        elif cross == '79 新市街-上塘路':
            cross_lst = crossinfo_79.objects.filter(
                Q(detectorID = detector),
                Q(date = ddate)
            )
            crosslst = crossinfo_79.objects.filter(detectorID=detector)
        elif cross == '106 环城北路-朝晖路':
            cross_lst = crossinfo_106.objects.filter(
                Q(detectorID = detector),
                Q(date = ddate)
            )
            crosslst = crossinfo_106.objects.filter(detectorID=detector)
        elif cross == '157 上塘路-朝晖路':
            cross_lst = crossinfo_157.objects.filter(
                Q(detectorID = detector),
                Q(date = ddate)
            )
            crosslst = crossinfo_157.objects.filter(detectorID=detector)
        elif cross == '194 绍兴路-德胜路':
            cross_lst = crossinfo_194.objects.filter(
                Q(detectorID = detector),
                Q(date = ddate)
            )
            crosslst = crossinfo_194.objects.filter(detectorID=detector)
        elif cross == '195 绍兴路400弄-绍兴路':
            cross_lst = crossinfo_195.objects.filter(
                Q(detectorID = detector),
                Q(date = ddate)
            )
            crosslst = crossinfo_195.objects.filter(detectorID=detector)
        elif cross == '199 东新路-德胜路':
            cross_lst = crossinfo_199.objects.filter(
                Q(detectorID = detector),
                Q(date = ddate)
            )
            crosslst = crossinfo_199.objects.filter(detectorID=detector)
        elif cross == '201 西文街-东新路':
            cross_lst = crossinfo_201.objects.filter(
                Q(detectorID = detector),
                Q(date = ddate)
            )
            crosslst = crossinfo_201.objects.filter(detectorID=detector)
        elif cross == '263 中诸葛路-德胜中路':
            cross_lst = crossinfo_263.objects.filter(
                Q(detectorID = detector),
                Q(date = ddate)
            )
            crosslst = crossinfo_263.objects.filter(detectorID=detector)
        elif cross == '319 石大路-华中路':
            cross_lst = crossinfo_319.objects.filter(
                Q(detectorID = detector),
                Q(date = ddate)
            )
            crosslst = crossinfo_319.objects.filter(detectorID=detector)
        elif cross == '320 石祥路-长浜路':
            cross_lst = crossinfo_320.objects.filter(
                Q(detectorID = detector),
                Q(date = ddate)
            )
            crosslst = crossinfo_320.objects.filter(detectorID=detector)
        elif cross == '321 石祥路-东新路':
            cross_lst = crossinfo_321.objects.filter(
                Q(detectorID = detector),
                Q(date = ddate)
            )
            crosslst = crossinfo_321.objects.filter(detectorID=detector)
        elif cross == '336 新汇路-石大路':
            cross_lst = crossinfo_336.objects.filter(
                Q(detectorID = detector),
                Q(date = ddate)
            )
            crosslst = crossinfo_336.objects.filter(detectorID=detector)
        elif cross == '405 中山北路-朝晖路':
            cross_lst = crossinfo_405.objects.filter(
                Q(detectorID = detector),
                Q(date = ddate)
            )
            crosslst = crossinfo_405.objects.filter(detectorID=detector)
        elif cross == '471 颜三路-碧桃巷':
            cross_lst = crossinfo_471.objects.filter(
                Q(detectorID = detector),
                Q(date = ddate)
            )
            crosslst = crossinfo_471.objects.filter(detectorID=detector)
        elif cross == '472 颜三路-白石巷':
            cross_lst = crossinfo_472.objects.filter(
                Q(detectorID = detector),
                Q(date = ddate)
            )
            crosslst = crossinfo_472.objects.filter(detectorID=detector)
        elif cross == '474 西文街-白石巷':
            cross_lst = crossinfo_474.objects.filter(
                Q(detectorID = detector),
                Q(date = ddate)
            )
            crosslst = crossinfo_474.objects.filter(detectorID=detector)
        elif cross == '475 香积寺路-碧桃巷':
            cross_lst = crossinfo_475.objects.filter(
                Q(detectorID = detector),
                Q(date = ddate)
            )
            crosslst = crossinfo_475.objects.filter(detectorID=detector)
        elif cross == '564 新汇路-八角亭街':
            cross_lst = crossinfo_564.objects.filter(
                Q(detectorID = detector),
                Q(date = ddate)
            )
            crosslst = crossinfo_564.objects.filter(detectorID=detector)
        elif cross == '565 新汇路-长华街':
            cross_lst = crossinfo_565.objects.filter(
                Q(detectorID = detector),
                Q(date = ddate)
            )
            crosslst = crossinfo_565.objects.filter(detectorID=detector)
        elif cross == '647 重机巷-新天地街':
            cross_lst = crossinfo_647.objects.filter(
                Q(detectorID = detector),
                Q(date = ddate)
            )
            crosslst = crossinfo_647.objects.filter(detectorID=detector)
        elif cross == '754 华中南路-长城街':
            cross_lst = crossinfo_754.objects.filter(
                Q(detectorID = detector),
                Q(date = ddate)
            )
            crosslst = crossinfo_754.objects.filter(detectorID=detector)
        elif cross == '755 俞章路-华中南路':
            cross_lst = crossinfo_755.objects.filter(
                Q(detectorID = detector),
                Q(date = ddate)
            )
            crosslst = crossinfo_755.objects.filter(detectorID=detector)
        elif cross == '920 东文路-安桥路':
            cross_lst = crossinfo_920.objects.filter(
                Q(detectorID = detector),
                Q(date = ddate)
            )
            crosslst = crossinfo_920.objects.filter(detectorID=detector)
        elif cross == '968 载歌巷-香积寺路':
            cross_lst = crossinfo_968.objects.filter(
                Q(detectorID = detector),
                Q(date = ddate)
            )
            crosslst = crossinfo_968.objects.filter(detectorID=detector)
        ds_lst = list(cross_lst.values_list('DS',flat=True))
        actualVolume_lst = list(cross_lst.values_list('actualVolume',flat=True))
        pl_lst = list(cross_lst.values_list('PL',flat=True))
        time_lst = list(cross_lst.values_list('time',flat=True))
        
        dic0 = crosslst.aggregate(Max('DS'))
        dic1 = crosslst.aggregate(Max('actualVolume'))
        for k in dic0:
            M_MAX_DS=dic0[k]
        for k in dic1:
            M_MAX_actualVolume=dic1[k]
        dic2 = cross_lst.aggregate(Max('DS'))         
        for k in dic2:
            MAX_DS=dic2[k]
        dic3 = cross_lst.aggregate(Max('actualVolume'))   
        for k in dic3:
            MAX_actualVolume=dic3[k]
        precent1=round(int(MAX_DS)/int(M_MAX_DS)*100,2)
        precent2=round(int(MAX_actualVolume)/int(M_MAX_actualVolume)*100,2)                  
        return render(request,"dashboard_2.html",{"years":years,"months":months,"days":days,
                                                  "cross_lst":cross_lst,"ds_lst":ds_lst,"actualVolume_lst":actualVolume_lst,"pl_lst":pl_lst,"time_lst":time_lst,
                                                  "ddate":ddate,"cross":cross,"detector":detector,
                                                  "M_MAX_DS":M_MAX_DS,"M_MAX_actualVolume":M_MAX_actualVolume,
                                                  "MAX_DS":MAX_DS,"MAX_actualVolume":MAX_actualVolume,
                                                  "precent1":precent1,"precent2":precent2})

def Process(a,b,f):
    num_road=a
    dataa=[]
    for i in range(f.shape[0]):
#    if f[i,2]==11 or f[i,2]==12 or f[i,2]==13:
        if f[i,2]==3 or f[i,2]==4 or f[i,2]==9 or f[i,2]==12:
            continue
        dataa.append(f[i,:])
    dataa=np.array(dataa)
    dataa=dataa[:,[8,9,12]]
    data=np.zeros((dataa.shape[0],dataa.shape[1]))
    for i in range(dataa.shape[0]):
        data[i,0]=dataa[i,0]
        data[i,1]=dataa[i,1]
        if dataa[i,2]==1.3:
            data[i,2]=1
        if dataa[i,2]==2.3:
            data[i,2]=2
        if dataa[i,2]==3.3:
            data[i,2]=3
        if dataa[i,2]==4.3:
            data[i,2]=4
    data_in=[]
    for i in range(int(data.shape[0]/num_road)):
       data_in.append(data[i*num_road:(i+1)*num_road,:])
    data_in=np.array(data_in)
    data_in1=data_in[:,:,0]
    data_in2=data_in[:,:,1]
    label=data_in[:,0,2]
    scaler=MinMaxScaler()
    data_in1_=scaler.fit_transform(data_in1)
    data_in2_=scaler.fit_transform(data_in2)
    mid=[]
    in_data=[]
    for i in range(label.shape[0]):
        mid.append(data_in1_[i,:])
        mid.append(data_in2_[i,:])
        mid=np.array(mid)
        mid=mid.reshape(mid.shape[1],mid.shape[0])
        in_data.append(mid)
        mid=[]
    in_data=np.array(in_data)
    in_data_=in_data.reshape(in_data.shape[0],-1)
    lags=b
    label=label.reshape(label.shape[0],-1)
    num_class=4
    encoder=LabelEncoder()
    encoded_label=encoder.fit_transform(label)
    label_in=np_utils.to_categorical(encoded_label,num_class)
    input_cnn,input_lstm,input_label=[],[],[]
    for i in range(in_data.shape[0]-lags):
        input_cnn.append(in_data[i:i+lags,:,:])
        input_lstm.append(in_data_[i:i+lags,:])
        input_label.append(label_in[i:i+lags,:])
    input_cnn=np.array(input_cnn)
#    input_cnn-input_cnn.reshape(input_cnn.shape[0],b,a,2)
    input_lstm=np.array(input_lstm)
    input_label=np.array(input_label)
    input_cnn=input_cnn[-1,:,:,:]
    input_cnn=input_cnn.reshape(1,input_cnn.shape[0],input_cnn.shape[1],input_cnn.shape[2])
    input_lstm=input_lstm[-1,:,:]
    input_lstm=input_lstm.reshape(1,input_lstm.shape[0],input_lstm.shape[1])
    input_label=input_label[-1,:,:]
    input_label=input_label.reshape(1,input_label.shape[0],input_label.shape[1])
    return input_cnn,input_lstm,input_label

def Process_D(a,b,f_1,f_2):
    lags=b
    num_feature=2
    num_cross=2
    num_lane=a
    dataa1=[]
    for i in range(f_1.shape[0]):
        dataa1.append(f_1[i,:])
    dataa1=np.array(dataa1)
    dataa1=dataa1[:,[8,9,12]]
    data1=np.zeros((dataa1.shape[0],dataa1.shape[1]))
    for i in range(dataa1.shape[0]):
        data1[i,0]=dataa1[i,0]
        data1[i,1]=dataa1[i,1]
        if dataa1[i,2]==1.1 or dataa1[i,2]==1.3 or dataa1[i,2]==1.2 or dataa1[i,2]==1.4:
            data1[i,2]=1
        if dataa1[i,2]==2.1 or dataa1[i,2]==2.3 or dataa1[i,2]==2.2 or dataa1[i,2]==2.4:
            data1[i,2]=2
        if dataa1[i,2]==3.1 or dataa1[i,2]==3.3 or dataa1[i,2]==3.2 or dataa1[i,2]==3.4:
            data1[i,2]=3
        if dataa1[i,2]==4.1 or dataa1[i,2]==4.3 or dataa1[i,2]==4.2 or dataa1[i,2]==4.4:
            data1[i,2]=4
    data_in1=[]
    for i in range(int(data1.shape[0]/num_lane)):
        data_in1.append(data1[i*num_lane:(i+1)*num_lane,:])
    data_in1=np.array(data_in1)
    data_in1=data_in1[:,:-1,:]
    
    dataa2=[]
    for i in range(f_2.shape[0]):
        dataa2.append(f_2[i,:])
    dataa2=np.array(dataa2)
    dataa2=dataa2[:,[8,9,12]]
    data2=np.zeros((dataa2.shape[0],dataa2.shape[1]))
    for i in range(dataa2.shape[0]):
        data2[i,0]=dataa2[i,0]
        data2[i,1]=dataa2[i,1]
        if dataa2[i,2]==1.1 or dataa2[i,2]==1.3 or dataa2[i,2]==1.2 or dataa2[i,2]==1.4:
            data2[i,2]=1
        if dataa2[i,2]==2.1 or dataa2[i,2]==2.3 or dataa2[i,2]==2.2 or dataa2[i,2]==2.4:
            data2[i,2]=2
        if dataa2[i,2]==3.1 or dataa2[i,2]==3.3 or dataa2[i,2]==3.2 or dataa2[i,2]==3.4:
            data2[i,2]=3
        if dataa2[i,2]==4.1 or dataa2[i,2]==4.3 or dataa2[i,2]==4.2 or dataa2[i,2]==4.4:
            data2[i,2]=4
    data_in2=[]
    for i in range(int(data2.shape[0]/num_lane)):
        data_in2.append(data2[i*num_lane:(i+1)*num_lane,:])
    data_in2=np.array(data_in2)
    data_in2=data_in2[:,:-1,:]
    data_cnn=[]
    data_mid=[]
    for i in range(data_in1.shape[0]):
        data_mid.append(data_in1[i,:,:])
        data_mid.append(data_in2[i,:,:])
        data_mid=np.array(data_mid)
        data_cnn.append(data_mid)
        data_mid=[]
    data_cnn=np.array(data_cnn)
    data_lstm=data_cnn[:,:,:,:-1]
    data_lstm=data_lstm.reshape(data_lstm.shape[0],-1)
    scaler=MinMaxScaler()
    data_lstm=scaler.fit_transform(data_lstm)
    data_cnn_=data_lstm.reshape(data_cnn.shape[0],data_cnn.shape[1],data_cnn.shape[2],data_cnn.shape[3]-1)
    target=data_cnn[:,:,0,2]
    input_lstm,input_cnn=[],[]
    for i in range(data_lstm.shape[0]-lags-1):
        input_lstm.append(data_lstm[i:i+lags,:])
        input_cnn.append(data_cnn_[i:i+lags,:,:,:].reshape(-1,data_cnn_.shape[2],data_cnn_.shape[3]))
    input_lstm=np.array(input_lstm)
    input_cnn=np.array(input_cnn)
    input_cnn=input_cnn.reshape(input_cnn.shape[0],lags,num_cross*(num_lane-1),num_feature)
    mid_num=(int)(input_lstm.shape[2]/2)
    input_lstm1=input_lstm[:,:,:mid_num]
    input_lstm2=input_lstm[:,:,mid_num:]
    target1=target[:,0]
    target2=target[:,1]
    num_class=4
    encoder=LabelEncoder()
    encoded_label1=encoder.fit_transform(target1)
    target1_onehot=np_utils.to_categorical(encoded_label1,num_class)
    encoded_label2=encoder.fit_transform(target2)
    target2_onehot=np_utils.to_categorical(encoded_label2,num_class)
    input_label1,input_label2=[],[]
    for i in range(target1.shape[0]-lags-1):
        input_label1.append(target1_onehot[i:i+lags,:])
        input_label2.append(target2_onehot[i:i+lags,:])
    input_label1=np.array(input_label1)
    input_label2=np.array(input_label2)
    input_cnn=input_cnn[input_cnn.shape[0]-1:input_cnn.shape[0],:,:,:]
    input_lstm1=input_lstm1[input_lstm1.shape[0]-1:input_lstm1.shape[0],:,:]
    input_lstm2=input_lstm2[input_lstm2.shape[0]-1:input_lstm2.shape[0],:,:]
    input_label1=input_label1[input_label1.shape[0]-1:input_label1.shape[0],:,:]
    input_label1=input_label2[input_label2.shape[0]-1:input_label2.shape[0],:,:]
    return input_cnn,input_lstm1,input_lstm2,input_label1,input_label2

def predict_1(request):
    if request.method == 'POST': 
        cross = request.POST.get('bcross','')
        if cross == '8':
            model = model_from_json(open('E:/大四下/毕设/预测模型/8/model_architecture.json').read())
            model.load_weights('E:/大四下/毕设/预测模型/8/model_weights.h5')
            file_path = 'E:\大四下\毕设\预测模型\解析1\\8单表单.xlsx'
            num_road=9
            f = pd.read_excel(file_path)
            f=np.array(f)
            lags=8
            test_cnn,test_lstm,test_label=Process(num_road,lags,f)
            act_2 = Model(inputs=model.input,outputs=model.get_layer('out_2').output)
            out_2 = act_2.predict([test_cnn,test_lstm,test_label])
            label_pre=np.argmax(out_2,axis=1)
            pre=label_pre[0]+1
            jingdu = 120.169922
            weidu = 30.276049
            result = pl_8.objects.get(id=pre)
        elif cross == '22':  
            model = model_from_json(open('E:/大四下/毕设/预测模型/22/model_architecture.json').read())
            model.load_weights('E:/大四下/毕设/预测模型/22/model_weights.h5')                        
            file_path = 'E:\大四下\毕设\预测模型\解析1\\22单表单.xlsx'
            num_road=8
            f = pd.read_excel(file_path)
            f=np.array(f)
            lags=8
            test_cnn,test_lstm,test_label=Process(num_road,lags,f)
            act_2 = Model(inputs=model.input,outputs=model.get_layer('out_2').output)
            out_2 = act_2.predict([test_cnn,test_lstm,test_label])
            label_pre=np.argmax(out_2,axis=1)
            pre=label_pre[0]+1
            jingdu = 120.178132
            weidu = 30.263773
            result= pl_22.objects.get(id=pre)
    return render(request,"predict.html",{"result":result,"jingdu":jingdu,"weidu":weidu,"pre":pre,"cross":cross})

def predict_2(request):
    if request.method == 'POST': 
        cross = request.POST.getlist('bcross')
        if cross[0] == '14' and cross[1] == '19':
            model_1 = model_from_json(open('E:/大四下/毕设/预测模型/14+19/model_architecture_14+19.json').read())
            model_1.load_weights('E:/大四下/毕设/预测模型/14+19/model_weights_14+19.h5')
            model_2 = model_from_json(open('E:/大四下/毕设/预测模型/14+19/model_architecture_14+19_2.json').read())
            model_2.load_weights('E:/大四下/毕设/预测模型/14+19/model_weights_14+19_2.h5')
            file_path1 = 'E:\大四下\毕设\预测模型\解析1\\14单表单.xlsx'
            f1 = pd.read_excel(file_path1)
            f1=np.array(f1)
            file_path2 = 'E:\大四下\毕设\预测模型\解析1\\19单表单.xlsx'
            f2 = pd.read_excel(file_path2)
            f2=np.array(f2)
            num_lane=17
            lags=8
            input_cnn,input_lstm1,input_lstm2,input_label1,input_label2=Process_D(num_lane,lags,f1,f2)
            act_1 = Model(inputs=model_1.input,outputs=model_1.get_layer('output').output)
            out_1 = act_1.predict([input_cnn,input_lstm1,input_lstm2,input_label1,input_label2])
            label_pre1=np.argmax(out_1,axis=1)
            pre1=label_pre1[0]+1
            act_2 = Model(inputs=model_2.input,outputs=model_2.get_layer('output').output)
            out_2 = act_2.predict([input_cnn,input_lstm1,input_lstm2,input_label1,input_label2])
            label_pre2=np.argmax(out_2,axis=1)
            pre2=label_pre2[0]+1
            jingdu1 = 120.170173
            weidu1 = 30.269684
            jingdu2 = 120.170546
            weidu2 = 30.264225
            result1 = pl_14.objects.get(id=pre1)
            result2 = pl_19.objects.get(id=pre2) 
            return render(request,"predict_m.html",{"jingdu1":jingdu1,"weidu1":weidu1,"jingdu2":jingdu2,"weidu2":weidu2,
                                                    "pre1":pre1,"cross":cross,"pre2":pre2,"result1":result1,"result2":result2})