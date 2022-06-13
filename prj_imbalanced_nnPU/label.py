import numpy as np
import pandas as pd
import os
import tarfile
import wfdb
import pywt
import ast

import torch
from sklearn.datasets import fetch_openml

def del_noise(data, level):

    coeffs = pywt.wavedec(data=data, wavelet='db5', level=level)
    if level == 6:
        cA6,cD6,cD5,cD4,cD3,cD2,cD1=coeffs
        threshold=(np.median(np.abs(cD1))/0.6745)*(np.sqrt(2*np.log(len(cD1))))

        cD1.fill(0)
    else:
        cA9,cD9,cD8,cD7,cD6,cD5,cD4,cD3,cD2,cD1=coeffs
        threshold=(np.median(np.abs(cD1))/0.6745)*(np.sqrt(2*np.log(len(cD1))))

        cD1.fill(0)
        cD2.fill(0)

    for i in range(0,len(coeffs)-2):
        coeffs[i]=pywt.threshold(coeffs[i],threshold)

    rdata = pywt.waverec(coeffs=coeffs,wavelet='db5')

    return rdata

def get_aha_st_data(number,X_data,Y_data,ecgClassSet,channel,path):
    
    print("Reading "+number+" recording...")
    record,_=wfdb.rdsamp(path+"/"+number,channels=[channel])
    annotation=wfdb.rdann(path+"/"+number, 'atr')

    data=record.flatten()

    #小波去噪
    clean_data=del_noise(data=data,level=9)

    sam_location=annotation.sample
    sym_class=annotation.symbol

    #去掉前后不稳定数据
    start=10
    end=5
    i=start
    j=len(annotation.symbol)-end

    #只选NAVLR,其他去除
    #x_data截取R波前后300
    #y_data按NAVLR按顺序转换到01234
    while i<j:
        try:
            label=ecgClassSet.index(sym_class[i])
            x_train=clean_data[sam_location[i]-99:sam_location[i]+101]
            X_data.append(x_train)
            Y_data.append(label)
            i+=1
        except ValueError:
            i+=1
    #不需要return，因为数据已经在X_data和Y_data里面
    return

def get_mit_data(number,X_data,Y_data,ecgClassSet,channel,path):
	
	print("Reading "+number+" recording...")
	record,_=wfdb.rdsamp(path+"/"+number,channels=[channel])
	annotation=wfdb.rdann(path+"/"+number, 'atr')

	data=record.flatten()

	#小波去噪
	clean_data=del_noise(data=data,level=9)

	sam_location=annotation.sample
	sym_class=annotation.symbol

	#去掉前后不稳定数据
	start=10
	end=5
	i=start
	j=len(annotation.symbol)-end

	#只选NAVLR,其他去除
	#x_data截取R波前后300
	#y_data按NAVLR按顺序转换到01234
	while i<j:
		try:
			label=ecgClassSet.index(sym_class[i])
			x_train=clean_data[sam_location[i]-99:sam_location[i]+201]
			X_data.append(x_train)
			Y_data.append(label)
			i+=1
		except ValueError:
			i+=1
	#不需要return，因为数据已经在X_data和Y_data里面
	return

def get_ptb_data(df,sampling_rate,path,channel):
    if sampling_rate==100:
        data=[wfdb.rdsamp(path+f,channels=[channel]) for f in df.filename_lr]
    else:
        data=[wfdb.rdsamp(path+f,channels=[channel]) for f in df.filename_hr]

    data=np.array([meta for meta in data])  #data.p_signals get signals
    return data

def load_aha():
    number_set=[]

    path='/home/Student/s4570079/aha-1.0.0/aha1.0.0'
    os.chdir(path)
    for file in os.listdir():
        if(file.endswith(".dat")):
            number_set.append(file.split(".")[0])

    dataSet=[]
    labelSet=[]
    ecgClassSet=['N','V']
    channel=0
    for n in number_set:
        get_aha_st_data(n,dataSet,labelSet,ecgClassSet,channel,path)
    print("Read all files done")
    dataSet=np.array(dataSet).reshape(-1,200)
    labelSet=np.array(labelSet).reshape(-1,1)

    train_dataset=np.hstack((dataSet,labelSet))

    X=train_dataset[:,:200].reshape(-1,200,1)
    Y=train_dataset[:,200]

    shuffle_index = np.random.permutation(len(X))
    test_length = int(0.1 * len(shuffle_index)) #Ratio=0.1
    test_index = shuffle_index[:test_length]
    train_index = shuffle_index[test_length:]
    X_train, Y_train = X[train_index], Y[train_index]
    X_test, Y_test = X[test_index], Y[test_index]

    '''test_length = int(0.1 * len(X)) #Ratio=0.1
    X_train, Y_train = X[test_length:], Y[test_length:]
    X_test, Y_test = X[:test_length], Y[:test_length]'''
    

    print(np.shape(X_test))
    print(np.shape(Y_test))
    print(np.shape(X_train))
    print(np.shape(Y_train))

    return (X_train, Y_train), (X_test, Y_test)

def load_mit():
	number_set=[]

	path='/home/Student/s4570079/mit-1.0.0/mit-bih1.0.0'
	os.chdir(path)
	for file in os.listdir():
		if(file.endswith(".dat")):
			number_set.append(file.split(".")[0])

	dataSet=[]
	labelSet=[]
	ecgClassSet=['N','A','V','L','R']
	channel=0
	for n in number_set:
		get_mit_data(n,dataSet,labelSet,ecgClassSet,channel,path)
	print("Read all files done")
	dataSet=np.array(dataSet).reshape(-1,300)
	labelSet=np.array(labelSet).reshape(-1,1)

	train_dataset=np.hstack((dataSet,labelSet))

	X=train_dataset[:,:300].reshape(-1,300,1)
	Y=train_dataset[:,300]

	shuffle_index = np.random.permutation(len(X))
	test_length = int(0.3 * len(shuffle_index)) #Ratio=0.3
	test_index = shuffle_index[:test_length]
	train_index = shuffle_index[test_length:]
	X_train, Y_train = X[train_index], Y[train_index]
	X_test, Y_test = X[test_index], Y[test_index]

	'''test_length = int(0.3 * len(X)) #Ratio=0.3
	X_train, Y_train = X[test_length:], Y[test_length:]
	X_test, Y_test = X[:test_length], Y[:test_length]'''

	print("X_train:", np.shape(X_train))
	print("Y_train:", np.shape(Y_train))
	print("X_test:", np.shape(X_test))
	print("Y_test: ",np.shape(Y_test))

	return (X_train, Y_train), (X_test, Y_test)

def load_st():
    number_set=[]

    path='/home/Student/s4570079/st-1.0.0/st1.0.0'
    os.chdir(path)
    for file in os.listdir():
        if(file.endswith(".dat")):
            number_set.append(file.split(".")[0])

    dataSet=[]
    labelSet=[]
    ecgClassSet=['N','a','s','J','S','V','F','Q']
    channel=0
    for n in number_set:
        get_aha_st_data(n,dataSet,labelSet,ecgClassSet,channel,path)
    print("Read all files done")
    dataSet=np.array(dataSet).reshape(-1,200)
    labelSet=np.array(labelSet).reshape(-1,1)
    train_dataset=np.hstack((dataSet,labelSet))

    X=train_dataset[:32567,:200].reshape(-1,200,1)
    Y=train_dataset[:32567,200]

    shuffle_index = np.random.permutation(len(X))
    test_length = int(0.3 * len(shuffle_index)) #Ratio=0.3
    test_index = shuffle_index[:test_length]
    train_index = shuffle_index[test_length:]
    X_train, Y_train = X[train_index], Y[train_index]
    X_test, Y_test = X[test_index], Y[test_index]

    '''test_length = int(0.3 * len(X)) #Ratio=0.3
    X_train, Y_train = X[test_length:], Y[test_length:]
    X_test, Y_test = X[:test_length], Y[:test_length]'''

    print(np.shape(X_train))
    print(np.shape(Y_train))
    print(np.shape(X_test))
    print(np.shape(Y_test))

    return (X_train, Y_train), (X_test, Y_test)

def load_ptb():
    ecgClassSet=['NORM','UNDEFINE','STTC','NST_','IMI','AMI','LVH','LAFB/LPFB','ISC_','IRBBB','_AVB','IVCD','ISCA','CRBBB','CLBBB','LAO/LAE',
                    'ISCI','LMI','RVH','RAO/RAE','WPW','ILBBB','SEHYP','PMI']
    channel=9
    sampling_rate=100 # 500: hr,5000    100:lr,1000
    path="/home/Student/s4570079/ptb-1.0.1/ptb1.0.1/"

    #load and convert annotation data
    Y=pd.read_csv(path+'ptbxl_database.csv',index_col='ecg_id')
    Y.scp_codes=Y.scp_codes.apply(lambda x:ast.literal_eval(x))

    #load raw signal data
    X=get_ptb_data(Y,sampling_rate,path,channel)
    print("Read all files done")

    # load scp_statements for diagnostic aggregation
    agg_df=pd.read_csv(path+'scp_statements.csv',index_col=0)
    agg_df=agg_df[agg_df.diagnostic == 1]

    def aggregate_diagnostic(y_dic):
        key=max(y_dic, key=y_dic.get)
        diagnosis='UNDEFINE'
        if key in agg_df.index:
            diagnosis=agg_df.loc[key].diagnostic_subclass

        return diagnosis

    #apply diagnostic superclass
    Y['diagnostic_class']=Y.scp_codes.apply(aggregate_diagnostic)
    print("Labels are defined")

    test_fold=10
    X_train_meta=X[np.where(Y.strat_fold != test_fold)]
    Y_train_list=Y[(Y.strat_fold != test_fold)].diagnostic_class


    X_test_meta=X[np.where(Y.strat_fold == test_fold)]
    Y_test_list=Y[(Y.strat_fold == test_fold)].diagnostic_class

    X_train=[]
    for meta_num in range(0,len(X_train_meta)):
        signal_train,_=X_train_meta[meta_num]
        X_train_cleaned=del_noise(data=signal_train.flatten(),level=6)
        X_train.append((X_train_cleaned))

    X_test=[]
    for meta_num in range(0,len(X_test_meta)):
        signal_test,_=X_test_meta[meta_num]
        X_test_cleaned=del_noise(data=signal_test.flatten(),level=6)
        X_test.append((X_test_cleaned))

    Y_train=[]
    for i in range(0,Y_train_list.size):
        label=ecgClassSet.index(Y_train_list.iloc[i])
        Y_train.append(label)

    Y_test=[]
    for i in range(0,Y_test_list.size):
        label=ecgClassSet.index(Y_test_list.iloc[i])
        Y_test.append(label)

    if sampling_rate==100:
        X_test=np.reshape(np.asarray(X_test),(2203,1000,1))
        Y_test=np.asarray(Y_test)
        X_train=np.reshape(np.asarray(X_train),(19634,1000,1))
        Y_train=np.asarray(Y_train)
    else:
        X_test=np.reshape(np.asarray(X_test),(2203,5000,1))
        Y_test=np.asarray(Y_test)
        X_train=np.reshape(np.asarray(X_train),(19634,5000,1))
        Y_train=np.asarray(Y_train)

    print(np.shape(X_train))
    print(np.shape(Y_train))
    print(np.shape(X_test))
    print(np.shape(Y_test))

    return (X_train, Y_train), (X_test, Y_test)

def binarize_class(y_train, y_test):
    y_train_bin = np.ones(len(y_train), dtype=np.int32)
    #norm=-1, that's negative
    y_train_bin[(y_train == 0)] = -1
    y_test_bin = np.ones(len(y_test), dtype=np.int32)
    y_test_bin[(y_test == 0)] = -1
    return y_train_bin, y_test_bin

def binarize_ptb_class(y_train, y_test):
    y_train_bin = np.ones(len(y_train), dtype=np.int32)
    #norm, undefine, imi=-1, negative
    y_train_bin[(y_train == 0) | (y_train == 1) | (y_train == 4)] = -1
    y_test_bin = np.ones(len(y_test), dtype=np.int32)
    y_test_bin[(y_test == 0) | (y_test == 1) | (y_test == 4)] = -1
    return y_train_bin, y_test_bin


def conv_data2image(data):
    return np.rollaxis(data.reshape((3, 32, 32)), 0, 3)


def make_dataset(dataset, n_labeled, n_unlabeled, seed):
    def make_pu_dataset_from_binary_dataset(x, y, seed,labeled=n_labeled, unlabeled=n_unlabeled):
        labels = np.unique(y)
        print("train labels bin: ", labels)
        positive, negative = labels[1], labels[0]
        print("positive: ", positive)
        print("negative: ", negative)
        x, y = np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.int32)
        assert(len(x) == len(y))
        perm = np.random.permutation(len(y))
        #为了将标签打乱
        x, y = x[perm], y[perm]
        #各数据数量
        n_p = (y == positive).sum()
        print("no. of pos: ",n_p)
        #人工定义的数量，用在positive部分
        n_lp = labeled
        print("no. of labeled pos: ",n_lp)
        n_n = (y == negative).sum()
        print("no. of neg: ",n_n)
        n_u = unlabeled
        print("no. of unlabeled: ",n_u)
        if labeled + unlabeled == len(x):
            n_up = n_p - n_lp
        elif unlabeled == len(x):
            n_up = n_p
        else:
            raise ValueError("Only support |P|+|U|=|X| or |U|=|X|.")
        print("no. of unlabeled positive: ",n_up)
        _prior = float(n_up) / float(n_u)
        print("prior:",_prior)
        #数据集划分
        xlp = x[y == positive][:n_lp]
        print("x labeled postive: ",np.shape(xlp))
        xup = np.concatenate((x[y == positive][n_lp:], xlp), axis=0)[:n_up]
        print("x unlabled positive: ",np.shape(xup))
        xun = x[y == negative]
        print("x unlabled negative: ",np.shape(xun))
        #pos+u_pos+u_n
        x_positive = xlp
        print("x labeled positive: ",np.shape(x_positive))
        x_unlabel = np.asarray(np.concatenate((xup, xun), axis=0), dtype=np.float32)
        print("x unlabeled: ",np.shape(x_unlabel))

        #pos+un: label
        y_positive = np.asarray((np.ones(n_lp)), dtype=np.int32)
        print("y labeled positive:", np.shape(y_positive))
        y_unlabel = np.asarray( -np.ones(n_u), dtype=np.int32)
        print("y unlabeled:", np.shape(y_unlabel))
        #y = np.asarray(np.concatenate((np.ones(n_lp), -np.ones(n_u))), dtype=np.int32)
        perm1 = np.random.permutation(len(y_positive))
        x_positive, y_positive = x_positive[perm1], y_positive[perm1]
        print(np.shape(x_positive))
        print(np.shape(y_positive))
        perm2 = np.random.permutation(len(y_unlabel))
        x_unlabel ,y_unlabel = x_unlabel[perm2], y_unlabel[perm2]
        print(np.shape(x_unlabel))
        print(np.shape(y_unlabel))
        return x_positive, x_unlabel, y_positive, y_unlabel, _prior

    def make_pn_dataset_from_binary_dataset(x, y):
        labels = np.unique(y)
        positive, negative = labels[1], labels[0]
        X, Y = np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.int32)
        n_p = (Y == positive).sum()
        n_n = (Y == negative).sum()
        Xp = X[Y == positive][:n_p]
        Xn = X[Y == negative][:n_n]
        X_pos = np.asarray(Xp, dtype=np.float32)
        X_neg = np.asarray(Xn, dtype=np.float32)
        Y_pos = np.asarray(np.ones(n_p), dtype=np.int32)
        Y_neg = np.asarray((-np.ones(n_n)), dtype=np.int32)
        perm1 = np.random.permutation(len(Y_pos))
        X_pos, Y_pos = X_pos[perm1], Y_pos[perm1]
        perm2 = np.random.permutation(len(Y_neg))
        X_neg, Y_neg = X_neg[perm2], Y_neg[perm2]
        return X_pos, X_neg, Y_pos, Y_neg
#看这块
    (x_train, y_train), (x_test, y_test) = dataset
    x_positive, x_unlabel, y_positive, y_unlabel, prior = make_pu_dataset_from_binary_dataset(x_train, y_train, seed)
    X_pos, X_neg, Y_pos, Y_neg = make_pn_dataset_from_binary_dataset(x_test, y_test)
    print("training:{}".format(x_train.shape))
    print("test:{}".format(x_test.shape))

    print("x_positive",np.shape(x_positive))
    print("y_positive",np.shape(y_positive))
    print("x_unlabel",np.shape(x_unlabel))
    print("y_unlabel",np.shape(y_unlabel))
    print("X_pos",np.shape(X_pos))
    print("Y_pos",np.shape(Y_pos))
    print("X_neg",np.shape(X_neg))
    print("Y_neg",np.shape(Y_neg))

    return list(zip(x_positive, y_positive)), list(zip(x_unlabel, y_unlabel)),list(zip(X_pos, Y_pos)),list(zip(X_neg, Y_neg)), prior


def load_dataset(dataset_name, n_labeled, n_unlabeled,seed):
    if dataset_name == "mit":
        (x_train, y_train), (x_test, y_test) = load_mit()
        y_train, y_test = binarize_class(y_train, y_test)
    elif dataset_name == "aha":
        (x_train, y_train), (x_test, y_test) = load_aha()
        y_train, y_test = binarize_class(y_train, y_test)
    elif dataset_name == "st":
        (x_train, y_train), (x_test, y_test) = load_st()
        y_train, y_test = binarize_class(y_train, y_test)
    elif dataset_name == "ptb":
        (x_train, y_train), (x_test, y_test) = load_ptb()
        y_train, y_test = binarize_ptb_class(y_train, y_test)
    else:
        raise ValueError("dataset name {} is unknown.".format(dataset_name))
    x_train=np.expand_dims(x_train,axis=1)
    x_test=np.expand_dims(x_test,axis=1)
    xy_pos_train, xy_unlabel_train, xy_pos_test, xy_neg_test, prior = make_dataset(((x_train, y_train), (x_test, y_test)), n_labeled, n_unlabeled,seed)
    return xy_pos_train, xy_unlabel_train, xy_pos_test, xy_neg_test, prior
#test如果需要更改，和上边一样的思路

