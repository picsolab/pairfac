from scipy import sparse
import os.path
import numpy as np
import itertools
import csv
import scipy.sparse as sps
from random import randrange
import scipy.optimize as opt
import functions as fn
import gc
import time
import json
import math
import sys
import pandas as pd
from numpy import random
import pdb
from sktensor import ktensor
from sktensor import dtensor
from sktensor.core import khatrirao
from sktensor import sptensor
from sktensor.sptensor import fromarray
from sktensor import dtensor, cp_als

from pyspark import SparkContext, SparkConf
from heapq import heappush, heappop, heapify
from collections import defaultdict
import copy


from pairfac import PAIRFAC
from pairfac import SDCDT
from sktensor import dtensor, cp_als
global CLAMPING_THRESHOLD
CLAMPING_THRESHOLD= 0

import logging
logging.basicConfig(level=logging.INFO)
_log = logging.getLogger('JNTF')




def _compute_tensor_values(block,X_U0,Y_U0,U1,U2,R,m,n,o):
    all_values_X = []
    all_values_Y = []
    all_values_ZX = []
    all_values_ZY = []
    all_values_S = []
    idx_list_str = []
                
    for i in block:
        for j in range(0,n):
            for k in range(0,o):
                value_X = np.sum([X_U0[i,r] * U1[j,r] * U2[k,r] for r in range(R)])
                value_Y = np.sum([Y_U0[i,r] * U1[j,r] * U2[k,r] for r in range(R)])
                value_ZX = clampping(value_X - value_Y)
                value_ZY = clampping(value_Y - value_X)
                value_S = 1 if abs(value_X - value_Y) == 0 else 0
                if value_X > 0 or value_Y > 0 or value_ZX > 0 or value_ZY > 0:
                    idx = '{},{},{}'.format(i,j,k)              
                    all_values_X.append(value_X)
                    all_values_Y.append(value_Y)
                    all_values_ZX.append(value_ZX)
                    all_values_ZY.append(value_ZY)
                    all_values_S.append(value_S)
                    idx_list_str.append(idx)
    return all_values_X, all_values_Y, all_values_ZX, all_values_ZY ,idx_list_str, all_values_S

def roll(sides, bias_list):
    assert len(bias_list) == sides
    number = random.uniform(0, sum(bias_list))
    current = 0
    for i, bias in enumerate(bias_list):
        current += bias
        if number <= current:
            return i + 1

def clampping(x):
    return max(x,CLAMPING_THRESHOLD)

def clampping1(x):
    return np.exp(-np.absolute(x))

def read_city_data(date1,date2,dims,city,type,grid_size,k, nb_data_points, bootstrap_seed = 0):

    from pprint import pprint
    from pyspark import SparkConf, SparkContext
    conf = SparkConf().set("spark.driver.maxResultSize", "10g").setAppName("ReadInData")
    sc = SparkContext(conf=conf)
    _log.info(date1)
    _log.info(date2)
    # _log.info(dims)
    _log.info(city)
    _log.info(grid_size)
    if grid_size == "mg":
        folder_dir = "content_topic_concept"
    elif grid_size == "od":
        folder_dir = "month_weekday_neighbor"
    elif grid_size in ["alto","altotb", "sci"]:
        folder_dir = "stage_app_topic"
    elif grid_size.startswith("wpi_"):
        folder_dir = "attempt_kc_response" 
        if grid_size.endswith("4d"):
            folder_dir = "problem_name_kc_problem_view_duration"

    elif grid_size.startswith("ha_"):
        folder_dir = "education_country_daysq" 
        if grid_size.endswith("4d"):
            folder_dir = "education_country_daysq_age"

    elif grid_size.startswith("xt_"):
        folder_dir = "day_event_source"         
        if grid_size.endswith("4d"):
            folder_dir = "weekday_hour_event_source"
    else:
        folder_dir = "timeHashing_hour_locationHashing"
    idx_list_file_X = "/media/ext01/xidao/project/hipairfac/output/"+str(folder_dir)+"/exportR/"+str(date1)+"/"+city+"_"+type+"_dimensions_grid_"+str(grid_size) + "_" +str(bootstrap_seed)+"*"
    idx_list_file_Y = "/media/ext01/xidao/project/hipairfac/output/"+str(folder_dir)+"/exportR/"+str(date2)+"/"+city+"_"+type+"_dimensions_grid_"+str(grid_size) + "_" +str(bootstrap_seed)+"*"
    X_file = "/media/ext01/xidao/project/hipairfac/output/"+str(folder_dir)+"/exportR/"+str(date1)+"/"+city+"_"+type+"_values_grid_"+str(grid_size) + "_" +str(bootstrap_seed)+"*"
    Y_file = "/media/ext01/xidao/project/hipairfac/output/"+str(folder_dir)+"/exportR/"+str(date2)+"/"+city+"_"+type+"_values_grid_"+str(grid_size) + "_" +str(bootstrap_seed)+"*"
    X_values = sc.textFile(X_file).repartition(1).map(float).cache()
    Y_values = sc.textFile(Y_file).repartition(1).map(float).cache()
    idx_list_str_X = sc.textFile(idx_list_file_X).repartition(1).map(lambda x: '_'.join(i for i in x.split(','))).cache()
    idx_list_str_Y = sc.textFile(idx_list_file_Y).repartition(1).map(lambda x: '_'.join(i for i in x.split(','))).cache()
    
    idx_list_str_dict_X = idx_list_str_X.zip(X_values).collectAsMap()
    idx_list_str_dict_Y = idx_list_str_Y.zip(Y_values).collectAsMap()
    idx_list_str_X = idx_list_str_X.zip(X_values).sortBy(lambda a: (a[1]), ascending=False).map(lambda x:x[0]).take(nb_data_points)
    idx_list_str_Y = idx_list_str_Y.zip(Y_values).sortBy(lambda a: (a[1]), ascending=False).map(lambda x:x[0]).take(nb_data_points)



    idx_list_all = list(set().union(idx_list_str_X,idx_list_str_Y))

    value_list_X = []
    value_list_Y = []
    value_list_ZX = []
    value_list_ZY = []
    value_list_S = []
    epsilon_1 = 1e-5
    for each_indx in idx_list_all:
        if each_indx not in idx_list_str_dict_X:
            idx_list_str_dict_X[each_indx] = 0
        if each_indx not in idx_list_str_dict_Y:
            idx_list_str_dict_Y[each_indx] = 0
        diff_Y = idx_list_str_dict_Y[each_indx] - idx_list_str_dict_X[each_indx]
        diff_X = idx_list_str_dict_X[each_indx] - idx_list_str_dict_Y[each_indx]

        comm =  1 if np.absolute(diff_X) <= epsilon_1 else 0
        value_list_X.append(idx_list_str_dict_X[each_indx])
        value_list_Y.append(idx_list_str_dict_Y[each_indx])
        value_list_ZX.append(diff_X if diff_X > 0 else 0)
        value_list_ZY.append(diff_Y if diff_Y > 0 else 0)
        value_list_S.append(comm)

    del X_values
    del Y_values
    del idx_list_str_dict_X
    del idx_list_str_dict_Y
    del idx_list_str_X
    del idx_list_str_Y
    gc.collect()
    idx_list_all = [[int(x) for x in idx.split('_')] for idx in idx_list_all]
    sc.stop()
    return idx_list_all, value_list_X,value_list_Y,value_list_ZX,value_list_ZY,value_list_S

def _run_MG_case_study_incremental(R1=10, R2=10,k=2, reg_par=1e+2,stop_criteria=1e-6,location_reg=1e-2,verbose=2):
    # _log = logging.getLogger('')
    import os
    iter_cnt = 10
    nb_trial = 3 # check performance

    save_embedding = True
    
    alg_names = [PAIRFAC]
    alg_names = [SDCDT]
    iters = [iter_cnt] * len(alg_names) 
    CLAMPING_THRESHOLD = 0

    R = 10
    noise_sparsity = 0.2
    
    grid_size = "mg"
    grid_size = "od"
    grid_size = "alto"
    grid_size = "altotb"
    # grid_size = "wpi_2004"
    # grid_size = "wpi_2005"
    # grid_size = "wpi_4m"
    grid_size = "wpi_5m"
    grid_size = "wpi_5p"
    
    # grid_size = "wpi_{}mp4d".format(str(sys.argv[4]))    
    grid_size = "ha_{}mp4d".format(str(sys.argv[4]))    
    is_common = False

    R_check = 12
    d = 0
    delta = 0
    num_workers = 1
    nb_points = 40000 # nyc
    nb_points = 10000 # paris_week: neighbor_checkins
    nb_points = 16000 # paris_month: neighbor_checkins_month
    nb_points = 19000 # paris_month: neighbor_traffic_month
    delta,noise = 0,0
    np.random.seed(1)
    train_proportions = [10]
    
    type = "function"   
    distance = "" 
    k = "adjacent"
    date1 = ""
    date2 = ""
    city = "nyc"
 
    # if grid_size in ["ha_cs12mp","ha_6x12mp","ha_6x13mp", "ha_6x12gmp"]:
    if grid_size.startswith("ha_"):
        nb_points = 4000        
        if grid_size.endswith("4d"):
            dims = [5,34,6,54]        

        city = "mooc"
        date1 = "good"
        date2 = "bad"        
        layers = [0]
        R_set = {0:6}
        R_check = 6
        
    m, n, o = dims[0],dims[1],dims[2]

    noise = grid_size
    k = city + "_" + grid_size
    
    if alg_names[0].__name__ in ["SDCDT"]:
        distance = int(sys.argv[1])
        alpha = float(sys.argv[2])
        beta = float(sys.argv[3])

        alpha_pars = [alpha]
        beta_pars = [beta]
        gamma_pars = [1e+0]
        delta_pars = [1e+0]

    else:
        distance = 3
        alpha = float(sys.argv[1])
        beta = float(sys.argv[2])
        gamma = float(sys.argv[3])                
        alpha_pars = [alpha]
        beta_pars = [beta]
        gamma_pars = [gamma]
        delta_pars = [1e-8]
        

    bootstrap_seed_list = [0,1,2,3,4]
    for bootstrap_seed in bootstrap_seed_list:
        
        _log.info("Starting...")
        _log.info("distance:{}".format(distance))
        _log.info("bootstrap_seed:{}".format(bootstrap_seed))
        from datetime import datetime     
        idx_list, value_list_X,value_list_Y,value_list_ZX,value_list_ZY,value_list_S = read_city_data(date1,date2,dims,city,type,grid_size,k,nb_points,bootstrap_seed = bootstrap_seed)
        X = sptensor(tuple(np.asarray(idx_list).T), value_list_X,shape=dims, dtype=np.float)
        Y = sptensor(tuple(np.asarray(idx_list).T), value_list_Y,shape=dims, dtype=np.float)

        conf = SparkConf().setAppName("HiPairFac...")
        sc = SparkContext(conf=conf)            
        
        _log.info("Starting MultiResolution-HiPairFac..")
        _log.info("Runnig...")
            
        Lambda_all = [alpha_pars, beta_pars, gamma_pars, delta_pars, train_proportions]
        Lambda_all = list(itertools.product(*Lambda_all))

        len_paraset = len(Lambda_all)
        cur_para_index = 0

        train_cur_proportion = float(10)
        test_portion = (10.0 - train_cur_proportion) / 2.0
        np.random.seed(2)
        Weight = np.random.choice([0, 1, 2, 3], size=(len(value_list_ZX),), p=[test_portion/10, test_portion/10, train_cur_proportion/10, (10.0 - train_cur_proportion - test_portion*2)/10])
        train_index_1 = train_index_2 = np.where(Weight == 2)
        validation_index_1 = validation_index_2 = np.where(Weight == 1)
        test_index_1 = test_index_2 = np.where(Weight == 0)


        value_list_X_train = list(np.asarray(value_list_X)[train_index_1[0]])
        value_list_Y_train = list(np.asarray(value_list_Y)[train_index_1[0]])
        value_list_ZX_train = list(np.asarray(value_list_ZX)[train_index_1[0]])
        value_list_ZY_train = list(np.asarray(value_list_ZY)[train_index_1[0]])
        value_list_S_train = list(np.asarray(value_list_S)[train_index_1[0]])

        value_list_X_test = list(np.asarray(value_list_X)[test_index_1[0]])
        value_list_Y_test = list(np.asarray(value_list_Y)[test_index_1[0]])
        value_list_ZX_test = list(np.asarray(value_list_ZX)[test_index_1[0]])
        value_list_ZY_test = list(np.asarray(value_list_ZY)[test_index_1[0]])
        value_list_S_test = list(np.asarray(value_list_S)[test_index_1[0]])

        value_list_X_validation = list(np.asarray(value_list_X)[validation_index_1[0]])
        value_list_Y_validation = list(np.asarray(value_list_Y)[validation_index_1[0]])
        value_list_ZX_validation = list(np.asarray(value_list_ZX)[validation_index_1[0]])
        value_list_ZY_validation = list(np.asarray(value_list_ZY)[validation_index_1[0]])
        value_list_S_validation = list(np.asarray(value_list_S)[validation_index_1[0]])

        X_train = sptensor(tuple(np.asarray(idx_list)[train_index_1[0]].T), value_list_X_train,shape=dims, dtype=np.float)
        Y_train = sptensor(tuple(np.asarray(idx_list)[train_index_1[0]].T), value_list_Y_train,shape=dims, dtype=np.float)
        ZX_train = sptensor(tuple(np.asarray(idx_list)[train_index_1[0]].T), value_list_ZX_train,shape=dims, dtype=np.float)
        ZY_train = sptensor(tuple(np.asarray(idx_list)[train_index_1[0]].T), value_list_ZY_train,shape=dims, dtype=np.float)
        S_train = sptensor(tuple(np.asarray(idx_list)[train_index_1[0]].T), value_list_S_train,shape=dims, dtype=np.float)

        non_zero_idxs=np.asarray(idx_list)[train_index_1[0]]
        D_matrix = np.zeros((X_train.shape[0],X_train.shape[0]))
        W_matrix = np.zeros((X_train.shape[0],X_train.shape[0]))

        for alg_name, iter_cnt in zip(alg_names, iters):
            for each_lambda in Lambda_all:
                progress = cur_para_index*1.0 / len_paraset
                cur_para_index += 1
                _log.info('[Hi-PAIRFAC] Running the progress {}...'.format(progress))
                # sc.clearFiles()
                alg = alg_name()
                Lambda = list(each_lambda)
                P_all = None

                cur_paras = '_'.join([str(x) for x in each_lambda])
                fname = 'weight_s_t_2_{}_layer_0_distance_{}_seed_{}_R_{}'.format(alg.__class__.__name__, distance,bootstrap_seed,R_check)

                if is_common:
                    sub_dir = "ranking"
                else:
                    sub_dir = "classification"

                directory_ = "/media/ext01/xidao/project/hipairfac/src/dtenfac/output_" + k + "_"+sub_dir+""


                layer_fileName = directory_ + "/weights/" + str(cur_paras) + "/" + str(fname)
                
                embeddings_dir = directory_ + "/embeddings/"+str(alg_names[0].__name__)+"/"+str(bootstrap_seed)+"/" + cur_paras
                if not os.path.exists(embeddings_dir):
                    os.makedirs(embeddings_dir)
                weights_dir = directory_ + "/weights/" + cur_paras
                if not os.path.exists(weights_dir):
                    os.makedirs(weights_dir)

                _log.info(layer_fileName)
                if os.path.exists(layer_fileName):
                    _log.info("{} exists".format(layer_fileName))
                    continue

                alg.run_multi_trials( sc, X_train, Y_train, ZX_train, ZY_train, S_train, k,k, Lambda, D_matrix, W_matrix, \
                    reg_par=0, location_reg=0, num_trial=nb_trial, max_iter=iter_cnt, verbose=2,noise=0.01,nb_points=nb_points,\
                    non_zero_idxs=non_zero_idxs, \
                    test_idx=[],test_data=[], \
                    validation_idx=[],validation_data=[],\
                    num_workers=num_workers,distance=distance, R_set = R_set,layers = layers,bootstrap_seed = bootstrap_seed, 
                    ranking = is_common, save_embedding = save_embedding)

                gc.collect()


        sc.stop()


if __name__ == '__main__':

    _run_MG_case_study_incremental()


