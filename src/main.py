import os

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

PROJECT_DIR = ''
EPSILON = 1e-5

import logging
logging.basicConfig(level=logging.INFO)
_log = logging.getLogger('TF')



def clampping(x):
    return max(x,CLAMPING_THRESHOLD)

def read_domain_data(source1,source2,dims,domain,type,case_study,k, nb_data_points, bootstrap_seed = 0):

    from pprint import pprint
    from pyspark import SparkConf, SparkContext
    conf = SparkConf().set("spark.driver.maxResultSize", "10g").setAppName("ReadInData")
    sc = SparkContext(conf=conf)
    _log.info(source1)
    _log.info(source2)
    # _log.info(dims)
    _log.info(domain)
    _log.info(case_study)
    if case_study == "mg":
        folder_dir = "content_topic_concept"
    elif case_study == "od":
        folder_dir = "month_weekday_neighbor"
    elif case_study in ["alto","altotb", "sci"]:
        folder_dir = "stage_app_topic"
    elif case_study.startswith("wpi_"):
        folder_dir = "attempt_kc_response" 
        if case_study.endswith("4d"):
            folder_dir = "problem_name_kc_problem_view_duration"

    elif case_study.startswith("ha_"):
        folder_dir = "education_country_daysq" 
        if case_study.endswith("4d"):
            folder_dir = "education_country_daysq_age"

    elif case_study.startswith("xt_"):
        folder_dir = "day_event_source"         
        if case_study.endswith("4d"):
            folder_dir = "weekday_hour_event_source"
    else:
        folder_dir = "timeHashing_hour_locationHashing"
    idx_list_file_X = PROJECT_DIR + "/output/"+str(folder_dir)+"/exportR/"+str(source1)+"/"+domain+"_"+type+"_dimensions_grid_"+str(case_study) + "_" +str(bootstrap_seed)+"*"
    idx_list_file_Y = PROJECT_DIR + "/output/"+str(folder_dir)+"/exportR/"+str(source2)+"/"+domain+"_"+type+"_dimensions_grid_"+str(case_study) + "_" +str(bootstrap_seed)+"*"
    X_file = PROJECT_DIR + "/output/"+str(folder_dir)+"/exportR/"+str(source1)+"/"+domain+"_"+type+"_values_grid_"+str(case_study) + "_" +str(bootstrap_seed)+"*"
    Y_file = PROJECT_DIR + "/output/"+str(folder_dir)+"/exportR/"+str(source2)+"/"+domain+"_"+type+"_values_grid_"+str(case_study) + "_" +str(bootstrap_seed)+"*"
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
    for each_indx in idx_list_all:
        if each_indx not in idx_list_str_dict_X:
            idx_list_str_dict_X[each_indx] = 0
        if each_indx not in idx_list_str_dict_Y:
            idx_list_str_dict_Y[each_indx] = 0
        diff_Y = idx_list_str_dict_Y[each_indx] - idx_list_str_dict_X[each_indx]
        diff_X = idx_list_str_dict_X[each_indx] - idx_list_str_dict_Y[each_indx]

        comm =  1 if np.absolute(diff_X) <= EPSILON else 0
        value_list_X.append(idx_list_str_dict_X[each_indx])
        value_list_Y.append(idx_list_str_dict_Y[each_indx])
        value_list_ZX.append(diff_X if diff_X > 0 else 0)
        value_list_ZY.append(diff_Y if diff_Y > 0 else 0)
        value_list_S.append(comm)

    gc.collect()
    idx_list_all = [[int(x) for x in idx.split('_')] for idx in idx_list_all]
    sc.stop()
    return idx_list_all, value_list_X,value_list_Y,value_list_ZX,value_list_ZY,value_list_S

def run_case_study():
    iter_cnt = 10
    nb_trial = 3 
    
    alg_names = [PAIRFAC]
    alg_names = [SDCDT]
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

    iters = [iter_cnt] * len(alg_names) 
    CLAMPING_THRESHOLD = 0

    case_study = "ha_{}mp4d".format(str(sys.argv[4]))    
    sub_dir = "classification"
    num_workers = 1
    nb_points = 40000 # nyc
    train_proportions = [10]
    
    _type = "function"   

    if case_study.startswith("ha_"):
        nb_points = 4000        
        dims = [5,34,6,54]        
        domain = "mooc"
        source1 = "good"
        source2 = "bad"        
        layers = [0]
        R_set = {0:6}
        R_check = 6
        
    k = domain + "_" + case_study
    
    bootstrap_seed_list = [0,1,2,3,4]
    for bootstrap_seed in bootstrap_seed_list:
        
        _log.info("distance:{}".format(distance))
        _log.info("bootstrap_seed:{}".format(bootstrap_seed))
        from datetime import datetime     
        idx_list, value_list_X,value_list_Y,value_list_ZX,value_list_ZY,value_list_S = read_domain_data(source1,source2,dims,domain,_type,case_study,k,nb_points,bootstrap_seed = bootstrap_seed)
        X = sptensor(tuple(np.asarray(idx_list).T), value_list_X,shape=dims, dtype=np.float)
        Y = sptensor(tuple(np.asarray(idx_list).T), value_list_Y,shape=dims, dtype=np.float)

        conf = SparkConf().setAppName("PairFac...")
        sc = SparkContext(conf=conf)            
        
            
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
                _log.info('[{}] Running {}...'.format(alg_names[0].__name__,progress))
                alg = alg_name()
                Lambda = list(each_lambda)
                cur_paras = '_'.join([str(x) for x in each_lambda])
                fname = 'weight_s_t_2_{}_layer_0_distance_{}_seed_{}_R_{}'.format(alg.__class__.__name__, distance,bootstrap_seed,R_check)
                directory_ = PROJECT_DIR + "/output/output_" + k + "_"+sub_dir+""
                if not os.path.exists(directory_):
                    os.makedirs(directory_)

                layer_fileName = directory_ + "/weights/" + str(cur_paras) + "/" + str(fname)
                
                embeddings_dir = directory_ + "/embeddings/"+str(alg_names[0].__name__)+"/"+str(bootstrap_seed)+"/" + cur_paras
                if not os.path.exists(embeddings_dir):
                    os.makedirs(embeddings_dir)
                weights_dir = directory_ + "/weights/" + cur_paras
                if not os.path.exists(weights_dir):
                    os.makedirs(weights_dir)
                if os.path.exists(layer_fileName):
                    _log.info("{} exists".format(layer_fileName))
                    continue

                alg.run_multi_trials( sc, X_train, Y_train, ZX_train, ZY_train, S_train, k,k, Lambda, D_matrix, W_matrix, \
                    num_trial=nb_trial, max_iter=iter_cnt, verbose=2,
                    noise=0.01,nb_points=nb_points,
                    non_zero_idxs=non_zero_idxs,                    
                    num_workers=num_workers,distance=distance, R_set = R_set,layers = layers,bootstrap_seed = bootstrap_seed)

                gc.collect()


        sc.stop()


if __name__ == '__main__':

    run_case_study()


