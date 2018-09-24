import numpy as np
import csv
import scipy.sparse as sps
from random import randrange
import scipy.optimize as opt
import functions as fn
import time
import json
import sys
import math
import scipy as sp

import pandas as pd
from numpy import random
import pdb
from sktensor.core import khatrirao
from sktensor.core import teneye
from sktensor.sptensor import fromarray
from sktensor import sptensor, ktensor, dtensor, cp_als

from random import randint
from pyspark import SparkContext, SparkConf
import os
import logging

logging.basicConfig(level=logging.INFO)
_log = logging.getLogger('JNTF')
_log.setLevel(logging.INFO)



def quiet_logs( sc ):
  logger = sc._jvm.org.apache.log4j
  logger.LogManager.getLogger("org"). setLevel( logger.Level.ERROR )
  logger.LogManager.getLogger("akka").setLevel( logger.Level.ERROR )

class TensorFactorization_Base(object):

    """ Base class
    Specific algorithms need to be implemented by deriving from this class.
    """
    default_max_iter = 100
    default_max_time = np.inf

    def __init__(self):
        raise NotImplementedError(
            'TensorFactorization_Base is a base class that cannot be instantiated')

    def set_default(self, default_max_iter, default_max_time):
        self.default_max_iter = default_max_iter
        self.default_max_time = default_max_time

    def assignBlock(self, i, s, X,Y, Z1, Z2, S, tensor_dim_ceiling, subs_idx, num_workers, tensor_dim_size):
        _dict = {}
        num_ways = len(tensor_dim_ceiling)
        strata_index = [int(math.floor(i + sum([float(s) / num_workers**way_index for way_index in range(way_index+1)]))) % num_workers for way_index in range(num_ways)]
        strata_range = [range(int(math.ceil(strata_index[way_index] * tensor_dim_ceiling[way_index])), int(math.ceil((strata_index[way_index]+1) * tensor_dim_ceiling[way_index]))) for way_index in range(num_ways)]
        strata_range = [[o for o in each_range if o < tensor_dim_size[index]] for index, each_range in enumerate(strata_range)]        
        strata_range = [range(each_range[0], (each_range[-1] + 1)) for each_range in strata_range]
        total_nb_points = len(subs_idx.value)
        subs = [idx for idx in subs_idx.value if all([idx[way_index] in strata_range[way_index] for way_index in range(num_ways)])]
        subs_x = [tuple(idx) for idx in subs_idx.value if all([idx[way_index] in strata_range[way_index] for way_index in range(num_ways)])]

        X_vals = []
        Y_vals = []
        ZX_vals = []
        ZY_vals = []
        S_vals = []
        if len(subs_x) > 0:
            for i in range(len(subs_x)):
                tensor_index = tuple(np.array(subs_x[i]).T)
                X_vals.append(X[tensor_index][0])
                Y_vals.append(Y[tensor_index][0])
                ZX_vals.append(Z1[tensor_index][0])
                ZY_vals.append(Z2[tensor_index][0])
                S_vals.append(S[tensor_index][0])
            X_subs = sptensor(tuple(np.array(subs_x).T), X_vals,shape=tensor_dim_size, dtype=np.float)
            Y_subs = sptensor(tuple(np.array(subs_x).T), Y_vals,shape=tensor_dim_size, dtype=np.float)
            ZX_subs = sptensor(tuple(np.array(subs_x).T), ZX_vals,shape=tensor_dim_size, dtype=np.float)
            ZY_subs = sptensor(tuple(np.array(subs_x).T), ZY_vals,shape=tensor_dim_size, dtype=np.float)
            S_subs = sptensor(tuple(np.array(subs_x).T), S_vals,shape=tensor_dim_size, dtype=np.float)

            _dict['ratio'] = len(subs_x) / float(total_nb_points)
            _dict['X_subs'] = X_subs
            _dict['Y_subs'] = Y_subs
            _dict['ZX_subs'] = ZX_subs
            _dict['ZY_subs'] = ZY_subs
            _dict['S_subs'] = S_subs
            _dict['subs'] = subs        
            return _dict
        else:
            return None

    def run(self, sc, num_workers, all_blocks, X, Y, Z1, Z2, S, R1, R2, k, \
        Lambda, D_matrix, W_matrix, init=None,reg_par=1e+2, \
        location_reg=0, stop_criteria=1e-6, max_iter=None, \
        max_time=None, verbose=2,noise=0, trial=1,nb_points=10000, \
        non_zero_idxs=[], distance = 1,tree_group=None, \
        level = None, R_set = None):

        """ Run an algorithm with random initial values 
            and return the factor matrices

        Parameters
        ----------
        X : original tensor before
        Y : original tensor after
        Z1: discriminative signals before
        Z2: diescriminative signals after
        S : common signals
        R1: rank for before tensor
        R2: rank for after tensor
        k : shared number of rank in the ground truth (not necessary for proposed model)
        ground_truth_k : same as above
        Lambda : set of parameters in the paper ([lambda_1, lambda_2, lambda_3, lambda_4])
        trial : the trial index
        noise : default noise added to the ground truth factor matrices
        distance : the tree distance between the two trees (1, 2, 3)
        nb_points : number of points in the tensor
        non_zero_idx : the indexes of the none zero points

        Optional Parameters
        -------------------
        max_iter : int - maximum number of iterations for each trial.
                    If not provided, default maximum for each algorithm is used.
        max_time : int - maximum amount of time in seconds for each trial.
                    If not provided, default maximum for each algorithm is used.
        verbose : whether to print the resutl
        D_matrix, W_matrix : auxiliary matrix (default None)
        reg_par : regularzation parameter for location (default not used)
        num_workers : number of workers (used for distributed version)

        Returns
        -------
        [U] : Obtained factor matrix for each tensor
        [w] : Obtained weight vector for each tensor
        [cost_log] : the convergence log
        """

        alpha, beta, gamma, delta, train_proportion = Lambda

        quiet_logs( sc )

        info = {'R1': R1,
                'R2': R2,
                'k': k,
                'distance':distance,
                'alg': str(self.__class__),
                'X_type': str(X.__class__),
                'Y_type': str(Y.__class__),
                'alpha': alpha,
                'beta': beta,
                'gamma': gamma,
                'delta': delta,
                'train_proportion': train_proportion,
                'num_workers': num_workers,
                'nb_points': len(non_zero_idxs),
                'noise': noise,
                'location_reg': location_reg,
                'max_iter': max_iter if max_iter is not None else self.default_max_iter,
                'verbose': verbose,
                'max_time': max_time if max_time is not None else self.default_max_time}
        # X_init = []
        for way_index in range(len(X.shape)):
            info['X_dim_{}'.format(way_index)] = X.shape[way_index]
            info['Y_dim_{}'.format(way_index)] = Y.shape[way_index]

        # pdb.set_trace()
        if init != None:
            X_init = init[0:len(X.shape)]
            Y_init = init[len(X.shape):]
            info['init'] = 'user_provided'
        else:
            X_init = []
            Y_init = []
            for way_index in range(len(X.shape)):
                X_init.append(random.rand(X.shape[way_index], R1))
                Y_init.append(random.rand(Y.shape[way_index], R2))
            info['init'] = 'uniform_random'
        # verbose = 1
        if verbose >= 2:
             _log.info(('[{}] Running: '.format(self.__class__.__name__)))
             _log.info(json.dumps(info, indent=4, sort_keys=True))
        L_matrix = D_matrix - W_matrix
        X_init = [fn.normalize_column(each_init, by_norm='2')[0] for each_init in X_init]
        Y_init = [fn.normalize_column(each_init, by_norm='2')[0] for each_init in Y_init]        
        norm_X = X.norm()
        norm_Y = Y.norm()
        tensor_dim_size = X.shape
        num_ways = len(X.shape)

        Teneye_X = np.zeros([R1]*len(X.shape))
        for i in range(R1): Teneye_X[tuple([i]*num_ways)] = float(1.0)/R1
        Teneye_X = dtensor(Teneye_X)
        Teneye_Y = Teneye_X
        Teneye_Z_Y = Teneye_S_Y = Teneye_Y
        Teneye_Z_X = Teneye_S_X = Teneye_X
        previous_cost_all = previous_cost = 1e+10
        total_time = 0

        his = {'iter': [], 'elapsed': [], 'cost': []}
        Y_X = [None] * R1
        Y_Y = [None] * R2
        Z_weights_X = Teneye_X
        Z_weights_Y = Teneye_Y
        PairFac_list = ["PAIRFAC"]
        if str(self.__class__.__name__) in PairFac_list:
            Y_X = X_init
            Y_Y = Y_init
            Z_weights_X = Teneye_X
            Z_weights_Y = Teneye_Y
        start = time.time()

        tensor_dim_ceiling = [int(math.ceil(each_dim / float(num_workers))) for each_dim in X.shape]

        (X_init, Y_init) = self.initializer(X_init, Y_init)
        weights_X = tuple(Teneye_X[tuple([i]*num_ways)] for i in range(R1))
        weights_X = weights_X/sum(weights_X)
        weights_Y = tuple(Teneye_Y[tuple([i]*num_ways)] for i in range(R2))
        weights_Y = weights_Y/sum(weights_Y)

        E_X = 0
        E_Y = 0
        alpha_k = 1
        random.seed(trial)
        P_Z = P_all = None
        X_factors = X_init
        Y_factors = Y_init
        for i in range(1, info['max_iter'] + 1):
            start_iter = time.time()        

            (X_factors, Y_factors, Teneye_X, Teneye_Y,E_X, E_Y, Y_X, Y_Y, alpha_k,
                Z_weights_X,Z_weights_Y, P_all, P_Z) = self.iter_solver(sc, 
                all_blocks, X, Y, Z1,Z2, S, X_factors, Y_factors, R1, R2, k, 
                norm_X, norm_Y, Lambda, location_reg, D_matrix, 
                W_matrix,Teneye_X, Teneye_Y,E_X,E_Y, Y_X, Y_Y,alpha_k,previous_cost_all,
                Z_weights_X,Z_weights_Y,num_workers, distance=distance, 
                current_iter = i, tree_group = tree_group, 
                P_new = P_all, P_Z = P_Z, \
                Teneye_S_X = Teneye_S_X, Teneye_S_Y = Teneye_S_Y, \
                level = level)

            Save_Lambda = Lambda + [tensor_dim_size[0]] + [distance] + [nb_points] 

            elapsed = time.time() - start_iter

            cost = fn.jtnorm_fro_err_nways(self,X, Y, X_factors, Y_factors, norm_X, norm_Y)
            delta = cost - previous_cost

            previous_cost_all = cost
            delta = np.abs(delta)

            his['iter'].append(i)
            his['elapsed'].append(elapsed)
            his['cost'].append(cost)

            if verbose >= 2:
                _log.info('[%3d] cost: %0.6f | delta: %7.1e | secs: %.5f' % (
                    i, cost, delta, elapsed
                ))              
            
            total_time += elapsed
            if total_time > info['max_time'] or delta < stop_criteria:
                break
            previous_cost = cost
        his_dataframe = pd.DataFrame(his)
        his_dataframe = his_dataframe.transpose()
        Save_Lambda = Lambda + [tensor_dim_size[0]] + [distance] + [nb_points] 

        weights_X = tuple(Teneye_X[tuple([i]*num_ways)] for i in range(R1))
        weights_X = weights_X/sum(weights_X)
        weights_Y = tuple(Teneye_Y[tuple([i]*num_ways)] for i in range(R2))
        weights_Y = weights_Y/sum(weights_Y)
        weights_S_X = tuple(Teneye_S_X[tuple([i]*num_ways)] for i in range(R1))
        weights_S_X = weights_S_X/sum(weights_S_X)
        weights_S_Y = tuple(Teneye_S_Y[tuple([i]*num_ways)] for i in range(R1))
        weights_S_Y = weights_S_Y/sum(weights_S_Y)

        final = {}
        final['norm_X'] = norm_X
        final['norm_Y'] = norm_Y
        final['cost'] = cost
        final['iterations'] = i
        final['elapsed'] = time.time() - start

        rec = {'info': info, 'final': final,'his':his}
        if verbose >= 2:
             _log.info('[NTF] Completed: ')
             _log.info(json.dumps(final, indent=4, sort_keys=True))
        return (X_factors,Y_factors,[weights_X,weights_Y],rec,[weights_S_X,weights_S_Y], P_Z)


    def iter_solver(self, X, Y,  X_factors, Y_factors, F, H, G, L, R1, R2, k, norm_X, norm_Y, alpha, beta, gamma):
        raise NotImplementedError

    def initializer(self,  X,Y):
        return ( X, Y)

    def run_multi_trials(self, sc, X, Y, Z1, Z2, S, 
                        k, ground_truth_k, Lambda, 
                        D_matrix, W_matrix,reg_par=1e+2, 
                        stop_criteria=1e-6,location_reg=0,num_trial=20, max_iter=None, 
                        max_time=None, verbose=1,noise=0,distance=1, nb_points=10000,non_zero_idxs=[],
                        test_idx=[],test_data=[],validation_idx=[],validation_data=[],
                        num_workers=4,case_study_mode_index=None, 
                        layers = None, R_set = None, bootstrap_seed = 0, 
                        ranking = False, save_embedding = False, save_factor = False):

        """ Run an algorithm several times with random initial values 
            and return the RMSE on the testing and validation set

        Parameters
        ----------
        X : original tensor before
        Y : original tensor after
        Z1: discriminative signals before
        Z2: diescriminative signals after
        S : common signals
        R1: rank for before tensor
        R2: rank for after tensor
        k : shared number of rank in the ground truth (not necessary for proposed model)
        ground_truth_k : same as above
        Lambda : set of parameters in the paper ([lambda_1, lambda_2, lambda_3, lambda_4])
        num_trial : number of runs with different inital values
        distance : the tree distance between the two trees (1, 2, 3)
        nb_points : number of points in the tensor
        non_zero_idx : the indexes of the none zero points
        test_idx : the indexes of the testing points
        test_data : the entries in the tensor for the test_idx
        validation : the indexes of the validation points
        validation_data : the entries in the tensor for the validation idx

        Optional Parameters
        -------------------
        max_iter : int - maximum number of iterations for each trial.
                    If not provided, default maximum for each algorithm is used.
        max_time : int - maximum amount of time in seconds for each trial.
                    If not provided, default maximum for each algorithm is used.
        verbose : whether to print the resutl
        D_matrix, W_matrix : auxiliary matrix (default None)
        reg_par : regularzation parameter for location (default not used)
        num_workers : number of workers (used for distributed version)

        Returns
        -------
        [U] : Obtained factor matrix for each tensor
        [w] : Obtained weight vector for each tensor
        [cost_log] : the convergence log
        """
        self.factor_matrix_name = str(self.__class__.__name__)+"_layer_{}_distance_" + str(distance) + "_seed_" + str(bootstrap_seed) + "_{}" 
        self.weight_matrix_name = "weights/{}/weight_{}_t_{}_"+str(self.__class__.__name__)+"_layer_{}_distance_" + str(distance) + "_seed_" + str(bootstrap_seed)
        self.cost_matrix_name = '_'+'d_'+str(distance)+str(self.__class__.__name__) + "_seed_" + str(bootstrap_seed)

        result = []
        result_matrix = []
        cost_log = pd.DataFrame()
        cost_log_final = []
        weights_all = pd.DataFrame()
        num_ways = len(X.shape)
        tensor_dim_size = X.shape
        tensor_dim_ceiling = [int(math.ceil(dim_size / float(num_workers))) for dim_size in X.shape]
        X_original = X
        Y_original = Y
        Z1_original = Z1
        Z2_original = Z2
        S_original = S
        non_zero_idxs_original = non_zero_idxs
        all_results = [None, None]
        for t in range((num_trial)):
            X = X_original
            Y = Y_original
            Z1 = Z1_original
            Z2 = Z2_original
            S = S_original
            non_zero_idxs = non_zero_idxs_original
            try:
                init = []

                R1 = R_set[0]
                R2 = R_set[0]
                random.seed(1*(t+1))
                init.append(random.rand(X.shape[0],R1))
                random.seed(20*(t+1))           
                init.append(random.rand(X.shape[1],R1))
                random.seed(30*(t+1))
                init.append(random.rand(X.shape[2],R1))
                for way_index in range(len(X.shape) - 3):
                    random.seed((30+way_index+1)*(t+1))
                    init.append(random.rand(X.shape[3 + way_index],R1))
                random.seed(40*(t+1))       
                init.append(random.rand(Y.shape[0],R2))
                random.seed(50*(t+1))           
                init.append(random.rand(Y.shape[1],R2))
                random.seed(60*(t+1))
                init.append(random.rand(Y.shape[2],R2))
                for way_index in range(len(Y.shape) - 3):
                    random.seed((60+way_index+1)*(t+1))
                    init.append(random.rand(Y.shape[3 + way_index],R1))
                
                all_blocks = {}

                subs_idx = non_zero_idxs
                subs_idx = sc.broadcast(subs_idx)
                for s in range(num_workers**(num_ways - 1)):
                    all_blocks[s] = sc.parallelize(range(num_workers),num_workers) \
                                    .map(lambda x: self.assignBlock(x, s, X, Y, Z1, Z2, S, 
                                        tensor_dim_ceiling,
                                        subs_idx, num_workers, tensor_dim_size)).filter(lambda x: x is not None).collect()

                if verbose >= 0:
                    _log.info('[TensorFactorization] Running the {0}/{1}-th trial...'.format(t + 1, num_trial))
                
                this = self.run(sc, num_workers, all_blocks, X,Y, Z1, Z2, S, R1, R2, k, 
                    Lambda, D_matrix, W_matrix,init = init, reg_par=reg_par,
                     location_reg=location_reg, stop_criteria=stop_criteria,max_iter= max_iter, 
                     verbose=(-1 if verbose is 0 else verbose),noise=noise,trial=t, 
                     nb_points=nb_points,non_zero_idxs=non_zero_idxs,distance=distance,R_set = R_set)
                
                cost_each = pd.read_json(json.dumps(this[3]['his']))
                cost_each = self.format_output(Lambda, ground_truth_k, noise, distance, t, k, cost_each)                
                cost_log = cost_log.append(cost_each)


                weight_each_Z = pd.DataFrame(this[2])        
                weight_each_Z['tensor'] = ['X','Y']                
                weight_each_Z = self.format_output(Lambda, ground_truth_k, noise, distance, t, k, weight_each_Z)
                weights_all = weights_all.append(weight_each_Z)

                weight_each_S = pd.DataFrame(this[4])        
                weight_each_S['tensor'] = ['X','Y']
                weight_each_S = self.format_output(Lambda, ground_truth_k, noise, distance, t, k, weight_each_S)
                weights_all = weights_all.append(weight_each_S)

                Save_Lambda = Lambda + [tensor_dim_size[0]] + [distance] + [nb_points] + [bootstrap_seed]

                cur_paras_ = '_'.join([str(x) for x in Lambda])
                this[3]['trial_ID'] = t
                result.append(this[3])
                result_matrix.append(this[:3])
                fn.saveFileLambda(weight_each_Z,R1,k ,Lambda,
                    self.weight_matrix_name.format(cur_paras_,"z", t, 0),
                    case_study = k, ranking = ranking)
                fn.saveFileLambda(weight_each_S,R1,k ,Lambda,
                    self.weight_matrix_name.format(cur_paras_,"s", t, 0),
                    case_study = k, ranking = ranking)

                if save_factor:
                    fn.saveAllFactorMatricesLambda(this[0], \
                       R1,k,Save_Lambda, t, 
                       self.factor_matrix_name.format(0, '0'),
                       case_study = k, ranking = ranking)

                    fn.saveAllFactorMatricesLambda(this[1], \
                       R1,k,Save_Lambda, t, 
                       self.factor_matrix_name.format(0, '1'),
                       case_study = k, ranking = ranking)

                all_results[0] = this

                norm_X_1 = X_original.norm()
                norm_Y_1 = Y_original.norm()
                alpha, beta, gamma, delta, train_proportion = Lambda


                cost = fn.jtnorm_fro_err_nways(self,X_original, Y_original, 
                    all_results[0][0],all_results[0][1],
                    norm_X_1, norm_Y_1)

                _log.info("final cost:{}".format(cost))
                cost_log_final.append([t, 0, cost,t, 
                    ground_truth_k, k, noise, train_proportion, 
                    alpha, beta, delta, gamma, distance])                        

            except Exception as e:
                _log.info(e)
                # sys.exit(1)
                raise
                # continue

        cost_log_final = pd.DataFrame(cost_log_final, 
            columns=['iter', 'elapsed', 'cost', 
            'trial_ID','ground_truth_k', 'k', 'noise', 'train_proportion', 
            'alpha', 'beta', 'delta', 'gamma', 'distance'])
        alpha, beta, gamma, delta, train_proportion = Lambda

        fn.saveCSV(cost_log_final, R1, k, ground_truth_k, alpha, beta, gamma, delta,
            self.cost_matrix_name,
            case_study = k, ranking = ranking)

        return cost_log_final

    def kronecker(self, matrices, tensor):
        K = len(matrices)
        x = tensor

        for k in range(K):
            M = matrices[k]
            x = x.ttm(M, k)
        return x

    def format_output(self, Lambda, ground_truth_k, noise, distance, t, k, weight_each):
        alpha, beta, gamma, delta, train_proportion = Lambda
        weight_each['trial_ID'] = t
        weight_each['ground_truth_k'] = ground_truth_k
        weight_each['k'] = k
        weight_each['noise'] = noise
        weight_each['alpha'] = alpha
        weight_each['beta'] = beta
        weight_each['delta'] = delta
        weight_each['gamma'] = gamma
        weight_each['distance'] = distance
        return weight_each


    def iter_solver(self, X, Y,  X_factors, Y_factors, F, H, G, L, R1, R2, k, norm_X, norm_Y, alpha, beta, gamma):
        raise NotImplementedError

    def initializer(self,  X_factors, Y_factors):
        return ( X_factors, Y_factors)

class SDCDT(TensorFactorization_Base): #Baseline 3

    """ JNMF algorithm: 
    Baseline 3-Batch processing
    Block Coordinate Descent Framework + column regularization
    KDD model: Block Coordinate Descent Framework + column regularization
    Kim, Choo, Kim, Reddy and Park. Simultaneous Discovery of Common and Discriminative Topics via Joint non-negative Matrix factorization.
    """

    def __init__(self, default_max_iter=100, default_max_time=np.inf):
        self.eps = 1e-16
        self.set_default(default_max_iter, default_max_time)

    def iter_solver(self, sc, blocks, X, Y, Z1, Z2, S, \
        X_factors,Y_factors, R1, R2, k, norm_X, norm_Y, \
        Lambda, location_reg, D_matrix, W_matrix,Teneye_Z_X, Teneye_Z_Y, \
        E_X, E_Y, Z_X, Z_Y,alpha_k,cost_current,Z_weights_X,Z_weights_Y,\
        num_workers,distance = 1, current_iter = 1, tree_group = None,
        P_new = None, P_Z = None, Teneye_S_X = None, Teneye_S_Y = None, 
        level = None):


        alpha, beta, gamma, delta, train_proportion = Lambda
        by_norm = '2'
        X_itr = X_factors    
        Y_itr = Y_factors
        num_ways = len(X_itr)
        n1 = norm_X
        n2 = norm_Y
        
        k = distance

        X_d = [np.sum(each_factor[:,k:],axis=1) for each_factor in X_itr]
        Y_d = [np.sum(each_factor[:,k:],axis=1) for each_factor in Y_itr]

        X_new = ktensor(X_itr).totensor()
        Y_new = ktensor(Y_itr).totensor()
        
        X_FF_iter = []
        Y_FF_iter = []
        XtW_iter = []
        YtW_iter = []
        for way_index in range(num_ways):
            ways = list(range(num_ways))
            ways.remove(way_index)
            X_FF = np.ones((R1,R1))
            Y_FF = np.ones((R2,R2))
            # pdb.set_trace()           
            for w in ways:
                X_FF = X_FF * X_itr[w].T.dot(X_itr[w])
                Y_FF = Y_FF * Y_itr[w].T.dot(Y_itr[w])
            X_FF_iter.append(X_FF)
            Y_FF_iter.append(Y_FF)
            XtW_iter.append(X.uttkrp(X_itr, way_index))
            YtW_iter.append(Y.uttkrp(Y_itr, way_index))

        for l in range(R1):
            for way_index in range(num_ways):
                if l < k:
                    X_itr[way_index][:,l] = X_factors[way_index][:,l] * (X_FF_iter[way_index][l,l]) / (X_FF_iter[way_index][l,l] + n1*alpha) + (XtW_iter[way_index][:,l] - X_factors[way_index].dot(X_FF_iter[way_index])[:,l] + (n1*alpha)*Y_factors[way_index][:,l]) / (X_FF_iter[way_index][l,l] + n1*alpha + self.eps)
                    Y_itr[way_index][:,l] = Y_factors[way_index][:,l] * (Y_FF_iter[way_index][l,l]) / (Y_FF_iter[way_index][l,l] + n2*alpha) + (YtW_iter[way_index][:,l] - Y_factors[way_index].dot(Y_FF_iter[way_index])[:,l] + (n2*alpha)*X_factors[way_index][:,l]) / (Y_FF_iter[way_index][l,l] + n2*alpha + self.eps)

                else:

                    X_itr[way_index][:,l] = X_factors[way_index][:,l] + (XtW_iter[way_index][:,l] - X_factors[way_index].dot(X_FF_iter[way_index])[:,l] - (n1*beta/2)*Y_d[way_index]) / (X_FF_iter[way_index][l,l] + self.eps)
                    Y_itr[way_index][:,l] = Y_factors[way_index][:,l] + (YtW_iter[way_index][:,l] - Y_factors[way_index].dot(Y_FF_iter[way_index])[:,l] + (n2*beta/2)*X_d[way_index]) / (Y_FF_iter[way_index][l,l] + self.eps)

                X_itr[way_index][:,l][X_itr[way_index][:,l] < self.eps] = self.eps
                Y_itr[way_index][:,l][Y_itr[way_index][:,l] < self.eps] = self.eps


        X_itr = [fn.normalize_column(each_factor,by_norm='2')[0] if way_index < (num_ways - 1) else each_factor for way_index, each_factor in enumerate(X_itr)]
        Y_itr = [fn.normalize_column(each_factor,by_norm='2')[0] if way_index < (num_ways - 1) else each_factor for way_index, each_factor in enumerate(Y_itr)]

        alpha_k_new = 0
        return (X_itr,Y_itr,Teneye_S_X, Teneye_S_Y, 
            E_X, E_Y, Z_X, Z_Y,
            alpha_k_new,Z_weights_X,Z_weights_Y, P_new, P_Z)        

class PAIRFAC(TensorFactorization_Base): 

    """ PairFac algorithm: 
    Wen, X., Lin, Y. R., & Pelechrinis, K. (2016, October). 
    Pairfac: Event analytics through discriminant tensor factorization. 
    In Proceedings of the 25th ACM International on Conference on 
    Information and Knowledge Management (pp. 519-528). ACM.    
    """

    def __init__(self, default_max_iter=100, default_max_time=np.inf):
        self.eps = 1e-16
        self.set_default(default_max_iter, default_max_time)


    def computeGradient(self, block, i, X_U_new,Y_U_new):
        X_sub, Y_sub, Z1_sub, Z2_sub, S_sub = block['X_subs'],block['Y_subs'],block['ZX_subs'],block['ZY_subs'],block['S_subs']
        
        return X_sub.uttkrp(X_U_new, i), Y_sub.uttkrp(Y_U_new, i), \
                Z1_sub.uttkrp(X_U_new, i), Z2_sub.uttkrp(Y_U_new, i), \
                S_sub.uttkrp(X_U_new, i), S_sub.uttkrp(Y_U_new, i)

    def computeGradientZ(self, block, X_U_new,Y_U_new):
        Z1_sub, Z2_sub, S_sub = block['ZX_subs'],block['ZY_subs'],block['S_subs']
        X_U_new = [U.T for U in X_U_new]
        Y_U_new = [U.T for U in Y_U_new]
        return self.kronecker(X_U_new, Z1_sub), self.kronecker(Y_U_new, Z2_sub), self.kronecker(X_U_new, S_sub), self.kronecker(Y_U_new, S_sub)

    def iter_solver(self, sc, blocks, X, Y, Z1, Z2, S, 
        X_factors,Y_factors, R1, R2, k, norm_X, norm_Y, 
        Lambda, location_reg, D_matrix, W_matrix,Teneye_X, Teneye_Y, 
        E_X, E_Y, Z_X, Z_Y,alpha_k,cost_current,Z_weights_X,Z_weights_Y,
        num_workers,distance = 1, current_iter = 1, tree_group = None,
        P_new = None, P_Z = None, Teneye_S_X = None, Teneye_S_Y = None, 
        level = None):

        alpha, beta, gamma, delta, train_proportion = Lambda

        by_norm = '2'
        X_U = X_factors
        Y_U = Y_factors
        num_ways = len(X_U)
        
        lambda_1 = alpha
        lambda_2 = beta
        lambda_3 = gamma

        n1 = 1./norm_X
        n2 = 1./norm_Y
        X_U_new = X_U
        Y_U_new = Y_U
        weights_X = tuple(Teneye_X[tuple([i]*num_ways)] for i in range(R1))
        weights_X = weights_X/sum(weights_X)
        weights_Y = tuple(Teneye_Y[tuple([i]*num_ways)] for i in range(R2))
        weights_Y = weights_Y/sum(weights_Y)
        alpha_k_new = (1+np.sqrt(4*(alpha_k**2)+1)) / 2     
        
        for way_index in range(num_ways):
            ways = list(range(num_ways))
            ways.remove(way_index)
            X_FF = np.ones((R1,R1))
            Y_FF = np.ones((R2,R2))
            for w in ways:
                X_FF = X_FF * X_U_new[w].T.dot(X_U_new[w])
                Y_FF = Y_FF * Y_U_new[w].T.dot(Y_U_new[w])

            all_results = []
            results_X = np.zeros((X.shape[way_index],R1))
            results_Y = np.zeros((Y.shape[way_index],R2))
            results_Z1 = np.zeros((Z1.shape[way_index],R1))
            results_Z2 = np.zeros((Z2.shape[way_index],R2))
            results_S1 = np.zeros((S.shape[way_index],R1))
            results_S2 = np.zeros((S.shape[way_index],R2))

            results_ratio = 0
            for s in range(num_workers**(num_ways - 1)): 
                results = sc.parallelize(blocks[s], num_workers).map(lambda x: self.computeGradient(x, way_index, X_U_new,Y_U_new)).collect()
                for X_, Y_, Z1_, Z2_, S1_, S2_ in results:
                    results_X += X_
                    results_Y += Y_
                    results_Z1 += Z1_
                    results_Z2 += Z2_
                    results_S1 += S1_
                    results_S2 += S2_

            for l in range(R1): 

                column_regularization_X  = n1*lambda_3 * ((1-weights_X[l]) * X_U_new[way_index][:,l] - (1-weights_Y[l]) * Y_U_new[way_index][:,l]) * (1 - weights_X[l])
                column_regularization_Y  = n2*lambda_3 * ((1-weights_X[l]) * X_U_new[way_index][:,l] - (1-weights_Y[l]) * Y_U_new[way_index][:,l]) * (0 - (1 - weights_Y[l]))
                reg_X = n1*lambda_1 * (np.asarray(weights_X[l]) * (np.asarray(weights_X[l]) * X_U_new[way_index].dot(X_FF)[:,l] - results_Z1[:,l])) # Z term
                reg_Y = n2*lambda_1 * (np.asarray(weights_Y[l]) * (np.asarray(weights_Y[l]) * Y_U_new[way_index].dot(Y_FF)[:,l] - results_Z2[:,l])) # Z term

                reg_X += n1*lambda_2 * (np.asarray(1 - weights_X[l]) * (np.asarray(1 - weights_X[l]) * X_U_new[way_index].dot(X_FF)[:,l] - results_S1[:,l])) # S term
                reg_Y += n2*lambda_2 * (np.asarray(1 - weights_Y[l]) * (np.asarray(1 - weights_Y[l]) * Y_U_new[way_index].dot(Y_FF)[:,l] - results_S2[:,l])) # S term
                X_U_new[way_index][:,l] = Z_X[way_index][:,l] - (X_U_new[way_index].dot(X_FF)[:,l] - results_X[:,l] + reg_X + column_regularization_X) / (fn.norm_fro(X_FF)) # fn.norm_fro(X_FF)
                Y_U_new[way_index][:,l] = Z_Y[way_index][:,l] - (Y_U_new[way_index].dot(Y_FF)[:,l] - results_Y[:,l] + reg_Y + column_regularization_Y) / (fn.norm_fro(Y_FF))

                if way_index < (num_ways - 1):
                    X_U_new[way_index][:,l] = fn.normalize_column(X_U_new[way_index][:,l],by_norm='2')[0]
                    Y_U_new[way_index][:,l] = fn.normalize_column(Y_U_new[way_index][:,l],by_norm='2')[0]


                X_U_new[way_index][:,l][X_U_new[way_index][:,l] < self.eps] = 0
                Y_U_new[way_index][:,l][Y_U_new[way_index][:,l] < self.eps] = 0

            Z_X[way_index] = X_U_new[way_index] + ((alpha_k-1) / alpha_k_new) * (X_U_new[way_index] - X_U[way_index])
            Z_Y[way_index] = Y_U_new[way_index] + ((alpha_k-1) / alpha_k_new) * (Y_U_new[way_index] - Y_U[way_index])

        X_FF = np.ones((R1,R1))
        Y_FF = np.ones((R2,R2))
        for way_index in range(num_ways):
            X_FF = X_FF * X_U_new[way_index].T.dot(X_U_new[way_index])
            Y_FF = Y_FF * Y_U_new[way_index].T.dot(Y_U_new[way_index])

        results_Z1_ = np.zeros([R1]*num_ways)
        results_Z2_ = np.zeros([R2]*num_ways)
        results_S1_ = np.zeros([R1]*num_ways)
        results_S2_ = np.zeros([R2]*num_ways)

        for s in range(num_workers**(num_ways - 1)): 
            results = sc.parallelize(blocks[s], num_workers).map(lambda x: self.computeGradientZ(x, X_U_new,Y_U_new)).collect()
            for Z1_, Z2_, S1_, S2_ in results:
                results_Z1_ += Z1_
                results_Z2_ += Z2_
                results_S1_ += S1_
                results_S2_ += S2_

        weights_X_regularization = np.zeros((R1, R1))
        weights_Y_regularization = np.zeros((R1, R1))
        for way_index in range(num_ways):
            weights_X_regularization += -((X_U_new[way_index].T.dot((np.ones(R1) - np.asarray(weights_X))*X_U_new[way_index] - (np.ones(R1) - np.asarray(weights_Y))*Y_U_new[way_index])))
            weights_Y_regularization += ((Y_U_new[way_index].T.dot((np.ones(R1) - np.asarray(weights_X))*X_U_new[way_index] - (np.ones(R1) - np.asarray(weights_Y))*Y_U_new[way_index])))

        Teneye_X_new_tmp = np.zeros([R1]*num_ways)
        Teneye_Y_new_tmp = np.zeros([R2]*num_ways)
        for i in range(R1): Teneye_X_new_tmp[tuple([i]*num_ways)] = np.sum(weights_X_regularization[i,i])
        for i in range(R1): Teneye_Y_new_tmp[tuple([i]*num_ways)] = np.sum(weights_Y_regularization[i,i])

        deriviative_X = lambda_1 * (np.asarray(weights_X)*X_FF - results_Z1_)
        deriviative_X += lambda_2 * -((np.ones(R1) - np.asarray(weights_X))*X_FF - results_S1_)
        deriviative_X += lambda_3 * Teneye_X_new_tmp

        deriviative_Y = lambda_1 * (np.asarray(weights_Y)*Y_FF - results_Z2_)
        deriviative_Y += lambda_2 * -((np.ones(R1) - np.asarray(weights_Y))*Y_FF - results_S2_)
        deriviative_Y += lambda_3 * Teneye_Y_new_tmp

        Teneye_X_new = Z_weights_X - (deriviative_X) / (fn.norm_fro(X_FF))
        Teneye_Y_new = Z_weights_Y - (deriviative_Y) / (fn.norm_fro(Y_FF))
        Teneye_X_new[Teneye_X_new < self.eps] = 0
        Teneye_Y_new[Teneye_Y_new < self.eps] = 0

        weights_X_new = tuple(Teneye_X_new[tuple([i]*num_ways)] for i in range(R1))
        weights_X_new = weights_X_new/sum(weights_X_new)
        weights_Y_new = tuple(Teneye_Y_new[tuple([i]*num_ways)] for i in range(R2))
        weights_Y_new = weights_Y_new/sum(weights_Y_new)

        Teneye_X_new = np.zeros([R1]*num_ways)
        Teneye_Y_new = np.zeros([R2]*num_ways)

        for i in range(R1): Teneye_X_new[tuple([i]*num_ways)] = weights_X_new[i]
        for i in range(R1): Teneye_Y_new[tuple([i]*num_ways)] = weights_Y_new[i]

        for i in range(R1): Teneye_X[tuple([i]*num_ways)] = weights_X[i]
        for i in range(R1): Teneye_Y[tuple([i]*num_ways)] = weights_Y[i]


        Z_weights_X = Teneye_X_new + ((alpha_k-1) / alpha_k_new) * (Teneye_X_new - Teneye_X)
        Z_weights_Y = Teneye_Y_new + ((alpha_k-1) / alpha_k_new) * (Teneye_Y_new - Teneye_Y)

        return (X_U_new,Y_U_new,
            Teneye_X_new, Teneye_Y_new, E_X, E_Y, Z_X, Z_Y,
            alpha_k_new,Z_weights_X,Z_weights_Y, P_new, P_Z)



