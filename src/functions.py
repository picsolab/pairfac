import numpy as np
from numpy import zeros, ones, diff, kron, tile, any, all, linalg
from functools import reduce
import numpy.linalg as nla
import time
from sktensor import sptensor, ktensor, dtensor,cp_als
from sktensor.core import khatrirao
import pandas as pd
import pdb
import logging
import sys,os,csv
import math
import scipy
import scipy.sparse as sps
import random
from termcolor import colored
import spm1d

PROJECT_DIR = '/media/ext01/xidao/project/hipairfac'


def jtnorm_fro_err_nways(self, X, Y, X_all, Y_all, norm_X, norm_Y):
    """ Compute the approximation error in Frobeinus norm

    norm(X - W.dot(H.T)) is efficiently computed based on trace() expansion 
    when W and H are thin.

    Parameters
    ----------
    X : numpy.array or scikit tensor, shape (m,n,o)
    X_U : numpy.array, shape (m,R1)
    norm_X : precomputed norm of X

    Returns
    -------
    float
    """
    F_ktensor_X = ktensor(X_all)
    F_ktensor_Y = ktensor(Y_all)
    error_X = getError(X,F_ktensor_X,norm_X)
    error_Y = getError(Y,F_ktensor_Y,norm_Y)
    cost = ((math.sqrt(np.maximum(error_X, 0)))/norm_X + (math.sqrt(np.maximum(error_Y, 0)))/norm_Y) / 2.

    return cost

def saveCSV(data, R,K,ground_truth_k,alpha, beta, gamma, noise,fname, case_study = "python_alto", ranking = False, sub_dir = "classification"):

    folder_loc = PROJECT_DIR + "/output/output_"+case_study+"_"+sub_dir+""
    fileName = folder_loc + "/" + str(fname)+ "_R_" + str(R) + "_gr_" +str(ground_truth_k) + "_k_"+ str(K)+ "_alpha_" + str(alpha)+ "_beta_" + str(beta)+  "_gamma_" + str(gamma) + "_noise_" + str(noise)
    pd.DataFrame(data).to_csv(fileName,sep=',',index=False,header=True) 

def saveFileLambda(data, R,K,Lambda, fname, case_study = "python_alto", ranking = False, sub_dir = "classification"):
    

    folder_loc = PROJECT_DIR + "/output/output_"+case_study+"_"+sub_dir+""

    paras = '_'.join([str(x) for x in Lambda])
    fileName = folder_loc + "/" + str(fname)+ "_R_" + str(R)
    pd.DataFrame(data).to_csv(fileName,sep=',',index=False,header=True) 
    
def saveAllFactorMatricesLambda(U_all, R,K,Lambda, iteration,fname, case_study = "python_alto", ranking = False, sub_dir = "classification"):

    folder_loc = PROJECT_DIR + "/output/output_"+case_study+"_"+sub_dir+""
    paras = '_'.join([str(x) for x in Lambda])
    for i in range(len(U_all)):
        fileName = "{0}/factor_matrices/_U{1}_R{2}_k_{3}_{4}_iter_{5}_{6}".format(folder_loc, i, R, K, paras, iteration, fname)
        pd.DataFrame(U_all[i]).to_csv(fileName,sep=',',index=False,header=False)

def saveProjectionLambda(P_X, P_Y, R,K,Lambda, iteration,fname, case_study = "python_alto", ranking = False, sub_dir = "classification"):

    folder_loc = PROJECT_DIR + "/output/output_"+case_study+"_"+sub_dir+""

    paras = '_'.join([str(x) for x in Lambda])
    fileName = folder_loc + "/projection_matrices/projection_X_" + str(R) + "_k_" + str(K)+ "_" +paras+ "_iter_" + str(iteration) + "_" + str(fname)
    pd.DataFrame(P_X).to_csv(fileName,sep=',',index=False,header=False)
    fileName = folder_loc + "/projection_matrices/projection_Y_" + str(R) + "_k_" + str(K)+ "_" +paras+ "_iter_" + str(iteration) + "_" + str(fname)
    pd.DataFrame(P_Y).to_csv(fileName,sep=',',index=False,header=False)

def getError(X, F_kten, norm_X):
    
    return norm_X ** 2 + F_kten.norm() ** 2 - 2 * F_kten.innerprod(X)

def norm_fro(X):
    """ Compute the Frobenius norm of a matrix

    Parameters
    ----------
    X : numpy.array or scipy.sparse matrix

    Returns
    -------
    float
    """
    if sps.issparse(X):     # scipy.sparse array
        return math.sqrt(X.multiply(X).sum())
    else:                   # numpy array
        return nla.norm(X)

def issparsetensor(X):
    return str(X.__class__) == 'sktensor.sptensor.sptensor'

def solve(AtA, AtB):
    try:
        soln = nla.solve(AtA, AtB)
    except np.linalg.LinAlgError:
        soln = nla.lstsq(AtA, AtB)[0]
    except Exception as e:
        raise e
    return soln

def save(path, fig, ext='png', close=True, verbose=True):
    """Save a figure from pyplot.
    Parameters
    ----------
    path : string
        The path (and filename, without the extension) to save the
        figure to.
    ext : string (default='png')
        The file extension. This must be supported by the active
        matplotlib backend (see matplotlib.backends module).  Most
        backends support 'png', 'pdf', 'ps', 'eps', and 'svg'.
    close : boolean (default=True)
        Whether to close the figure after saving.  If you want to save
        the figure multiple times (e.g., to multiple formats), you
        should NOT close it in between saves or you will have to
        re-plot it.
    verbose : boolean (default=True)
        Whether to print information about when and where the image
        has been saved.
    """
    
    # Extract the directory and filename from the given path
    directory = os.path.split(path)[0]
    filename = "%s.%s" % (os.path.split(path)[1], ext)
    if directory == '':
        directory = '.'

    # If the directory does not exist, create it
    if not os.path.exists(directory):
        os.makedirs(directory)

    # The final path to save to
    savepath = os.path.join(directory, filename)

    if verbose:
        print("Saving figure to '%s'..." % savepath),

    # Actually save the figure
    fig.savefig(savepath,dpi=(450))
    
    # Close it
    # if close:
        # fig.close()

    if verbose:
        print("Done")

def plot(F,G,H, L, M, N, R1,R2,m,n,o, it):

    # if it == 90 or it == 100 or it == 110 or it == 110:
    # saveFactorIntermediaOutput(F,G,it)
    fig,((ax1,ax2),(ax3,ax4),(ax5,ax6)) = plt.subplots(nrows=3,ncols=2,figsize=(20,20))

    ax1.matshow(F, origin="upper",extent=[0,R1,0,m],aspect=float(R1)/m)
    ax2.matshow(G, origin="upper",extent=[0,R2,0,m],aspect=float(R1)/m)
    ax3.matshow(H, origin="upper",extent=[0,R1,0,n],aspect=float(R1)/n)
    ax4.matshow(L, origin="upper",extent=[0,R2,0,n],aspect=float(R1)/n)
    ax5.matshow(M, origin="upper",extent=[0,R2,0,o],aspect=float(R1)/o)
    ax6.matshow(N, origin="upper",extent=[0,R2,0,o],aspect=float(R1)/o)

    fname = "fig/jntf_step_" + str(it)
    save(fname,fig, ext="png",close=True,verbose=True)
    plt.clf()
    plt.close()

    # plt.clf()

def showplot(F,G,H, L, M, N, R1,R2,m,n,o):

    # if it == 90 or it == 100 or it == 110 or it == 110:
    # saveFactorIntermediaOutput(F,G,it)
    fig,((ax1,ax2),(ax3,ax4),(ax5,ax6)) = plt.subplots(nrows=3,ncols=2,figsize=(20,20))

    ax1.matshow(F, origin="upper",extent=[0,R1,0,m],aspect=float(R1)/m)
    ax2.matshow(G, origin="upper",extent=[0,R2,0,m],aspect=float(R1)/m)
    ax3.matshow(H, origin="upper",extent=[0,R1,0,n],aspect=float(R1)/n)
    ax4.matshow(L, origin="upper",extent=[0,R2,0,n],aspect=float(R1)/n)
    ax5.matshow(M, origin="upper",extent=[0,R2,0,o],aspect=float(R1)/o)
    ax6.matshow(N, origin="upper",extent=[0,R2,0,o],aspect=float(R1)/o)

    # fname = "fig/jntf_step_" + str(it)
    # save(fname,fig, ext="png",close=True,verbose=True)
    plt.show()
    # plt.close()

    # plt.clf()

def column_norm(X, by_norm='2'):
    """ Compute the norms of each column of a given matrix

    Parameters
    ----------
    X : numpy.array or scipy.sparse matrix

    Optional Parameters
    -------------------
    by_norm : '2' for l2-norm, '1' for l1-norm.
              Default is '2'.

    Returns
    -------
    numpy.array
    """
    if sps.issparse(X):
        if by_norm == '2':
            norm_vec = np.sqrt(X.multiply(X).sum(axis=0))
        elif by_norm == '1':
            norm_vec = X.sum(axis=0)
        return np.asarray(norm_vec)[0]
    else:
        if by_norm == '2':
            norm_vec = np.sqrt(np.sum(X * X, axis=0))
        elif by_norm == '1':
            norm_vec = np.sum(X, axis=0)
        return norm_vec

def normalize_column(X, by_norm='2'):
    """ Column normalization

    Scale the columns of X so that they have unit l2-norms.
    The normalizing coefficients are also returned.

    Side Effect
    -----------
    X given as input are changed and returned

    Parameters
    ----------
    X : numpy.array or scipy.sparse matrix

    Returns
    -------
    ( X, weights )
    X : normalized matrix
    weights : numpy.array, shape k 
    """
    if sps.issparse(X):
        weights = column_norm(X, by_norm)
        # construct a diagonal matrix
        dia = [1.0 / w if w > 0 else 1.0 for w in weights]
        N = X.shape[1]
        r = np.arange(N)
        c = np.arange(N)
        mat = sps.coo_matrix((dia, (r, c)), shape=(N, N))
        Y = X.dot(mat)
        return (Y, weights)
    else:
        norms = column_norm(X, by_norm)
        toNormalize = norms > 0
        X[:, toNormalize] = X[:, toNormalize] / norms[toNormalize]
        weights = np.ones(norms.shape)
        weights[toNormalize] = norms[toNormalize]
        # pdb.set_trace()
        return (X, weights)
