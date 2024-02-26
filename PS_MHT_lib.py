import numpy as np
from scipy.stats import ncx2
from scipy.stats import chi2

import os, sys

import pandas as pd
import geopandas as gp
from shapely.geometry import Point

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
import datetime

def get_lambda_o_est(alpha,q,gamma):# gamma could be a list
    crtvlue=chi2.ppf(1-alpha, df=q)#chi2inv(1-a,q)
    beta_o = 1 - np.array(gamma)
    Err=5e-5
    lambda_o = np.empty((len(gamma)))
    for i in range(len(gamma)):
        low, high = 0, 100
        mid=(low+high)/2
        
        while high-low>Err:
            if ncx2.cdf(crtvlue,q,mid) > beta_o[i]:
                low=mid
                mid=(high+low)/2
            elif ncx2.cdf(crtvlue,q,mid) < beta_o[i]:
                high = mid
                mid=(high+low)/2
            else:
                high = mid
                low = mid
        lambda_o[i] = mid
    return lambda_o


def get_K_alpha(B_temp):
    alpha = 1/(2*B_temp.size)
    #Kritical = chi2.ppf(1-alpha,df = B_temp.size-1);
    gamma = 0.5
    lambda1 = get_lambda_o_est(alpha,1,[gamma])
    
    crivalue_q_dict = {}
    
    crivalue_q_dict[1] = ncx2.ppf(1-gamma, 1, lambda1[0])
    crivalue_q_dict[2] = ncx2.ppf(1-gamma, 2, lambda1[0])
    crivalue_q_dict[3] = ncx2.ppf(1-gamma, 3, lambda1[0])
    crivalue_q_dict[4] = ncx2.ppf(1-gamma, 4, lambda1[0])
    crivalue_q_dict[5] = ncx2.ppf(1-gamma, B_temp.size-1, lambda1[0])
    
    #print(crivalue_q)
    return [alpha, crivalue_q_dict]

def get_A_matrix(B_temp, choice=1):
    #choice (int) = 1: default model: defomation with linear velocity
    if choice==1:
        A = np.vstack((B_temp, np.ones(B_temp.size))).T
        #A = np.kron(np.identity(num_points), A)
    
    if choice==2:
        omega =2*np.pi/0.2 #lambda = 1
        A = np.sin(omega*B_temp)[np.newaxis,...].T
        #A = np.vstack((np.sin(omega*B_temp))).T
        #A = np.kron(np.identity(num_points), A)
        
    if choice==3:
        omega =2*np.pi/0.2 #lambda = 1
        #A = np.sin(omega*B_temp)[np.newaxis,...].T
        A = np.vstack((np.sin(omega*B_temp), np.ones(B_temp.size))).T
        #A = np.kron(np.identity(num_points), A)
        
    if choice==4:
        omega =2*np.pi/0.2 #lambda = 1
        
        A = np.vstack((B_temp, np.sin(omega*B_temp), np.ones(B_temp.size))).T
        #A = np.kron(np.identity(num_points), A)
    return A

def least_sq(y, A, Q):
    Q_inv = np.linalg.inv(Q)
    Q_x = np.linalg.inv(A.T @ Q_inv @ A)
    x_blue = Q_x @ A.T @ Q_inv @ y
    return x_blue

def get_A_matrix(B_temp, choice=1):
    #choice (int) = 1: default model: defomation with linear velocity
    if choice==1:
        A = np.vstack((B_temp, np.ones(B_temp.size))).T
        #A = np.kron(np.identity(num_points), A)
    
    if choice==2:
        omega =2*np.pi/0.2 #lambda = 1
        A = np.sin(omega*B_temp)[np.newaxis,...].T
        #A = np.vstack((np.sin(omega*B_temp))).T
        #A = np.kron(np.identity(num_points), A)
        
    if choice==3:
        omega =2*np.pi/0.2 #lambda = 1
        #A = np.sin(omega*B_temp)[np.newaxis,...].T
        A = np.vstack((np.sin(omega*B_temp), np.ones(B_temp.size))).T
        #A = np.kron(np.identity(num_points), A)
        
    if choice==4:
        omega =2*np.pi/0.2 #lambda = 1
        
        A = np.vstack((B_temp, np.sin(omega*B_temp), np.ones(B_temp.size))).T
        #A = np.kron(np.identity(num_points), A)
        
    if choice==5:
        omega = 2*np.pi
        A = np.vstack((B_temp, np.sin(omega*B_temp), np.cos(omega*B_temp), np.ones(B_temp.size))).T
    
    return A

def least_sq(y, A, Q):
    Q_inv = np.linalg.inv(Q)
    Q_x = np.linalg.inv(A.T @ Q_inv @ A)
    x_blue = Q_x @ A.T @ Q_inv @ y
    return x_blue

def MHT(B_temp, defo_df):
    #THis function works on the def series per point.
    #tests fitting of various deformation trend model of the data
    #uses stochastic model
    
    sd = np.sqrt(5)#0.01
    
    num_points, epochs = defo_df.shape
    A = get_A_matrix(B_temp, choice=1)
    Q = sd * np.identity(epochs)
    Q_inv = np.linalg.inv(Q)
    Q_x = np.linalg.inv(A.T @ Q_inv @ A)
    Q_e = Q - A @ Q_x @ A.T
    
    K_a = get_K_alpha(B_temp)
    print('K_a', K_a)
    alpha, crivalue_q_dict = K_a
    
    model_selector={}
    Tr_q3_list=[]
    Tr_qbp_list=[]
    
    M_7 = np.tril(np.ones((B_temp.size-1, B_temp.size-1)))
    M_7 = np.vstack((np.zeros((1,B_temp.size-1)), M_7))
    #print(M_7)
    #for Heaviside
    Ling4 = []
    for epo in range(epochs-1):
        #Ling4(i,:,:) = inv(Q)*M_7(:,i)*inv(M_7(:,i)'*inv(Q)*Q_e*inv(Q)*M_7(:,i))*M_7(:,i)'*inv(Q);
        M_7_test_epo = M_7[:,epo][...,np.newaxis]
        
        temp_inv = 1/(M_7_test_epo.T @ Q_inv @ Q_e @ Q_inv @ M_7[:,epo])
        #print(temp_inv)
        poora = Q_inv @ M_7_test_epo *temp_inv @ M_7_test_epo.T @ Q_inv
        
        Ling4.append(poora)
    
    ##for breakpoint
    Ling5 = []
    for epo in range(epochs-3):# changed epochs-2 to epochs-3
        M_bp_i = np.hstack((np.zeros((epo+2)), B_temp[epo+2:]-B_temp[epo+1]))# changed epo+2 -> epo +_1 and vice versa
        #M_bp.append(Q_inv @ M_bp_i)
        M_bp_i = M_bp_i[...,np.newaxis]
        
        temp_inv = 1/(M_bp_i.T @ Q_inv @ Q_e @ Q_inv @ M_bp_i)
        #print(temp_inv)
        poora = Q_inv @ M_bp_i *temp_inv @ M_bp_i.T @ Q_inv
        
        #plt.imshow(poora)
        #plt.show()
        
        Ling5.append(poora)
    
    #Step 1: apply linear model
    e_o_arr = np.zeros((num_points, epochs))
    for i in range(num_points):
        y_bar = defo_df.iloc[i] #observation matrix
        #fit linear model:
        #A = get_A_matrix(B_temp, choice=1) #1:default model, 2: sin model
        x_lin = least_sq(y_bar, A, Q)
        #calculate error in the default model
        e_o_arr[i,:] =  y_bar - A @ x_lin
    
    #Step 2: correct the error of the reference point
    ref_point_err = e_o_arr.mean(axis=0)
    
    # Step 3: subtract the error
    defo_df = defo_df - ref_point_err
    
    #Step 4: recompute the residuals
    T_lin_arr = np.zeros((num_points))
    T_sin_arr = np.zeros((num_points))
    T_q3_arr = np.zeros((num_points, epochs-1))
    T_qbp_arr = np.zeros((num_points, epochs-3))
    
    for i in range(num_points):
        #Q_e = Q - A @ Q_x_lin @ A.T
        y_bar = defo_df.iloc[i] #observation matrix
        #fit linear model:
        #A = get_A_matrix(B_temp, choice=1) #1:default model, 2: sin model
        x_lin = least_sq(y_bar, A, Q)
        #calculate error in the default model
        e_o =  y_bar - A @ x_lin
        
        '''
        plt.plot(B_temp, y_bar, '.')
        plt.plot(B_temp, A @ x_lin)
        #plt.ylim((min(y_bar),max(y_bar)))
        plt.plot(B_temp, e_o, 'x')
        plt.show()
        '''
        T_1 = e_o.T @ Q_inv @ e_o
        #T_1_list.append(T_1)
        #print(T_1)
        
        #sinusoidal
        A_sin = get_A_matrix(B_temp, choice=5)
        x_sin = least_sq(y_bar, A_sin, Q)
        e_sin = y_bar - A_sin @ x_sin
        '''
        plt.plot(B_temp, y_bar, '.')
        plt.plot(B_temp, A_sin @ x_sin)
        #plt.ylim((min(y_bar),max(y_bar)))
        #plt.plot(B_temp, e_o, 'x')
        plt.show()
        '''
        T_sin = e_sin.T @ Q_inv @ e_sin
        
        #continue
        #print(i)
        if (T_1 < crivalue_q_dict[5]):
            #print('linear model selected')
            #model_selector[point_id]=x_lin
            T_lin_arr[i] = T_1
            #print('T_1', T_1)
        elif T_sin < crivalue_q_dict[5]:# and T_1 >= crivalue_q_dict[5]:
            T_sin_arr[i] = T_sin
            #print('T_sin', T_sin)
        
        else:
        #if ~((T_1 < crivalue_q_dict[5]) | (T_sin < crivalue_q_dict[5])):
            crivalue_new = crivalue_q_dict[1]
            #if(model_choice=='breakpoint'):
            for p in range(epochs-3):
                #T_qbp.append(e_o.values[...,np.newaxis].T @ Ling5[p] @ e_o.values)
                T_qbp_arr[i, p] = e_o.values[...,np.newaxis].T @ Ling5[p] @ e_o.values
            #if(model_choice=='Heaviside'):
            for p in range(epochs-1):
                #T_q3.append(e_o.values[...,np.newaxis].T @ Ling4[p] @ e_o.values)
                T_q3_arr[i, p] = e_o.values[...,np.newaxis].T @ Ling4[p] @ e_o.values
        
    crivalue_new = crivalue_q_dict[1]
    
    Tr_q3_arr = T_q3_arr/crivalue_new
    Tr_qbp_arr = T_qbp_arr/crivalue_new
    Tr_lin_arr = T_lin_arr/crivalue_new
    Tr_sin_arr = T_sin_arr/crivalue_q_dict[2]
    '''
    plt.subplot(121)
    plt.plot(B_temp[:-1], Tr_q3_arr.T, '.-')
    plt.xlabel('Temporal Baseline (yr)')
    plt.ylabel('Test Ratio')
    plt.title('Heaviside')
    plt.subplot(122)
    plt.plot(B_temp[:-3], Tr_qbp_arr.T, '.-')
    plt.title('Breakpoint')
    plt.show()
    '''
    #Return test statistic scores
    #return T_q3_arr, T_qbp_arr, T_lin_arr, T_sin_arr
    
    #Return test ratios
    return Tr_q3_arr, Tr_qbp_arr, Tr_lin_arr, Tr_sin_arr
