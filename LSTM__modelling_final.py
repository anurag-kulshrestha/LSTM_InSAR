#/////////////////////////////////////////////////////
#               LIBRARY IMPORTS
#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
import os, argparse, sys
from doris_DePSI_Post_lib import read_PS_defo_Btemp_data_1, read_mrm_file, read_depsi_param_file, plot_mrm_slc_ps_defo, get_param_file_name
from PS_MHT_lib import MHT, get_A_matrix, least_sq
import matplotlib.pyplot as plt
import numpy as np
from LSTM_modelling import plot_real_anomaly_samples, put_jump_threshold, craft_real_samples, divide_train_test_samples, LSTM_class_1, LSTM_class, analyze_pol_decomp, plot_decomposition_RGB
#from time_series_augmentation.utils import augmentation as aug
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
from matplotlib import cm
import pandas as pd
from scipy.cluster.vq import kmeans2 as kmeans



def apply_percentile_thresholds(Tr_q3_Heaviside, Tr_q3_breakpoint, Tr_q3_lin, Tr_q3_sin, percentile_anomaly, percentile_anomaly_low):
    hea_perc_th = Tr_q3_Heaviside > np.percentile(Tr_q3_Heaviside,percentile_anomaly)
    hea_low_th = Tr_q3_Heaviside < np.percentile(Tr_q3_Heaviside, percentile_anomaly_low)
    
    brp_perc_th = Tr_q3_breakpoint > np.percentile(Tr_q3_breakpoint,percentile_anomaly)
    brp_low_th = Tr_q3_breakpoint < np.percentile(Tr_q3_breakpoint, percentile_anomaly_low)
    #lin_model_pt_th = np.where(Tr_q3_lin!=0)#
    #print('np.percentile(Tr_q3_lin, percentile_lin)', np.nanpercentile(Tr_q3_lin, percentile_lin))
    lin_model_pt_th = Tr_q3_lin < np.nanpercentile(Tr_q3_lin, percentile_lin) #np.where(Tr_q3_lin!=0)
    sin_model_pt_th = Tr_q3_sin < np.nanpercentile(Tr_q3_sin, percentile_sin) #np.where(Tr_q3_sin!=0)
    
    #find points where at least one data point passes through the threshold
    hea_perc_th_pts = np.intersect1d(np.unique(np.where(hea_perc_th)[0]), np.unique(np.where(brp_low_th)[0]))
    brp_perc_th_pts = np.intersect1d(np.unique(np.where(brp_perc_th)[0]), np.unique(np.where(hea_low_th)[0])) 
    
    hea_perc_th_pts = np.unique(np.where(hea_perc_th)[0])
    brp_perc_th_pts = np.unique(np.where(brp_perc_th)[0])
    lin_perc_th_pts = np.unique(np.where(lin_model_pt_th)[0])
    sin_perc_th_pts = np.unique(np.where(sin_model_pt_th)[0])
    
    #find the pointwise epoch where test ratio is maximum for that point
    heaviside_max_epochs = Tr_q3_Heaviside[hea_perc_th_pts].argmax(axis=1)
    breakpoint_max_epochs = Tr_q3_breakpoint[brp_perc_th_pts].argmax(axis=1)
    
    return lin_perc_th_pts, sin_perc_th_pts, hea_perc_th_pts, brp_perc_th_pts, heaviside_max_epochs, breakpoint_max_epochs

def epoch_thresh(hea_perc_th_pts, brp_perc_th_pts, heaviside_max_epochs, breakpoint_max_epochs):
    
    hea_epoch_th = (heaviside_max_epochs > epoch_low_th) & (heaviside_max_epochs < epoch_high_th)
    brp_epoch_th = (breakpoint_max_epochs > epoch_low_th) & (breakpoint_max_epochs < epoch_high_th)
    
    #apply the threshold
    #Heaviside
    hea_perc_th_pts = hea_perc_th_pts[hea_epoch_th]
    heaviside_max_epochs = heaviside_max_epochs[hea_epoch_th]
    #breakpoint
    brp_perc_th_pts = brp_perc_th_pts[brp_epoch_th]
    breakpoint_max_epochs = breakpoint_max_epochs[brp_epoch_th]
    
    return hea_perc_th_pts, brp_perc_th_pts, heaviside_max_epochs, breakpoint_max_epochs

def plot_lin_sample(ps_gdf, defo_df, B_temp, lin_perc_th_pts, Tr_q3_lin, sd=5, plotting=True, printing=True):
    for lin_pts in lin_perc_th_pts:
        y = defo_df.iloc[lin_pts]
        A = get_A_matrix(B_temp, choice=1)
        x = least_sq(y, A, sd * np.identity(B_temp.size))
        if printing:
            print('OMT value = ', Tr_q3_lin[lin_pts])
            if ps_gdf is not None:
                print(ps_gdf.iloc[lin_pts])
                print('Ens_coh = ' , ps_gdf['ens_coh'].iloc[lin_pts])
            
        if plotting:
            fig, ax1 = plt.subplots()
            #ax1.plot(B_temp, y, '.k')
            #ax1.plot(B_temp, A @ x, '-k')
            
            ax1.plot(np.arange(B_temp.size), y, '.k')
            ax1.plot(np.arange(B_temp.size), A @ x, '-k')
            
            #ax2.plot(B_temp[:-epoch_diff], Tr_q3[hea_pt])
            ax1.set_ylim(-50,15)
            #ax1.set_xlabel('Temporal baseline (yrs)')
            ax1.set_xlabel('InSAR Epochs')
            ax1.set_ylabel('Deformation (mm)')
            plt.tight_layout()
            plt.savefig('/home/anurag/Documents/PhDProject/Papers/Sec_paper/Images/python_high_res_plots/{}_PSX_{}_Y{}.png'.format('Lin', int(ps_gdf.iloc[lin_pts]['X']), int(ps_gdf.iloc[lin_pts][['Y']])), dpi=300)
            plt.show()
            
def plot_sin_sample(ps_gdf, defo_df, B_temp, sin_perc_th_pts, sd=5, plotting=True):
    for sin_pts in lin_perc_th_pts:
        y = defo_df.iloc[sin_pts]
        A = get_A_matrix(B_temp, choice=5)
        x = least_sq(y, A, sd * np.identity(B_temp.size))
        if plotting:
            fig, ax1 = plt.subplots()
            ax1.plot(B_temp, y, '.k')
            
            ax1.plot(B_temp, A @ x, '-k')
            
            #ax2.plot(B_temp[:-epoch_diff], Tr_q3[hea_pt])
            ax1.set_ylim(-50,15)
            ax1.set_xlabel('Temporal baseline (yrs)')
            ax1.set_ylabel('Deformation (mm)')
            plt.tight_layout()
            plt.savefig('/home/anurag/Documents/PhDProject/Papers/Sec_paper/Images/python_high_res_plots/{}_PSX_{}_Y{}.png'.format('Sin', int(ps_gdf.iloc[sin_pts]['X']), int(ps_gdf.iloc[sin_pts][['Y']])), dpi=300)
            plt.show()

def get_jumps_vel_thetas(defo_df, B_temp, perc_th_pts, max_epochs, Tr_q3, sd=5, plotting=False, printing=False, ps_gdf = None, label='Hea'):
    #jumps = defo_df.values[perc_th_pts,max_epochs] - defo_df.values[perc_th_pts,max_epochs+1]
    jumps = []
    vel_diff, vel_diff_perc, theta = [], [], []
    epoch_diff = B_temp.size - Tr_q3.shape[1]
    for count, (hea_pt, hea_epoch) in enumerate(zip(perc_th_pts, max_epochs)):
        #print(hea_epoch)
        #if count<21:#45
            #continue
        
            
        y_before, y_after = defo_df.iloc[hea_pt,:hea_epoch], defo_df.iloc[hea_pt,hea_epoch:]
        A_bef, A_af = get_A_matrix(B_temp[:hea_epoch], choice=1), get_A_matrix(B_temp[hea_epoch:], choice=1)
        x_bef, x_af = least_sq(y_before, A_bef, sd * np.identity(hea_epoch)), least_sq(y_after, A_af, sd * np.identity(B_temp.size-hea_epoch))
        
        m1, m2 = x_bef[0], x_af[0]
        jump = y_before.iloc[-1] - y_after.iloc[0]
        _theta = np.arctan((m2-m1)/(1+m1*m2))*180/np.pi
        jumps.append(jump)
        theta.append(_theta)
        vel_diff.append(m2-m1)
        vel_diff_perc.append((m2-m1)/m1)
        
        if ps_gdf is not None:
            print(ps_gdf.iloc[hea_pt])
            print('Ens_coh = ' , ps_gdf['ens_coh'].iloc[hea_pt])
        
        if printing:
            print('jump', jump)
            print('hea_epoch', hea_epoch)
            print('Velocities', m1, m2)
            print('velocity difference', m2 - m1)
            #print('velocity difference perc', (m1-m2)/m1)
            print('theta', _theta)
        
        if plotting:
            fig, ax1 = plt.subplots()
            ax2 = ax1.twinx()
            #ax1.plot(B_temp[:hea_epoch], y_before, '.k')
            #ax1.plot(B_temp[hea_epoch:], y_after, '.k')
            #ax1.plot(B_temp[:hea_epoch], A_bef @ x_bef, '-k')
            #ax1.plot(B_temp[hea_epoch:], A_af @ x_af, '-k')
            #ax2.plot(B_temp[:-epoch_diff], Tr_q3[hea_pt])
            
            ax1.plot(np.arange(hea_epoch), y_before, '.k')
            ax1.plot(np.arange(hea_epoch, B_temp.size), y_after, '.k')
            ax1.plot(np.arange(hea_epoch), A_bef @ x_bef, '-k')
            ax1.plot(np.arange(hea_epoch, B_temp.size), A_af @ x_af, '-k')
            #ax2.plot(np.arange(B_temp.size-3), Tr_q3[hea_pt])
            
            ax1.set_ylim(-50,15)
            ax1.set_xlabel('InSAR Epochs')
            ax1.set_ylabel('Deformation (mm)')
            ax2.set_ylabel('Model test statistic')
            plt.tight_layout()
            #plt.savefig('/home/anurag/Documents/PhDProject/Papers/Sec_paper/Images/python_high_res_plots/{}_PSX_{}_Y{}.png'.format(label, int(ps_gdf.iloc[hea_pt]['X']), int(ps_gdf.iloc[hea_pt][['Y']])), dpi=300)
            plt.show()
            
    return np.array(jumps), np.array(vel_diff), np.array(vel_diff_perc), np.array(theta)

def jump_vel_gra_threshold(hea_jump_min, hea_jump_max, brp_jump_max, hea_theta_max, brp_theta_min, brp_theta_max, defo_df, B_temp, hea_perc_th_pts, brp_perc_th_pts, heaviside_max_epochs, breakpoint_max_epochs):
    
    jumps_brp, vel_diff_brp, vel_diff_perc_brp, theta_brp = get_jumps_vel_thetas(defo_df, B_temp, brp_perc_th_pts, breakpoint_max_epochs, Tr_q3_breakpoint, plotting=False, printing=False)
    jumps_hea, vel_diff_hea, vel_diff_perc_hea, theta_hea = get_jumps_vel_thetas(defo_df, B_temp, hea_perc_th_pts, heaviside_max_epochs+1, Tr_q3_Heaviside, plotting=False)
    
    hea_vel_thresh = abs(theta_hea) < hea_theta_max
    hea_jump_thresh = (abs(jumps_hea) > hea_jump_min) & (abs(jumps_hea)<hea_jump_max)
    hea_thresh = hea_vel_thresh*hea_jump_thresh
    #apply_thresh
    hea_perc_th_pts = hea_perc_th_pts[hea_thresh]
    heaviside_max_epochs = heaviside_max_epochs[hea_thresh]
    
    #Brp VELOCITY AND JUMP THRESHOLD
    brp_vel_thresh = (abs(theta_brp)>brp_theta_min) & (abs(theta_brp)<brp_theta_max)
    brp_jump_thresh = abs(jumps_brp)<brp_jump_max
    brp_thresh = brp_vel_thresh*brp_jump_thresh
    #apply_thresh
    brp_perc_th_pts = brp_perc_th_pts[brp_thresh]
    breakpoint_max_epochs = breakpoint_max_epochs[brp_thresh]
    
    return hea_perc_th_pts, brp_perc_th_pts, heaviside_max_epochs, breakpoint_max_epochs

def plot_jumps_slope(hea_perc_th_pts, brp_perc_th_pts, heaviside_max_epochs, breakpoint_max_epochs):
    jumps_brp, vel_diff_brp, vel_diff_perc_brp, theta_brp = get_jumps_vel_thetas(defo_df, B_temp, brp_perc_th_pts, breakpoint_max_epochs, Tr_q3_breakpoint, plotting=False, printing=False)
    jumps_hea, vel_diff_hea, vel_diff_perc_hea, theta_hea = get_jumps_vel_thetas(defo_df, B_temp, hea_perc_th_pts, heaviside_max_epochs+1, Tr_q3_Heaviside, plotting=False)
    
    from scipy.stats import gaussian_kde
    x, y = abs(jumps_brp), abs(theta_brp)
    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)
    fig, ax = plt.subplots()
    brp_sc = ax.scatter(x, y, c=z, s=50, label='Breakpoint', cmap='Greens', alpha = .5)
    #brp_cax = fig.add_axes([.6, .7, 0.2, 0.05])
    #cb = plt.colorbar(brp_sc, orientation='horizontal', cax=brp_cax)
    
    x, y = abs(jumps_hea), abs(theta_hea)
    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)
    hea_sc = ax.scatter(x, y, c=z, s=50, marker='x', label='Heaviside', cmap='Blues', alpha =.3)
    #hea_cax = fig.add_axes([.6, .6, 0.2, 0.05])
    #cb = plt.colorbar(hea_sc, orientation='horizontal', cax=hea_cax)
    
    ax.set_xlabel('Jumps (mm)')
    ax.set_ylabel('Slope Change Angle (degrees)')
    ax.legend()
    #plt.colorbar(sc)
    plt.show()
    
    
    
    '''
    #HEA VELOCITY THRESHOLD
    hea_vel_thresh = abs(theta_hea) < 10
    hea_jump_thresh = (abs(jumps_hea)>5) & (abs(jumps_hea)<15)
    hea_thresh = hea_vel_thresh*hea_jump_thresh
    #apply_thresh
    hea_perc_th_pts = hea_perc_th_pts[hea_thresh]
    heaviside_max_epochs = heaviside_max_epochs[hea_thresh]
    
    #Brp VELOCITY AND JUMP THRESHOLD
    brp_vel_thresh = (abs(theta_brp)>20) & (abs(theta_brp)<70)
    brp_jump_thresh = abs(jumps_brp)<5
    brp_thresh = brp_vel_thresh*brp_jump_thresh
    #apply_thresh
    brp_perc_th_pts = brp_perc_th_pts[brp_thresh]
    breakpoint_max_epochs = breakpoint_max_epochs[brp_thresh]
    
    #REPEAT
    
    jumps_brp, vel_diff_brp, vel_diff_perc_brp, theta_brp = get_jumps_vel_thetas(defo_df, B_temp, brp_perc_th_pts, breakpoint_max_epochs, Tr_q3_breakpoint, plotting=False, printing=False)
    jumps_hea, vel_diff_hea, vel_diff_perc_hea, theta_hea = get_jumps_vel_thetas(defo_df, B_temp, hea_perc_th_pts, heaviside_max_epochs+1, Tr_q3_Heaviside, plotting=False, printing=False)
    
    print_samples(lin_perc_th_pts, sin_perc_th_pts,hea_perc_th_pts, brp_perc_th_pts)
    
    #plt.scatter(abs(jumps_brp), abs(theta_brp), marker='.', label='Breakpoint', c='k')
    #plt.scatter(abs(jumps_hea), abs(theta_hea), marker='x', label='Heaviside', c='k')
    #plt.legend()
    #plt.show()
    
    brp_hea_intersect = np.intersect1d(hea_perc_th_pts, brp_perc_th_pts)
    print('brp_hea_intersect.size', brp_hea_intersect.size)
    
    #REPEAT
    
    #better method
    hea_intersect = np.in1d(hea_perc_th_pts, brp_hea_intersect)
    hea_perc_th_pts = hea_perc_th_pts[~hea_intersect]
    heaviside_max_epochs = heaviside_max_epochs[~hea_intersect]
    brp_intersect = np.in1d(brp_perc_th_pts, brp_hea_intersect)
    brp_perc_th_pts = brp_perc_th_pts[~brp_intersect]
    breakpoint_max_epochs = breakpoint_max_epochs[~brp_intersect]
    
    print_samples(lin_perc_th_pts, sin_perc_th_pts,hea_perc_th_pts, brp_perc_th_pts)
    
    brp_hea_intersect = np.intersect1d(hea_perc_th_pts, brp_perc_th_pts)
    print('brp_hea_intersect.size', brp_hea_intersect.size)
    '''

def remove_intersecting_points(lin_perc_th_pts, sin_perc_th_pts, hea_perc_th_pts, brp_perc_th_pts, heaviside_max_epochs, breakpoint_max_epochs):
    brp_hea_intersect = np.intersect1d(hea_perc_th_pts, brp_perc_th_pts)
    print('brp_hea_intersect.size', brp_hea_intersect.size)
    lin_sin_intersect = np.intersect1d(lin_perc_th_pts, sin_perc_th_pts)
    print('lin_sin_intersect.size', lin_sin_intersect.size)
    #REPEAT
    
    #better method
    hea_intersect = np.in1d(hea_perc_th_pts, brp_hea_intersect)
    hea_perc_th_pts = hea_perc_th_pts[~hea_intersect]
    heaviside_max_epochs = heaviside_max_epochs[~hea_intersect]
    brp_intersect = np.in1d(brp_perc_th_pts, brp_hea_intersect)
    brp_perc_th_pts = brp_perc_th_pts[~brp_intersect]
    breakpoint_max_epochs = breakpoint_max_epochs[~brp_intersect]
    
    #linear and sin
    lin_intersect = np.in1d(lin_perc_th_pts, lin_sin_intersect)
    lin_perc_th_pts = lin_perc_th_pts[~lin_intersect]
    sin_intersect = np.in1d(sin_perc_th_pts, lin_sin_intersect)
    sin_perc_th_pts = sin_perc_th_pts[~sin_intersect]
    
    lin_sin_intersect = np.intersect1d(lin_perc_th_pts, sin_perc_th_pts)
    print('lin_sin_intersect.size', lin_sin_intersect.size)
    brp_hea_intersect = np.intersect1d(hea_perc_th_pts, brp_perc_th_pts)
    print('brp_hea_intersect.size', brp_hea_intersect.size)
    
    return lin_perc_th_pts, sin_perc_th_pts, hea_perc_th_pts, brp_perc_th_pts, heaviside_max_epochs, breakpoint_max_epochs

def print_samples(lin_perc_th_pts, sin_perc_th_pts,hea_perc_th_pts, brp_perc_th_pts):
    print('lin_perc_th_pts', lin_perc_th_pts.size)
    print('hea_perc_th_pts', hea_perc_th_pts.size)
    print('brp_perc_th_pts', brp_perc_th_pts.size)
    print('sin_perc_th_pts', sin_perc_th_pts.size)

def norm_data_epochs(defo_df, insar_dates):
    
    defo_df.columns = insar_dates
    defo_df = defo_df.T.resample('12D').ffill().T
    
    return defo_df
    
def analyze_B_temp(B_temp):
    
    #print()
    B_temp_delta = np.round((B_temp[1:]-B_temp[:-1])*365)
    #print(B_temp_delta)
    print(np.where(B_temp_delta==24))
    
    print(plt.hist(B_temp_delta))
    plt.show()

def plot_anomalies(PS_X, PS_Y, intersect_indices_dict, vv_amp_arr, vh_amp_arr):
    
    vv_mrm_log = 10*np.log10(vv_amp_arr)
    vh_mrm_log = 10*np.log10(vh_amp_arr)
    plot_decomposition_RGB(np.dstack((vv_mrm_log, vh_mrm_log)), True)
    
    
    norm = mpl.colors.Normalize(vmin=0, vmax=3)
    #cmap = mpl.cm.get_cmap('jet_r')
    
    #MARKERS = ['s','P','P','P','^','^','^']
    lin_marker = [(-1,1), (1,-1), (.9,-1.1), (-1.1,.9)]
    hea_marker = [(-1,1),(0,1), (0,-1), (1,-1), \
        (1,-.9), (0.1,-0.9), (0.1,1.1),(-1,1.1)]
    brp_marker = [(-1,0),(0,0), (1,-1), \
        (1,-1.1), (0,-.1), (-1,-.1)]
    MARKERS = [lin_marker, hea_marker,hea_marker,hea_marker, brp_marker,brp_marker,brp_marker]
    #MARKERS = [hea_marker,hea_marker,hea_marker, brp_marker,brp_marker,brp_marker, lin_marker]
    low_vol_c, high_vol_c, dbl_c = cm.jet(norm(1)), cm.jet(norm(2)), cm.jet(norm(3))
    c_bar_choices = [dbl_c, low_vol_c, high_vol_c, dbl_c, low_vol_c, high_vol_c, dbl_c]
    #c_bar_choices = [low_vol_c, high_vol_c, dbl_c, low_vol_c, high_vol_c, dbl_c, dbl_c]
    LABELS = ['Lin', 'Hea_low_vol', 'Hea_low_vol', 'Hea_dbl', 'Brp_low_vol', 'Brp_low_vol', 'Brp_dbl']
    #LABELS = ['Hea_low_vol', 'Hea_low_vol', 'Hea_dbl', 'Brp_low_vol', 'Brp_low_vol', 'Brp_dbl', 'Lin']
    
    for class_index, marker, c_bar, label in zip(intersect_indices_dict.keys(), MARKERS, c_bar_choices, LABELS):
    #for class_index, marker, c_bar, label in zip([1,2,3,4,5,6,0], MARKERS, c_bar_choices, LABELS):
        #plt.imshow(mrm, cmap='gray')#rfor_4class[800:1800,11300:12300]
        indices = intersect_indices_dict[class_index]
        plt.scatter(PS_X[indices]-1, PS_Y[indices]-1, c=c_bar,marker=marker, s=200, alpha = 1, linewidth=2, label=label)
    #plt.legend()
    
    plt.xlabel('Range')
    plt.ylabel('Azimuth')
    
    plt.show()

def clustering(PS_X, PS_Y, vv_amp_arr, vh_amp_arr, hea_perc_th_pts, brp_perc_th_pts, heaviside_max_epochs, breakpoint_max_epochs, lin_perc_th_pts, epochs):
    num_clusters = 5
    jumps_brp, vel_diff_brp, vel_diff_perc_brp, theta_brp = get_jumps_vel_thetas(defo_df, B_temp, brp_perc_th_pts, breakpoint_max_epochs, Tr_q3_breakpoint, plotting=False, printing=False)
    jumps_hea, vel_diff_hea, vel_diff_perc_hea, theta_hea = get_jumps_vel_thetas(defo_df, B_temp, hea_perc_th_pts, heaviside_max_epochs+1, Tr_q3_Heaviside, plotting=False)
    
    
    #find the points which are divided in classes: lin, hea, break, and for those points
    #choice of features: PS_X, PS_Y, anomaly epochs (normalized), anomaly extent (as % of maximum)
    
    X_points = np.append(PS_X[hea_perc_th_pts], PS_X[brp_perc_th_pts])
    Y_points = np.append(PS_Y[hea_perc_th_pts], PS_Y[brp_perc_th_pts])
    anomaly_epochs = np.append(heaviside_max_epochs, breakpoint_max_epochs)/epochs
    anomaly_extent = np.append(jumps_hea/max(jumps_hea), vel_diff_brp/max(vel_diff_brp))
    
    input_arr = np.vstack((anomaly_epochs, anomaly_extent)).T #X_points/max(PS_X), Y_points/max(PS_Y), 
    
    kmeans_cd_bk, label = kmeans(input_arr, num_clusters)
    
    print(kmeans_cd_bk)
    print(label)
    
    norm_clusters = mpl.colors.Normalize(vmin=0, vmax=num_clusters)
    
    vv_mrm_log = 10*np.log10(vv_amp_arr)
    vh_mrm_log = 10*np.log10(vh_amp_arr)
    fig, ax = plt.subplots()
    plot_decomposition_RGB(np.dstack((vv_mrm_log, vh_mrm_log)), True)
    ax.scatter(X_points, Y_points, c=label, s=50, cmap='jet_r')
    plt.show()

def anomaly_map_analysis(PS_X, PS_Y, vv_amp_arr, vh_amp_arr, hea_perc_th_pts, brp_perc_th_pts, heaviside_max_epochs, breakpoint_max_epochs, lin_perc_th_pts):
    
    jumps_brp, vel_diff_brp, vel_diff_perc_brp, theta_brp = get_jumps_vel_thetas(defo_df, B_temp, brp_perc_th_pts, breakpoint_max_epochs, Tr_q3_breakpoint, plotting=False, printing=False)
    jumps_hea, vel_diff_hea, vel_diff_perc_hea, theta_hea = get_jumps_vel_thetas(defo_df, B_temp, hea_perc_th_pts, heaviside_max_epochs+1, Tr_q3_Heaviside, plotting=False)
    
    vv_mrm_log = 10*np.log10(vv_amp_arr)
    vh_mrm_log = 10*np.log10(vh_amp_arr)
    fig, ax = plt.subplots()
    plot_decomposition_RGB(np.dstack((vv_mrm_log, vh_mrm_log)), True)
    #plt.show()
    
    lin_marker = [(-1,1), (1,-1), (.9,-1.1), (-1.1,.9)]
    hea_marker = [(-1,1),(0,1), (0,-1), (1,-1), \
        (1,-.9), (0.1,-0.9), (0.1,1.1),(-1,1.1)]
    brp_marker = [(-1,0),(0,0), (1,-1), \
        (1,-1.1), (0,-.1), (-1,-.1)]    
    
    norm_jumps = mpl.colors.Normalize(vmin=5, vmax=14)
    norm_vel_ch = mpl.colors.Normalize(vmin=20, vmax=70)
    
    plt.scatter(PS_X[lin_perc_th_pts], PS_Y[lin_perc_th_pts], c='w', marker=lin_marker, s=200, label='Linear')
    
    #hea_sc = ax.scatter(PS_X[hea_perc_th_pts], PS_Y[hea_perc_th_pts], c=abs(jumps_hea), marker=hea_marker, s=200, cmap='Blues')
    hea_sc = ax.scatter(PS_X[hea_perc_th_pts], PS_Y[hea_perc_th_pts], c=cm.Blues(norm_jumps(abs(jumps_hea))), marker=hea_marker, s=200)#, cmap='Blues')
    
    #brp_cax = fig.add_axes([.6, .7, 0.2, 0.05])
    #cb = plt.colorbar(hea_sc, orientation='horizontal', cax=brp_cax)
    
    #brp_sc = ax.scatter(PS_X[brp_perc_th_pts], PS_Y[brp_perc_th_pts], c=abs(theta_brp), marker=brp_marker, s=200, cmap='Greens')
    brp_sc = ax.scatter(PS_X[brp_perc_th_pts], PS_Y[brp_perc_th_pts], c=cm.Greens(norm_vel_ch(abs(theta_brp))), marker=brp_marker, s=200)#, cmap='Greens')
    #brp_cax = fig.add_axes([.6, .6, 0.2, 0.05])
    #cb = plt.colorbar(brp_sc, orientation='horizontal', cax=brp_cax)
    
    #plt.scatter(PS_X[hea_perc_th_pts], PS_Y[hea_perc_th_pts], c=heaviside_max_epochs+1, marker=hea_marker, s=200, cmap='YlOrRd', label='Heaviside')
    #plt.scatter(PS_X[brp_perc_th_pts], PS_Y[brp_perc_th_pts], c=breakpoint_max_epochs, marker=brp_marker, s=200, cmap='YlOrRd', label='Breakpoint')
    #plt.scatter(PS_X[lin_perc_th_pts], PS_Y[lin_perc_th_pts], c='w', marker=lin_marker, s=200, label='Linear')
    
    #plt.colorbar(orientation='horizontal')
    ax.set_xlabel('Range')
    ax.set_ylabel('Azimuth')
    
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    #plt.legend()
    #plt.axis([11300,12300,1800,800])
    #plt.axis([15000,16000,1000,0])
    plt.show()

def thresh_filter(PS_X, PS_Y, crp_list):
    x_th = (PS_X>crp_list[2]) & (PS_X<crp_list[3])
    y_th = (PS_Y>crp_list[0]) & (PS_Y<crp_list[1])
    xy_th = x_th*y_th
    return PS_X[xy_th], PS_Y[xy_th]

def count_anomalies(PS_X, PS_Y, defo_threshold_points_list, defo_anomaly_max_epochs, vv_amp_arr, vh_amp_arr):
    
    crp_anom = [800,1800,11300,12300]
    crp_top_right = [0,1000,15000,16000]
    
    vv_mrm_log = 10*np.log10(vv_amp_arr)
    vh_mrm_log = 10*np.log10(vh_amp_arr)
    
    lin_perc_th_pts, hea_perc_th_pts, brp_perc_th_pts = defo_threshold_points_list
    
    #plot_decomposition_RGB(np.dstack((vv_mrm_log, vh_mrm_log)), True)
    x_lin_crp_anom, y_lin_crp_anom = thresh_filter(PS_X[lin_perc_th_pts], PS_Y[lin_perc_th_pts], crp_anom)
    x_hea_crp_anom, y_hea_crp_anom = thresh_filter(PS_X[hea_perc_th_pts], PS_Y[hea_perc_th_pts], crp_anom)
    x_brp_crp_anom, y_brp_crp_anom = thresh_filter(PS_X[brp_perc_th_pts], PS_Y[brp_perc_th_pts], crp_anom)
    
    print(x_lin_crp_anom.size,x_hea_crp_anom.size,x_brp_crp_anom.size)
    
    plt.bar(0, x_lin_crp_anom.size, label='Linear', color='k', width=0.4)
    plt.bar(1, x_hea_crp_anom.size, label='Hea', color='k', width=0.4)
    plt.bar(2, x_brp_crp_anom.size, label='Brp', color='k', width=0.4)
    
    plt.ylabel('Frequency')
    plt.xlabel('Deformation Class')
    plt.xticks([0,1,2], ['Linear', 'Heaviside', 'Breakpoint'])
    
    #plt.scatter(x_lin_crp, y_lin_crp)
    
    plt.show()

    
if __name__=='__main__':
    
    mpl.rcParams.update({'font.size': 20}) # increasing font size 
    #Load base directories where Inputs and outputs are stored, i.e. workind directory
    data_dir = '/home/anurag/Documents/PhDProject/Papers/Sec_paper/data/ireland'
    
    #change to working directory
    os.chdir(data_dir)
    
    #Load pre-saved files
    mrm = np.load('mrm.npy') #load multi-reflectivity files
    
    #Load the locations of PS points
    PS_X, PS_Y = np.load('PS_X_25.npy'), np.load('PS_Y_25.npy')
    
    #Load the corresponding deformation time series.
    defo_df = pd.DataFrame(np.load('defo_stack_25.npy'))
    
    print('defo_df loaded with shape', defo_df.shape)
    
    PS_vel = np.load('PS_vel_25.npy') #load PS velocity, if saved
    ens_coh_list = np.load('ens_coh_list_25.npy')
    
    rfor_4class = np.load('rforest_pred_deburst_classes_4_depsi_crop_vv_vh_ent.npy')
    
    vv_amp_arr = np.load('vv_arr_mrm_deburst_depsi_crop.npy')
    vh_amp_arr = np.load('vh_arr_mrm_deburst_depsi_crop.npy')
    
    B_temp = np.load('B_temp.npy')
    epochs = B_temp.size
    insar_dates = np.load('dates.npy', allow_pickle=True)
    
    
    #Applying epoch normalization
    #analyze_B_temp(B_temp)
    defo_df = norm_data_epochs(defo_df, insar_dates) #set deformation w.r.t. first epoch
    
    #print('defo_df.shape', defo_df.shape)
    dates = defo_df.columns
    new_B_temp = np.array([(date - dates[0]).days for date in dates])/365
    B_temp = new_B_temp
    
    #np.save('B_temp_epoch_norm.npy', B_temp)
    #analyze_B_temp(new_B_temp)
    
    Tr_q3_Heaviside, Tr_q3_breakpoint, Tr_q3_lin, Tr_q3_sin = MHT(B_temp, defo_df)
    
    #end = time.time()
    
    #print(end - start)
    
    #np.save('T_q3_Heaviside_-25mm_5mm_burst_joined.npy',Tr_q3_Heaviside)
    #np.save('T_q3_breakpoint_-25mm_5mm_burst_joined.npy',Tr_q3_breakpoint)
    #np.save('T_q3_lin_-25mm_5mm_burst_joined_1.npy',Tr_q3_lin)
    #np.save('T_q3_sin_-25mm_5mm_burst_joined_1.npy',Tr_q3_sin)
    
    #sys.exit()
    
    #Tr_q3_Heaviside = np.load(os.path.join(data_dir, 'T_q3_Heaviside_-25mm_5mm_burst_joined.npy'))
    #Tr_q3_breakpoint = np.load(os.path.join(data_dir,'T_q3_breakpoint_-25mm_5mm_burst_joined.npy'))
    #Tr_q3_lin = np.load(os.path.join(data_dir,'T_q3_lin_-25mm_5mm_burst_joined_1.npy'))
    #Tr_q3_sin = np.load(os.path.join(data_dir,'T_q3_sin_-25mm_5mm_burst_joined_1.npy'))
    
    percentile_anomaly, percentile_anomaly_low = 95, 10
    percentile_lin = 95
    percentile_sin = 30
    epoch_low_th, epoch_high_th = np.array([1, 9])*epochs//10
    hea_jump_min, hea_jump_max, brp_jump_max, hea_theta_max, brp_theta_min, brp_theta_max = 7.5, 14, 2.5, 10, 40, 70 #5, 14, 5, 20, 20, 70 # #
    
    
    #plt.imshow(mrm, cmap='gray')#rfor_4class[800:1800,11300:12300]
    
    #plt.scatter(PS_X-1, PS_Y-1, c=PS_vel, cmap='RdYlGn', s=20)
    #plt.show()
    
    #sys.exit()
    Tr_q3_lin = np.ma.masked_where(Tr_q3_lin==0, Tr_q3_lin)
    Tr_q3_lin = np.ma.filled(Tr_q3_lin, np.nan)
    Tr_q3_sin = np.ma.masked_where(Tr_q3_sin==0, Tr_q3_sin)
    Tr_q3_sin = np.ma.filled(Tr_q3_sin, np.nan)
    
    print('Number of samples immediately after MHT')
    print('Heaviside', Tr_q3_Heaviside.shape)
    print('Breakpoint', Tr_q3_breakpoint.shape)
    print('Linear', Tr_q3_lin.shape)
    print('Sinusoidal', Tr_q3_sin.shape)
    '''
    #*************************************
    #Test ratio percentile thresholding
    #*************************************
    lin_perc_th_pts, sin_perc_th_pts, hea_perc_th_pts, brp_perc_th_pts, heaviside_max_epochs, breakpoint_max_epochs = apply_percentile_thresholds(Tr_q3_Heaviside, Tr_q3_breakpoint, Tr_q3_lin, Tr_q3_sin, percentile_anomaly, percentile_anomaly_low)
    print('samples after percentile threshold')
    print_samples(lin_perc_th_pts, sin_perc_th_pts,hea_perc_th_pts, brp_perc_th_pts)
    
    #*************************************
    #**********Epoch thresholding**********
    #*************************************
    hea_perc_th_pts, brp_perc_th_pts, heaviside_max_epochs, breakpoint_max_epochs = epoch_thresh(hea_perc_th_pts, brp_perc_th_pts, heaviside_max_epochs, breakpoint_max_epochs)
    
    print('samples after epoch threshold')
    print_samples(lin_perc_th_pts, sin_perc_th_pts,hea_perc_th_pts, brp_perc_th_pts)
    
    #***************************
    #HEAVISIDE JUMP AND BREAKPOINT vel. change THRESHOLDING
    #***************************
    
    #plot_jumps_slope(hea_perc_th_pts, brp_perc_th_pts, heaviside_max_epochs, breakpoint_max_epochs)
    
    hea_perc_th_pts, brp_perc_th_pts, heaviside_max_epochs, breakpoint_max_epochs = jump_vel_gra_threshold(hea_jump_min, hea_jump_max, brp_jump_max, hea_theta_max, brp_theta_min, brp_theta_max, defo_df, B_temp, hea_perc_th_pts, brp_perc_th_pts, heaviside_max_epochs, breakpoint_max_epochs)
    print('samples after jump_veltheta threshold')
    print_samples(lin_perc_th_pts, sin_perc_th_pts,hea_perc_th_pts, brp_perc_th_pts)
    #***************************
    #Removing Intersecting points
    #***************************
    lin_perc_th_pts, sin_perc_th_pts, hea_perc_th_pts, brp_perc_th_pts, heaviside_max_epochs, breakpoint_max_epochs = remove_intersecting_points(lin_perc_th_pts, sin_perc_th_pts, hea_perc_th_pts, brp_perc_th_pts, heaviside_max_epochs, breakpoint_max_epochs)
    print('samples after removing intersecting samples')
    print_samples(lin_perc_th_pts, sin_perc_th_pts,hea_perc_th_pts, brp_perc_th_pts)
    
    np.save('lin_perc_th_pts-25mm_5mm_burst_joined_strict_thresh_lin_high_14.npy', lin_perc_th_pts)
    np.save('sin_perc_th_pts-25mm_5mm_burst_joined_strict_thresh_lin_high_14.npy', sin_perc_th_pts)
    np.save('hea_perc_th_pts-25mm_5mm_burst_joined_strict_thresh_lin_high_14.npy', hea_perc_th_pts)
    np.save('brp_perc_th_pts-25mm_5mm_burst_joined_strict_thresh_lin_high_14.npy', brp_perc_th_pts)
    np.save('heaviside_max_epochs-25mm_5mm_burst_joined_strict_thresh_lin_high_14.npy', heaviside_max_epochs)
    np.save('breakpoint_max_epochs-25mm_5mm_burst_joined_strict_thresh_lin_high_14.npy', breakpoint_max_epochs)
    
    plot_jumps_slope(hea_perc_th_pts, brp_perc_th_pts, heaviside_max_epochs, breakpoint_max_epochs)
    
    #sys.exit()
    '''
    
    #lin_perc_th_pts = np.load('lin_perc_th_pts-25mm_5mm_burst_joined_lin_high_14.npy')
    #sin_perc_th_pts = np.load('sin_perc_th_pts-25mm_5mm_burst_joined_lin_high_14.npy')
    #hea_perc_th_pts = np.load('hea_perc_th_pts-25mm_5mm_burst_joined_lin_high_14.npy')
    #brp_perc_th_pts = np.load('brp_perc_th_pts-25mm_5mm_burst_joined_lin_high_14.npy')
    #heaviside_max_epochs = np.load('heaviside_max_epochs-25mm_5mm_burst_joined_lin_high_14.npy')
    #breakpoint_max_epochs = np.load('breakpoint_max_epochs-25mm_5mm_burst_joined_lin_high_14.npy')
    
    lin_perc_th_pts = np.load('lin_perc_th_pts-25mm_5mm_burst_joined_strict_thresh_lin_high_14.npy')
    sin_perc_th_pts = np.load('sin_perc_th_pts-25mm_5mm_burst_joined_strict_thresh_lin_high_14.npy')
    hea_perc_th_pts = np.load('hea_perc_th_pts-25mm_5mm_burst_joined_strict_thresh_lin_high_14.npy')
    brp_perc_th_pts = np.load('brp_perc_th_pts-25mm_5mm_burst_joined_strict_thresh_lin_high_14.npy')
    heaviside_max_epochs = np.load('heaviside_max_epochs-25mm_5mm_burst_joined_strict_thresh_lin_high_14.npy')
    breakpoint_max_epochs = np.load('breakpoint_max_epochs-25mm_5mm_burst_joined_strict_thresh_lin_high_14.npy')
    
    print_samples(lin_perc_th_pts, sin_perc_th_pts,hea_perc_th_pts, brp_perc_th_pts)
    
    
    #plot_jumps_slope(hea_perc_th_pts, brp_perc_th_pts, heaviside_max_epochs, breakpoint_max_epochs)
    
    #analysis of anomalies over the map
    
    defo_threshold_points_list = [lin_perc_th_pts, hea_perc_th_pts, brp_perc_th_pts]
    
    defo_anomaly_max_epochs = [heaviside_max_epochs, breakpoint_max_epochs]
    
    #count_anomalies(PS_X, PS_Y, defo_threshold_points_list, defo_anomaly_max_epochs, vv_amp_arr, vh_amp_arr)
    
    #anomaly_map_analysis(PS_X, PS_Y, vv_amp_arr, vh_amp_arr, hea_perc_th_pts, brp_perc_th_pts, heaviside_max_epochs, breakpoint_max_epochs, lin_perc_th_pts)
    
    #clustering(PS_X, PS_Y, vv_amp_arr, vh_amp_arr, hea_perc_th_pts, brp_perc_th_pts, heaviside_max_epochs, breakpoint_max_epochs, lin_perc_th_pts, epochs)
    
    SM_class_name_dict = {1:'Surface', 2:'$Volume_{low}$', 3:'$Volume_{high}$', 4:'Dbl bounce'} #1:'Surface'
    #defo_class_dict = {1:'Linear', 2:'Sinusoidal', 3:'Heaviside', 4:'Breakpoint'}
    defo_class_dict = {1:'Linear', 2:'Heaviside', 3:'Breakpoint'}
    print(np.unique(rfor_4class, return_counts=True))
    #PS_X = np.clip(PS_X, 0,16000)
    ps_gdf = pd.DataFrame(data = np.vstack((PS_X.astype(np.int), PS_Y.astype(np.int), PS_vel, rfor_4class[PS_Y.astype(np.int)-1, PS_X.astype(np.int)-1]+1, ens_coh_list)).T, columns=['X', 'Y', 'linear1', 'Class', 'ens_coh'])
    print(ps_gdf)
    
    #analyze_pol_decomp(mrm, 0, ps_gdf, defo_df,(0,0), rfor_4class, B_temp, dates = insar_dates)
    #jumps_brp, vel_diff_brp, vel_diff_perc_brp, theta_brp = get_jumps_vel_thetas(defo_df, B_temp, brp_perc_th_pts, breakpoint_max_epochs, Tr_q3_breakpoint, plotting=True, printing=True, ps_gdf=ps_gdf, label='Brp')
    #jumps_hea, vel_diff_hea, vel_diff_perc_hea, theta_hea = get_jumps_vel_thetas(defo_df, B_temp, hea_perc_th_pts, heaviside_max_epochs+1, Tr_q3_Heaviside, plotting=True, printing=True, ps_gdf=ps_gdf, label='Hea')
    #plot_lin_sample(ps_gdf, defo_df, B_temp, lin_perc_th_pts, Tr_q3_lin, sd=5, plotting=True)
    
    #plot_sin_sample(ps_gdf, defo_df, B_temp, sin_perc_th_pts, sd=5, plotting=True)
    
    train_x_all_real, train_y_all_real, intersect_indices_dict = craft_real_samples(ps_gdf, defo_df, 0, 0, defo_threshold_points_list, \
        defo_anomaly_max_epochs, SM_class_name_dict, defo_class_dict, CLASS_CROSSING=True, num_features=1, MAX_TRAIN_SAMPLES=1000, MIN_TRAIN_SAMPLES=100)#min([len(i) for i in defo_threshold_points_list]))
    
    #np.save('train_x_all_-.25_ire_both_bursts_class_crossed.npy', train_x_all_real)
    #np.save('train_y_all_-.25I_ire_both_bursts_class_crossed.npy', train_y_all_real)
    
    
    plot_anomalies(PS_X, PS_Y, intersect_indices_dict, vv_amp_arr, vh_amp_arr)
    
    
    print(train_x_all_real.shape, train_y_all_real.shape)
    
    print(intersect_indices_dict)
    
    sys.exit()
    
    train_x_all_MULTI = train_x_all_real#np.load('train_x_all_MULTI_ire_both_bursts.npy')
    train_y_all_MULTI = train_y_all_real#np.load('train_y_all_MULTI_ire_both_bursts.npy')
    epochs = 112
    
    #plt.hist(train_y_all_MULTI.argmax(1))
    #plt.show()

    batch_size_list = np.array([30])
    LSTM_EPOCHS_list = train_x_all_MULTI.shape[0]//batch_size_list
    #LSTM_EPOCHS_list = [30]#, 40, 60]
    print('LSTM_EPOCHS_list', LSTM_EPOCHS_list)
    print('batch_size_list', batch_size_list)
    count=1
    kappa_dict = {}
    fig, ax = plt.subplots()
    for LSTM_NEURONS in [100]:
        for batch_size, LSTM_EPOCHS in zip(batch_size_list, LSTM_EPOCHS_list):
            kappa_list = []
            for i in range(15):
                train_x, train_y, test_x, test_y = divide_train_test_samples(train_x_all_MULTI, train_y_all_MULTI, TRAIN_RATIO=0.67)
                print('Training with LSTM_NEURONS = {}, batch_size = {}, LSTM_EPOCHS = {}'.format(LSTM_NEURONS, batch_size, LSTM_EPOCHS))
                
                y_pred, test_predict, kappa, score, model, class_report = LSTM_class_1(train_x, train_y, test_x, test_y, LOOK_BACK=epochs, LSTM_NEURONS = LSTM_NEURONS, LSTM_EPOCHS = LSTM_EPOCHS, batch_size=batch_size, B_temp=0)#=B_temp)
                kappa_list.append(kappa)
                print(class_report)
                
            kappa_dict[count] = kappa_list
            print(kappa_list)
            count+=1
            
        #pos = np.arange(len(treatments)) + 1
    print(list(kappa_dict.values()))
    #bp = plt.boxplot(list(kappa_dict.values()), sym='k+', notch=1, bootstrap=None, positions=np.arange(1,count))#,usermedians=medians,conf_intervals=conf_intervals)
    bp = ax.boxplot(list(kappa_dict.values()), kappa_dict.keys())
    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['whiskers'], color='black')
    plt.setp(bp['fliers'], color='red', marker='+')
    plt.show()
