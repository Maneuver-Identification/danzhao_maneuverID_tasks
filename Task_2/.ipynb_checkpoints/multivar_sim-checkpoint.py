import os
import sys
import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
import joblib
import torch.nn.init as init
import seaborn as sns
import tslearn

from tslearn.metrics import cdist_dtw
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy.special import softmax
from torch.utils.data import WeightedRandomSampler
from PIL import Image
from tqdm import tqdm
from torch.autograd import Variable
from sklearn.model_selection import StratifiedKFold
from collections import defaultdict

raw_vars = ["xeast", "ynorth", "zup", "vx", "vy", "vz", "head", "pitch", "roll"]
d1_vars = [i + "_d1" for i in raw_vars]

## will need to change directory, this currently only works with tsv from sample maneuver data
home_dir = "/home/gridsan/dzhao/ManeuverID_shared/dan_zhao"
tsv_dir = home_dir + "/ManeuverID_archive/SampleManeuversData/SampleData/VanceAFB_tsv"
task_dir = home_dir + "/Task_2"

### read in data for categorized maneuvers, assuming each tsv describes the time-series of one maneuever each

if "recog_maneuvers.joblib" not in os.listdir(task_dir):
    recog_maneuvers = defaultdict()

    for m_name in os.listdir(tsv_dir):
        if m_name.endswith(".tsv"):
            m_key = m_name.replace(" ", "").split('_')[-1][:-8]
            recog_maneuvers[m_key] = -1
    i = 0
    for di in os.listdir(tsv_dir):
        if di.endswith(".tsv"):

            di_name = di.replace(" ", "").split('_')[-1][:-8]

            df_temp = pd.read_csv(tsv_dir + di, sep = '\t', index_col=0)
            df_temp.columns.values[0] = "time"
            df_temp.columns.values[1] = "xeast"
            df_temp.columns.values[2] = "ynorth"           
            df_temp.columns.values[3] = "zup"
            df_temp.columns.values[4] = "vx"
            df_temp.columns.values[5] = "vy"
            df_temp.columns.values[6] = "vz"           
            df_temp.columns.values[7] = "head"
            df_temp.columns.values[8] = "pitch"
            df_temp.columns.values[9] = "roll"


            for n, _ in enumerate(d1_vars):
                df_temp[d1_vars[n]] = df_temp[raw_vars[n]].diff()

            recog_maneuvers[di_name] = df_temp
    joblib.dump(recog_maneuvers, task_dir + "/recog_maneuvers.joblib", compress=3)

if "smpl_mnvrcorr.joblib" not in os.listdir(task_dir):
    recog_maneuvers = defaultdict()
    smpl_mnvrcorr = defaultdict()

    for k, v in recog_maneuvers.items():
        cor_raw = np.array(v[raw_vars].dropna().corr(method="kendall"))
        cor_d1 = np.array(v[d1_vars].dropna().corr(method="kendall"))

        smpl_mnvrcorr[k] = defaultdict()
        smpl_mnvrcorr[k]["raw"] = cor_raw
        smpl_mnvrcorr[k]["d1"] = cor_d1

    joblib.dump(smpl_mnvrcorr, task_dir + "/smpl_mnvrcorr.joblib", compress=3)
    
recog_maneuvers = joblib.load(task_dir + "/recog_maneuvers.joblib")
smpl_mnvrcorr = joblib.load(task_dir + "/smpl_mnvrcorr.joblib")

root_dir2 = home_dir+"/ManeuverID_archive/ObservedTrajectoryData/Sorted/path_tsvdata"
## using 1200... folder (can extend to 1300...)
good_dir = root_dir2 + "/12000000000_tsv_good" 
bad_dir = root_dir2 + "/12000000000_tsv_bad"

### mapping table for reference to tell what file (by file name) is in what class (good or bad)
good_fileidx = [f.replace(" ", "").split(".")[0] for f in os.listdir(good_dir)]
bad_fileidx = [f.replace(" ", "").split(".")[0] for f in os.listdir(bad_dir)]

if "obs_trajcorr.joblib" not in os.listdir(task_dir):

    obs_trajcorr = defaultdict()

    result_good = pd.DataFrame()
    result_bad = pd.DataFrame()

    covs_good = []
    covs_bad = []

    t1 = time.time()
    i = 0
    for di in os.listdir(good_dir):
        if di.endswith(".tsv"):

            df_temp = pd.read_csv(good_dir + di, sep = '\t', index_col=0)
            df_temp.columns.values[0] = "time"
            df_temp.columns.values[1] = "xeast"
            df_temp.columns.values[2] = "ynorth"           
            df_temp.columns.values[3] = "zup"
            df_temp.columns.values[4] = "vx"
            df_temp.columns.values[5] = "vy"
            df_temp.columns.values[6] = "vz"           
            df_temp.columns.values[7] = "head"
            df_temp.columns.values[8] = "pitch"
            df_temp.columns.values[9] = "roll"

            for n, _ in enumerate(d1_vars):
                df_temp[d1_vars[n]] = df_temp[raw_vars[n]].diff()

            cor_raw = np.array(df_temp[raw_vars].dropna().corr(method="kendall"))
            cor_d1 = np.array(df_temp[d1_vars].dropna().corr(method="kendall"))

    #         covs_good.append(cov)

            df_temp["obsTrajID"] = i
            df_temp["TrajClass"] = "good"
            file_idx = di.replace(" ", "").split(".")[0]
            df_temp["fileIdx"] = file_idx

            obs_trajcorr[file_idx] = defaultdict()
            obs_trajcorr[file_idx]["raw"] = cor_raw                         
            obs_trajcorr[file_idx]["d1"] = cor_d1                                                               

            result_good = pd.concat([result_good, df_temp])

            i+=1

    i = 0
    for di in os.listdir(bad_dir):
        if di.endswith(".tsv"):

            df_temp = pd.read_csv(bad_dir + di, sep = '\t', index_col=0)
            df_temp.columns.values[0] = "time"
            df_temp.columns.values[1] = "xeast"
            df_temp.columns.values[2] = "ynorth"           
            df_temp.columns.values[3] = "zup"
            df_temp.columns.values[4] = "vx"
            df_temp.columns.values[5] = "vy"
            df_temp.columns.values[6] = "vz"           
            df_temp.columns.values[7] = "head"
            df_temp.columns.values[8] = "pitch"
            df_temp.columns.values[9] = "roll"

            for n, _ in enumerate(d1_vars):
                df_temp[d1_vars[n]] = df_temp[raw_vars[n]].diff()

            cor_raw = np.array(df_temp[raw_vars].dropna().corr(method="kendall"))
            cor_d1 = np.array(df_temp[d1_vars].dropna().corr(method="kendall"))

    #         covs_bad.append(cov)   

            df_temp["obsTrajID"] = i ## id within good/bad class
            df_temp["TrajClass"] = "bad" ## good or bad
            file_idx = di.replace(" ", "").split(".")[0] ## corresponding file id                           
            df_temp["fileIdx"] = file_idx

            obs_trajcorr[file_idx] = defaultdict()
            obs_trajcorr[file_idx]["raw"] = cor_raw                         
            obs_trajcorr[file_idx]["d1"] = cor_d1                                                               

            result_bad = pd.concat([result_bad, df_temp])

            i+=1
        
    t2 = time.time()
    print(t2-t1) ## takes about 830 seconds to run through cell (13 min, likely due to kendall corr mats)
    
    
    result_good.reset_index(inplace = True, drop=True)
    result_bad.reset_index(inplace = True, drop=True)
    
    ### combine good and bad observed trajectories into one dataset
    allobs_trajectories = pd.concat([result_good, result_bad])
    allobs_trajectories.reset_index(inplace=True)
    
    allobs_trajectories.to_csv(task_dir + "/tabdf_all.csv", sep = '\t', index=False)
    result_good.to_csv(task_dir + "/good_tabdf_all.csv", sep = '\t', index=False)
    result_bad.to_csv(task_dir + "/bad_tabdf_all.tsv", sep = '\t', index=False)
    
    joblib.dump(obs_trajcorr, "obs_trajcorr.joblib", compress=3)

result_good = pd.read_csv(task_dir+"/good_tabdf_all.csv")
result_bad = pd.read_csv(task_dir+"/bad_tabdf_all.csv")
allobs_trajectories = pd.read_csv(task_dir+"/tabdf_all.tsv", sep = "\t") 
obs_trajcorr = joblib.load(task_dir+"/obs_trajcorr.joblib")


#### Time-series similarity and profiling
all_ids = list(allobs_trajectories["fileIdx"].astype(str).unique())

### mappings to help with looking up results array
smplmnvrID_map = dict(zip(list(recog_maneuvers.keys()), np.arange(len(recog_maneuvers))))
obstrajID_map = dict(zip(list(all_ids), np.arange(len(all_ids))))

raw_vars2 = ["vx", "vy", "vz", "head", "pitch", "roll"]
d1_vars2 = [i + "_d1" for i in raw_vars2]

def corr_sim(A,B):
#     d = 1 - np.trace(A @ B) / (np.linalg.norm(A, ord="fro") * np.linalg.norm(B, ord="fro"))
    d = np.trace(A @ B) / (np.sum(A ** 2) ** 0.5 * np.sum(B ** 2) ** 0.5)
    return d

def KL(A,B):
    KL_ab = 0.5 * (np.trace(np.linalg.pinv(A) @ B) - np.log((np.linalg.det(B)/(np.linalg.det(A) + 1e-5)) + 1e-5))
    KL_ba = 0.5 * (np.trace(np.linalg.pinv(B) @ A) - np.log((np.linalg.det(A)/(np.linalg.det(B) + 1e-5)) + 1e-5))
    KL_sym = 0.5 * KL_ab + 0.5 * KL_ba
    return KL_sym

multivar_simraw = defaultdict(str)
multivar_simd1 = defaultdict(str)

for f in list(obs_trajcorr.keys()):
    multivar_simraw[f] = {}
    multivar_simd1[f] = {}
    for i in list(recog_maneuvers.keys()):
        multivar_simraw[f][i] = -999
        multivar_simraw[f][i] = -999  

assert len(raw_vars2) == len(d1_vars2)

t1 = time.time()
for n, k in enumerate(recog_maneuvers.keys()):    
    mvr_corr_raw = smpl_mnvrcorr[k]["raw"]
    mvr_corr_d1 = smpl_mnvrcorr[k]["d1"]
        
    for num, obsID in enumerate(list(obs_trajcorr.keys())):
        obs_corr_raw = obs_trajcorr[obsID]["raw"] 
        obs_corr_d1 = obs_trajcorr[obsID]["d1"]
        
        mvr_corr_raw[np.isnan(mvr_corr_raw)] = 0
        mvr_corr_d1[np.isnan(mvr_corr_d1)] = 0
        obs_corr_raw[np.isnan(obs_corr_raw)] = 0
        obs_corr_d1[np.isnan(obs_corr_d1)] = 0
                
        multivar_simraw[obsID][k] = KL(mvr_corr_raw, obs_corr_raw)
        multivar_simd1[obsID][k] = KL(mvr_corr_d1, obs_corr_d1)         

joblib.dump(multivar_simraw, task_dir+"/multivar_simraw.joblib", compress=3)
joblib.dump(multivar_simd1, task_dir+"/multivar_simd1.joblib", compress=3)

print("done,", str(time.time() - t1)) ## about 325.52 seconds (5 min) with fast DTW algo