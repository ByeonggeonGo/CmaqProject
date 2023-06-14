import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
from jax import grad, jit, vmap
from jax import random
import optax
import geopandas as gpd
import pandas as pd
import os
from shapely.geometry import Point
import numpy as np
from glob import glob
from netCDF4 import Dataset

def get_yearly_dataset():

    smoke_chem_list =['SO2', 'PM2_5', 'NOx', 'VOCs', 'NH3', 'CO']
    avg_conc_path_list = glob(os.path.join('exampledata',"concentration","*"))
    avg_conc_path_info_df = pd.DataFrame()
    avg_conc_path_info_df.loc[:,'path'] = avg_conc_path_list
    avg_conc_path_info_df.loc[:,'Run'] = [int(path.split("/")[-1].split(".")[1]) for path in avg_conc_path_list]

    avg_ems_path_list = glob(os.path.join('exampledata',"emission","*"))
    avg_ems_path_info_df = pd.DataFrame()
    avg_ems_path_info_df.loc[:,'path_ems'] = avg_ems_path_list
    avg_ems_path_info_df.loc[:,'Run'] = [int(path.split("/")[-1].split(".")[1]) for path in avg_ems_path_list]

    merged_info_df = pd.merge(avg_conc_path_info_df, avg_ems_path_info_df, how='left', on='Run')
    merged_info_df = merged_info_df.sort_values(by=['Run'])
    merged_info_df.index = range(len(merged_info_df))

    target_arr = np.concatenate([np.array(Dataset(path, 'r').variables['PM2_5'][0,0,:,:])[np.newaxis,:,:] for path in merged_info_df.path.tolist()])[:,:,:,None]

    # smoke_list = [Dataset(path, 'r') for path in merged_info_df.path_ems.tolist()]
    smoke_val_list = []
    for i in smoke_chem_list:
        smoke_arr_i = np.concatenate([np.array(Dataset(path, 'r').variables[i][0,0,:,:])[np.newaxis,:,:] for path in merged_info_df.path_ems.tolist()])[:,:,:,None]
        smoke_val_list.append(smoke_arr_i)

    smoke_arr = np.concatenate(smoke_val_list, axis = 3)
    smoke_arr.shape, target_arr.shape
    return smoke_arr, target_arr