#!/usr/bin/env python
# -*- coding: utf-8 -*-

# common
import os
import os.path as op
import subprocess as sp

# pip
import numpy as np
import pandas as pd
import xarray as xr
from scipy import signal as sg
from tqdm import tqdm

p_cases = r'/Users/albaricondo/Documents/wrapswash-1d/data/demo/cases'
ds_sta = postprocessing(p_cases)
ds_sta.to_netcdf(op.join(p_cases, '..'))

def get_run_folders(p_cases):
        'Return sorted list of project cases folders'
        ldir = sorted(os.listdir(p_cases))
        fp_ldir = [op.join(p_cases, c) for c in ldir]

        return [p for p in fp_ldir if op.isdir(p)]

def postprocessing(p_cases):
        ''' Concat folder results '''
        
        # get sorted execution folders
        run_dirs = get_run_folders(p_cases)
        depth = - np.loadtxt(op.join(p_cases, '0000', 'depth.bot')
        tendc = 3600
        warmup = 0.15*tendc
        dxinp = 1
        Gate_Q = np.argsort(depth) / dxinp        
        
        ru2, Q = [], []
      
        # exctract output case by case and concat in list
        for case_id, p_run in enumerate(run_dirs):

            xds_out = output_points(p_run)   # output.tab

            wp = np.where(xds_out.Tsec.values > warmup)[0]

            # overtopping (swash m2/s)
            q = xds_out.isel(Tsec=wp, Xp=int(Gate_Q)).Qmag.values
            q[np.where(q == -9999.0)[0]] = 0
            q = q[np.isnan(q)==False]
            q = q[np.where(q > 0)]
            Q.append(np.nansum(q)*1000/tendc) # [l/s/m]
            
            # runup
            g = xds_out.isel(Tsec=wp).Runlev.values
            g[np.where(g == -9999.0)] = np.nan
            g = g[np.isnan(g)==False]
         
            if len(g) > 0 and np.percentile(g,98) < depth[int(Gate_Q)]:
                ru2.append(np.percentile(g, 98))
            else:
                ru2.append(depth[int(Gate_Q)])
            
        # Stadistics Ru2% - qmean
        ds_sta = pd.DataFrame(
                    {
                        'Ru2': ws['ru2'],
                        'q': ws['q']
                    }
        )

        pass
        return(ds_sta)
            
def output_points(p_case):
         
        'read table_outpts.tab output file and returns xarray.Dataset'
        
        p_dat = op.join(p_case, 'output.tab')
        p_run = op.join(p_case, 'run.tab')
        
        ds1 = read_file(p_dat)
        ds2 = read_file(p_run)
        
        ds1 = ds1.set_index(['Xp', 'Yp','Tsec']) #, coords = Time, Xp, Yp
        ds1 = ds1.to_xarray()
        
        ds2 = ds2.set_index(['Tsec']) #, coords = Time, Xp, Yp
        ds2 = ds2.to_xarray()

        ds = xr.merge([ds1, ds2], compat='no_conflicts')
        
        return(ds)
         
    def read_file(p_file):
        'Read output file p_file path'
        
        # Read head colums (variables names)
        f = open(p_file,"r")
        lineas = f.readlines()
        
        names = lineas[4].split()
        names = names[1:] # Eliminate '%'
        f.close()        
        
        # Read data rows
        data = np.loadtxt(p_file, skiprows=7)
        
        ds1 = pd.DataFrame({})
        for p, t in enumerate(names):
            ds1[t] = data[:,p]
        
        return(ds1)

        
        
        
        