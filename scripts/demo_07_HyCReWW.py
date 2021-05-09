#!/usr/bin/env python
# -*- coding: utf-8 -*-

# common 
import sys
import os
import os.path as op

import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.io import loadmat

# dev
sys.path.insert(0, op.join(op.dirname(__file__), '..'))

# SAWSH wrap
from wswash.wrap import SwashProject, SwashWrap
from wswash.plots import SwashPlot

# --------------------------------------------------------------------------- #
# Reef morphologic configurations and offshore wave and water level conditions
p_data = op.abspath(op.join(op.dirname(__file__), '..', 'data'))
p_demo = op.join(p_data, 'demo')

p_waves = op.join(p_demo, 'subset.pkl')
p_depth = op.join(p_demo, 'depth.bot')

waves = pd.read_pickle(p_waves)

# --------------------------------------------------------------------------- #
# SWASH project (config bathymetry, parameters, computational grid)
p_proj = op.join(p_data, 'projects')  # swash projects main directory
n_proj = 'test_2'             # project name

sp = SwashProject(p_proj, n_proj)
sp.Nonhydrostatic = True

sp.warmup = 300

# parameters (sea level, jonswap gamma)
sp.par_jonswap_gamma = 3
sp.par_sea_level = 0
sp.vert = 2           # vertical layers
sp.tbegc = 0          # initial time (compute)
sp.deltc = 0.001       # time step (compute)
sp.tendc = 2100       # final time (compute) # sea state 3600s
sp.tbegtbl = 0        # initial time in fields output
sp.delttbl = 0.5        # time interval between fields
sp.delttbl_ = 'SEC'   # time units
        
# configure SWASH bathymetry
sp.dp_xpc = 0         # x origin
sp.dp_ypc = 0         # y origin
sp.dp_alpc = 0        # x-axis direction 
sp.dp_ylenc = 0       # grid length in y
sp.dp_myc = 0         # number mesh y
sp.dp_dxinp = 1       # size mesh x
sp.dp_dyinp = 1       # size mesh y

# SWASH Computational grid
sp.cc_xpc = 0         # x origin
sp.cc_ypc = 0         # y origin
sp.cc_alpc = 0.00     # x-axis direction 
sp.cc_xlenc = 100     # grid length in x
sp.cc_ylenc = 0       # grid length in y
sp.cc_mxc = 400      # number mesh x
sp.cc_myc = 0         # number mesh y
sp.cc_dxinp = 0.25     # size mesh x
sp.cc_dyinp = 1       # size mesh y

# --------------------------------------------------------------------------- #
# SWASH wrap (create case files, launch SWASH num. model, extract output)
sw = SwashWrap(sp)

# build cases from waves data
waves = waves.iloc[range(1)]
waves = sw.build_cases_hycr(waves)

# run SWASH
# sw.run_cases()

# xds_table -> General output (Tsec, Xp, Yp, Botlev, Watlev)
# xds_quant -> Compute variables as Hs and Setup (not used - only to compare)
# df -> compute setup dynamic and mean setup in case of bichromatic waves

xds_table, msetup, df = sw.extract_output_points(waves, df=False, msetup=False)

# fft calculations -> Hs, Hrms, Hmax
# xds_table = xds_table.isel(case=0).squeeze()
# df_Hi = sw.fft_wafo(xds_table)
waves_ru = sw.Ru2(xds_table, waves)

sw.plot(waves_ru)

# Create figures and save in folder p_run
# sw.plot_Hycr(waves, xds_table)
# sp.depth = np.loadtxt(r'/media/administrador/DiscoHD/Escritorio/gitLab/mw_deltares/data/projects/test_1/0000/depth.bot')
# sw.plot(xds_table, msetup, df, df_Hi)

# Create videos from p_run
# sw.create_video('stat_Jonswap.mp4', 'nonstat_Jonswap.mp4')
print('\noutput')




