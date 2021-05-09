#!/usr/bin/env python
# -*- coding: utf-8 -*-

# common 
import sys
import os
import os.path as op

import numpy as np
import pandas as pd
from scipy import interpolate

# dev
sys.path.insert(0, op.join(op.dirname(__file__), '..'))

# SWASH wrap
from wswash.wrap import SwashProject, SwashWrap
from wswash.plots import SwashPlot

# --------------------------------------------------------------------------- #
# data
p_data = op.abspath(op.join(op.dirname(__file__), '..', 'data'))
p_demo = op.join(p_data, 'demo')

# test data: csiro point 
# p_waves_demo = op.join(p_demo, 'waves_csiro_demo.nc')
p_depth = op.join(p_demo, 'case_larger.bot')


# --------------------------------------------------------------------------- #
# SWASH project (config bathymetry, parameters, computational grid)
p_proj = op.join(p_data, 'projects')  # swash projects main directory
n_proj = 'Jonswap'             # project name

sp = SwashProject(p_proj, n_proj)
sp.Nonhydrostatic = True
sp.wind = False

sp.warmup = 800

# parameters (sea level, jonswap gamma)
sp.par_jonswap_gamma = 3
sp.par_sea_level = 0
sp.vert = 1           # vertical layers
sp.tbegc = 0          # initial time (compute)
sp.deltc = 0.1        # time step (compute)
sp.tendc = 4400       # final time (compute) # sea state 3600s
sp.tbegtbl = 0        # initial time in fields output
sp.delttbl = 0.1        # time interval between fields
sp.delttbl_ = 'SEC'   # time units
sp.nlocs = 2000       # number stations along curve (-1)
        
# configure SWASH bathymetry
sp.depth = np.linspace(11, -2, 601)
sp.dp_xpc = 0         # x origin
sp.dp_ypc = 0         # y origin
sp.dp_alpc = 0        # x-axis direction 
sp.dp_xlenc = 600     # grid length in x
sp.dp_ylenc = 0       # grid length in y
sp.dp_mxc = 600       # number mesh x
sp.dp_myc = 0         # number mesh y
sp.dp_dxinp = 1       # size mesh x
sp.dp_dyinp = 1       # size mesh y

# SWASH Computational grid
sp.cc_xpc = 0         # x origin
sp.cc_ypc = 0         # y origin
sp.cc_alpc = 0.00     # x-axis direction 
sp.cc_xlenc = 600     # grid length in x
sp.cc_ylenc = 0       # grid length in y
sp.cc_mxc = 1200      # number mesh x
sp.cc_myc = 0         # number mesh y
sp.cc_dxinp = 0.5     # size mesh x
sp.cc_dyinp = 0       # size mesh y

# Obtain the intersection free-surface - profile
f = interpolate.interp1d(sp.depth, np.linspace(0,600,601))
sp.gate = f(0)

# --------------------------------------------------------------------------- #
# SWASH case input: waves_event 

# now we generate the wave event 
we = pd.DataFrame({'hs':[1],
                   'tp':[10],
                   'dir':[270],
                   'spr':[15],
                   'wvel':[30],
                   'wdir':[270]})

# --------------------------------------------------------------------------- #
# SWASH wrap (create case files, launch SWASH num. model, extract output)
sw = SwashWrap(sp)

# build cases from waves data
sw.build_cases(we)

# run SWASH
sw.run_cases()


# xds_table -> General output (Tsec, Xp, Yp, Botlev, Watlev)
# xds_quant -> Compute variables as Hs and Setup (not used - only to compare)
# df -> compute setup dynamic and mean setup in case of bichromatic waves

xds_table, xds_quant, df = sw.extract_output_points(df=False)

# fft calculations -> Hs, Hrms, Hmax
df_Hi = sw.fft_wafo(xds_table)

# Create figures and save in folder p_run
sw.plot(xds_table, xds_quant, df, df_Hi)

# Create videos from p_run
sw.create_video('stat_Jonswap.mp4', 'nonstat_Jonswap.mp4')
print('\noutput')




