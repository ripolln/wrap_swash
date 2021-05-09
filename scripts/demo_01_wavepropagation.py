#!/usr/bin/env python
# -*- coding: utf-8 -*-

# common 
import sys
import os
import os.path as op

import numpy as np
import pandas as pd
import xarray as xr

# dev
sys.path.insert(0, op.join(op.dirname(__file__), '..'))

# SWASH wrap
from wswash.wrap import SwashProject, SwashWrap

# --------------------------------------------------------------------------- #
# data
p_data = op.abspath(op.join(op.dirname(__file__), '..', 'data'))
p_demo = op.join(p_data, 'demo')

# test data: csiro point 
p_waves_demo = op.join(p_demo, 'waves_csiro_demo.nc')

xds_waves = xr.open_dataset(p_waves_demo)
xds_waves = xds_waves.squeeze()   # remove lon,lat dim (len=1)
waves = xds_waves.to_dataframe()  # xarray --> pandas

# --------------------------------------------------------------------------- #
# SWASH project (config bathymetry, parameters, computational grid)
p_proj = op.join(p_data, 'projects')  # swash projects main directory
n_proj = '01_hydrostatic'             # project name

sp = SwashProject(p_proj, n_proj)
sp.Nonhydrostatic = True
sp.wind = True

# parameters (sea level, jonswap gamma)
sp.par_jonswap_gamma = 1.9
sp.par_sea_level = 10
sp.vert = 1           # vertical layers
sp.tbegc = 0          # initial time (compute)
sp.deltc = 0.01       # time step (compute)
sp.tendc = 30         # final time (compute)
sp.tbegtbl = 0        # initial time in fields output
sp.delttbl = 0.1      # time interval between fields
sp.delttbl_ = 'SEC'   # time units
sp.nlocs = 5          # number stations along curve (-1)

# configure SWASH bathymetry
sp.depth = np.ones((31,)) * 0.4
sp.dp_xpc = 0         # x origin
sp.dp_ypc = 0         # y origin
sp.dp_alpc = 0        # x-axis direction 
sp.dp_xlenc = 30      # grid length in x
sp.dp_ylenc = 0       # grid length in y
sp.dp_mxc = 30        # number mesh x
sp.dp_myc = 0         # number mesh y
sp.dp_dxinp = 1       # size mesh x
sp.dp_dyinp = 1       # size mesh y

# SWASH Computational grid
sp.cc_xpc = 0         # x origin
sp.cc_ypc = 0         # y origin
sp.cc_alpc = 0.00     # x-axis direction 
sp.cc_xlenc = 30      # grid length in x
sp.cc_ylenc = 0       # grid length in y
sp.cc_mxc = 1200      # number mesh x
sp.cc_myc = 0         # number mesh y
sp.cc_dxinp = 30      # size mesh x
sp.cc_dyinp = 0       # size mesh y


# --------------------------------------------------------------------------- #
# SWASH case input: waves_event 

# now we generate the wave event 
vs = ['hs', 't02', 'dir', 'spr', 'U10', 'V10']
we = waves['2000-01-02 00:00':'2000-01-03 00:00'][vs]  # data subset 
we['wvel'] = 30   # wind speed
we['wdir'] = 270  # wind direction
we['forcing'] = 'no'

print('\ninput wave events')
print(we)

# --------------------------------------------------------------------------- #
# SWASH wrap (create case files, launch SWASH num. model, extract output)
sw = SwashWrap(sp)

# build cases from waves data
sw.build_cases(we)

# run SWASH
sw.run_cases()

# extract output from cases (Block, Table)
xds_table = sw.extract_output_points()

print('\noutput')

