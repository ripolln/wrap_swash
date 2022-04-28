#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import xarray as xr
from scipy import signal

from .spectra import spectral_analysis
from .statistics import upcrossing


class Postprocessor(object):
    'SWASH numerical model customized output postprocessor operations'

    def __init__(self, swash_proj, swash_input, swash_output):
        '''
        initialize postprocessor

        swash_proj  - SwanProject instance, contains project parameters
        swash_input - SwanInput instance, contains case input parameters
        swash_output - xarray.Dataset, cointains case output (from io.output_points())
        '''

        self.swash_proj = swash_proj
        self.swash_input = swash_input
        self.swash_output = swash_output

    def calculate_overtopping(self):
        '''
        Calculates overtopping at maximum bathymetry elevation point

        returns acumulated overtopping (l/s/m)
        '''

        # get data from project, input and output
        depth = - np.array(self.swash_proj.depth)
        dx = self.swash_proj.b_grid.dx
        out = self.swash_output
        warmup = self.swash_input.waves_parameters['warmup']
        tendc = self.swash_input.waves_parameters['tendc']

        # remove warmup time from output
        t = np.where(out['Tsec'].values > warmup)[0]
        outn = out.isel(Tsec = t) 

        # get overtopping at max bathymetry elevation point
        ix_gate_q = int(np.argmax(depth, axis=None, out=None) * dx)
        q = outn.isel(Xp = ix_gate_q).Qmag.values

        # process overtopping nodatavalues and nanvalues
        q[np.where(q == -9999.0)[0]] = np.nan
        q = q[~np.isnan(q)]
        q = q[np.where(q > 0)]

        # acumulated overtopping (units as l/s/m)
        Q = np.nansum(q) * 1000 / tendc

        return Q, q

    def calculate_reflection(self, flume_f=0.25):
        '''
        Calculates waves reflection using Welch's method

        flume_f - fraction of profile length to compute kr

        returns
        Kr - reflection coefficient
        '''

        # get data from project, input and output
        depth = - np.array(self.swash_proj.depth)
        delttbl = self.swash_proj.delttbl
        H = self.swash_input.waves_parameters['H']
        out = self.swash_output

        # set flume as depth/4 (default flume_f = 0.25)
        flume = int(len(depth) * flume_f)

        # output water level
        sw = out.isel(Xp = flume)['Watlev'].values
        sw = sw[~np.isnan(sw)]

        # estimate power spectral density using Welch's method
        fout, Eout = signal.welch(sw, fs = 1/delttbl , nfft = 512, scaling='density')
        m0out = np.trapz(Eout, x=fout)
        Hsout = 4 * np.sqrt(m0out)

        # calculate reflection coefficient
        Kr = np.sqrt((Hsout/H)-1)

        return Kr

    def calculate_runup(self):
        '''
        Calculates runup

        returns runup-02
        '''

        # get data from project, input and output
        depth = - np.array(self.swash_proj.depth)
        dx = self.swash_proj.b_grid.dx
        out = self.swash_output
        warmup = self.swash_input.waves_parameters['warmup']

        # remove warmup time from output
        t = np.where(out['Tsec'].values > warmup)[0]
        outn = out.isel(Tsec = t) 

        # get runup
        g = outn.Runlev.values

        # remove runup nodatavalues and nanvalues
        g[np.where(g == -9999.0)] = np.nan
        g = g[~np.isnan(g)]

        # calculate ru2 (98 percentile)
        ru2 = np.percentile(g, 98)

        # locate max bathymetry elevation point (index)
        ix_gate_q = int(np.argmax(depth, axis=None, out=None))
        depth_gate = depth[ix_gate_q]

        # check ru2 under depth at max bathymetry elevation point 
        if len(g) > 0 and np.percentile(g, 98) < depth_gate:
            ru2 = np.percentile(g, 98)
        else:
            ru2 = depth_gate

        return ru2, g

    def calculate_setup(self):
        '''
        Calculate mean set-up

        returns setup xarray.Dataset
        '''

        # get data from project, input and output
        out = self.swash_output
        warmup = self.swash_input.waves_parameters['warmup']

        # remove warmup time from output
        t = np.where(out['Tsec'].values > warmup)[0]
        outn = out.isel(Tsec = t) 

        #wp = ws.warmup.values

        outn = outn.squeeze()
        Tsec = outn.Tsec.values

        setup_m = []
        for i, j in enumerate(outn.Xp.values):
            sx = outn.isel(Xp=i)['Watlev'].values
            sx[np.where(sx == -99.)] = np.nan
            #sx = sx[np.where(Tsec > wp)]
            set_d = np.nanmean(sx)
            setup_m.append(set_d)

        # return mean setup
        ds = xr.Dataset({'Setup': ('Xp', setup_m)}, {'Xp': outn.Xp.values})

        return(ds)

    def calculate_spectral_analysis(self):
        '''
        makes a water level spectral analysis (scipy.signal.welch)
        then separates incident waves, infragravity waves, very low frequency
        waves.

        returns

        df         - pandas dataframe with analysis output
        ds_fft_hi  -
        '''

        # get data from project, input and output
        delttbl = self.swash_proj.delttbl
        warmup = self.swash_input.waves_parameters['warmup']
        xds_table = self.swash_output

        # time values
        Tsec = xds_table.Tsec.values
        pt = np.where((Tsec > warmup))[0]
        xds_table = xds_table.squeeze()

        # initialize output holders
        Hm, Hr = [], []
        env_ma, env_mi = [], []
        x = []
        Hs, Hic, Hig, Hvlf = [], [], [], []

        # Save Hi for each X value
        ds_stat = xr.Dataset(
            data_vars = {
            },
            coords = {
                'Xp': [],
                'ix_wave': [],
            },
        )

        # For each sensor, extract Watlev and supress warmp up 
        for i, j in enumerate(xds_table.Xp.values):

            sw = xds_table.isel(Xp=i, Tsec=pt).Watlev.values
            sw = sw[np.where(sw != -99.)]

            if len(sw)>0:

                # perform spectral analysis
                Hsi, HIC, HIG, HVLF = spectral_analysis(sw, delttbl)

                # perform statistical analysis 
                Ti, Hi = upcrossing(pt, sw)

                # append output to datasets
                if len(Hi) > 0:
                    Hmax = max(Hi)
                    Hrmsi = 0
                    for h in Hi:
                        Hrmsi += h**2

                    Hr.append(np.sqrt(Hrmsi/len(Hi)))
                    Hm.append(Hmax)
                    Hs.append(Hsi)
                    Hig.append(HIG)
                    Hic.append(HIC)
                    Hvlf.append(HVLF)

                else:
                    Hr.append(np.nan)
                    Hm.append(np.nan)
                    Hs.append(np.nan)
                    Hig.append(np.nan)
                    Hic.append(np.nan)
                    Hvlf.append(np.nan)

                env_ma.append(np.nanmax(sw))
                env_mi.append(np.nanmin(sw))
                x.append(xds_table.Xp.values[i])

                # merge stats for output
                ds_Hi = xr.Dataset(
                    {
                        'Hi':(['Xp', 'ix_wave'], np.array(Hi).reshape(1, -1)),
                    },
                    coords = {
                        'ix_wave':np.arange(0, len(Hi), 1).reshape(-1),
                        'Xp': [j],
                    }
                )

                ds_stat = xr.merge([ds_stat, ds_Hi])

        # mount output dataframe
        df_spec = pd.DataFrame({
            'Xi': x,
            'Hmax': Hm,
            'Hrms': Hr,
            'Hs': Hs,
            'HIC': Hic,
            'HIG': Hig,
            'HVLF': Hvlf,
            'E_max': env_ma,
            'E_min': env_mi,
        })

        # clean output dataframe
        df_spec['E_min'][df_spec['E_min']==-99.0] = np.nan
        df_spec['E_max'][df_spec['E_max']==-99.0] = np.nan

        return(df_spec, ds_stat)

