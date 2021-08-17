#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import xarray as xr
from scipy import signal


class Postprocessor(object):
    'SWASH numerical model customized output postprocessor operations'

    def __init__(self, swash_proj, swash_input, swash_output):
        '''
        initialize postprocessor

        swash_proj  - SwanProject instance, contains project parameters
        swash_input - SwanInput instance, contains case input parameters
        swash_output - xarray.Dataset, cointains case output 
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

    def calculate_reflection(self):
        '''
        Calculates waves reflection using Welch's method

        returns reflection coefficient
        '''

        # get data from project, input and output
        depth = - np.array(self.swash_proj.depth)
        delttbl = self.swash_proj.delttbl
        H = self.swash_input.waves_parameters['H']
        out = self.swash_output

        # set flume as depth/4
        flume = int(len(depth)/4)

        # output water level
        sw = out.isel(Xp = int(flume/2))['Watlev'].values
        sw = sw[~np.isnan(sw)]

        # estimate power spectral density using Welch's method
        fout, Eout = signal.welch(sw, fs = 1/delttbl , nfft = 512, scaling='density')
        m0out = np.trapz(Eout, x=fout)
        Hsout = 4 * np.sqrt(m0out)

        # calculate reflection coefficient
        Kr = np.sqrt((Hsout/H)-1)
        # TODO: Hsout < H asi que sqrt de numero negativo

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

    def calculate_HT(self):
        '''
        TODO: add description 
        '''

        # TODO ds_fft_hi sale vacio sin ninguna variable

        # TODO: revisar motivo warning y arreglar (en vez de ocultar)
        import warnings
        warnings.filterwarnings('ignore')

        # aux function
        def upcrossing(time, watlev):
            '''
            Performs fast fourier transformation FFt by extracting
            the wave series mean and interpolating to find the cross-zero

            time : time series
            watlev : surface elevation series

            return - Wave heigths and periods
            '''

            watlev = watlev - np.mean(watlev)
            neg = np.where(watlev < 0)[0]

            neg1 = []
            r = []
            H = []

            for i in range(len(neg)-1):
                if neg[i+1] != neg[i] + 1:
                    neg1.append(neg[i])

            for i in range(len(neg1)-1):
                p = np.polyfit(time[slice(neg1[i],neg1[i]+1)], watlev[slice(neg1[i], neg1[i]+1)], 1)
                r.append(np.roots(p))

            r = [item for sublist in r for item in sublist]
            r = np.abs(r)
            for i in np.arange(2, len(r), 1):
                H.append(max(watlev[np.where((time < r[i]) & (time > r[i-1]))]) - min(watlev[np.where((time < r[i]) & (time > r[i-1]))]))

            T = np.diff(r)

            return(T, H)


        # get data from project, input and output
        delttbl = self.swash_proj.delttbl
        warmup = self.swash_input.waves_parameters['warmup']
        out = self.swash_output

        # TODO refactor
        xds_table = out


        # Time values
        Tsec = xds_table.Tsec.values
        pt = np.where((Tsec > warmup))[0]
        xds_table = xds_table.squeeze()

        Hm, Hr = [], []
        env_ma, env_mi = [], []
        x = []
        Hs, Hic, Hig, Hvlf = [], [], [], []

        # Save Hi for each X value
        ds_fft_hi = xr.Dataset({'Xp':[]})

        # For each sensor, extract Watlev and supress warmp up 
        for i, j in enumerate(xds_table.Xp.values):

            sw = xds_table.isel(Xp=i, Tsec=pt).Watlev.values
            sw = sw[np.where(sw != -99.)]

            if len(sw)>0:
                f, E = signal.welch(sw, fs = 1/delttbl , nfft = 512, scaling='density')
                # Slice frequency to divide energy into components
                # Incident waves IC
                fIC = f[np.where(f > 0.04)]
                EIC = E[np.where(f > 0.04)]
                mIC =  np.trapz(EIC, x=fIC)

                # Infragravity waves IG
                fIG = f[(np.where(f > 0.004) and np.where(f < 0.04))]
                EIG = E[(np.where(f > 0.004) and np.where(f < 0.04))]
                mIG =  np.trapz(EIG, x=fIG)

                # Very low frequency waves VLF
                fVLF = f[(np.where(f > 0.001) and np.where(f < 0.004))]
                EVLF = E[(np.where(f > 0.001) and np.where(f < 0.004))]
                mVLF =  np.trapz(EVLF, x=fVLF)

                m0 = np.trapz(E, x=f)

                Hsi = 4 * np.sqrt(m0)
                HIC = 4 * np.sqrt(mIC)
                HIG = 4 * np.sqrt(mIG)
                HVLF = 4 * np.sqrt(mVLF)

                Ti, Hi = upcrossing(pt, sw)
                if len(Hi)>0:
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

                ds_Hi = pd.DataFrame({'Xp':np.ones(len(Hi))*j, 'Hi':Hi})
                ds_Hi = ds_Hi.set_index('Xp')
                ds_Hi = ds_Hi.to_xarray()

                ds_fft_hi = xr.concat([ds_fft_hi, ds_Hi], dim='Xp')

        df = pd.DataFrame({'Xi':x, 'Hmax':Hm, 'Hrms':Hr, 'Hs':Hs, 'HIC':Hic,
                           'HIG':Hig, 'HVLF':Hvlf, 'E_max':env_ma,'E_min':env_mi})

        df['E_min'][df['E_min']==-99.0] = np.nan
        df['E_max'][df['E_max']==-99.0] = np.nan
        df['Hs'][df['Hs'] > 30] = np.nan
        df['HIC'][df['HIC'] > 30] = np.nan
        df['HIG'][df['HIG'] > 30] = np.nan
        df['HVLF'][df['HVLF'] > 30] = np.nan

        return(df, ds_fft_hi)

