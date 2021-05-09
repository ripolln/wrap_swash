#!/usr/bin/env python
# -*- coding: utf-8 -*-

# common
import os
import os.path as op

# pip
import numpy as np
from scipy import signal as sg
from scipy.signal import hilbert
from scipy.stats import norm, pareto
from scipy import interpolate

# matplotlib import
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
from matplotlib import colors, patches
import matplotlib.mlab as mlab
import cmocean



class SwashPlot(object):
    'SWASH numerical model plotting'

    def __init__(self, swash_proj):

        # needs SwashProject 
        self.proj = swash_proj

    def plot_interpolated(self, ax, xnew, ynew):
        'axes plot interpolated bathymetry'
        
        # fill axs
        ax.fill_between(xnew, - ynew[0],  np.zeros((len(ynew))), facecolor="deepskyblue", alpha=0.5, zorder=1)
        ax.fill_between(xnew, np.zeros((len(ynew))) - ynew[0],  -ynew, facecolor="wheat", alpha=1, zorder=2)
        ax.plot(xnew,  -ynew, color = 'k', zorder=3)
        
        # customize axes
        ax.set_ylabel('$Depth$ $[m]$', fontweight='bold')
        ax.set_xlabel('$X$ $[m]$', fontweight='bold')
        ax.set_xlim(0, xnew[-1])
        ax.set_ylim(-ynew[0], -min(ynew))


    def plot_computational(self, ws):
        '''
        Plot computational grid and interpolated bathymetry
        
        ws - waves-dataset iterrow
        '''
        
        # import depth and spacing values
        depth = np.array(self.proj.depth)
        dxinp = self.proj.dxinp     # input bathymetry spacing
        dxL = self.proj.dxL         # user request nodos/L
        dx = ws['dx']               # calculated grid spacing from nodos/L

        # interpolate bathymetry to computational grid
        x = np.arange(0, len(depth) * dxinp, dxinp)
        xnew = np.arange(0, len(depth)*dxinp, dx)
        xnew = xnew[np.where(xnew <= x[-1])]  # avoid xnew values bellow the interpolation range
        f = interpolate.interp1d(x, depth)
        ynew = f(xnew)
        yp = np.arange(np.min(-depth), np.max(-depth), dx)
        
        # plot
        fig, axs = plt.subplots(2, figsize = (12, 8), sharex=True)
        self.plot_interpolated(axs[0], xnew, ynew)
        self.plot_interpolated(axs[1], xnew, ynew)
        
        # explicit grid
        for h in yp:
            axs[0].plot(xnew, np.zeros((len(xnew)))+h, color='black', linestyle='-', linewidth=0.5, zorder=3)
            
        for v in xnew:
            axs[0].plot(np.zeros((len(yp)))+v, yp, color='black', linestyle='-', linewidth=0.5, zorder=3)
            
        axs[0].set_title('$Computational$ $grid$ ${0}$ $nodes/L$'.format(dxL))
        axs[1].set_title('$Interpolated$ $bathymetry$')

    
    def plot_depthfile(self):
        'Plot bathymetry data including friction or vegetation area in case active commands'
        
        depth = self.proj.depth
        depth = np.array(depth)
   
        x = np.array(range(len(depth)))
       
        fig, ax = plt.subplots(1, figsize = (12, 4))
        ax.fill_between(range(len(depth)), - depth[0],  np.zeros((len(depth))), facecolor="deepskyblue", alpha=0.5, zorder=1)
        ax.fill_between(range(len(depth)), np.zeros((len(depth))) - depth[0],  -depth, facecolor="wheat", alpha=1, zorder=2)
        ax.plot(range(len(depth)),  -depth, color = 'k', zorder=3)
        
        # friction
        if self.proj.friction != None:
            
            cf_ini = self.proj.cf_ini
            cf_fin = self.proj.cf_fin
            
            pos = [(x >= cf_ini) & (x <= cf_fin)][0]
            
            # plot
            ax.fill_between(x[pos], -depth[pos], -depth[pos] + np.abs(depth[0])/30,
                    facecolor='crimson', alpha=0.5, edgecolor='crimson',  zorder=3, label='Friction')
            ax.legend(loc='lower right')

        # vegetation
        if self.proj.vegetation != None:
            
            np_ini = self.proj.np_ini
            np_fin = self.proj.np_fin
            height = self.proj.height
            cmap = cmocean.cm.algae_r
            
            pos = [(x >= np_ini) & (x <= np_fin)][0]
            
            nc = 100
            
            x = x[pos]
            y1 = -depth[pos]
            y2 = (-depth[pos] + height)

            z = np.linspace(0, 10, nc)
            normalize = colors.Normalize(vmin=z.min(), vmax=z.max())
            
            for ii in range(len(x)-1):
                y = np.linspace(y1[ii], y2[ii], nc)
                yn = np.linspace(y1[ii+1], y2[ii+1], nc)
                for kk in range(nc - 1):
                    p = patches.Polygon([[x[ii], y[kk]], 
                                         [x[ii+1], yn[kk]], 
                                         [x[ii+1], yn[kk+1]], 
                                         [x[ii], y[kk+1]]], color=cmap(normalize(z[kk])), label='Vegetation area')
                    ax.add_patch(p)

            patch = mpatches.Patch(color='green', alpha=1, label='$Vegetation$', edgecolor=None)
            ax.legend(handles=[patch], loc='lower right')
        
        # customize axes
        ax.set_ylabel('$Depth$ $[m]$', fontweight='bold')
        ax.set_xlabel('$X$ $[m]$', fontweight='bold')
        ax.set_xlim(0, range(len(depth))[-1])
        ax.set_ylim(-depth[0], -min(depth)+3)
    

    def plot_waves(self, waves, series):
        '''
        Plot wave series
        
        waves:   DataFrame with wave vars
        series:  surface elevation series
        '''
        
        # basic import
        tendc = self.proj.tendc
        deltat = self.proj.deltat
        warmup = waves['warmup'].values
        
        # hilbert transformation
        time = np.arange(0, tendc+int(warmup), deltat)
        envelope = np.abs(hilbert(series))
        
        # plot series, group envelope
        fig, ax = plt.subplots(1, figsize=(15,3))
        ax.plot(time, series, c='k')
        ax.fill_between(time, -envelope, envelope, facecolor='dodgerblue', alpha=0.3, label='Group envelope')
        ax.plot(time, envelope, '--', c='k')
        ax.plot(time, -envelope, '--', c='k')
        ax.set_xlim(0, time[-1])
        ax.set_xlabel('$Time$ $[s]$')
        ax.set_ylabel('$Elevation$ $[m]$')
        ax.legend(loc='lower right')
        
        
    def output_gate_spectrum(self, ax, ws, xds_table, gate):
        'axes plot Power Spectral Density PSD in (x = gate)'
        
        delttbl = self.proj.delttbl  
        warmup = ws['warmup'].values
        
        wt = xds_table.Tsec.values
        wt = xds_table.Tsec.values[np.where(wt >= warmup)[0]]
        
        data = xds_table.isel(Xp = gate)
        data = data.sel(Tsec=wt).Watlev.values
   
        # Estimate power spectral density using Welch’s method.        
        f, E = sg.welch(data, fs = 1/delttbl , nfft = 512)
        
        # plot
        ax.loglog(f, E, c='dodgerblue')
        ax.grid(which='both', color='lightgray', linestyle='-', linewidth=0.9)
        # ax.set_xlim(min(f), max(f))
        ax.set_xlim(0,1)
        ax.set_ylim(0, 1.1)
        ax.set_xlabel('f [Hz]', fontweight='bold')
        # ax.yaxis.set_label_position("right")
        ax.set_ylabel(r'PDS $[m^{2} s]$', fontweight='bold')
        
        
    def output_gate_series(self, ax, ws, xds_table, gate):
        'axes plot watlev for (x = gate)'
        
        warmup = ws['warmup'].values
        hs = ws['H'].values
        
        wt = xds_table.Tsec.values
        wt = xds_table.Tsec.values[np.where(wt >= warmup)[0]]
       
        data = xds_table.isel(Xp = gate)
        data = data.sel(Tsec=wt).Watlev.values
        
        # Water level series
        ax.plot(xds_table.Tsec.values, xds_table.isel(Xp = gate).Watlev.values, c='k')
        ax.set_xlim(xds_table.Tsec.values[0], xds_table.Tsec.values[-1])
        ax.set_ylabel('\u03B7 $[m] (WG = {0})$'.format(gate), fontweight='bold')
        ax.set_xlabel('Time [m]', fontweight='bold')
        ax.set_ylim(-1.5*np.float(hs), 1.5*np.float(hs))
        

    def output_forcing(self, ax, ws, xds_table, p, t):
        'axes plot wave forcing in the west boundary inluding spinup time'
        
        # read vars
        wt = xds_table.Tsec.values
        waves = xds_table.sel(Xp=0, Tsec=wt).Watlev.values.squeeze()
        pw = np.where(xds_table.isel(Tsec=p).Watlev.values!=-99)
            
        # set axis to fill between
        y1 = -xds_table.sel(Tsec=t).Botlev.values[pw[0]]
        y2 = xds_table.sel(Tsec=t).Watlev.values[pw[0]]
            
        y1 = np.reshape(y1,(len(y1,)))
        y2 = np.reshape(y2,(len(y2,)))

        # customize axes          
        ax.plot(wt, waves, 'k', linewidth=1, zorder=1)
        ax.scatter(wt[p], waves[p], c='red', s=20, marker='o', zorder=2)
        ax.set_ylabel('\u03B7 $[m] (x = 0)$', fontweight='bold', horizontalalignment='left')
        ax.set_xlim(wt[0], wt[-1])
        

    def output_profile(self, ax, ws, xds_table, t, depth):
        'axes plot watlev from spaced stations'

        h0 = ws['h0'].values
        
        pw = np.where(xds_table.sel(Tsec=t).Watlev.values != -99.0)  
        x = xds_table.Xp.values[pw[0]]
            
        # Set axis to fill between
        y1 = -xds_table.sel(Tsec=t).Botlev.values[pw[0]]
        y2 = xds_table.sel(Tsec=t).Watlev.values[pw[0]]
            
        y1 = np.reshape(y1,(len(y1,)))
        y2 = np.reshape(y2,(len(y2,)))
            
        # water level
        ax.fill_between(x, y1, y2, facecolor="deepskyblue", alpha=0.5)
        ax.fill_between(xds_table.Xp.values, np.ones((len(xds_table.Xp.values)))*-h0, -xds_table.sel(Tsec=t).Botlev.values, facecolor="wheat", alpha=1)
        ax.plot(xds_table.Xp.values, -xds_table.sel(Tsec=t).Botlev.values, color='k')
           
        # customize plot
        ax.set_ylabel('y [m]', fontweight='bold')
        ax.set_xlim(xds_table.Xp.values[0], xds_table.Xp.values[-1])
        ax.set_ylim(-h0, np.nanmax(-xds_table.sel(Tsec=0).Botlev.values)+2)
       
        
    def output_gate_runrup(self, ax, ws, xds_table, p, t):
        ' axes plot output runup from intersection free surface-profile'
        
        # read vars
        wt = xds_table.Tsec.values
        runup = xds_table.sel(Tsec=wt).Runlev.values*100

        # runrup
        ax.plot(wt, runup, 'b', linewidth=1, zorder=1)
        # ax.scatter(wt[p], runup[p], c='r', s=25, marker='o', zorder=2)
        ax.set_xlim(wt[0], wt[-1])
        ax.set_ylabel('Ru $[cm]$', fontweight='bold', horizontalalignment='left')
        
        # Validation box
        
        if np.isnan(runup).all() == True:
            props = dict(boxstyle='round', facecolor='azure', alpha=1, edgecolor='lightcyan')
            ax.text(0.03, 0.8, '$No$ $runup$ $computed$', 
                    transform=ax.transAxes, 
                    fontsize=10, 
                    horizontalalignment='left',
                    bbox=props)
        
        
    def output_gate_disch(self, ax, ws, xds_table, p, t):
        'axes plot output instantaneous discharge magnitude'
        
        gate = ws['Gate_Q'].values
        q = ws['q'].values
        
        # find position in dataset
        tr = np.where(xds_table.Xp.values >= gate)[0][0]
        gate = xds_table.Xp.values[tr]
        
        # read vars
        wt = xds_table.Tsec.values
        disch = xds_table.sel(Tsec=wt, Xp=gate).Qmag.values

        # Discharge
        ax.plot(wt, disch, 'crimson', linewidth=1, zorder=1)
        ax.scatter(wt[p], disch[p], marker='o', c='b', s=25, zorder=2)
        ax.set_xlim(wt[0], wt[-1])
        ax.set_ylim(0, max(disch)+0.1)
        ax.set_ylabel('Q $[m^{2}/s]$', fontweight='bold')

        # Validation box
        props = dict(boxstyle='round', facecolor='azure', alpha=1, edgecolor='lightcyan')

        if np.isnan(q).all() == True:
            ax.text(0.03, 0.8, '$No$ $overtopping$ $computed$'.format(np.float(q)), 
                transform=ax.transAxes, 
                fontsize=10, 
                horizontalalignment='left',
                bbox=props)
        else:
            ax.text(0.03, 0.8, '$q$ $Swash$ $=$ ${0:.2f}$ $[m/l/s]$'.format(np.float(q)), 
                transform=ax.transAxes, 
                fontsize=10, 
                horizontalalignment='left',
                bbox=props)
    
    def output_contribution(self, ax, ws, df_Hi):
        'axes plot energy contribution of different frequency waves (IC, IG, VLF) measured by Hs2'

        ic = df_Hi.HIC.values
        ig = df_Hi.HIG.values
        vlf = df_Hi.HVLF.values
        
        ic[np.where(np.isnan(ic)==True)] = 0
        ig[np.where(np.isnan(ig)==True)] = 0
        vlf[np.where(np.isnan(vlf)==True)] = 0
        
        ax.fill_between(df_Hi.Xi.values, ic**2, color='yellowgreen', edgecolor=None, alpha=0.6)
        ax.fill_between(df_Hi.Xi.values, ic**2, ic**2+ig**2, color='navy',  edgecolor=None, alpha=0.5)
        ax.fill_between(df_Hi.Xi.values, ic**2+ig**2, ig**2+ic**2+vlf**2, edgecolor=None,  color='navy', alpha=1)
        
        # customize axes
        patchIC = mpatches.Patch(color='yellowgreen', alpha=0.6, label='$H_{IC}^{2}$ $[{m}^{2}]$', edgecolor=None)
        patchIG = mpatches.Patch(color='navy', alpha=0.5, label='$H_{IG}^{2}$ $[{m}^{2}]$', edgecolor=None)
        patchVLF = mpatches.Patch(color='navy', alpha=1, label='$H_{VLF}^{2}$ $[{m}^{2}]$', edgecolor=None)
        ax.legend(handles=[patchIC, patchIG, patchVLF], loc='lower left')
        ax.set_xlim(df_Hi.Xi.values[0], df_Hi.Xi.values[-1])
        ax.set_ylim(0, max(ig**2+ic**2+vlf**2)+0.5)
        
        
    def output_fft(self, ax, ws, df_Hi):
        'axes plot output fourier transformation Hi, Hs, Hmax'
        
        ax.plot(df_Hi.Xi.values, df_Hi.HIC.values, c='r', label='$H_{IC}$')
        ax.plot(df_Hi.Xi.values, df_Hi.HIG.values, c='k', label='$H_{IG}$')
        ax.plot(df_Hi.Xi.values, df_Hi.HVLF.values, '--', c='k', label='$H_{VLF}$')
        
        # customize axes
        ax.set_xlim(df_Hi.Xi.values[0], df_Hi.Xi.values[-1])
        ax.legend(loc ='lower left')

        
    def output_env(self, ax, ws, xds_table, df, df_Hi, depth):
        'axes plot maximun and minimun free surface'
        
        ru2 = ws['ru2'].values
        WL = ws['WL'].values
        h0 = ws['h0'].values
        dx = ws['dx'].values
        kr = ws['kr'].values
        hini = -xds_table.isel(Tsec=0).Botlev.values[0]
        
        # Set axis to fill between
        x = xds_table.Xp.values[np.where(xds_table.Xp.values <= df_Hi.Xi.values[-1])]
        y = -xds_table.Botlev.isel(Tsec=0).values[np.where(xds_table.Xp.values <= df_Hi.Xi.values[-1])[0]]
        y = np.reshape(y,(len(y),))
        
        # Interpolate between bathymetry values
        f = interpolate.interp1d(x, y, kind='linear')
        ynew = f(df_Hi.Xi)
        
        # water level
        ax.fill_between(df_Hi.Xi, df_Hi.E_min, df_Hi.E_max,  facecolor="deepskyblue", alpha=1, label='\u03B7 [m]')
        ax.fill_between(df_Hi.Xi, ynew, df_Hi.E_min, facecolor='deepskyblue', alpha=.5)
        ax.fill_between(xds_table.Xp.values, np.zeros((len(xds_table.Xp.values)))+(hini), -xds_table.isel(Tsec=0).Botlev.values, facecolor="wheat", alpha=1)
        ax.plot(xds_table.Xp.values, -xds_table.isel(Tsec=0).Botlev.values, color='k')
        
        # mean setup
        ax.plot(df.Xp.values, df.Setup.values, c='blue', zorder=1, label='Setup')
 
        # runup elevation point
        if np.isnan(ru2) == False:
            idx1 = np.argwhere(np.diff(np.sign(depth - np.zeros(len(depth))+ru2-WL)) != 0).reshape(-1) + 0
            ax.scatter(range(len(depth))[idx1[0]], ru2-WL, marker='o', s= 20, c='coral', alpha=1, zorder=3, label='$Ru_{2}$')

        # customize axes
        ax.set_ylabel('y [m]', fontweight='bold')
        ax.set_xlim(xds_table.Xp.values[0], xds_table.Xp.values[-1])
        ax.set_ylim(-h0, np.nanmax(-xds_table.sel(Tsec=0).Botlev.values)+2)
        ax.legend(loc='lower right')
        props = dict(boxstyle='round', facecolor='snow', alpha=0.6, edgecolor='w')
        ax.text(0.03, 0.07, '  $Kr = {0:.2f}$ \n $h0 = {1:.2f} m$ \n  $dx = {2:.2f}$'.format(np.float(kr), np.float(h0), np.float(dx)), transform=ax.transAxes, fontsize=10, bbox=props)

    def hist_Hi(self, ax, df_Hi, ds_fft_hi, wg, x):        
         
        Hi = ds_fft_hi.sel(Xp=wg).Hi.values
        # print(ds_fft_hi)
        Hs = df_Hi.iloc[x].Hs
        m0 = (Hs/4)**2
        H13 = np.percentile(Hi, 67)

        # the histogram of the data
        num_bins = 25
        n, bins, patches = ax.hist(Hi, num_bins, color='b', alpha=0.5, density=1, rwidth=0.95, zorder=2)

        # add a 'best fit' line
        y = 4.01*(bins/Hs**2)*np.exp(-2.005*((bins**2)/(Hs**2)))

        ax.plot(bins, y, '--', c='k', label='Rayleigh distribution')
        ax.set_xlabel('$Normalized$ $wave$ $heigh$')
        ax.set_ylabel('$Probability$ $density$')
        ax.set_title('Wave heigh distribution', fontweight='bold')

        ax.legend(loc='upper right')
        
        props = dict(boxstyle='round', facecolor='b', alpha=0.05, edgecolor='w')
        ax.text(0.03, 0.85, '$Hs%$ $=$ ${0:.2f}$ $m$\n$H_13$ $=$ ${1:.2f}$ $m$'.format(np.float(Hs), H13), transform=ax.transAxes, fontsize=9, bbox=props)

        
        
    def hist_Ru(self, ax, ax1, ws, g):
        
        # the histogram of the data
        num_bins = 25
        n, bins, patches = ax.hist(g, num_bins, color='orangered', alpha=0.5, rwidth=0.95, density=1, zorder=2)
        
        ax.set_xlabel('$Normalized$ $runup$')
        ax.set_ylabel('$Probability$ $density$')
        ax.set_title('Runup distribution', fontweight='bold')
        
        props = dict(boxstyle='round', facecolor='orangered', alpha=0.10, edgecolor='w')
        ax.text(0.03, 0.90, '$Ru2%$ $=$ ${0:.2f}$ $m$'.format(np.float(ws['ru2'].values)), transform=ax.transAxes, fontsize=12, bbox=props)

    def hist_eta(self, ax, xds_table, ws, x):
        
        surface= xds_table.isel(Xp=x).Watlev.values
            
        # the histogram of the data
        num_bins = 25
        n, bins, patches = ax.hist(surface, num_bins, color='khaki', rwidth=0.95, density=1, zorder=2)  
            
        (mu, sigma) = norm.fit(surface)

        y = mlab.normpdf( bins, mu, sigma)
                    
        ax.plot(bins, y, '--', c='k', label='Normal distribution')
                    
        ax.set_xlabel('$Normalized$ $surface$ $elevation$')
        ax.set_ylabel('$Probability$ $density$')
        ax.set_title('Surface elevation distribution', fontweight='bold')
    
        ax.legend(loc='upper right')
    
    def hist_Q(self, ax, ax1, xds_table, ws, q, x):
        
        # the histogram of the data
        num_bins = 15
        n, bins, patches = ax.hist(q, num_bins,  color='crimson', rwidth=0.50, density=1, zorder=2)    
            
        ax.set_xlabel('$Normalized$ $Overtopping$')
        ax.set_ylabel('$Probability$ $density$')
        ax.set_title('Overtopping distribution', fontweight='bold')

    def limits_hi(self, ax, xds_table, ws, ds_fft_hi):
        
        maxh = 0
        minh = 100
        
        for wg in xds_table.Xp.values:
            Hi = ds_fft_hi.sel(Xp=wg).Hi.values
            if np.nanmax(Hi) > maxh: maxh = np.nanmax(Hi)
            if np.nanmin(Hi) < minh: minh = np.nanmin(Hi)
            
        ax.set_xlim(minh, maxh)    
    
    def limits_eta(self, ax, xds_table, ws):
        maxeta = 0
        mineta = 100
        
        for wg in xds_table.Xp.values:
            eta = xds_table.sel(Xp=wg).Watlev.values
            if np.nanmax(eta) > maxeta: maxeta = np.nanmax(eta)
            if np.nanmin(eta) < mineta: mineta = np.nanmin(eta)
            
        ax.set_xlim(mineta, maxeta)               

    def single_plot_stat(self, ws, xds_table, df, df_Hi, p_run, depth):
        '''
        Figure-frames for a X-dependent-video with:
            Hs, Hrms, Hmax
            evolution of free surface
            evolution of spectrum
            
        ws  -        waves-dataset iterrow
        xds_table -  Dataset with output vars
        df -         setup dataframe
        df_Hi -      FFt output
        p_run -      output file path
        depth -      bathymetry values
        runrup -     bool var 
        '''
        
        gate = ws['Gate_Q']
        hs = ws['H'].values
        tp = ws['T'].values
        WL = ws['WL'].values
        
        fig = plt.figure(figsize=(20, 9))
        gs = GridSpec(4, 10)
        gs.update(wspace=1.1,hspace=0.15)
        
        ax0 = fig.add_subplot(gs[0, :5])     # Hsi contribution
        ax1 = fig.add_subplot(gs[1, :5])     # HsIc, HsIg, HsVlf Validation
        ax2 = fig.add_subplot(gs[2:4, :5])   # Shore-cross profile
        ax3 = fig.add_subplot(gs[-1, -5:-2]) # Watlev series
        ax4 = fig.add_subplot(gs[-1, -2:])   # Spectrum
        ax5 = fig.add_subplot(gs[0, -5:])    # Forcing
        ax6 = fig.add_subplot(gs[1, -5:])    # Runup
        ax7 = fig.add_subplot(gs[2, -5:])    # Overtopping discharge

        # Save figures to create a video
        for figure, wg in enumerate(np.arange(0, int(gate), 5)):
        
            x = np.where(xds_table.Xp.values >= wg)[0][0]
            self.output_contribution(ax0, ws, df_Hi)
            self.output_fft(ax1, ws, df_Hi)
            self.output_env(ax2, ws, xds_table, df, df_Hi, depth)
            self.output_gate_series(ax3, ws, xds_table, x)
            self.output_gate_spectrum(ax4, ws, xds_table, x)
            self.output_forcing(ax5, ws, xds_table, 0, 0)
            self.output_gate_runrup(ax6, ws, xds_table, 0, 0)
            self.output_gate_disch(ax7, ws, xds_table, 0, 0)
            
            # plot in ax1 the WG indicator
            ax2.scatter(wg, 0, s=15, c='b', zorder=2)
            ax2.plot([wg, wg, wg, wg, wg], [0, 0.2, 0.3, 0.4, 0.5], color='deepskyblue', zorder=1)
            ax2.annotate('WG {0}'.format(wg), xy=(wg-4, 0.90), rotation='vertical')
            
            ax0.set_title('Jonswap $H_s={0}$ $m$ $T_p={1:.2f}$ $s$ $\u03B3$ $=$ $10$ $WL={2}$ $m$'.format(np.float(hs), np.float(tp), np.float(WL)), fontweight='bold')
            path = op.join(p_run, 'X')
            
            # save figure
            if not os.path.exists(path): os.mkdir(path)
            fig.savefig(op.join(path, "{0}.png".format(str(figure).zfill(3))),  pad_inches = 0)
            
            # clear axis
            ax0.clear()
            ax1.clear()
            ax2.clear()
            ax3.clear()
            ax4.clear()
            ax5.clear()
            ax6.clear()
            ax7.clear()
            
            plt.ioff()  # mode off interactive graphics
        plt.close(fig)
        

    def single_plot_nonstat(self, ws, xds_table, df, df_Hi, p_run, depth, t_video):
        '''
        Figure-frames for a Time-dependent-video with:
            Boundary forcing
            Waves breaking over profile
            Run-up excursion
            Discharge magnitude /m
        '''
        
        hs = ws['H'].values
        tp = ws['T'].values
        
        fig = plt.figure(figsize=(9, 13))
        gs = GridSpec(3, 1)
        ax0 = fig.add_subplot(gs[0, :])    # Forcing
        ax1 = fig.add_subplot(gs[1:3, :])  # Profile wave-breaking

        # save figures to create a video
        for figure, p in enumerate(np.arange(0, t_video, 2)):
            
            t = xds_table.Tsec.values[p]
            self.output_forcing(ax0, ws, xds_table, p, t)
            self.output_profile(ax1, ws, xds_table, t, depth)
            
            ax0.set_title('Jonswap $H_s={0}$ m $T_p={1:.2f}$ s $\u03B3 = 10$ Time = {2} s'.format(np.float(hs), np.float(tp), int(xds_table.Tsec.values[p])), fontweight='bold')
            path = op.join(p_run, 'Time')
            
            # save figure
            if not os.path.exists(path): os.mkdir(path)
            fig.savefig(op.join(path, "{0}.png".format(str(figure).zfill(3))), pad_inches = 0)
            
            # clear axis
            ax0.clear()
            ax1.clear()
            
            plt.ioff()  # mode off interactive graphics
        plt.close(fig)

    def attr_axes(self, ax):
        
        ax.grid(color='lightgrey', axis='both', which='major', zorder=1)
        # ax.grid(color='gainsboro', axis='both', which='minor', linestyle='--')        



    def histograms(self, ws, xds_table, g, q, df, df_Hi, ds_fft_hi, depth, p_run):
        '''
        Figure histogram evolution across the reef profile (Ru, Q, Hi)
        '''
        gate = ws['Gate_Q']
        hs = ws['H'].values
        tp = ws['T'].values
        WL = ws['WL'].values
        
        fig = plt.figure(figsize=(15, 7))
        gs = GridSpec(3, 2,  wspace=0.2, hspace=0.3)
        
        ax0 = fig.add_subplot(gs[0, 0])    # Profile
        ax1 = fig.add_subplot(gs[1, 0])    # Hi histogram
        ax2 = fig.add_subplot(gs[2, 0])    # Ru-Q Histogram
        ax3 = fig.add_subplot(gs[1, 1])    # Eta
        ax4 = fig.add_subplot(gs[1, 1])    # Ru-Q histogram
        
        # save figures to create a video
        for figure, wg in enumerate(np.arange(0, max(ds_fft_hi.Xp.values), 5)):
            
            self.attr_axes(ax1)
            self.attr_axes(ax2)
            self.attr_axes(ax3)
            self.attr_axes(ax4)
        
            x = np.where(xds_table.Xp.values >= wg)[0][0]
            self.hist_Hi(ax1, df_Hi, ds_fft_hi, wg, x)
            self.hist_eta(ax3, xds_table, ws, x)
            
            if np.isnan(q).all() == True:
                self.hist_Ru(ax2, ax4, ws, g)
            else:
                self.hist_Q(ax2, ax4, xds_table, ws, q, x)
                
            self.output_env(ax0, ws, xds_table, df, df_Hi, depth)
            
            # ax0.set_title('Jonswap $H_s={0}$ m $T_p={1:.2f}$ s $\u03B3 = 10$ Time = {2} s'.format(np.float(hs), np.float(tp), int(xds_table.Tsec.values[p])), fontweight='bold')
            path = op.join(p_run, 'Hist')
            
            # plot in ax1 the WG indicator
            ax0.scatter(wg, 0, s=15, c='b', zorder=2)
            ax0.plot([wg, wg, wg, wg, wg], [0, 0.2, 0.3, 0.4, 0.5], color='deepskyblue', zorder=1)
            ax0.annotate('WG {0}'.format(wg), xy=(wg-4, 0.90), rotation='vertical')
            
            # Keep constant limits
            self.limits_hi(ax1, xds_table, ws, ds_fft_hi)
            self.limits_eta(ax3, xds_table, ws)
            
            # save figure
            if not os.path.exists(path): os.mkdir(path)
            fig.savefig(op.join(path, "{0}.png".format(str(figure).zfill(3))), pad_inches = 0)

            # clear axis
            ax0.clear()
            ax1.clear()
            ax2.clear()
            ax3.clear()
            ax4.clear()
            
            plt.ioff()  # mode off interactive graphics
        plt.close(fig)        
        
        
        
        
        
        
        
        
        
        
        
