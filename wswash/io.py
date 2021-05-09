#!/usr/bin/env python
# -*- coding: utf-8 -*-

# common
import os
import os.path as op

# pip
import numpy as np
import pandas as pd
from scipy import signal as sg
from scipy import interpolate
import xarray as xr
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# swash
from .plots import SwashPlot


class SwashIO(object):
    'SWASH numerical model input/output'

    def __init__(self, swash_proj):

        # needs SwashProject 
        self.proj = swash_proj
        self.plots = SwashPlot(self.proj)   # swash plotting tool

    def DispersionLonda(self, T, h):
        'Solve the wave dispersion relation'

        L1 = 1
        L2 = ((9.81*T**2)/2*np.pi)*np.tanh(h*2*np.pi/L1)
        umbral = 1

        while umbral > 0.1:
            L2 = ((9.81*T**2)/(2*np.pi))*np.tanh((h*2*np.pi)/L1)
            umbral = np.abs(L2-L1)
            L1 = L2

        L = L2
        k = (2*np.pi)/L
        c = np.sqrt(9.8*np.tanh(k*h)/k)
        return(L,k,c)

    def print_wrap(self, case_id):
        'Print "input.sws" text'

        p_cases = self.proj.p_cases

        path = op.join(p_cases, case_id)
        os.chdir(path)
        f = open('input.sws')

        # print(lines)
        lines = f.read().splitlines()
        f.close()
        for line in lines:
            print(line)

    def make_project(self):
        'Make swash project folder and subfolders'

        if not op.isdir(self.proj.p_main): os.makedirs(self.proj.p_main)
        if not op.isdir(self.proj.p_cases): os.makedirs(self.proj.p_cases)

    def build_case(self, case_id, waves_ss):
        '''
        Build SWASH case input files for given wave sea state (H, T, WL)

        case_id  - SWASH case index (int)
        waves_ss - wave sea state (H, T, WL)
        '''

        # SWASH case path
        p_case = op.join(self.proj.p_cases, case_id)

        # make execution dir
        if not op.isdir(p_case): os.makedirs(p_case)

        # make depth file
        p_depth = op.join(p_case, 'depth.bot')
        self.make_depth(p_depth)

        # make wave file
        # self.make_wave(op.join(p_case, 'waves.bnd'))

        # make friction file
        if self.proj.friction == True:
            p_friction = op.join(p_case, 'friction.txt')
            self.make_friction(p_friction, p_depth, waves_ss)

        # make vegetation  file
        if self.proj.vegetation == True:
            p_vegetation = op.join(p_case, 'plants.txt')
            self.make_vegetation(p_vegetation, p_depth, waves_ss)

        # make input.sws file
        save = op.join(p_case, 'input.sws')

        waves = self.make_input(save, case_id, waves_ss)

        return(waves)

    def energy_spectrum(self, waves, warmup):
        '''
        spec: Dataset with vars for the whole partitions
        S(f,dir) = S(f) * D(dir)
        S(f) ----- D(dir)
        Meshgrid de x freqs - dir - z energy
        '''

        hs = waves.Hs
        tp = waves.Tp
        tendc = self.proj.tendc
        gamma = waves.gamma

        # Defining frequency series - tend length
        freqs = np.linspace(0.02,1,tendc+int(warmup))

        S = []
        fp = 1/tp

        for f in freqs:
            if f <= fp:
                sigma = 0.07
            if f > fp:
                sigma = 0.09

            Beta = (0.06238/(0.23+0.0336*gamma-0.185*(1.9+gamma)**-1))*(1.094-0.01915*np.log(gamma))
            Sf = Beta * (hs**2) * (tp**-4) * (f**-5)*np.exp(-1.25*(tp*f)**-4)*gamma**(np.exp((-(tp*f-1)**2)/(2*sigma**2)))
            S.append(Sf)

        return (S)

    def reef_profile(self, h0, Slope1, Slope2, Wreef, Wfore, bCrest, emsl):
        '''

        h0 :          offshore depth (m)
        Slope1 :      fore shore slope
        Slope2 :      inner shore slope
        Wreef :       reef bed width (m)
        Wfore :       flume length before fore toe (m)
        bCrest :      beach heigh (m)
        emsl :        mean sea level (m)

        return - depth data values
        '''

        # set bathymetry FILE spacing [m]
        dxinp = self.proj.dxinp

        # flume length
        W_inner = bCrest / Slope2
        W1 = int(h0 / Slope1)

        # sections length
        x1 = np.arange(0, Wfore,   dxinp)
        x2 = np.arange(0, W1,      dxinp)
        x3 = np.arange(0, Wreef,   dxinp)
        x4 = np.arange(0, W_inner, dxinp)

        # curve equation
        y_fore = np.zeros(len(x1)) + [h0]
        y1 = - Slope1 * x2 + h0
        y2 = np.zeros(len(x3)) + [0]
        y_inner = - Slope2 * x4

        # overtopping cases: an inshore plane beach to dissipate overtopped flux
        plane = 0.005 * np.arange(0, len(y_inner), 1) + y_inner[-1]

        d = [y_fore, y1 ,y2, y_inner, plane]
        depth = [item + emsl for sublist in d for item in sublist]

        return(depth)

    def linear_profile(self, h0, bCrest, m, Wfore):
        '''

        h0 : offshore depth (m)
        bCrest : beach heigh (m)
        m : profile slope
        Wfore : flume length before slope toe (m)

        return - bathymetry values
        '''

        # Bathymetry file spacing
        dxinp = self.proj.dxinp

        # Flume length
        W1 = int(h0 / m)
        W2 = int(bCrest / m)

        # Sections length
        x1 = np.arange(0, Wfore, dxinp)
        x2 = np.arange(0, W1, dxinp)
        x3 = np.arange(0, W2, dxinp)

        # Curve equation
        y_fore = np.zeros(len(x1)) + [h0]
        y1 = - m * x2 + h0
        y2 = - m * x3

        # Overtopping cases: an inshore plane beach to dissipate overtopped flux
        plane = 0.005 * np.arange(0, len(y2), 1) + y2[-1] # Length bed = 2 L

        d = [y_fore, y1, y2, plane]
        depth = [item  for sublist in d for item in sublist]

        return(depth)

    def parabolic_profile(self, h0, A, xBeach, bCrest):

        # Bathymetry file spacing
        dxinp = self.proj.dxinp

        lx = np.arange(1, xBeach, dxinp)
        y = - (bCrest/xBeach) * lx

        depth, xl = [], []
        x, z = 0, 0

        while z <= h0:
            z = A * x**(2/3)
            depth.append(z)
            xl.append(x)
            x += dxinp

        f = interpolate.interp1d(xl, depth)
        xnew = np.arange(0, int(np.round(len(depth)*dxinp)), 1)
        ynew = f(xnew)
        d = [ynew[::-1], y]

        # set project depth and plot
        self.proj.depth = [item  for sublist in d for item in sublist]
        self.plots.plot_depthfile()

    def make_depth(self, p_file):
        'Export depth_grid to plain text depth file (SWASH compatible)'

        np.savetxt(p_file, self.proj.depth, fmt='%.2f')

    def biparabolic_profile(self, h0, hsig, omega_surf_list, TR):
        'Bernabeu et al. 2013'

        # Plot the profile
        fig, ax = plt.subplots(1, figsize = (12, 4))

        # Discontinuity point
        hr = 1.1 * hsig + TR

        # Legal point
        ha = 3 * hsig + TR

        # Empirical adjusted parameters
        A = 0.21 - 0.02 * omega_surf_list
        B = 0.89 * np.exp(-1.24 * omega_surf_list)
        C = 0.06 + 0.04 * omega_surf_list
        D = 0.22 * np.exp(-0.83*omega_surf_list)

        # Different values for the height
        h = np.linspace(0, h0, 150)
        h_cont = []

        # Important points for the profile
        xr = (hr/A)**(3/2) + (B/(A**(3/2)))*hr**3

        # Lines for the profile
        x, Xx, X, xX  = [], [], [], []

        for hs in h: # For each vertical point
            if hs < hr:
                x_max = 0
                xapp = (hs/A)**(3/2) + (B/(A**(3/2)))*hs**3
                x.append(xapp)
                x_max = max(xapp, x_max)
                if hs>(hr-1.5):
                    Xxapp = (hs/C)**(3/2) + (D/(C**(3/2)))*hs**3
                    Xx.append(Xxapp)
                    h_cont.append(hs)
            else:
                Xapp = (hs/C)**(3/2) + (D/(C**(3/2)))*hs**3
                if (hs-hr)<0.1:
                    x_diff = x_max - Xapp
                X.append(Xapp)
                if hs<(hr+1.5):
                    xXapp = (hs/A)**(3/2) + (B/(A**(3/2)))*hs**3
                    xX.append(xXapp)
                    h_cont.append(hs)

        h_cont = np.array(h_cont)
        x_tot  = np.concatenate((np.array(x), np.array(X)+x_diff))
        x_cont = np.concatenate((np.array(Xx)+x_diff, np.array(xX)))

        # Centering the y-axis in the mean tide
        xnew = np.arange(0, x_tot[-1], 1)
        xnew_border = np.arange(x_tot[-1]-x_cont[0], x_cont[-1]-x_cont[-1], 1)
        depth = (h - TR/2)
        border = (-h_cont+TR/2)

        f = interpolate.interp1d(x_tot, depth)
        f1 = interpolate.interp1d(x_cont, border)
        ynew = f(xnew)[::-1]
        ynew_border = f1(xnew_border)[::-1]

        depth = (h - TR/2)[::-1]
        border = (-h_cont+TR/2)[::-1]

        # plot
        ax.plot(xnew, -ynew, color='k', zorder=3)
        ax.fill_between(xnew, np.zeros((len(xnew)))+(-ynew[0]), -ynew,facecolor="wheat", alpha=2, zorder=2)
        ax.scatter(x_tot[-1]-xr, -hr+TR/2, s=30, c='red', label='Discontinuity point', zorder=5)
        ax.fill_between(xnew, -ynew, np.zeros(len(xnew)), facecolor="deepskyblue", alpha=0.5, zorder=1)
        ax.axhline(-ha+TR/2, color='grey', ls='-.', label='Available region')
        ax.axhline(TR/2, color='silver', ls='--', label='HT')
        ax.axhline(0, color='lightgrey', ls='--', label='MSL')
        ax.axhline(-TR/2, color='silver', ls='--', label='LT')
        ax.scatter(xnew_border, -ynew_border, c='k', s=1, marker='_', zorder=4)

        # attrbs
        ax.set_ylim(-ynew[0], -ynew[-1]+1)
        ax.set_xlim(0, x_tot[-1])
        set_title  = '$\Omega_{sf}$ = ' + str(omega_surf_list)
        set_title += ', TR = ' + str(TR)
        ax.set_title(set_title)
        ax.legend(loc='upper left')
        ax.set_ylabel('$Depth$ $[m]$', fontweight='bold')
        ax.set_xlabel('$X$ $[m]$', fontweight='bold')

        return(ynew)

    def custom_profile(self, emsl, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6):

        # set bathymetry FILE spacing [m]
        dxinp = self.proj.dxinp

        # flume length
        X = [0, x2, x3, x4, x5, x6]
        Y = [y1, y2, y3, y4, y5, y6]

        xnew = np.arange(0, x6, dxinp)
        f = interpolate.interp1d(X, Y)
        ynew = f(xnew)
        self.proj.depth = -ynew

        ax = self.plots.plot_depthfile()
        ax.scatter(np.array(X), -np.array(Y), c='r', s=20)

        # Add building parameters for validation
        # return(depth)

    def make_Jonswap(self, p_waves, waves):
        'Surface elevation from PSD df = 1/tendc'

        # basci import
        tendc = self.proj.tendc
        deltat = self.proj.deltat

        warmup = waves.warmup

        S = self.energy_spectrum(waves, warmup)

        freqs = np.linspace(0.02,1,tendc + int(warmup))
        delta_f = freqs[1] - freqs[0]

        Time = np.arange(0, tendc+int(warmup), deltat)
        El = np.zeros((len(Time)))

        # calculate aij
        for f in range(len(freqs)):
            ai = np.sqrt(S[f] * 2 * delta_f )
            eps = np.random.rand() * (2*np.pi)

            # calculate elevation
            El = El + ai * np.cos(2*np.pi*freqs[f] * Time + eps)

        # surface elevation series from PDS
        waves = pd.DataFrame({})
        waves['time'] = Time
        waves['value'] = El

        # export and save
        p_file = op.join(p_waves, 'waves.bnd')
        np.savetxt(p_file, np.array(waves), fmt='%f %e')

        return El

    def make_regular(self, p_waves, ws, forcing):
        'Bichromatic waves'
        'Generate and export sea level for boundary conditions'

        # depth = self.proj.depth
        tendc = self.proj.tendc
        deltat = self.proj.deltat

        warmup = ws['warmup']
        T = ws['T']
        H = ws['H']

        time = np.arange(0, tendc + int(warmup), deltat)

        if forcing == 'bi':
            T2 = ws['T2']
            teta = (H/2) * np.cos((2*np.pi/T)*time) + (H/2) * np.cos((2*np.pi/T2)*time)

        else:
            teta = (H/2) * np.cos((2*np.pi/T)*time)

        biwave = np.zeros((len(time), 2))
        biwave[:,0] = time
        biwave[:,1] = teta

        # Export and save
        p_file = op.join(p_waves, 'waves.bnd')
        np.savetxt(p_file, teta, fmt='%e')

        return teta

    def make_friction(self, p_friction, p_depth, ws):

        depth = self.proj.depth
        cf_ini = self.proj.cf_ini
        cf_fin = self.proj.cf_fin
        cf = self.proj.friction

        mx = len(depth)

        f = pd.DataFrame({})
        f1 = np.zeros((int(cf_ini))) + 0
        f2 = np.zeros((int(cf_fin-cf_ini))) + cf
        f3 = np.zeros((int(mx-cf_fin))) + 0

        f['value'] = [i for i in np.concatenate((f1, f2, f3), axis=0)]
        fr = np.array(f)

        # Export and save
        np.savetxt(p_friction, fr, fmt='%e')

    def make_vegetation(self, p_vegetation, p_depth, ws):

        depth = self.proj.depth
        np_ini = self.proj.np_ini
        np_fin = self.proj.np_fin
        nstems = self.proj.nstems

        mx = len(depth)

        f = pd.DataFrame({})
        f1 = np.zeros((int(np_ini))) + 0
        f2 = np.zeros((int(np_fin-np_ini))) + nstems
        f3 = np.zeros((int(mx-np_fin))) + 0

        f['value'] = [i for i in np.concatenate((f1, f2, f3), axis=0)]
        vg = np.array(f)

        # Export and save
        np.savetxt(p_vegetation, vg, fmt='%e')

    def GetTime(self, s):
        'Convert seconds time to hours, minutes and seconds'

        sec = timedelta(seconds = s)
        d = datetime(1,1,1) + sec
        horas = d.hour + (d.day-1)*24
        return (horas, d.minute, d.second)

    def cal_step(self, ws):
        'Calculate time step and computational grid spacing'
        depth = self.proj.depth
        dxL = self.proj.dxL
        deep = np.abs(depth[0])
        tp = ws['T']

        # Assuming there is always 1m of setup due to (IG, VLF)
        Ls,ks,cs = self.DispersionLonda(tp, 1)
        L,k,c = self.DispersionLonda(tp, deep)
        dx = Ls / dxL
        step = 0.5 * dx / (np.sqrt(9.806*deep) + np.abs(c))

        return(step, dx)

    def make_input(self, p_file, id_run, ws):
        '''
        Writes input.sws file from waves sea state for computation

        p_file  - input.sws file path
        ws      - wave sea state (hs, tp, dr, spr)
        bnd     - wave sea state active boundaries

        more info: http://swash.sourceforge.net/download/zip/swashuse.pdf
        '''
        step, dx = self.cal_step(ws)

        depth = self.proj.depth

        # Set Gate discharge as maximun bathymetry elevation
        Gate_Q = np.argmin(depth, axis=None, out=None)

        # .sws file parameters
        WL = ws['WL']                                 # water level
        warmup = ws['warmup']                         # spinup time 
        vert = self.proj.vert                         # multilayered mode
        Nonhydrostatic = self.proj.Nonhydrostatic     # non Hydrostatic pressure

        # friction
        Cf = self.proj.Cf                             # friction coefficient
        friction = self.proj.friction                 # bool var to activate friction
        friction_file = self.proj.friction_file       # bool friction coefficient defined by file

        # vegetation
        vegetation = self.proj.vegetation             # bool var to activate vegetation 
        vegetation_file = self.proj.vegetation_file   # bool vegetation density defined by file
        height = self.proj.height                     # plant height  
        diamtr = self.proj.diamtr                     # plant diameter   
        nstems = self.proj.nstems                     # nº plants/m2
        drag = self.proj.drag                         # drag coefficient

        # wind
        wind = self.proj.wind                         # bool var to activate wind
        Vel = self.proj.Vel                           # wind velocity
        Wdir = self.proj.Wdir                         # wind direction
        Ca = self.proj.Ca                             # wind drag

        tbegtbl = 0                                   # initial time in fields output
        delttbl = 1                                   # time interval between fields
        delttbl_ = 'SEC'                              # time units
        tendc = self.proj.tendc + warmup              # final time

        # bathymetry grid 
        xpinp = 0                                     # x origin
        ypinp = 0                                     # y origin
        alpinp = 0                                    # orientation
        mxinp = len(depth)-1                          # bathymetry spacing in x-dir
        myinp = 0                                     # bathymetry spacing in y-dir
        xlenc = mxinp                                 # length bathymetry x-dir

        # computational grid
        xpc = 0                                       # x origin
        ypc = 0                                       # y origin
        alpc = 0                                      # orientation grid x-dir
        ylenc = 0                                     # length grid y-dir
        myc = 0                                       # nº meshes in y-dir
        mxc = int(mxinp/dx)                           # nº meshes in x-dir
        dxinp = 1                                     # mesh size in x-dir
        dyinp = 1                                     # mesh size in y-dir


        Hr, Mn, Sc = self.GetTime(tendc)

        # .sws text file
        t = "$Project name\n"
        t += "PROJECT '{0}' '{1}'\n$\n".format(self.proj.name, id_run)

        t += "$Set water level\n"
        t += 'SET LEVEL={0}\n$\n'.format(WL)

        # MODE: requests a 1D-mode / 2D-mode of SWASH
        t += "$(1D-mode, flume) or (2D-mode, basin)\n"
        t += 'MODE DYNanic ONED\n'

        # to choose between Cartesian and spherical coordinates
        t += 'COORD CARTesian\n'

        # computational grid
        t += "$Computational grid: geographic location, size, resolution and orientation\n"
        t += 'CGRID {0} {1} {2} {3} {4} {5} {6}\n$\n'.format(
            xpc, ypc, alpc, xlenc, ylenc, mxc, myc)

        t += "$Multi-layered mode\n"
        t += 'VERT {0}\n$\n'.format(vert)

        # bathymetry
        t += "$Reading bathymetry values from file\n"
        t += 'INPGRID BOTTOM {0} {1} {2} {3} {4} {5:.2f} {6}\n'.format(
            xpinp, ypinp, alpinp, mxinp, myinp, dxinp, dyinp)
        t += "READINP BOTTOM 1 '{0}' 1 0 FREE\n$\n".format('depth.bot')

        # friction
        if friction_file:
            t += "$Reading friction values from file\n"
            t += 'INPGRID NPLANTS {0} {1} {2} {3} {4} {5} {6}\n'.format(
                xpinp, ypinp, alpinp, mxinp, myinp, dxinp, dyinp)
            t += "READINP NPLANTS 1 '{0}' 1 0 FREE\n$\n".format('plants.txt')
            t += "FRIC MANNING\n"

        # vegetation file
        if vegetation_file:
            t += "$Reading friction values from file\n"
            t += 'INPGRID FRICTION {0} {1} {2} {3} {4} {5} {6}\n'.format(
                xpinp, ypinp, alpinp, mxinp, myinp, dxinp, dyinp)
            t += "READINP FRICTION 1 '{0}' 1 0 FREE\n$\n".format('friction.txt')
            t += "FRIC MANNING\n"

        t += "$Initial values for flow variables\n"
        t += 'INIT ZERO\n'
        t += "$\n"

        # waves 
        t += "$Hydraulic boundary conditions\n"
        t += "BOU SIDE W CCW BTYPE WEAK CON SERIES 'waves.bnd'\n"
        t += 'BOU SIDE E CCW BTYPE RADIATION \n'

        t += "$\n"

        # numerics
        t += "$Physics\n"
        t += 'BREAK\n'

        if friction == False:
            t += "FRIC MANNING {0}\n".format(Cf)
        if vegetation == False:
            t += "VEGETATION {0} {1} {2} {3}\n".format(height, diamtr, nstems, drag)

        if wind:
            t += "WIND {0} {1}\n$\n".format(Vel, Wdir, Ca)

        if Nonhydrostatic:
            t += "$Numerics\n"
            t += 'NONHYDrostatic\n$\n'

        t += "$Output quantities\n"
        t += 'DISCRET UPW MOM\n'
        t += 'DISCRET UPW UMOM H NONE\n'
        t += 'DISCRET UPW WMOM H NONE\n$\n'

        t += "$Time integration\n"
        t += 'TIMEI 0.1 0.5\n$\n'

        t += 'QUANTITY XP hexp=10\n'
        t += 'QUANT RUNUP delrp 0.01\n$\n'

        # output
        t += "CURVE 'line' {0} {1} {2} {3} {4}\n".format(xpc, ypc, xlenc, xpc + xlenc, ypc + ylenc)

        # general output
        t += "TABLE 'line' HEAD 'output.tab' TSEC XP YP BOTL WATL QMAG OUTPUT {0} {1} {2}\n$\n".format(tbegtbl, delttbl, delttbl_)

        # runup
        t += "TABLE 'NOGRID' HEAD 'run.tab' TSEC RUNUP OUTPUT {0} {1} {2}\n$\n".format(tbegtbl, delttbl, delttbl_)

        # compute
        t += "$Starts computation\n"
        t += 'TEST  1,0\n'
        t += 'COMPUTE 000000.000 {0:.2f} SEC {1}{2}{3}.000\n'.format(step, str(Hr).zfill(2), str(Mn).zfill(2), str(Sc).zfill(2))
        t += 'STOP\n$\n'

        # write file:
        with open(p_file, 'w') as f:
            f.write(t)

        # add waves info
        ws['Gate_Q'] = Gate_Q
        ws['step'] = step
        ws['dx'] = dx

        return(ws)

    def read_file(self, p_file):
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


    def output_points(self, p_case):
        'read table_outpts.tab output file and returns xarray.Dataset'

        p_dat = op.join(p_case, 'output.tab')
        p_run = op.join(p_case, 'run.tab')

        ds1 = self.read_file(p_dat)
        ds2 = self.read_file(p_run)

        ds1 = ds1.set_index(['Xp', 'Yp','Tsec']) #, coords = Time, Xp, Yp
        ds1 = ds1.to_xarray()

        ds2 = ds2.set_index(['Tsec']) #, coords = Time, Xp, Yp
        ds2 = ds2.to_xarray()

        ds = xr.merge([ds1, ds2], compat='no_conflicts')

        return(ds)

    def output_quant(self, p_case):
        'read table_outpts.tab output file and returns xarray.Dataset'

        p_dat = op.join(p_case, 'setup.tab')

        # Read head colums (variables names)
        f = open(p_dat,"r")
        lineas = f.readlines()
        names = lineas[4].split()
        names = names[1:] # Eliminate '%'
        f.close()

        # Read data rows
        data = np.loadtxt(p_dat, skiprows=7)

        # l_xds_pts = []
        df = pd.DataFrame({})
        for p, t in enumerate(names):
            df[t] = data[:,p]

        # Calculate and save runup
        df = df.set_index(['Xp', 'Yp','Tsec']) #, coords = Time, Xp, Yp
        df = df.to_xarray()

        return(df)

    def cal_setup(self, ws, xds_table):
        'Calculate mean set-up'

        wp = ws.warmup.values

        xds_table = xds_table.squeeze()
        Tsec = xds_table.Tsec.values

        setup_m = []

        for i, j in enumerate(xds_table.Xp.values):
            sx = xds_table.isel(Xp=i).Watlev.values
            sx[np.where(sx == -99.)] = np.nan
            sx = sx[np.where(Tsec > wp)]
            set_d = np.nanmean(sx)

            setup_m.append(np.nanmean(set_d))

        # Mean setup
        ds = xr.Dataset({'Setup': ('Xp', setup_m)}, {'Xp': xds_table.Xp.values})

        return(ds)

    def cal_HT(self, ws, xds_table):
        'Calculate df_Hi dataframe from FFt transformation'

        delttbl = self.proj.delttbl
        wp = ws.warmup.values

        # Time values
        Tsec = xds_table.Tsec.values
        pt = np.where((Tsec > wp))[0]
        xds_table = xds_table.squeeze()

        Hm, Hr = [], []
        env_ma, env_mi = [], []
        x = []
        Hs, Hic, Hig, Hvlf = [], [], [], []

        # Save Hi for each X value
        ds_fft_hi = xr.Dataset({})

        # For each sensor, extract Watlev and supress warmp up 
        for i, j in enumerate(xds_table.Xp.values):

            sw = xds_table.isel(Xp=i, Tsec=pt).Watlev.values
            sw = sw[np.where(sw != -99.)]

            if len(sw)>0:
                f, E = sg.welch(sw, fs = 1/delttbl , nfft = 512, scaling='density')
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

                Ti, Hi = self.upcrossing(pt, sw)
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

                ds_fft_hi = xr.auto_combine([ds_fft_hi, ds_Hi], concat_dim='Xp')

        df = pd.DataFrame({'Xi':x, 'Hmax':Hm, 'Hrms':Hr, 'Hs':Hs, 'HIC':Hic,
                           'HIG':Hig, 'HVLF':Hvlf, 'E_max':env_ma,'E_min':env_mi})

        df['E_min'][df['E_min']==-99.0] = np.nan
        df['E_max'][df['E_max']==-99.0] = np.nan
        df['Hs'][df['Hs'] > 30] = np.nan
        df['HIC'][df['HIC'] > 30] = np.nan
        df['HIG'][df['HIG'] > 30] = np.nan
        df['HVLF'][df['HVLF'] > 30] = np.nan

        return(df, ds_fft_hi)

    def upcrossing(self, time, watlev):
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

