#!/usr/bin/env python
# -*- coding: utf-8 -*-

# common
import os
import os.path as op
import subprocess as sp

# pip
import numpy as np
import pandas as pd
from scipy import signal as sg

# swash
from .io import SwashIO
from .plots import SwashPlot

class SwashProject(object):
    'SWASH numerical model project parameters. used inside swashwrap and swashio'

    def __init__(self, p_proj, n_proj):
        '''
        SWASH project information will be stored here

        http://swash.sourceforge.net/download/zip/swashuse.pdf
        '''

        self.p_main = op.join(p_proj, n_proj)    # project path
        self.name = n_proj                       # project name

        # sub folders 
        self.p_cases = op.join(self.p_main)  # project cases

        # bathymetry depth value (2D numpy.array)
        self.depth = None
        self.gate = None

        # input friction
        self.friction = None
        self.friction_file = None
        self.Cf = None
        self.cf_ini = None
        self.cf_fin = None

        # vegetation
        self.vegetation = None
        self.vegetation_file = None
        self.height = None
        self.diamtr = None
        self.nstems = None
        self.drag = None
        self.np_ini = None
        self.np_fin = None

        # Wind
        self.Vel = None
        self.Wdir = None
        self.Ca = None

        # input.swn file parameters
        self.input_waves = True
        self.vert = None
        self.par_cdcap = None
        self.par_jonswap_gamma = None
        self.WL = None
        self.vert = None        # vertical layers

        self.Nonhydrostatic = False
        self.wind = False
        self.coords_spherical = False  # True spherical, False cartesian
        sp.specfilename = None

        # computational grid parameters
        self.cc_xpc = None      # x origin
        self.cc_ypc = None      # y origin
        self.cc_alpc = None     # x-axis direction 
        self.cc_xlenc = None    # grid length in x
        self.cc_ylenc = None    # grid length in y
        self.cc_mxc = None      # number mesh x
        self.cc_myc = None      # number mesh y
        self.cc_dxinp = None    # size mesh x
        self.cc_dyinp = None    # size mesh y

        # bathymetry grid parameters
        self.dp_xpc = None      # x origin
        self.dp_ypc = None      # y origin
        self.dp_alpc = None     # x-axis direction 
        self.dp_xlenc = None    # grid length in x
        self.dp_ylenc = None    # grid length in y
        self.dp_mxc = None      # number mesh x
        self.dp_myc = None      # number mesh y
        self.dp_dxinp = None    # size mesh x
        self.dp_dyinp = None    # size mesh y

        self.flume = None
        self.plane = None
        self.Gate_Q = None
        self.dx = None
        self.h0 = None
        self.Xfore = None
        self.Xinner = None
        self.Rc = None
        self.step = None
        self.warmup = None
        self.delttbl = None


class SwashWrap(object):
    'SWASH numerical model wrap for multi-case handling'

    def __init__(self, swash_proj):
        '''
        initializes wrap

        swash_proj - SwanProject() instance, contains project parameters
        '''

        self.proj = swash_proj              # swash project parameters
        self.io = SwashIO(self.proj)        # swash input/output 
        self.plots = SwashPlot(self.proj)   # swash plotting tool

        # resources
        p_res = op.join(op.dirname(op.realpath(__file__)), 'resources')

        # swan bin executable
        self.bin = op.abspath(op.join(p_res, 'swash_bin', 'swash_ser.exe'))

    def build_cases(self, waves_dataset):
        '''
        generates all files needed for swash multi-case execution

        waves_dataset - pandas.dataframe with "n" boundary conditions setup
        '''

        # make main project directory
        self.io.make_project()

        new_waves = pd.DataFrame()

        if waves_dataset['forcing'].values[0] == "Jonswap":
                waves_dataset.rename(columns={"Tp": "T", "Hs": "H"}, inplace=True)

        # one stat case for each wave sea state
        for ix, (_, ws) in enumerate(waves_dataset.iterrows()):

            # build stat case 
            case_id = '{0:04d}'.format(ix)
            waves = self.io.build_case(case_id, ws)
            waves = waves.to_frame()

            new_waves = pd.concat([new_waves, waves], axis=1)

        return(new_waves.transpose())

    def make_reef(self, waves_dataset):

        # make main project directory
        self.io.make_project()

        coral_config = pd.DataFrame()

        # one stat case for each wave sea state
        for ix, (_, ws) in enumerate(waves_dataset.iterrows()):

            # build stat case 
            case_id = '{0:04d}'.format(ix)

            # SWASH case path
            p_case = op.join(self.proj.p_cases, case_id)

            # make execution dir
            if not op.isdir(p_case): os.makedirs(p_case)

            waves_reef, depth = self.io.make_reef(p_case, case_id, ws)

            waves_reef = waves_reef.to_frame()

            coral_config = pd.concat([coral_config, waves_reef], axis=1)

        return(coral_config.transpose(), depth)

    def make_waves_series(self, waves_dataset):
        'Irregular waves series (forcing): monochromatic, bichromatic, Jonswap'

        # make main project directory
        self.io.make_project()

        # one stat case for each wave sea state
        for ix, (_, ws) in enumerate(waves_dataset.iterrows()):

            # build stat case 
            case_id = '{0:04d}'.format(ix)

            # SWASH case path
            p_case = op.join(self.proj.p_cases, case_id)

            # make execution dir
            if not op.isdir(p_case): os.makedirs(p_case)

            if ws['forcing'] == "Bichromatic":
                series = self.io.make_regular(p_case, ws, 'bi')
            elif ws['forcing'] == "Monochromatic":
                series = self.io.make_regular(p_case, ws, 'mono')
            else:
                ws = ws.rename(columns={"Tp": "T", "Hs": "H"})
                series = self.io.make_Jonswap(p_case, ws)

        return(series.transpose())

    def get_run_folders(self):
        'return sorted list of project cases folders'

        ldir = sorted(os.listdir(self.proj.p_cases))
        fp_ldir = [op.join(self.proj.p_cases, c) for c in ldir]

        return [p for p in fp_ldir if op.isdir(p)]

    def run_cases(self):
        'run all cases inside project "cases" folder'

        # TODO: improve log / check execution ending status

        # get sorted execution folders
        run_dirs = self.get_run_folders()
        for p_run in run_dirs:

            # run case
            self.run(p_run)

            # log
            p = op.basename(p_run)
            print('SWASH CASE: {0} SOLVED'.format(p))

    def run(self, p_run):
        'Bash execution commands for launching SWASH'

        # aux. func. for launching bash command
        def bash_cmd(str_cmd, out_file=None, err_file=None):
            'Launch bash command using subprocess library'

            _stdout = None
            _stderr = None

            if out_file:
                _stdout = open(out_file, 'w')
            if err_file:
                _stderr = open(err_file, 'w')

            s = sp.Popen(str_cmd, shell=True, stdout=_stdout, stderr=_stderr)
            s.wait()

            if out_file:
                _stdout.flush()
                _stdout.close()
            if err_file:
                _stderr.flush()
                _stderr.close()

        # ln input file and run swan case
        cmd = 'cd {0} && ln -sf input.sws INPUT && {1} INPUT'.format(
            p_run, self.bin)
        bash_cmd(cmd)

        # windows launch
        #cmd = 'cd {0} && swashrun input && {1} input'.format(
        #    p_run, self.bin)
        #bash_cmd(cmd)



    def fft_wafo(self, df):
        ''

        return(self.io.cal_HT(df))

    def reflection(self, ws, xds_out):
        'calculate reflection coefficient from incident and outgoing energy'

        depth = self.proj.depth
        delttbl = self.proj.delttbl

        flume = int(len(depth)/4)
        hs = np.float(ws['H'].values)

        sw_out = xds_out.isel(Xp = int(flume/2)).Watlev.values
        sw_out = sw_out[np.isnan(sw_out) == False]
        fout, Eout = sg.welch(sw_out, fs = 1/delttbl , nfft = 512, scaling='density')

        m0out = np.trapz(Eout, x=fout)
        Hsout = 4 * np.sqrt(m0out)
        Kr = np.sqrt((Hsout/hs)-1)

        return(Kr)

    def postprocessing(self, waves, t_video):
        '''
        Calculate setup, significant wave heigh and print outputs

        waves   -  DataFrame with waves vars
        t_video -  Duration output video
        '''

        # get sorted execution folders
        run_dirs = self.get_run_folders()

        Gate_Q = waves.Gate_Q
        WL = waves.WL

        ru2, Q = [], []

        # exctract output case by case and concat in list
        for case_id, p_run in enumerate(run_dirs):

            xds_out = self.io.output_points(p_run)   # output.tab
            depth = np.loadtxt(op.join(p_run, 'depth.bot'))

            print("\033[1m" +'\nOutput table\n' + "\033[0m")
            print(xds_out)

            ws = waves.iloc[[case_id]]
            ws['h0'] = np.abs(depth[0])
            warmup = ws.warmup.values

            wp = np.where(xds_out.Tsec.values > warmup)[0]

            # overtopping
            q = xds_out.isel(Tsec=wp, Xp=int(Gate_Q)).Qmag.values
            q[np.where(q == -9999.0)] = np.nan
            q = q[np.isnan(q)==False]
            q = q[np.where(q > 0)]

            ws['q'] = np.nanmean(q)*1000
            Q.append(np.nanmean(q)*1000) # [l/s/m] 

            # reflection coefficient
            ws['kr'] = self.reflection(ws, xds_out)

            # runup
            g = xds_out.isel(Tsec=wp).Runlev.values
            g[np.where(g == -9999.0)] = np.nan
            g = g[np.isnan(g)==False]

            if len(g) > 0:
                ws['ru2'] = np.percentile(g, 98) + WL
                ru2.append(np.percentile(g, 98) + WL)

            else:
                ws['ru2'] = np.nan
                ru2.append(np.nan)

            # statistical and spectral hi
            df_Hi, ds_fft_hi = self.io.cal_HT(ws, xds_out)

            print("\033[1m" + "\nProcessed data: Fft transformation\n" + "\033[0m")
            print(df_Hi)

            # calculate mean setup
            df = self.io.cal_setup(ws, xds_out)

            xds_out = xds_out.squeeze()

            # histograms Ru, Q, Hi
            self.plots.histograms(ws, xds_out, g, q, df, df_Hi, ds_fft_hi, depth, p_run)

            # save results into folder
            self.plots.single_plot_stat(ws, xds_out, df, df_Hi, p_run, depth)
            self.plots.single_plot_nonstat(ws, xds_out, df, df_Hi, p_run, depth, t_video)

            print("\033[1m" + '\nEnd postprocessing case {0}\n'.format(case_id) + "\033[0m")

    def print_wraps(self, waves_dataset):
        'Print "input.sws" files'

        # make main project directory
        self.io.make_project()

        # one stat case for each wave sea state
        for ix, (_, ws) in enumerate(waves_dataset.iterrows()):

            case_id = '{0:04d}'.format(ix)
            self.io.print_wrap(case_id)

    def plot_grid(self, waves_dataset):
        "Plot computational grids"

        # make main project directory
        self.io.make_project()

        # one stat case for each wave sea state
        for ix, (_, ws) in enumerate(waves_dataset.iterrows()):

            self.plots.plot_computational(ws)

