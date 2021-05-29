"""
Module attrs
"""

__version__     = '0.1.0'
__author__      = 'GeoOcean(UC)'
__contact__     = 'ripolln@unican.es'
__url__         = 'https://gitlab.com/geoocean/bluemath/numerical-models-wrappers/wrap_swash'
__description__ = 'SWASH numerical model python wrap'
__keywords__    = 'SWASH waves numerical'

import os.path as op
import shutil

from . import io
from . import wrap
from . import plots
from . import postprocessor
from . import profiles
from . import waves


def set_swash_binary_file(bin_file):
    'copy swash bin_file to wrap bin location'

    # swan bin executable
    p_res = op.join(op.dirname(op.realpath(__file__)), 'resources')
    p_bin = op.abspath(op.join(p_res, 'swash_bin', 'swash_ser.exe'))

    # copy file
    shutil.copyfile(bin_file, p_bin)

    print('bin file copied to {0}'.format(p_bin))

