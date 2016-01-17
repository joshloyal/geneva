#!/usr/bin/env python
import os
import sys

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy as np

pkg_include_dirs = [
    '/usr/include',
    '/usr/local/Cellar/openblas/0.2.15/include/',
    '/System/Library/Frameworks/vecLib.framework/Versions/A/Headers',
    np.get_include(),
    '.',
]

def strip_start(path):
    return path.split('/', 1)[1]

def scan_dir(dir, files=[]):
    for file in os.listdir(dir):
        path = os.path.join(dir, file)
        if os.path.isfile(path) and path.endswith(".pyx"):
            path = strip_start(path)
            files.append(path.replace(os.path.sep, ".")[:-4])
        elif os.path.isdir(path):
            scan_dir(path, files)
    return files

def make_extension(ext_name):
    ext_path = ext_name.replace(".", os.path.sep) + '.pyx'
    return Extension(
        ext_name,
        [ext_path],
        include_dirs=pkg_include_dirs,
        extra_compile_args=['-O3', '-Wall', '-fopenmp'],
        extra_link_args=['-Wl'],
        libraries=['blas'],
        library_dirs=['/usr/lib']
    )

ext_names =  scan_dir('.')
extensions = [make_extension(name) for name in ext_names]

setup(
    name='geneva',
    ext_modules=extensions,
    cmdclass={'build_ext': build_ext},
)
