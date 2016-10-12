from distutils.core import setup
from Cython.Build import cythonize
import numpy

# use command 'python setup.py build_ext --inplace' in file location path to compile ga_functions.pyx

setup(ext_modules=cythonize("ga_functions.pyx"), include_dirs=[numpy.get_include()])
