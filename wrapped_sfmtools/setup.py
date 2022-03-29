from distutils.core import setup, Extension
from Cython.Build import cythonize

ext = Extension('PySfmWrapper',
                sources=['PySfmWrapper.pyx', './sfmtools/src/triangulation.cc', './sfmtools/src/misc.cc'],
                include_dirs=['./sfmtools/include'],
                extra_compile_args=['-std=c++11']
                )

setup(
    name = "PySfmWrapper",
    ext_modules = cythonize(ext),
)

#python3.5 setup.py build_ext --inplace
