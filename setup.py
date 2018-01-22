from distutils.core import setup
from Cython.Build import cythonize

setup(
    name='Feature matching app',
    ext_modules=cythonize("feature_matcher.pyx"),
)
