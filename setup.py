from distutils.core import setup
from Cython.Build import cythonize

setup(
    name='extUnionFind',
    version='1.0',
    description='Fast disjoint set forest data structure, with extended features (add clusters)',
    author='Alberto Bailoni',
    author_email='alberto.bailoni@iwr.uni-heidelberg.de',
    ext_modules = cythonize("utils/extUnionFind.pyx")
)