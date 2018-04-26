from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='dagnn_cpp',
    ext_modules=[
        CppExtension('dagnn_cpp', ['dagnn.cpp']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
