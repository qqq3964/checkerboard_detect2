from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
import glob
import subprocess

__version__ = "0.1.0"

opencv_cflags = subprocess.getoutput("pkg-config --cflags opencv4").split()
opencv_libs = subprocess.getoutput("pkg-config --libs opencv4").split()

source_files = glob.glob("corner_detect/src/*.cpp")

ext_modules = [
    Pybind11Extension(
        "tcar",
        sources=source_files,
        include_dirs=["corner_detect/src"],
        define_macros=[("VERSION_INFO", __version__)],
        extra_compile_args=opencv_cflags,
        extra_link_args=opencv_libs,
        cxx_std=14,
    ),
]

setup(
    name="tcar",
    version=__version__,
    author="Taewan Kim",
    description="Chessboard detection with pybind11",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.7",
    options={
        'build': {
            'build_base': 'corner_detect/build'
        },
        'egg_info': {
            'egg_base': 'corner_detect' 
        }
    },
)
