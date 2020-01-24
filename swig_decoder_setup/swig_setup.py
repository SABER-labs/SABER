"""Script to build and install decoder package."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# from setuptools import setup, Extension, distutils
from distutils.core import setup
from distutils.extension import Extension
import glob
import platform
import os, sys
import multiprocessing.pool
import argparse

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--num_processes",
    default=1,
    type=int,
    help="Number of cpu processes to build package. (default: %(default)d)")
args = parser.parse_known_args()

# reconstruct sys.argv to pass to setup below
sys.argv = [sys.argv[0]] + args[1]

def compile_test(header, library):
    dummy_path = os.path.join(os.path.dirname(__file__), "dummy")
    command = "bash -c \"g++ -include " + header \
                + " -l" + library + " -x c++ - <<<'int main() {}' -o " \
                + dummy_path + " >/dev/null 2>/dev/null && rm " \
                + dummy_path + " 2>/dev/null\""
    return os.system(command) == 0

# FILES += glob.glob('sentencepiece/src/*.cc')

FILES = glob.glob('kenlm/util/*.cc') \
        + glob.glob('kenlm/lm/*.cc') \
        + glob.glob('kenlm/util/double-conversion/*.cc')

FILES += glob.glob('openfst-1.6.3/src/lib/*.cc')

FILES = [
    fn for fn in FILES
    if not (fn.endswith('main.cc') or fn.endswith('test.cc') or fn.endswith(
        'unittest.cc'))
]

LIBS = ['stdc++', 'sentencepiece', 'protobuf-lite', 'sentencepiece_train']
ARGS = ['-g', '-O0', '-DNDEBUG', '-DKENLM_MAX_ORDER=6', '-std=c++11', '-D_GLIBCXX_USE_CXX11_ABI=0']

if platform.system() != 'Darwin':
    LIBS.append('rt')

if platform.system() == 'Darwin':
    ARGS.append('-stdlib=libc++')
    ARGS.append("-mmacosx-version-min=10.9")

if compile_test('zlib.h', 'z'):
    ARGS.append('-DHAVE_ZLIB')
    LIBS.append('z')

if compile_test('bzlib.h', 'bz2'):
    ARGS.append('-DHAVE_BZLIB')
    LIBS.append('bz2')

if compile_test('lzma.h', 'lzma'):
    ARGS.append('-DHAVE_XZLIB')
    LIBS.append('lzma')

os.system('swig -py3 -python -c++ ./decoders.i')

decoders_module = [
    Extension(
        name='_swig_decoders',
        sources=FILES + glob.glob('*.cxx') + glob.glob('*.cpp'),
        language='c++',
        include_dirs=[
            '.',
            'kenlm',
            'openfst-1.6.3/src/include',
            'ThreadPool'
            'sentencepiece/src',
            # 'sentencepiece',
            # '/tts_data/build/sentencepiece/build/src/',
            # '/tts_data/build/sentencepiece/build'
        ],
        cython_directives={'language_level': "3"},
        libraries=LIBS,
        extra_compile_args=ARGS)
]

setup(
    name='swig_decoders',
    version='1.1',
    description="""CTC decoders""",
    ext_modules=decoders_module,
    py_modules=['swig_decoders'], )