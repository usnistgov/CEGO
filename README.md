# CEGO

CEGO (pronounced sea-go) is C++11 Evolutionary Global Optimization.  It allows for:

* A flexible C++11 architecture for doing parallel global optimization with multithreading
    * Also allows for new evolutionary optimization techniques to be specified with a minimum of code
    * Uses the age-layered approach
* A C++ datatype (``CEGO::numberish``) that can be either an integer or a floating double precision value
* Python wrappers of the core of the library

## Examples:

Try it in your browser: [![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/usnistgov/CEGO/master)

Statically rendered examples are provided as Jupyter notebooks served on nbviewer ([link to folder](https://nbviewer.jupyter.org/github/usnistgov/CEGO/tree/master/notebooks)), roughly sorted in terms of complexity of the example:

* [Hundred-digit challenge]()
* [Griewangk]() (10-dimensional double precision optimization)
* [Inverse Gaussian bumps]()

## License

*MIT licensed (see LICENSE for specifics), not subject to copyright in the USA.

## Dependencies

* Unmodified [Eigen](https://eigen.tuxfamily.org/dox/) for matrix operations
* Unmodified [nlohmann::json](https://github.com/nlohmann/json) for JSON management
* Unmodified [pybind11](https://github.com/pybind/pybind11) for C++ <-> Python interfacing
* Unmodified [ThreadPool2](https://github.com/stfx/ThreadPool2) for thread pooling

## Contributing/Getting Help

If you would like to contribute to ``CEGO`` or report a problem, please open a pull request or submit an issue.  Especially welcome would be additional tests.

## Installation

### Prerequisites

You will need:

* cmake (on windows, install from cmake, on linux ``sudo apt install cmake`` should do it, on OSX, ``brew install cmake``)
* Python (the anaconda distribution is used by the authors)
* a compiler (on windows, Visual Studio 2015+ (express version is fine), g++ on linux/OSX)

If on linux you use Anaconda and end up with an error something like
```
ImportError: /home/theuser/anaconda3/bin/../lib/libstdc++.so.6: version `GLIBCXX_3.4.20' not found (required by /home/theuser/anaconda3/lib/python3.6/site-packages/CEGO.cpython-35m-x86_64-linux-gnu.so)
```
it can be sometimes fixed by installing ``libgcc`` with conda: ``conda install libgcc``.  [This is due to an issue in Anaconda](https://github.com/ContinuumIO/anaconda-issues/issues/483)

## To install in one line from github (easiest)

This will download the sources into a temporary directory and build and install the python extension so long as you have the necessary prerequisites:
```
pip install git+git://github.com/usnistgov/CEGO.git
```

### From a cloned repository

Alternatively, you can clone (recursively!) and run the ``setup.py`` script

```
git clone --recursive https://github.com/usnistgov/CEGO
cd CEGO
python setup.py install
```

to install, or 

```
python setup.py develop
```

to use a locally-compiled version for testing.  If you want to build a debug version, you can do so with

```
python setup.py build -g develop
```
With a debug build, you can step into the debugger to debug the C++ code, for instance.  

### Cmake build

Starting in the root of the repo (a debug build with the default compiler, here on linux):

``` 
git clone --recursive https://github.com/usnistgov/CEGO
cd CEGO
mkdir build
cd build
cmake ..
cmake --build .
```
For those using Anaconda on Linux, please use the following for cmake:
```
mkdir build
cd build
cmake .. -DPYTHON_EXECUTABLE=`which python`
cmake --build .
```
For Visual Studio 2015 (64-bit) in release mode, you would do:
``` 
git clone --recursive https://github.com/usnistgov/CEGO
cd CEGO
mkdir build
cd build
cmake .. -G "Visual Studio 14 2015 Win64"
cmake --build . --config Release
```

If you need to update your submodules (pybind11 and friends)

```
git submodule update --init
```

For other options, see the cmake docs
