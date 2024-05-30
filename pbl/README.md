#  🖼️ ⇨ 📍📍  pbl - Part-based localizer

[![Build Status](https://travis-ci.com/fhkiel-mlaip/pbl.svg?token=WHxqMPs4z8VYjCAuFKAb&branch=master)](https://travis-ci.com/fhkiel-mlaip/pbl)

A localizer specifically designed to detect and localize spatially correlated
landmarks utilizing a conditional random field with unary potentials in form of
ML-driven independent localizers.

Works in 2D and 3D as well as with different (multi-channel) modalities.

## Installation

### Manual

The installation requires a **Python 3.6 64-bit** installation:

1. Clone the [hiwi], [rfl] and [pbl] repositories

    ```bash
    git clone git@github.com:fhkiel-mlaip/hiwi.git
    git clone git@github.com:fhkiel-mlaip/rfl.git
    git clone git@github.com:fhkiel-mlaip/pbl.git
    ```

2. Install all public requirements:

    ```bash
    pip install -r pbl/requirements.txt

    # for linux
    pip install https://github.com/b52/opengm/releases/download/v2.5/opengm-2.5-py3-none-manylinux1_x86_64.whl
    # for windows
    pip install https://github.com/b52/opengm/releases/download/v2.5/opengm-2.5-cp36-cp36m-win_amd64.whl
    ```

3. Install our packages in **development mode**

    ```bash
    pip install -e hiwi/
    pip install -e rfl/
    pip install -e pbl/
    ```

4. You should now be able to use the `pbl` command to train and evaluate
   models etc.

    ```bash
    # show all commands
    pbl --help

    # show the train command
    pbl train --help
    ```

### Automated (for servers)

In server environments where you quickly want to iterate different versions
and run different configurations side-by-side, the `bootstrap_pbl.sh` script
should be use.

The script requires a passed-through SSH agent (for the public key
authentication with the GitHub git servers).

The script simply needs a directory where to create/update the 
Python environment and an optional commit hash/tag/branch of the `pbl` software
to use (defaults to master). Note that the environment can exist already, in
which case it simply updates the software, instead of doing a clean install,
which is useful for quick changes to pbl.

```
# bootstrap_pbl.sh ENVIRONMENT [COMMIT/TAG/BRANCH]
bootstrap_pbl.sh env/ 7f58b009910454665bf60ee371536d964544f6b5
```

### Standalone distribution

Using [PyInstaller] it is possible to create distribution binary packages
of the package for Windows and Linux. Note that the target machine still
needs a CUDA driver and/or OpenCL driver.

After installing `pbl` as described previously and also installing PyInstaller,
one can use the following command to create a distributable folder `dist/pbl`:

```shell script
pyinstaller \
    --noupx --clean \
    --onedir --name pbl \
    --hidden-import sklearn.neighbors.typedefs \
    --hidden-import sklearn.neighbors.quad_tree \
    --hidden-import sklearn.tree._utils \
    --hidden-import sklearn.tree \
    --hidden-import sklearn.utils._cython_blas \
    --hidden-import scipy._lib.messagestream \
    --exclude-module FixTk \
    --exclude-module tcl \
    --exclude-module tk \
    --exclude-module _tkinter \
    --exclude-module tkinter \
    --exclude-module Tkinter \
    --additional-hooks-dir hooks \
    --add-data `python -c 'import rfl; import os; print(os.path.dirname(rfl.__file__) + "/kernel.cl")'`:rfl \
    `python -c 'import pbl; import os; print(os.path.dirname(pbl.__file__) + "/__main__.py")'`
```

This commands requires there exists a `hooks/` directory containing a file
named `hook-itk.py` with the following contents:

```python
# This hook only works when ITK is pip installed. It
# does not work when using ITK directly from its build tree.

from PyInstaller.utils.hooks import collect_data_files

hiddenimports = ['new']

itk_datas = collect_data_files('itk', include_py_files=True)
datas = [x for x in itk_datas if '__pycache__' not in x[0]]
```

If using this on Windows, you have to adjust the path slahes to `\` and change
`:rfl` to `;rfl` in the `--add-data` parameter.

### Docker

To build everything into a Docker image, run the provided script:

```bash
./build_docker.sh
```

**Note** This requires a `GH_TOKEN` environment variable in order to
download all the private dependencies from GitHub.

Once this is done, you can run the pbl command like this:

```
nvidia-docker run --rm -ti pbl --help

# simpler
alias pbl-docker='nvidia-docker run --rm -ti pbl'
pbl-docker --help
```

## Training a new model

Once the software is installed, you can use the command

```
pbl train ...
```

to create a new model from a set of training images. Run `pbl train --help`
to get an exhaustive list of options to configure the training.

### General recommendations

The following is a list of useful points to keep in mind when training a new
model. This also includes useful options in certain scenarios.

- When training a sparse spatial model, i.e., not fully connected, it might
  make sense to not drop potentials if you have very few binary potentials for
  key point pairs. Dropping all connectivity increases the chance of false
  positive detections. In this case, use `--sgd-min-weight 1.0e-6` to ensure
  that you not drop potentials and maintain the connectivity.

[hiwi]: https://github.com/fhkiel-mlaip/hiwi
[rfl]: https://github.com/fhkiel-mlaip/rfl
[pbl]: https://github.com/fhkiel-mlaip/pbl
[PyInstaller]: https://www.pyinstaller.org/