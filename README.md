# Setup

PBL needs python 3.6 because opengm needs python 3.6.
I am not aware if any other newer or older version also works (maybe 3.10).

## Windows
```
pip install https://github.com/b52/opengm/releases/download/v2.5/opengm-2.5-cp36-cp36m-win_amd64.whl
```

## Linux

```
pip install https://github.com/b52/opengm/releases/download/v2.5/opengm-2.5-py3-none-manylinux1_x86_64.whl
```

## Both
```
pip install -e rfl_install
streamlit run streamlit_app.py
```