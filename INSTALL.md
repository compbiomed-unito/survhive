# Install

## installing with Conda


The installation of the environment requires Conda >= 23.1.0.
Given the large number of dependencies one should use the libmamba solver.


### Install Conda

Download the Miniconda installer using the command:

```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

Run the installer with the path of installation:

```
bash Miniconda3-latest-Linux-x86_64.sh -p /path/to/installation
```

Activate the environment by sourcing the activate script:

```

source /path/to/installation/bin/activate
```

Configure Conda to use the libmamba solver:

```
conda config --set solver libmamba
```

### Creating an Environment

To install the requirements for the latest stable version from the main branch, use:

```
conda env create -f conda-reqs/hive-env.yml
```

This will create an environment named "hive" with all necessary components.

## Installing using pip**

Of course, it's much better to perform the installation inside a [virtual environment](https://docs.python.org/3/library/venv.html).
It is necessary to have Python >= 3.8. If missing, consider installing [pyenv](https://github.com/pyenv/pyenv).

It is possible to install directly from the repo using:

```
python -m pip install "survhive @ git+ssh://git@github.com/compbiomed-unito/survhive.git"
```

For development: 

```
python -m pip install "survhive @ git+ssh://git@github.com/compbiomed-unito/survhive.git"[dev]
```

## Legacy install mode

If conda or virtualenv do not work for you, you can try using the deprecated requirement-file method:

* `pip install -r requirements.txt`
* `pip install --no-deps (path-to-repo)` For development, you can also add --editableassistant





