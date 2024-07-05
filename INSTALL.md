# Installation how-to

## Installing using pip

The simplest way to install SurvHive is directly from the Github repository, using the following command:

```
python -m pip install "survhive @ git+ssh://git@github.com/compbiomed-unito/survhive.git"
```

Of course, it's much better to perform the installation inside a [virtual environment](https://docs.python.org/3/library/venv.html).
It is necessary to have Python >= 3.8. If missing, consider installing [pyenv](https://github.com/pyenv/pyenv).

To install the development requirements, use: 

```
python -m pip install "survhive[dev] @ git+https://git@github.com/compbiomed-unito/survhive.git"
```

A Notebook dependency (jupyter lab) is part of the [dev] dependencies only, since it
is not necessary to run the package and because everybody have their own favorite notebook platform.

If you want a notebook environment for execution, "pip install" it after SurvHive installation.
 
## Installing with Conda


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

If you get an error saying "conda do not support solver" it means that your
conda version is rather old. It will work anyway, but it will take a very long
time to solve dependencies. It is probably better to update to a recent conda
version. 

### get conda environment specs 

Download the [conda spec file](https://raw.githubusercontent.com/compbiomed-unito/survhive/main/conda-reqs/hive-env.yml) from the repository and save it.

For example, using curl:

```
curl -o hive-env.yml https://raw.githubusercontent.com/compbiomed-unito/survhive/main/conda-reqs/hive-env.yml
```


### Creating an Environment

To install the requirements for the latest stable version from the main branch, use:

```
conda env create -f (path_to_the downloaded_file)/hive-env.yml
```

This will create an environment named "hive" with all necessary components.


## Legacy install mode

If conda or virtualenv do not work for you, you can try using the deprecated requirement-file method:

* `pip install -r requirements.txt`
* `pip install --no-deps (path-to-repo)` For development, you can also add --editableassistant





