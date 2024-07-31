# Conda Requirements


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

### If you use mamba

Just get the conda spec file as explained above and execute:


```
mamba env create -f (path_to_the downloaded_file)/hive-env.yml
```

