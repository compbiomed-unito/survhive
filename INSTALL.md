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


You will find some instruction [here](conda-reqs/README.md)


## Legacy install mode

If conda or virtualenv do not work for you, you can try using the deprecated requirement-file method:

* `pip install -r requirements.txt`
* `pip install --no-deps (path-to-repo)` For development, you can also add --editableassistant





