# Installazione

L'installazione dell' env richiede conda >= 23.1.0.

## installazione di conda

```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -p path_di_installazione
source path_di_installazione/bin/activate
conda config --set solver libmamba 
```

## creazione environment

Conda accetta requirements tramite il comando

```
conda env create -n my_env_name -f environment.yml
```

Ci sono diversi esempi (parziali) sotto conda-reqs/

Per installare i requisiti per l'ultima versione stabile dalla branch main, usare:

```
conda env create -f conda-reqs/hive-env.yml
```

che andrà a creare un environment chiamato "hive" con tutto quello che serve

## installazione con pip 

Ovviamente è molto meglio fare l'installazione all interno di un [virtual environment](https://docs.python.org/3/library/venv.html)

1.  E' necessario avere python > 3.8. Se manca considerare l'installazione di [pyenv](https://github.com/pyenv/pyenv)
2.  Clonare il repository git
3.  installazione 
    * per utilizzo standard: pip install (path-del-repo)/.
    * per sviluppo: pip install (path-del-repo)/.[dev]
    a.  installazione direttamente dal repo

        ~~~ {.bash}
        python -m pip install "survwrap @ git+ssh://git@github.com/compbiomed-unito/survwrap.git"
        ~~~

Se non funziona si puo' provare "a calci e pugni":
1. pip install -r requirements.txt
2. pip install --no-deps (path-del-repo)
Per sviluppo si puo' aggiungere anche --editable