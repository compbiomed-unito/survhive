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
conda env create -f conda-reqs/tosa-env.yml
```

che andrà a creare un environment chiamato "tosa" con tutto quello che serve

