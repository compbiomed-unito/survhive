# Installazione

L'installazione dell' env richiede conda >= 23.1.0.

## installazione di conda

```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -p path_di_installazione
source path_di_installazione/bin/activate
```

## creazione environment

```
conda env create -n my_env_name -f environment.yml
```

