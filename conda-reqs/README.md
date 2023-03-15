# Conda Requirements

Una serie di ambienti parziali creati con @conda list --export@ usati per
creare environments intermedi con il solver libmamba un layer alla volta.

Per creare ambienti da questi files usare:

```
conda create -n ilmioenv -c sebp -c conda-forge --file (nomefile.yml)
```

NB non essendo un export *i canali vanno specificate in command line in ordine di priorità*

## inventario

* base-env.yml : mamba git pip  
* sksurv-env.yml: la precedente + jupyterlab, scikit-survival
* survwrap-pycox-env.yml: la precedente + pycox (pytorch come dipendenza)

