from __future__ import annotations
from dataclasses import dataclass, field
from collections.abc import Sequence  # abc: Abstract Base Class
import math

class BaseParameter:
    transform: object = None #FIXME a scalar function really

    transforms = {
        'exp10': (lambda x: 10**x, math.log10),
        'exp2': (lambda x: 2**x, math.log2),
        'exp':  (math.exp, math.log),
        'sigmoid': (
            lambda x: 1 / (1 + math.exp(-x)),
            lambda y: -math.log(1 / y - 1),
        ),
        None: (lambda x: x, lambda y: y),
    }

    def apply_transform(self, x):
        if self.transform in self.transforms:
            return self.transforms[self.transform][0](x)
        else:
            return self.transform(x)


    def get_values(self):
        # generate values according to distribution type and parameters
        values = self.get_values_()

        # apply transform
        if self.transform:
            values = map(self.transform, values)

        return list(values)

    def get_value_optuna(self, trial, name):
        value = self.get_value_optuna_(trial, name)
        # apply transform
        if self.transform:
            value = self.transform(value)

        return value

@dataclass
class IntegerParameter(BaseParameter):
    min: int
    max: int
    transform: object = None #FIXME a scalar function really
    
    def get_values_(self):
        return range(self.min, self.max + 1)
    def get_value_optuna_(self, trial, name):
        return trial.suggest_int(name, self.min, self.max)

@dataclass
class FloatParameter(BaseParameter):
    min: float
    max: float
    num_steps: int
    transform: object = None #FIXME a scalar function really

    def get_values_(self):
            return (self.min + i * (self.max - self.min) / (self.num_steps - 1) for i in range(self.num_steps))
    def get_value_optuna_(self, trial, name):
            return trial.suggest_float(name, self.min, self.max)

@dataclass
class CategoricalParameter(BaseParameter):
    values: list
    transform: object = None #FIXME a scalar function really
    
    def get_values_(self):
        return self.values
    def get_value_optuna_(self, trial, name):
        return trial.suggest_categorical(name, self.values)

@dataclass
class NNShapeParameter(BaseParameter):
    min_hidden_layers: int = 0 
    max_hidden_layers: int = 4
    max_log2_size: int = 8
    shape: str = 'constant'

    def get_values_(self):
        assert self.shape == 'constant'
        return [
            [2**log2_size]*hidden_layers
            for hidden_layers in range(self.min_hidden_layers, self.max_hidden_layers + 1)
            for log2_size in (range(1, self.max_log2_size + 1) if hidden_layers > 0 else [0])
        ]
        
    def get_values_0(self): # generic version with multiple shapes, only constant implemented yet :-)
        acc = []
        for hidden_layers in range(self.min_hidden_layers, self.max_hidden_layers + 1):
            if hidden_layers == 0:
                acc.append([])
                continue
            for log2_size in range(1, self.max_log2_size + 1):
                if self.shape == 'constant':
                    acc.append([2**log2_size]*hidden_layers)
                #elif self.shape == 'decreasing':
                else:
                    raise ValueError(f'Unknown shape value `{self.shape}`')
        
    def get_value_optuna_(self, trial, name):
        hidden_layers = trial.suggest_int(name + '_hidden_layers', self.min_hidden_layers, self.max_hidden_layers)
        if hidden_layers > 0:
            log2_size = trial.suggest_int(name + '_log2_layer_size', 1, self.max_log2_size)
            return [2**log2_size]*hidden_layers
        else:
            return []


# funnel shape
def generate_decreasing_layer_sizes_optuna(trial, min_hidden_layers=0, max_hidden_layers=4, max_log2_size=8, prefix='', decreasing=True):
    layers = []
    for i in range(trial.suggest_int(f'{prefix}n_hidden', min_hidden_layers, max_hidden_layers)):
        size = trial.suggest_int(f'{prefix}log2_layer_{i}', 0, max_log2_size)
        layers.append(2**size)
        if decreasing:
            max_log2_size = size
    return layers

def get_values(parameter_dict):
    return {name: param.get_values() for name, param in parameter_dict.items()}

def get_value_optuna(parameter_dict, trial):
    return {name: param.get_value_optuna(trial, name) for name, param in parameter_dict.items()}


try:
    import optuna
    optuna_available = True
except ImportError:
    optuna_available = False
    #print('Optuna not available')



def make_all_params():
    if True:
        Cat = CategoricalParameter
        Int = IntegerParameter
        Num = FloatParameter
        NN = NNShapeParameter
    else:
        Cat = lambda **args: Parameter('categorical', **args)
        Int = lambda **args: Parameter('integer', **args)
        Num = lambda **args: Parameter('float', **args)
    import math
    exp10 = lambda x: 10**x
    exp2 = lambda x: 2**x
    log10 = math.log10
    log2 = math.log2

    #y = x/(1 + abs(x))
    #y = 0.5 + 0.5*x/math.sqrt(1 + x**2)
    sigmoid = lambda x: round(1/(1 + math.exp(-x)), 2)
    sigmoid = lambda x: 1 / (1 + math.exp(-x))
    sigmoid_inv = lambda y: -math.log(1 / y - 1)
    
    return dict(
        CoxNet=dict(
            #l1_ratio=P('categorical', values=[0.01, 0.1, 0.2, 0.5, 0.8, 0.9, 0.99]), # FIXME the sigmoid function in the next row gives sligtly different values from these
            l1_ratio=Num(min=sigmoid_inv(0.01), max=sigmoid_inv(0.99), num_steps=7, transform=sigmoid)
        ),
        DeepSurvivalMachines = dict(
            n_distr         = Int(min=1, max=6),
            distr_kind      = Cat(values=['LogNormal', 'Weibull']),
            layer_sizes     = NN(max_hidden_layers=4, max_log2_size=8, shape='constant'),
            learning_rate   = Num(min=log10(1e-5), max=log10(1e-1), num_steps=11, transform=exp10),
            validation_size = Num(min=log10(0.05), max=log10(0.5), num_steps=5, transform=exp10),
            batch_size      = Int(min=2, max=10, transform=exp2),
            elbo            = Cat(values=[True, False]),
            #optimizer       = Cat(values=['Adam', 'RMSProp', 'SGD']),
            max_epochs      = Int(min=3, max=10, transform=exp2),
        )
    )

param_grids = make_all_params()

# compute grid size
def check_size(param_dict):
    s = 1
    for values in param_dict.values():
        s *= len(values)
    return s

import numpy
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
def optimize(estimator, X, y, mode='sklearn-grid', n_jobs=None, cv=None, tries=None):
    #groups = None
    #cv = StratifiedKFold(n_splits=5)
    #if groups is None: # stratify by events
    #    groups = get_indicator(y)
    params = param_grids[estimator.__class__.__name__]
    print(params)
    print(get_values(params))
    print(check_size(get_values(params)))
    if mode.startswith('sklearn-'):
        args = (estimator, get_values(params))
        kwargs = dict(n_jobs=n_jobs, refit=True, cv=cv)
        if mode == 'sklearn-grid':
            gs = GridSearchCV(*args, **kwargs)
        elif mode == 'sklearn-random':
            gs = RandomizedSearchCV(*args, n_iter=tries, **kwargs)
        else:
            raise ValueError(f'unknown mode parameter: "{mode}"')
        gs.fit(X, y)
        return gs.best_estimator_, gs.best_params_, gs
    elif mode == 'optuna':
        #path = f'sqlite:///prova.optuna.sqlite3'
        #assert n_jobs is None, 'not implemented'
        path = None
        seed = 0

        study = optuna.create_study(
            direction='maximize', sampler=optuna.samplers.RandomSampler(seed=seed), study_name='prova', load_if_exists=True,
            storage=path)
        def objective_function(trial):
            x = get_value_optuna(params, trial)
            print('trial params:', x)
            estimator.set_params(**x)
            
            ss = cross_val_score(estimator, X, y, cv=cv)
            s = numpy.mean(ss) - numpy.std(ss)/numpy.sqrt(len(ss))
            print('score', s, ss)

            return s

        study.optimize(objective_function, n_trials=tries, n_jobs=1 if n_jobs is None else n_jobs)
        # qui manca ancora la selezione del modello migliore e un return!
        return study
    else:
        raise ValueError(f'unknown mode parameter: "{mode}"')


if __name__ == '__main__':
    tests = {
        'integer 3-7': IntegerParameter(min=3, max=7),
        'float 3-7 9 steps': FloatParameter(min=3, max=7, num_steps=9),
        'cats 3 values': CategoricalParameter(values=['A', 'B', 'C']),
    }

    for name, param in tests.items():
        print(name, param.get_values())
    
    for model, params in param_grids.items():
        print('Model', model)
        for name, param in params.items():
            print(name, param.get_values())