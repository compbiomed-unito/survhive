#!/usr/bin/env python
# coding: utf-8

# # Testing fastCPH

# In[50]:


import numpy


# In[51]:


import sklearn


# In[52]:


import survwrap


# In[53]:


#X, y = survwrap.load_test_data()
#X.shape, y.shape


# In[54]:


my_data='flchain'
my_data_df=survwrap.datasets.get_data(my_data)
seed=2312


# In[55]:


X, y= my_data_df.get_X_y()
X.shape, y.shape


# ### Generate a (stratified) train-test split and Scale the features (only) 

# First do the stratified splitting THEN do scaling, parameterized on X_train set ONLY 

# In[56]:


from sklearn.preprocessing import StandardScaler, RobustScaler


# In[57]:


X_train, X_test, y_train, y_test = survwrap.survival_train_test_split(X, y,rng_seed=2311)


# In[58]:


scaler = StandardScaler().fit(X_train)
[X_train, X_test] = [ scaler.transform(_) for _ in  [X_train, X_test] ]
X_train.shape, X_test.shape


# In[59]:


#X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y)


# balanced partitioning OK. Robst scaler damages the performance of DSM A LOT.
# maybe did something wrong. It is standard scaler for now.

# In[60]:


survwrap.get_indicator(y).sum(), survwrap.get_indicator(y_train).sum(), survwrap.get_indicator(y_test).sum(),


# In[61]:


splitter = survwrap.survival_crossval_splitter(X_train,y_train,n_splits=3, n_repeats=2,rng_seed=2309)
print([ (survwrap.get_indicator(y_train[_[1]]).sum()) for _ in splitter])


# ## check possible dimensionality reduction

# In[62]:


from sklearn.decomposition import PCA


# In[63]:


pca= PCA(n_components=0.995, random_state=2308).fit(X_train)
pca.n_components_


# Only a modest dimensionality reduction is possible using PCA

# In[64]:


## Stratified CV spliter for survival analysis


# In[65]:


from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold


# In[66]:


#fastcph 
import lassonet
from lassonet import LassoNetCoxRegressor


# In[67]:


fastcph=LassoNetCoxRegressor(tie_approximation='breslow',
                                hidden_dims=(8,8),
    lambda_start=0.001,
    path_multiplier=1.02,
    backtrack=True,
    #gamma=0.5,
    #val_size=0.15,                        
    verbose=2,
    random_state=seed,
    torch_seed=seed,
    )
fastcph


# In[68]:


#X_train[:5]


# In[69]:


#y_swap
#from pandas import DataFrame
#y_df= DataFrame({'time': survwrap.get_time(y_train),'event':survwrap.get_indicator(y_train)})
yup=numpy.column_stack((survwrap.get_time(y_train).astype('float32'),
                        survwrap.get_indicator(y_train).astype('float32')))
print(type(yup))
#yup=numpy.array(list(zip(survwrap.get_indicator(y_train).astype(float),survwrap.get_time(y_train).astype(float))))
yup[:10]
X_train.shape, yup.shape
#numpy.ndarray(y_train[1], y_train[0])


# In[70]:


fitted_fastcph = fastcph.fit(X_train,yup)
fitted_fastcph


# In[75]:


assert fastcph == fitted_fastcph


# In[76]:


fastcph.set_params()


# In[73]:


# MACHECAZZO!
#fastcph.predict(X_train)[:10], fastcph.predict(X_test)[:10]
#lassonet.plot_path(fastcph, fastcph.path, X_train, yup)


# In[77]:


fastcph.path_[1].objective


# In[78]:


#fitted_fastcph.path_.loss , fitted_fastcph.path_.lambda 

lambda_min = min(fitted_fastcph.path_, key=(lambda x: x.objective))
#lambda_min = min([(_.objective, _.lambda_)  for _ in fitted_fastcph.path_])
lambda_min 


# In[80]:


fastcph.set_params(lambda_seq=[lambda_min.lambda_],
                   random_state=seed,torch_seed=seed,
                   verbose=1)
refit_fastcph=fastcph.fit(X_train,yup)


# In[81]:


fastcph.predict(X_train)


# In[82]:


fastcph.score(X_train,yup)


# In[ ]:





# In[83]:


#y_swap
#from pandas import DataFrame
#y_df= DataFrame({'time': survwrap.get_time(y_train),'event':survwrap.get_indicator(y_train)})
yup_test=numpy.column_stack((survwrap.get_time(y_test).astype('float32'),
                        survwrap.get_indicator(y_test).astype('float32')))
print(type(yup_test))
X_test.shape, yup_test.shape
#numpy.ndarray(y_train[1], y_train[0])


# In[84]:


fastcph.score(X_test,yup_test)


# ## Test with CV-based lambda selection

# In[87]:


cv_set=survwrap.survival_crossval_splitter(X_train,y_train,
                                           n_repeats=1,
                                          rng_seed=seed)
cv_set


# In[88]:


fastcph.get_params()


# In[94]:


fastcph_cv=lassonet.LassoNetCoxRegressorCV(cv=cv_set,
                                          hidden_dims=(8,8),
                                          tie_approximation='breslow',
                                            path_multiplier=1.02,
                                          torch_seed=seed,random_state=seed)
fastcph_cv


# In[96]:


fastcph_cv.fit(X_train,yup)

