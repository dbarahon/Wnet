#######################################################################
# NASA GSFC Global Modeling and Assimilation Office (GMAO), Code 610.1
# code developed by Donifan Barahona and Katherine Breen
# last edited: 07.2023
# purpose: train/validate/test Wnet-prior, plot output
######################################################################

#### IMPORT PACKAGES ####

import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import xarray as xr
import dask as da
import xesmf as xe
from sklearn.metrics import mean_squared_error

import keras
from keras.models import Sequential
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint
from keras.models import load_model
from keras.utils import Sequence
import tensorflow as tf

#### FUNCTIONS ####

# globally define loss function
mse =  tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)

def standardize(ds, s=1, m=0):
  i = 0
  #['T', 'AIRD', 'U', 'V', 'W', 'KM', 'RI', 'QV', 'QI', 'QL'] 
  for v in  ds.data_vars:  
   ds[v] = (ds[v] - m[i])/s[i]
   i = i+1
  return ds
 
     
def get_random_files(pth, nts):
  lall = glob.glob(pth) 
  f_ind = np.random.randint(0, len(lall)-1, nts)
  fils = [lall[i] for i in f_ind] 
  return fils

def switch_var(V, fils):
  #switch strings in two steps
  old = '0.0625_deg/inst/inst30mn_3d_W_Nv'
  new = '0.5000_deg/tavg/tavg01hr_3d_' + V + '_Cv'
  flsv = [sub.replace(old, new) for sub in fils]
  old  = 'inst30mn_3d_W_Nv'
  new =  'tavg01hr_3d_' + V + '_Cv'
  flsv = [sub.replace(old, new) for sub in flsv]  
  return [flsv]

def dens (ds): 
    d = ds.PL/287.0/ds.T
    ds.PL.data = d
    return ds

def QCT (ds): 
    d = ds.QL + ds.QI    
    ds.QL.data = d
    return ds
           
def set_callbacks(nam = "WNet" ):
    # SET CALLBACKS
    early_stop = EarlyStopping(monitor='val_loss', 
                       min_delta=0.000000001,  
                       patience=20, 
                       verbose=1)  

    csv_logger = CSVLogger(nam + '.csv', append=True)

    model_checkpoint = ModelCheckpoint(nam + '.hdf5',
                               monitor='val_loss', 
                               verbose=1,
                               save_best_only=True, 
                               mode='min')
    cllbcks = [csv_logger, model_checkpoint, early_stop]   
    return cllbcks  



def build_wnet(hp):

    n_feat =  hp['n_features']
    input_dat = keras.Input(shape=(n_feat,))
    initializer = tf.keras.initializers.HeUniform()
  
    x =  input_dat
    for hidden_layer_size in hp['hidden_layer_sizes']:
        x = layers.Dense(hidden_layer_size, kernel_initializer=initializer)(x) 
        x = layers.LeakyReLU(alpha=0.2)(x)       
	
    output =  layers.Dense(1)(x)
    model = keras.Model(input_dat, output)
    opt = tf.keras.optimizers.Adam(learning_rate=hp['lr'], amsgrad=True)
    model.compile(loss=polyMSE, optimizer=opt)

    return model  
    
def polyMSE(ytrue,ypred):            
    st =  2
    mx =  14
    x = tf.where(ypred > 1e-6, ypred, 0)  #use obs mask
    y = tf.where(ytrue > 1e-6, ytrue, 0) 

    aux = 0.
    m1 =  0.
    m2 = 0.
    for  n in range(0, mx, st):
        k = tf.constant((n+st)*0.1)
        m1 =tf.pow(x, k) + m1
        m2 = tf.pow(y, k) + m2

    return  tf.reduce_mean(mse(m1, m2)) 


#### CLASSES ####
## Loads and stores data files for training and validation. 
class get_dts():
   def __init__(self, ndts =  1, nam ="def",  exp_out=1, batch_size = 32000, subsample=5):  #creates a class that will handle ndsts files
    yr =  "Y2006/" 
    mo = "M*/"
    dy =  "D*/*30z*"

    self.batch_size = batch_size
    self.lev1 = 1
    self.lev2 = 72
    self.vars_in = ['T', 'PL', 'U', 'V', 'W', 'KM', 'RI', 'QV', 'QI', 'QL'] 
    self.means= [243.9, 0.6, 6.3, 0.013, 0.0002, 5.04, 21.8, 0.002, 9.75e-7, 7.87e-6] #hardcoded from G5NR based on 100 time steps
    self.stds =[30.3, 0.42, 16.1, 7.9, 0.05, 20.6, 20.8, 0.0036, 7.09e-6, 2.7e-5]
    self.surf_vars =  ['AIRD', 'KM', 'RI', 'QV']
    self.feats = len(self.vars_in)+len(self.surf_vars)
    self.chk = { "lat": -1, "lon": -1, "lev":  -1, "time": 1} # needed so we can regrid
    self.in_dir  =  ""  # path to input feature data
    self.out_dir  = ""  # path to output target data
    self.path_out =  self.out_dir +  yr + mo + dy
    self.create_regridder =  True       
    self.name = nam
    #get nts files 
    self.fls = get_random_files(self.path_out,ndts)
   
   def get_fls_batch (self, dt_batch_size):
     for i in range(0, len(self.fls), dt_batch_size):
       yield self.fls[i:i + dt_batch_size]
       if i >= len(self.fls):
           i = 0
             
   def  get_data(self, this_fls):
     
      self.dat_out =  xr.open_mfdataset(this_fls, chunks=self.chk, parallel=True)
      self.dat_out =  self.dat_out.coarsen(lat=8, lon=8, boundary="trim").std() #coarsen to about half degree using standard deviation as lumping function      
      self.levs =  len(self.dat_out['lev'])
      vars_in  = self.vars_in
      self.n_features_in_ = len(vars_in)*self.levs
      self.feats = len(vars_in)  
      
      dat_in = []
      m=0
      for v in vars_in:  
         flsv = switch_var(v, this_fls) 
         dat =  xr.open_mfdataset(flsv, chunks=self.chk, parallel=True).sel(lev=slice(self.lev1,self.lev2))
      
         if m ==0:
           dat_in = dat
           m=1
         else:          
           dat_in =  xr.merge([dat_in, dat])#, join='exact') 
      dat.close()        
        
      ###Calculate density 
      dat_in = dat_in.unify_chunks()           
      da= xr.map_blocks(dens, dat_in, template=dat_in)
      dat_in = da.rename({"PL":"AIRD"})
      dat_in =  dat_in[['T', 'AIRD', 'U', 'V', 'W', 'KM', 'RI', 'QV', 'QI', 'QL']]  # ensure features are ordered correctly
       
      ### standardize inputs (make sure only time iss chunked)s
      self.dat_in= xr.map_blocks(standardize, dat_in, kwargs={"m":self.means, "s": self.stds}, template=dat_in)
      dat_in.close()
           
    
   def regrid_out(self, create_regridder = True):
    #regrid output
      if self.create_regridder:  
          self.regridder = xe.Regridder(self.dat_out, self.dat_in, 'bilinear', periodic=True) #make sure they are exactly the same grid 
          self.create_regridder =  False    #this is done bacause there is a bug in xesmf when two consecutive regridders are created on the fly 
      self.dat_out =  self.regridder(self.dat_out)
   
      
   def get_Xy(self, make_y_array =  True,  batch_size  =5120, test = False, x_reshape =False):
   
        self.batch_size = batch_size
        if test:
           self.get_data(self.fls) 

        self.regrid_out()
        
        Xall = self.dat_in
        levs = Xall.coords['lev'].values
        nlev =  len(levs)
	
	   #concatenate surface variables
        for v in self.surf_vars:        
            vv =     Xall[v]
            Xs =  vv.sel(lev=[71]) #level 1 above surface
            Xsfc =  Xs
            v2 =  v + "_sfc"
            for l in range(nlev-1):         
                Xsfc = xr.concat([Xsfc, Xs], dim ='lev')                 
            Xsfc =  Xsfc.assign_coords(lev=levs)
            Xall[v2] =  Xsfc
            
        Xall =  Xall.unify_chunks()
        Xall = Xall.to_array()
        Xall = Xall.stack( s = ('time', 'lat', 'lon', 'lev')) 
        Xall = Xall.rename({"variable":"ft"})                       
        Xall = Xall.squeeze()
        Xall = Xall.transpose()
        Xall = Xall.chunk({"ft":self.n_features_in_, "s": 102000}) #chunked this way aligns the blocks/chunks with the samples 
        yall = self.dat_out.stack(s = ('time', 'lat', 'lon', 'lev' ))
        yall =  yall.squeeze()
        yall =  yall.transpose()   
        yall =  yall.chunk({"s": 102000})
        
        Xall = Xall.chunk({"ft":self.n_features_in_, "s": batch_size})
        yall =  yall.chunk({"s": batch_size})
        self.Nsamples =  len(yall['s'].values) 
       
        if make_y_array:
            yall =  yall.to_array().squeeze()
        if x_reshape:
            Xall =  Xall.rename({"ft":"variable"})
            Xall = Xall.expand_dims("w")
            Xall = Xall.transpose('s', 'variable', 'w') 
        
        return Xall, yall
        

# Use dask generator for data stream ============ here we read data in batches of dtsbatch steps to save time 
# Each chunk of the dask array is a minibatch
class DaskGenerator(Sequence):
        def __init__(self, dts_gen, nepochs_dtbatch, dtbatch_size, batch_size):
            
            
            self.dt_batch_size =  dtbatch_size # number of time steps loaded at once
            self.nepochs_dtbatch = nepochs_dtbatch #number of epochs to train current files
            self.count_epochs =  1 # counts how many epochs this batch has trained on 
            self.dts_gen =  dts_gen #data streamer
            self.batch_size =  batch_size
            self.fls_batch =  self.dts_gen.get_fls_batch (self.dt_batch_size)
            self.dts_gen.get_data(this_fls = next(self.fls_batch))                 
            
            X_train, y_train = self.dts_gen.get_Xy(batch_size  =  self.batch_size) 
            X_train = X_train.persist()
            y_train = y_train.persist()
            self.Nsamples =  len(y_train['s'].values)
            self.sample_batches = X_train.data.to_delayed()
            self.class_batches = y_train.data.to_delayed()

            assert len(self.sample_batches) == len(self.class_batches), 'lengths of samples and classes do not match'
            assert self.sample_batches.shape[1] == 1, 'all columns should be in each chunk'

        def __len__(self):
            '''Total number of batches, equivalent to Dask chunks in 0th dimension'''
           return len(self.sample_batches)
     
        def __getitem__(self, idx):
                    
            '''Extract and compute a single chunk returned as (X, y). This is also a minibatch'''
            X, y = da.compute(self.sample_batches[idx, 0], self.class_batches[idx])
            X = np.asarray(X).squeeze()
            y = np.asarray(y).squeeze()
            
            return X, y
            
        def on_epoch_end(self): 
            self.count_epochs  =  self.count_epochs + 1 
            if self.count_epochs >  self.nepochs_dtbatch:
                #get a new batch and start over 
                print("___new__", self.dts_gen.name, '__batch__', self)
                self.count_epochs  = 1 
                self.fls_batch =  self.dts_gen.get_fls_batch (self.dt_batch_size)
                self.dts_gen.get_data(this_fls = next(self.fls_batch))     
                
                X_train, y_train = self.dts_gen.get_Xy(batch_size  =  self.batch_size) 
                X_train = X_train.persist()
                y_train = y_train.persist()
                self.sample_batches = X_train.data.to_delayed()
                self.class_batches = y_train.data.to_delayed()
              
#=========================================
#=========================================
#=========================================
if __name__ == '__main__':

    hp = {
        'Nlayers': 5,
        'Nnodes': 128,
        'lr': 0.0001,
        'n_features' : [],
        'hidden_layer_sizes' : [],
    }
    
    physical_devices = tf.config.list_physical_devices('GPU')
    print("====Num GPUs:", len(physical_devices))
    strategy = tf.distribute.MirroredStrategy()
    
    model_name =  "Wnet_prior" 
    hp['hidden_layer_sizes'] = (hp['Nnodes'],)*hp['Nlayers']
    
    batch_size = 2048 #actual batch size
   
    dtbatch_size =  3 # number of time steps loaded at once (use 2-3 to avoid overfitting)
    epochs_per_dtbatch =  5# number of epochs before loading new training files
    dtbatch_size_val =  1 # number of time steps loaded at once
    epochs_per_dtbatch_val = 10 # number of epochs before loading new validation files
    nepochs = 1000
    ndts_train = 200 #number of files to choose from 
    ndts_val  = 200
    ndts_test =  20 
    train_model = True
     
    train_data =  get_dts(exp_out=nexp, ndts=ndts_train, nam = 'train_data', batch_size =  batch_size)
    val_data =  get_dts(exp_out=nexp, ndts=ndts_val, nam = 'val_data', batch_size =  batch_size)
    test_data =  get_dts(exp_out=nexp, ndts=ndts_test, nam = 'test_data', batch_size =  102000) # use a large batch size for inference
    levs =  train_data.lev2-train_data.lev1 + 1
    hp['n_features']= train_data.feats

    print('===train==', train_data.fls)
    print('===val==', val_data.fls)
    print('===test==', test_data.fls)  
    
    if os.path.exists(model_name + '.hdf5'):
        checkpoint_path = model_name + '.hdf5'
        # Load best model from checkpoint:
        print('-----Checkpoint exists! Restarting training')
        model = load_model(checkpoint_path, compile=train_model)

    else:
        with strategy.scope():
        	model =  build_wnet(hp)
    
    if train_model: 
        # build the data generators
        train_gen = DaskGenerator(train_data, epochs_per_dtbatch , dtbatch_size, batch_size )
        val_gen = DaskGenerator(val_data, epochs_per_dtbatch_val, dtbatch_size_val, batch_size)
        steps  =  int(0.99*train_gen.Nsamples/batch_size) 

        history =model.fit(train_gen,
                           validation_data =val_gen,
                           steps_per_epoch=steps, 
                           epochs=nepochs, 
                           verbose=2, 
                           callbacks=set_callbacks(model_name), 
                           use_multiprocessing=True, 
                           workers=10
                           ) 

        #plot loss
        plt.switch_backend('agg')
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(model_name + '_loss.png')
   
    
    #=========================================
    #====================test=================
    #=========================================
    
    if os.path.exists(model_name + '.hdf5'): #get the best model
        checkpoint_path = model_name + '.hdf5'
        # Load model:
        print('-----Checkpoint exists! Restarting training')
        model = load_model(checkpoint_path, compile=False)
                
    test_x, test_y =  test_data.get_Xy( make_y_array = False, batch_size = 512*72*2, test = True, x_reshape =  False)

    #==error calculation
    y_t =  test_y.to_array(dim="W").squeeze().persist() 
    X =  test_x.load()
    y_hat = model.predict(X, batch_size=32768) 
    y_hat =  np.squeeze(y_hat) 
    test_loss = mean_squared_error(y_t, y_hat) 

    #==========save netcdf ================"
    y_pred=  test_y.copy(data={"W":y_hat}) 

    Wtrue = test_y.transpose().unstack("s").set_coords(['time', 'lev', 'lat', 'lon']).rename({"W":"Wvar"})
    Wpred = y_pred.transpose().unstack("s").set_coords(['time', 'lev', 'lat', 'lon']).rename({"W":"Wvar_pred"})

    W_true = Wtrue.transpose('time', 'lev', 'lat', 'lon')
    W_pred = Wpred.transpose('time', 'lev', 'lat', 'lon')
    
    enc={'Wvar': {'dtype': 'float32', '_FillValue': -9999}}
    W_true.to_netcdf(model_name+".nc", mode = "w", encoding=enc)
    enc={'Wvar_pred': {'dtype': 'float32', '_FillValue': -9999}}
    W_pred.to_netcdf(model_name+".nc", mode = "a", encoding=enc)
    
