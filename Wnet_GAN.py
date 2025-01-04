#######################################################################
# NASA GSFC Global Modeling and Assimilation Office (GMAO), Code 610.1
# code developed by Donifan Barahona and Katherine Breen
# last edited: 06.2023
# purpose: train/validate/test Wnet (GAN generator), plot output
######################################################################

########################################################
# IMPORT PACKAGES
########################################################


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import xarray as xr
import dask as da
import xesmf as xe
from sklearn.metrics import mean_squared_error
from random import shuffle, randint

import keras
from tensorflow.keras.models import Sequential
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint
from keras import layers
from keras.models import load_model
from keras.utils import Sequence
from keras import regularizers
import keras.backend as K
from keras.optimizers import Adam
import tensorflow as tf

###########################################################
# FUNCTIONS
###########################################################

bce = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
mse =  tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)  
mae =  tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE)
 
def standardize(ds):
  i = 0
  m= [243.9, 0.6, 6.3, 0.013, 0.0002, 5.04, 21.8, 0.002, 9.75e-7, 7.87e-6]  #hardcoded from G5NR
  s = [30.3, 0.42, 16.1, 7.9, 0.05, 20.6, 20.8, 0.0036, 7.09e-6, 2.7e-5]
  for v in  ds.data_vars:
   ds[v] = (ds[v] - m[i])/s[i]
   i = i+1   
  return ds
   
def outlier(x):
  return abs((x - x.mean(dim='time')) / x.std(dim='time'))

def random_pert(ds, percent = 1):
   func  = lambda x, percent: x*randint(1000-percent*10, 1000+percent*10)/1000. 
   return xr.apply_ufunc(func, ds, percent, dask='parallelized')      
   
def build_gen(prior, hp):
	
    hidden_layer_sizes = (hp['Nnodes'],)*hp['Nlayers']

    #we need to recreate the model
    model = Sequential()
    for layer in prior.layers[:-1]: # do not add the last layer
        model.add(layer)    
    # Freeze the layers 
    for layer in model.layers:
    	layer._name = layer.name + str("_prior")
    	layer.trainable = False
        
	#add layers on top	
    for hidden_layer_size in hidden_layer_sizes:
        model.add(layers.Dense(hidden_layer_size)) 
        model.add(layers.LeakyReLU(alpha=0.1)) 

    model.add(layers.Dense(1))

    return model   
   

def build_dis(hp): 

    lat_dim = hp['latent_dim']
    n_feat =  hp['n_features'] +  1
    input_dat = keras.Input(shape=(n_feat,))
  
    x = layers.Dense(lat_dim*8)(input_dat) 
    x = layers.LeakyReLU(alpha=0.2)(x)

    x = layers.Dense(lat_dim*4)(x) 
    x = layers.LeakyReLU(alpha=0.2)(x)
    
    x = layers.Dense(lat_dim*2)(x) 
    x = layers.LeakyReLU(alpha=0.2)(x)
    
    x = layers.Dense(lat_dim)(x) 
    x = layers.LeakyReLU(alpha=0.2)(x)    
    
    x = layers.Dense(lat_dim*2)(x) 
    x = layers.LeakyReLU(alpha=0.2)(x)
    
    x = layers.Dense(lat_dim*4)(x) 
    x = layers.LeakyReLU(alpha=0.2)(x)
    
    x = layers.Dense(lat_dim*8)(x) 
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.3)(x)

    outputs =  layers.Dense(1, activation='sigmoid')(x) 
    discriminator = keras.Model(input_dat, outputs)
    
    return discriminator

def cmloss(ytrue,ypred):
    tf.autograph.to_graph(cmloss)            
    loss =  bce(ytrue, ypred)
    return tf.reduce_mean(loss)
       
def get_data():
    
    sites  = []  # list of site names to use for refinement. Site and reanalysis data should be stored as <site>.nc 
    path_asr  = ""  # path to observational data
    path_merra = ""  # path to reanalysis  
    m= 0
    for site in sites:
    
        # Read in observations
        asrdata = path_asr  + site + ".nc"   
        print('Loading data from --', asrdata)
        
        dat_obs =  xr.open_mfdataset(asrdata,  parallel=True, chunks={"time": 4096})#["W_asr_std"][:, 23:71]  
        nam =  "Wstd_" + site
        
        dat_obs.load()
        
        #need to partition the last 20% of the data for testing****
        tim = dat_obs['time'].values
        tx = int(0.8*len(tim))
        test_dates =  pd.to_datetime(tim[tx:])
        dat_obs =  dat_obs.drop_sel (time  = test_dates)
        
        dat_obs = dat_obs.where(dat_obs != -9999.)
        dat_obs= dat_obs.where(dat_obs <10)
        
        # the following block of code is specific to our data and serves as an example for cleaning individual site data based on site/domain expertise
        '''
        if site == 'manus':
        	dat_obs = dat_obs.where((dat_obs["time.year"] < 2005) | (dat_obs["time.year"] > 2007), drop=True)

        if site == 'twp':
        	dat_obs = dat_obs.where((dat_obs["time.year"] < 2014), drop=True)

        if site == 'mao':
            d1 =  dat_obs.sel(time=slice('2014-02-01', '2015-01-10'))
            d2 =  dat_obs.sel(time=slice('2015-04-01', '2016-02-01'))
            dat_obs =  xr.concat([d1, d2], dim ='time')
            #dat_obs =  dat_obs.drop_sel(time=slice("2015-01-01", "2015-04-01"))
            #	dat_obs = dat_obs.where((dat_obs["time.year"] < 2015) | (dat_obs["time.year"] > 2007), drop=True)

        if site == 'ena':
            #dat_obs  =  dat_obs.where(dat_obs.time!=slice("2006-01-01", "2008-01-01"))
            dat_obs = dat_obs.where((dat_obs["time.year"] < 2017) | (dat_obs["time.year"] >= 2018), drop=True)
        '''
        
        #remove outliers####################
        dat_obs = dat_obs.where(dat_obs != -9999.)   
        kstd =  2.5 # defines outliers beyond 4stdev  
        dat_aux =  dat_obs.where(dat_obs > 0.0001)  
        dat_std =  dat_aux.groupby('time.month').map(outlier) # returns abs(anomaly/std)        
        dat_obs = dat_obs.where(dat_std < kstd) 
        
        # Read in reanalysis data
        Minp = path_merra + site +  ".nc"
        print('Loading data from --', Minp)

        dat_merra = xr.open_mfdataset(Minp, parallel=True, chunks={"time": 4096})
        dat_merra.load()
        
        #need to partition the last 20% of the data for testing****
        tim = dat_merra['time'].values
        tx = int(0.8*len(tim))
        test_dates =  pd.to_datetime(tim[tx:])
        dat_merra =  dat_merra.drop_sel (time  = test_dates)
        
        levs = dat_merra.coords['lev'].values
        nlev =  len(levs)
        dat_obs = dat_obs.rename({'height':'lev'})
        dat_obs =  dat_obs.assign_coords(lev=levs)
        
        # Merra is 3-hourly we have to resample. USe 5 minutes to make sure there are coincident times with the obs    
        dat_merra = dat_merra.resample(time="5min").interpolate("linear") # use       
        
        # align time steps with obs
        dat_merra, dat_obs = xr.align(dat_merra, dat_obs, exclude = {'height', 'lev'})
        
        # radar only works in-cloud - do this after aligning
        radar_lst = []  # list of sites using radar data
        if site in radar_lst:
          QCT = dat_merra.QL + dat_merra.QI 
          dat_obs =  dat_obs.where(QCT > 1e-9)	
        
        # Preprocess Wnet_prior input data  
        dat  =  dat_merra[['T', 'AIRD', 'U', 'V', 'W', 'KM', 'RI', 'QV', 'QI', 'QL']]     
        feat_0= xr.map_blocks(standardize, dat, template=dat)
        
        dat_obs =  dat_obs.to_array()
       
        #=====drop levels with all zeros
        feat_in =  feat_0.where(dat_obs > 0).squeeze()
        dat_obs =  dat_obs.where(dat_obs > 0).squeeze()
        
        feat_in  =  feat_in.dropna(dim="lev", how="all")
        dat_obs  =  dat_obs.dropna(dim="lev", how="all")
         
        #========add_surface_vars======
  
        surf_vars =  ['AIRD', 'KM', 'RI', 'QV']
        levs = feat_in.coords['lev'].values
        nlev =  len(levs)

        for v in surf_vars:        
            Xs =  feat_0[v].sel(lev=[71])  #level 1 above surface        
            Xsfc =  Xs
            v2 =  v + "_sfc"
            for l in range(nlev-1):         
            	Xsfc = xr.concat([Xsfc, Xs], dim ='lev') 
            Xsfc =  Xsfc.assign_coords(lev=levs)
            feat_in[v2] =  Xsfc
        
        feat_in =  feat_in.unify_chunks()
        
        ######Augment the cirrus/convective data
        repeat_aug = 4
        aug_lst = []  # list of sites to augment
        if site in aug_lst:
          for i in range(repeat_aug):
              percent =  1 #% of random pert
              feat_aug =  feat_in 
              obs_aug =   random_pert(dat_obs, percent) 
              feat_in= xr.concat([feat_in, feat_aug], dim="time",  join='override') 
              dat_obs = xr.concat([dat_obs, obs_aug], dim="time",  join='override')
              
        #stack and remove remaining zeros
        feat_in =  feat_in.to_array()
        feat_in = feat_in.stack(s=('time', 'lev'))
        feat_in =  feat_in.rename({"variable":"ft"})
        dat_obs = dat_obs.stack(s=('time', 'lev'))

        feat_in =  feat_in.where(dat_obs > 0, drop = True).squeeze()#.to_array()
        dat_obs =  dat_obs.where(dat_obs > 0, drop = True).squeeze()#, drop =  True)
 
        #Concat the data  sets
        if m<1:
          yall =  dat_obs
          Xall =  feat_in
        else:          
          Xall = xr.concat([Xall, feat_in],  dim="s", fill_value = 0, join='override')
          yall = xr.concat([yall, dat_obs],  dim="s", fill_value = 0,  join='override') #this is the right one
        Xall =  Xall.compute()
        yall  = yall.compute()
        m= m+1

    Xall = Xall.transpose()
    yall =  yall.expand_dims(dim={"y": 1}).transpose()
 
    #======================shuffle and partition validation data
 
    ntime = len(yall[:, 0])
    n1 = int(0.9*ntime) #use 10% for validation.
   
    ismpls = list(i for i in range(0,yall.shape[0]))
    shuffle(ismpls)
    Xall =  Xall [ismpls, :]
    yall =  yall [ismpls, :]

    Xtrain = Xall[:n1, :] 
    Xval = Xall[n1:, :]
    ytrain = yall[:n1, :]
    yval = yall[n1:, :]
    
    return Xtrain, ytrain, Xval, yval



###########################################################
# CLASSES
###########################################################  
  
# custom callback to save best gen/dis models
class epoch_cllbck(Callback):
    def __init__(self):
        self.min_loss =  1e6
        
    def on_epoch_end(self, epoch, logs=None):
    	
        monitor = "val_g_obs_loss"
        current_loss = logs.get(monitor)
       
        if epoch>0 and self.min_loss > current_loss:  # best loss
        	
            message = monitor + ' improved from ' + str(self.min_loss) + ' to ' + str(current_loss)  
            print(message, '--current epoch: ', epoch)
            self.min_loss =  current_loss
            
            # save the current state of the generator/discriminator
            dis = gan.discriminator
            gen = gan.generator            
            dis.save('best_discriminator.h5')
            gen.save('best_generator.h5')
                
def set_callbacks(mod_name):
    # SET CALLBACKS
 
    csv_logger = CSVLogger(mod_name +'.csv', append=True)
    cllbcks = [csv_logger, epoch_cllbck()]
    
    return cllbcks
    
# custom model to train gen and dis simultaneously    
class GAN(keras.Model):
    def __init__(self, discriminator, generator, hp=[]):
        super(GAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator   
        self.wh =  hp['wh']
        self.num_t_critic =  hp['num_t_critic']
        
    def compile(self, d_optimizer, g_optimizer):
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
               
    def train_step(self, datasets):
        merra_train, real_obs_train = datasets

         
        noise = 0.025 * tf.random.uniform(tf.shape(real_obs_train))*real_obs_train
        x =  real_obs_train + noise
        y =  merra_train 
  

        # Train the discriminator
        for it in range(self.num_t_critic):
            with tf.GradientTape() as d_tape:
                g_z = self.generator(y) 
                g_z = g_z*(1-self.wh) + self.wh*x     
                d_x = self.discriminator(tf.concat([x, y], 1)) #make it conditional #real output 
                d_gz = self.discriminator(tf.concat([g_z, y], 1)) #fake output	
                ones =  tf.ones_like(d_gz)
                zeros = tf.zeros_like(d_gz)
                loss1 = cmloss(ones, d_x)  
                loss2 = cmloss(zeros, d_gz)
                d_loss  =  (loss1 + loss2)
            
            gradients_of_discriminator = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
            self.d_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables)) 

        # Train the generator    
        with tf.GradientTape() as g_tape:
            g_z = self.generator(y)
            g_z = g_z*(1-self.wh) + self.wh*x   
            d_x = self.discriminator(tf.concat([x, y], 1)) #make it conditional #real output 
            d_gz = self.discriminator(tf.concat([g_z, y], 1)) #fake output
            ones =  tf.ones_like(d_gz)
            g_loss = cmloss(ones, d_gz) 
                     
        gradients_of_generator = g_tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))

        return {"d_loss": d_loss, "g_loss": g_loss}

    def test_step(self, datasets):
       
        y, x = datasets 
        
        g_z = self.generator(y, training = False)
        g_z = tf.where(x>0, g_z, 0) #filter for obs
        
        # add a more meaninful metric
        g_obs_loss = tf.reduce_mean(mae(g_z, x))        
       
        d_x = self.discriminator(tf.concat([x, y], 1),  training = False) #make it conditional #real output 
        d_gz = self.discriminator(tf.concat([g_z, y], 1), training = False) #fake output	     
        ones =  tf.ones_like(d_gz)
        zeros = tf.zeros_like(d_gz)
        loss1 = cmloss(ones, d_x)  
        loss2 = cmloss(zeros, d_gz)
        d_val_loss = (loss1+loss2) 
        g_val_loss = cmloss(ones, d_gz) 
       
        GAN_loss =  tf.abs(d_val_loss + g_val_loss)
        
        return {"d_loss": d_val_loss, "g_loss": g_val_loss, "GAN_loss": GAN_loss, 'g_obs_loss': g_obs_loss}


# DO THE WORK
if __name__ == '__main__':

    strategy = tf.distribute.MirroredStrategy()
    
    mod_name = 'GAN_singlelev'
    nepochs = 500
    hp = {
        'latent_dim': 8,
        'num_t_critic': 1, 
        'lr': 1e-5,
        'batch_size': 1024,
        'wh' :  0.0, 
        'n_features' : 14, 
        'Nlayers': 1,
        'Nnodes': 128,
    }
    
    ###########################################################
    # GET DATA
    ###########################################################
    
    merra_train, obs_train, merra_val, obs_val= get_data()
    
    merra_train = tf.cast(merra_train.values,tf.float32)
    merra_val = tf.cast(merra_val.values,tf.float32)

    obs_train = tf.cast(obs_train.values,tf.float32)
    obs_val = tf.cast(obs_val.values,tf.float32)

    dat_train = tf.data.Dataset.from_tensor_slices((merra_train, obs_train))
    dat_train = dat_train.prefetch(buffer_size=1024)
    dat_train =  dat_train.batch(hp['batch_size'])
    dat_train =  dat_train.cache()

    dat_val = tf.data.Dataset.from_tensor_slices((merra_val, obs_val))
    dat_val = dat_val.prefetch(buffer_size=1024)
    dat_val =  dat_val.batch(hp['batch_size'])
    dat_val =  dat_val.cache()

    if os.path.exists('best_discriminator.h5'):
        print('Checkpoint exists! Restarting training')
        model_d=load_model('best_discriminator.h5', compile=False)
        model_d.summary()
        model_g=load_model('best_generator.h5', compile=False)
        model_g.summary()
    else:                
        ###########################################################
        # DEFINE DISCRIMINATOR
        ###########################################################
        model_d = build_dis(hp)
        model_d._name = 'discriminator'
        model_d.summary()

        ###########################################################
        # DEFINE GENERATOR
        ###########################################################

        pth  = ""  # path to prior model
        prior=load_model(pth , compile=False)           

        model_g = build_gen(prior, hp)
        model_g._name = 'generator'
        model_g.summary()

    ###########################################################
    # DEFINE GAN
    ###########################################################

    gan = GAN(discriminator=model_d, generator=model_g, hp=hp)
    gan.compile(
    d_optimizer=keras.optimizers.Adam(learning_rate=hp['lr']),  
    g_optimizer=keras.optimizers.Adam(learning_rate=hp['lr'])   
    )

   
    ###########################################################
    # TRAIN GAN
    ###########################################################
    
    history = gan.fit(
        dat_train,
        validation_data=dat_val,
        epochs=nepochs, batch_size =  hp['batch_size'],
        callbacks=set_callbacks(mod_name),
        verbose=2, use_multiprocessing = True, workers=10)
           
    model_g.save(model_g.name+'.h5')	
    model_d.save(model_d.name+'.h5')
    
    #plot loss
    plt.switch_backend('agg')
    plt.plot(history.history['g_loss'])
    plt.plot(history.history['val_g_loss'])
    plt.plot(history.history['d_loss'])
    plt.plot(history.history['val_d_loss'])
    plt.plot(history.history['val_g_obs_loss'])
    
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['g', 'g_val', 'd', 'd_val', 'g_vs_obs'], loc='upper left')
    plt.savefig( mod_name+'_loss.png')
    
    exit()
