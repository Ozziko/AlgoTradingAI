# -*- coding: utf-8 -*-
"""@author: Oz Livneh (oz.livneh@gmail.com),
    all rights reserved, use at your own risk.

See the explanation and demonstration at the Colab notebook:
    http://bit.ly/OzTradingAI

Compatibility:
    developed on anaconda 5.2, containing:
        Python: 3.6.5 |Anaconda, Inc.| (default, Mar 29 2018, 13:32:41) [MSC v.1900 64 bit (AMD64)]
        numpy: 1.14.3
        pandas: 0.23.0
        matplotlib: 2.2.2
        sklearn: 0.19.1
        tensorflow: 1.10.0
        logging: 0.5.1.2
    tested on Colab notebook, containing:
        Python: 3.6.3 (default, Oct  3 2017, 21:45:48) [GCC 7.2.0]
        numpy: 1.14.5
        pandas: 0.22.0
        matplotlib: 2.1.2
        sklearn: 0.19.2
        tensorflow: 1.10.1
        logging: 0.5.1.2
"""
#%% initializing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from time import time
from datetime import timedelta, datetime
from sklearn import linear_model

filenames_time_format='%Y-%m-%d %H-%M-%S'
logging_time_format='%H:%M:%S'
trading_datetime_format='%Y/%m/%d'

logger=logging.getLogger('OzML logger')
logging.basicConfig(
        format='%(asctime)s OzML %(funcName)s (%(levelname)s): %(message)s',
        level=logging.INFO, # logging messages which are less severe than level will be ignored. Levels: NOTSET<DEBUG<INFO<WARNING<ERROR<CRITICAL
        datefmt=logging_time_format)

logger.info('OzML imported, logger created with logging level: INFO (change levels by calling OzML.set_logging_level)')
#%% defining functions, classes

def set_logging_level(logger_obj,level):
    """created on 2018/09/10
    """
    if level=='debug':
        logging_level=logging.DEBUG
    elif level=='info':
        logging_level=logging.INFO
    elif level=='warning':
         logging_level=logging.WARNING
    elif level=='error':
        logger.warn('the logger was set to error = muted, since in errors I raise a RuntimeError and never use the logger')
        logging_level=logging.ERROR
    else:
        raise RuntimeError('unsupported log_level!')     
        
    logger_obj.setLevel(logging_level)

def import_data(symbols,columns_list_to_use):
  """importing symbols data from csv files into a single dataframe
  returns: (data_df,attribute_names_list)
  
  reads symbols from csv files (example: for symbols=['GOOGL'] it reads 
    from GOOGL.txt) into pandas dataframes, indexing by 'Date' columns of the 
    original data, taking only columns_list_to_use from each dataframe,
    merging all into one data_df by intersection of the index of all dataframes,
    saving all column names (including symbols) to attribute_names_list,
    re-naming data_df colums to generic names 'x0','x1',...
  """
  df_list=[]
  for symbol in symbols:
      df=pd.read_csv(symbol+'.txt',parse_dates=['Date'],index_col=['Date'])
      df=df[columns_list_to_use]
      df.columns=[symbol+' '+col for col in df.columns]
      df_list.append(df)

  data_df=df_list[0]
  for df in df_list[1:]:
      data_df=data_df.merge(df,how='inner',
                                    left_index=True,right_index=True)
  attribute_names_list=list(data_df.columns)
  data_df.columns=['x%d'%ind for ind in range(len(data_df.columns))]
  logger.info('data imported! total length (number of rows): %d, number of columns: %d'%(
    len(data_df),len(attribute_names_list)))
  return (data_df,attribute_names_list)

def legend_styler(labels=None,loc='best',framealpha=0.6,fancybox=True,frameon=True,facecolor='white'):
    """created on 2018/10/03 to easily control legend style in all executions 
        from one place.
    """
    if labels==None:
        plt.legend(loc=loc,framealpha=framealpha,fancybox=fancybox,frameon=frameon,facecolor=facecolor)
    else:
        plt.legend(labels,loc=loc,framealpha=framealpha,fancybox=fancybox,frameon=frameon,facecolor=facecolor)

def data_plotting(data_df,attribute_names_list):
    """created on 2018/09/13 for data plotting used for verification,
        before building features.
   The data is normalized here as a whole without splitting to test,
       which is not legit for testing, only for data verification
    """
    temp_norm_df=(data_df-data_df.mean(axis=0))/(data_df.max(axis=0)-data_df.min(axis=0))
    temp_norm_df.columns=attribute_names_list
    temp_norm_df.plot()
    plt.title('normalized data (x-mean(x))/std(x)')
    legend_styler(loc='upper left')
    plt.grid(True,which='major',axis='both')

def examine_data_jumps(df,processing_mode,suspicious_jump_percents,attribute_names_list=0,plotting=True,bins=100):
    """created on 2018/09/16 to examine suspicious jumps in data values 
        (value[1:]/value[0:-1]-1), meaning absolute value jumps of more than 
        suspicious_jump_percents.
    
    if processing_mode=='% differences' then examining df itself, otherwise 
        creating diff_percents_df to examine.
    plotting=True: plots jumps histogram
    if attribute_names_list is given, naming diff_percents_df columns by it 
        (recommended).
    
    returns: suspicious_jumps_df
    """
    number_of_row_with_any_zeros=len(df[df.values==0])
    if number_of_row_with_any_zeros>0:
        raise RuntimeError(
                "there are %d row with any zeros, get rid of them before executing this (don't include volumes in df!)"%number_of_row_with_any_zeros)
    
    if attribute_names_list!=0:
        df.columns=attribute_names_list
    
    if processing_mode=='% differences':
        diff_percents_df=df
    else:
        diff_percents_df=pd.DataFrame(100*(df.values[1:,:]/df.values[0:-1,:]-1),
            index=df.index[1:],
#            columns=[col+' jumps %' for col in df.columns])
            columns=df.columns)
    
    if plotting:
        print('histograms of % jumps =100*(value[1:]/value[0:-1]-1)')
        diff_percents_df.hist(bins=100)
        plt.show()
    
    suspicious_price_jumps_df=diff_percents_df[
        abs(diff_percents_df.values)>suspicious_jump_percents]
    
    logging.info('there are %d price jumps higher in abs value than %d%%, saved in suspicious_price_jumps_df'%(
              len(suspicious_price_jumps_df),suspicious_jump_percents))
    
    return suspicious_price_jumps_df

def basic_feature_building(data_df,target_name,attribute_names_list,
                           include_target_history_as_features=True):
    """created on 2018/09/17
    returns (X_df,y_series) where X_df.values[idx,:] can be used to predict 
        y_series.values[idx], maintaining causality:
        X_df.values[idx,:] elements are taken from data_df.values[idx-1,:] but 
        indexed by data_df.index[idx],
        while y_series.values[idx] is taken from data_df.values[idx,:] and 
        indexed by data_df.index[idx].
    """
    target_idx=attribute_names_list.index(target_name)
    base_feature_names_list=list(data_df.columns)
    if not include_target_history_as_features:
        base_feature_names_list.remove(attribute_names_list[target_idx])
    
    X_temp=data_df[base_feature_names_list]
    X_df=pd.DataFrame(X_temp.values[0:-1,:],
                           columns=[col+'(t-dt)' for col in base_feature_names_list],
                           index=data_df.index[1:]) 
    X_df.index.name='t'
    
    y_series=data_df[base_feature_names_list[target_idx]].filter(items=X_df.index)
    y_series.name=base_feature_names_list[target_idx]
    y_series.index.name='t'
    
    data_duration_years=timedelta.total_seconds(y_series.index[-1]-y_series.index[0])/(365*24*60*60)
    
    logger.info('total data duration: %.1f years'%data_duration_years)
    
    return (X_df,y_series)

class train_test_split:
    """v1 created on 2018/08/31 (in OzML 1.0) based on FinancialML 9.0
        
        returns: split_data_obj
        
        splits X_df,y_series each into train and test periods and saves 
            into a split_data_obj object,
        
        Update log:
            v2 (OzML_v2)- 2018/09/09: splitting original __init__ into 
                different splitting methods
            v3 (OzML_v4)- 2018/09/16: taking into consideration processing_mode - 
                saving it in self, if self.processing_mode!='% differences'
                calculating them for the target when splitting,
                using it in plot target (normalized_hist)
        """
    def __init__(self,target_name,processing_mode):
        self.target_name=target_name
        self.processing_mode=processing_mode
        
    def split_by_dates(self,X_df,y_series,
                 train_start_datetime,train_duration_years,test_duration_years):
        """v1 split scheme:
            the train period starts on start_date, lasts test_duration_years, 
            then the test period starts and lasts test_duration_years.
        """
        
        splitting_datetime=train_start_datetime+timedelta(days=round(365*train_duration_years))
        test_end_datetime=splitting_datetime+timedelta(days=round(365*test_duration_years))
        if splitting_datetime>y_series.index[-1]:
            raise RuntimeError('splitting_datetime>y_series.index[-1], meaning train_duration_years>y_series_duration_years!')
        if test_end_datetime>y_series.index[-1]:
            test_end_datetime=y_series.index[-1]
            logger.warn('setting test_end_datetime=y_series.index[-1] because test_end_datetime>y_series.index[-1], meaning train_duration_years+test_duration_years>y_series_duration_years!')
        
        train_bool_array=(y_series.index>=train_start_datetime) & (y_series.index<splitting_datetime)
        y_train_series=y_series[train_bool_array]
        X_train_df=X_df[train_bool_array]
        len_train=len(y_train_series)
        
        test_bool_array=(y_series.index>=splitting_datetime) & (y_series.index<test_end_datetime)
        y_test_series=y_series[test_bool_array]
        X_test_df=X_df[test_bool_array]
        len_test=len(y_test_series)
        
        len_total=len_train+len_test
        logger.info('train,test data lengths: (%d,%d) = (%.1f,%.1f)%% from total length'
                     %(len_train,len_test,100*(len_train/len_total),100*(len_test/len_total)))
        logger.info('train period: %s to %s'%(
                y_train_series.index[0].strftime(trading_datetime_format),
                y_train_series.index[-1].strftime(trading_datetime_format)))
        logger.info('test period: %s to %s'%(
                y_test_series.index[0].strftime(trading_datetime_format),
                y_test_series.index[-1].strftime(trading_datetime_format)))
        self.y_train_series=y_train_series
        self.X_train_df=X_train_df
        self.y_test_series=y_test_series
        self.X_test_df=X_test_df
        
        if self.processing_mode=='% differences':
            self.y_train_diff_percents_series=self.y_train_series
            self.y_test_diff_percents_series=self.y_test_series
        else:
            (self.y_train_diff_percents_series,
             self.y_test_diff_percents_series)=create_target_diff_percents(
                     y_train_series,y_test_series,mode='continuous test')
        
    def split_by_idx_arrays(self,X_df,y_series,train_indices_array,test_indices_array):
        """created on 2018/09/09, to split by CV_split_v2.iterator() outputs -
            numpy arrays of traing and validation/test indices
        """
        self.y_train_series=y_series.iloc[train_indices_array]
        self.X_train_df=X_df.iloc[train_indices_array,:]
        self.y_test_series=y_series.iloc[test_indices_array]
        self.X_test_df=X_df.iloc[test_indices_array,:]
        
        if self.processing_mode=='% differences':
            self.y_train_diff_percents_series=self.y_train_series
            self.y_test_diff_percents_series=self.y_test_series
        else:
            (self.y_train_diff_percents_series,
             self.y_test_diff_percents_series)=create_target_diff_percents(
                     self.y_train_series,self.y_test_series,mode='continuous test')
                       
    def X_train_stats(self,max_rows_to_show=0):
        means=self.X_train_df.mean(axis=0)
        stds=self.X_train_df.std(axis=0)
        if max_rows_to_show>0 and max_rows_to_show<len(means):
            print('X_train_df mean (for %d first features):\n'%max_rows_to_show,
                  means[:max_rows_to_show])
            print('\nX_train_df std (for %d first features):\n'%max_rows_to_show,
                  stds[:max_rows_to_show])
        else:
            print('X_train_df mean:\n',means)
            print('\nX_train_df std:\n',stds)
        
    def X_test_stats(self,max_rows_to_show=0):
        means=self.X_test_df.mean(axis=0)
        stds=self.X_test_df.std(axis=0)
        if max_rows_to_show>0 and max_rows_to_show<len(means):
            print('X_test_df mean (for %d first features):\n'%max_rows_to_show,
                  means[:max_rows_to_show])
            print('\nX_test_df std (for %d first features):\n'%max_rows_to_show,
                  stds[:max_rows_to_show])
        else:
            print('X_test_df mean:\n',means)
            print('\nX_test_df std:\n',stds)
        
    def normalize(self):
        """narmalizes X_train_df and X_test_df by StandardScaler fit to X_train_df.
        """
        from sklearn.preprocessing import StandardScaler # http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler
        self.scaler=StandardScaler()
        self.scaler.fit(self.X_train_df)
        
        self.X_train_df=pd.DataFrame(self.scaler.transform(self.X_train_df.values),index=self.X_train_df.index,columns=self.X_train_df.columns)
        self.X_test_df=pd.DataFrame(self.scaler.transform(self.X_test_df.values),index=self.X_test_df.index,columns=self.X_test_df.columns)

        logger.info('normalized X_train_df and X_test_df by StandardScaler fit to X_train_df')
    
    def plot_target(self,bins=200,hist_mode='pdf'):
        """v2 (OzML_v4): if self.processing_mode!='% differences', 
            calculating % differences and plotting scaled_hist only for it 
            and not of the raw target, since the point is to compare the 
            distributions in train and test periods, and while prices can be 
            different - the price differences 
            should be similar!
        """
        plt.figure()
        plt.subplot(2,1,1)
        plt.plot(self.y_train_series,label='train')
        plt.plot(self.y_test_series,label='test')
        plt.grid(True,which='major',axis='both')
        plt.title('target: '+self.target_name)
        legend_styler()
        
        if self.processing_mode=='% differences':
            plt.ylabel('% differences')
        elif self.processing_mode=='raw':
            plt.ylabel('$')
        
        plt.subplot(2,1,2)
        """comparing probability densities is less useful when the total 
            number of counts is different between them, as typically 
            test_priod<<train_period -
            in this case, comaring the histograms when counts=counts/max(counts)
            is preferrable.
        """
        if hist_mode=='pdf':
            plt.hist(self.y_train_diff_percents_series,label='train',bins=bins,density=True)
            plt.hist(self.y_test_diff_percents_series,label='test',bins=bins,density=True,alpha=0.7)
            plt.ylabel('pdf (normalized to area=1)')
        elif hist_mode=='scaled':
            scaled_hist(self.y_train_diff_percents_series.values,label='train',bins=bins)
            scaled_hist(self.y_test_diff_percents_series.values,label='test',bins=bins,opacity=0.7)
            plt.ylabel('counts / max(counts)')
        plt.grid(True,which='major',axis='both')
        legend_styler()
        plt.xlabel('% differences')
        
def scaled_hist(np_array,label,bins,opacity=1):
    """created on 2018/09/08 to create a histogram with bar heights=counts/max(counts),
        since matplotlib 2.2.3 AND numpy >1.6 normed=True option for hist is deprecated and does the same as density=True,
        making a probability density histogram, normalizing the area below the graph to be unity
    
    based on the discussion on:
    https://stackoverflow.com/questions/3866520/plotting-histograms-whose-bar-heights-sum-to-1-in-matplotlib
    """
    heights,edges = np.histogram(np_array,bins=bins)
    binWidth=edges[1]-edges[0]
    plt.bar(edges[:-1],heights/np.max(heights),binWidth,label=label,alpha=opacity)

def create_target_diff_percents(y_train_series,y_test_series,mode='continuous test'):
    """created on 2018/09/16
    returns: (y_train_diff_percents_series,y_test_diff_percents_series)
    
    y_train_diff_percents_series.values=
        100*(y_train_series.values[1:]/y_train_series.values[0:-1]-1)
    mode -
        'non-continuous test': not assuming test-train continuity, 
            therefore calculating differences only for y_test[1:].
        'continuous test': assuming test-train continuity, therefore 
            calculating differences also for y_test[0] by using y_train[-1]!
    """
    y_train_diff_percents_series=pd.Series(
        100*(y_train_series.values[1:]/y_train_series.values[0:-1]-1),
        index=y_train_series.index[1:],
        name=y_train_series.name+' % differences')
    if mode=='non-continuous test':
        logger.warn('not assuming test-train continuity, therefore calculating differences only for y_test[1:]')
        y_test_diff_percents_series=pd.Series(
                100*(y_test_series.values[1:]/y_test_series.values[0:-1]-1),
                index=y_test_series.index[1:],
                name=y_test_series.name+' % differences')
    elif mode=='continuous test':
        logger.warn('assuming test-train continuity, therefore calculating differences also for y_test[0] by using y_train[-1]!')
        y_test_diff_percents_array=0*y_test_series.values
        y_test_diff_percents_array[0]=100*(y_test_series.values[0]/y_train_series.values[-1]-1)
        y_test_diff_percents_array[1:]=100*(y_test_series.values[1:]/y_test_series.values[0:-1]-1)
        y_test_diff_percents_series=pd.Series(y_test_diff_percents_array,
                index=y_test_series.index,
                name=y_test_series.name+' % differences')
    else:
        raise RuntimeError('unsupported mode!')
    return (y_train_diff_percents_series,y_test_diff_percents_series)

def scaled_hist(np_array,label,bins,opacity=1):
    """created on 2018/09/08 to create a histogram with bar heights=counts/max(counts),
        since matplotlib 2.2.3 AND numpy >1.6 normed=True option for hist is deprecated and does the same as density=True,
        making a probability density histogram, normalizing the area below the graph to be unity
    
    based on the discussion on:
    https://stackoverflow.com/questions/3866520/plotting-histograms-whose-bar-heights-sum-to-1-in-matplotlib
    """
    heights,edges = np.histogram(np_array,bins=bins)
    binWidth=edges[1]-edges[0]
    plt.bar(edges[:-1],heights/np.max(heights),binWidth,label=label,alpha=opacity)


def create_target_diff_percents(y_train_series,y_test_series,mode='continuous test'):
    """created on 2018/09/16
    returns: (y_train_diff_percents_series,y_test_diff_percents_series)
    
    y_train_diff_percents_series.values=
        100*(y_train_series.values[1:]/y_train_series.values[0:-1]-1)
    mode -
        'non-continuous test': not assuming test-train continuity, 
            therefore calculating differences only for y_test[1:].
        'continuous test': assuming test-train continuity, therefore 
            calculating differences also for y_test[0] by using y_train[-1]!
    """
    y_train_diff_percents_series=pd.Series(
        100*(y_train_series.values[1:]/y_train_series.values[0:-1]-1),
        index=y_train_series.index[1:],
        name=y_train_series.name+' % differences')
    if mode=='non-continuous test':
        logger.warn('not assuming test-train continuity, therefore calculating differences only for y_test[1:]. Notice to use only X_test_df.iloc[1:,:] for learning')
        y_test_diff_percents_series=pd.Series(
                100*(y_test_series.values[1:]/y_test_series.values[0:-1]-1),
                index=y_test_series.index[1:],
                name=y_test_series.name+' % differences')
    elif mode=='continuous test':
        logger.warn('assuming test-train continuity, therefore calculating differences also for y_test[0] by using y_train[-1]!')
        y_test_diff_percents_array=0*y_test_series.values
        y_test_diff_percents_array[0]=100*(y_test_series.values[0]/y_train_series.values[-1]-1)
        y_test_diff_percents_array[1:]=100*(y_test_series.values[1:]/y_test_series.values[0:-1]-1)
        y_test_diff_percents_series=pd.Series(y_test_diff_percents_array,
                index=y_test_series.index,
                name=y_test_series.name+' % differences')
    else:
        raise RuntimeError('unsupported mode!')
    return (y_train_diff_percents_series,y_test_diff_percents_series)


class regression:
    """v1 created on 2018/08/31 based on FinancialML 9.0
    
    Update log:
        v2 (OzML_v4)- 2018/09/16: takes into consideration processing_mode 
            that must be passed as input!
    """
    def __init__(self,split_data_obj=0,split_data_tuple=0):
        """can recieve data (y_train_series,X_train_df,
                y_test_series,X_test_df,processing_mode) in two ways:
            A) split_data_obj that contains the data as attributes, or
            B) split_data_tuple that contains the data in a tuple.
        """
        if split_data_tuple==0:
            self.y_train_series=split_data_obj.y_train_series
            self.X_train_df=split_data_obj.X_train_df
            self.y_test_series=split_data_obj.y_test_series
            self.X_test_df=split_data_obj.X_test_df
            
            self.processing_mode=split_data_obj.processing_mode
            self.y_train_diff_percents_series=split_data_obj.y_train_diff_percents_series
            self.y_test_diff_percents_series=split_data_obj.y_test_diff_percents_series
        elif split_data_obj==0:
            (self.y_train_series,self.X_train_df,self.y_test_series,
                 self.X_test_df,self.processing_mode)=split_data_tuple
            if self.processing_mode=='% differences':
                self.y_train_diff_percents_series=self.y_train_series
                self.y_test_diff_percents_series=self.y_test_series
            else:
                (self.y_train_diff_percents_series,
                 self.y_test_diff_percents_series)=create_target_diff_percents(
                         self.y_train_series,self.y_test_series)
        else:
            raise RuntimeError('both passed split_data_obj,split_data_tuple are zero, at least one should not be zero and contain data!')      
    
    def predict(self):
        """flattening regressor.predict output for comparibility with TF model prediction output, 
            which is a numpy (data_size,1) array, instead of the required (data_size,) array as sklearn outputs!
        """
        self.y_train_pred_series=pd.Series(
                self.regressor.predict(self.X_train_df.values).flatten(),
                name=self.y_train_series.name+' predicted',
                index=self.y_train_series.index)
        self.y_test_pred_series=pd.Series(
                self.regressor.predict(self.X_test_df.values).flatten(),
                name=self.y_test_series.name+' predicted',
                index=self.y_test_series.index)
        
        # calculating errors here to allow using regression.position_deciding() independently of regression.analyze()
        self.train_err_series=self.y_train_pred_series-self.y_train_series
        self.train_err_series.name='train error = predicted-target'
        self.test_err_series=self.y_test_pred_series-self.y_test_series
        self.test_err_series.name='test error = predicted-target'
        
        self.train_err_mean=np.mean(self.train_err_series.values)
        self.train_err_std=np.std(self.train_err_series.values)
        self.train_MSE=np.mean(self.train_err_series.values**2)
        self.test_err_mean=np.mean(self.test_err_series.values)
        self.test_err_std=np.std(self.test_err_series.values)
        self.test_MSE=np.mean(self.test_err_series.values**2)
        
        logging.info('notice: error = prediction-target, MSE=mean(error^2)!=var(error)=mean((error-mean(error))^2)')
        logger.info('TRAIN error: mean=%.3f, std=%.3f; sqrt(MSE)=%.3f'%(
                self.train_err_mean,self.train_err_std,self.train_MSE**0.5))
        logger.info('TEST error: mean=%.3f, std=%.3f; sqrt(MSE)=%.3f'%(
                self.test_err_mean,self.test_err_std,self.test_MSE**0.5))
                
    def plot_prediction(self,thick_linewidth=3,thin_linewidth=1,markersize=3,
                        rolling_window=50):
        """
        rolling window - window length (steps) to calculate and plot the rolling 
            (%d step) error std for test and train periods, plotted only if >0.
        """
        plt.figure()
        ax1=plt.subplot(211)
        plt.plot(self.y_train_series,'b.-',label='target (train)',
                 linewidth=thick_linewidth,markersize=markersize)
        plt.plot(self.y_train_pred_series,'c.--',label='predicted target (train)',
                 linewidth=thin_linewidth,markersize=markersize)
        plt.plot(self.y_test_series,'k.-',label='target (test)',
                 linewidth=thick_linewidth,markersize=markersize)
        plt.plot(self.y_test_pred_series,'r.--',label='predicted target (test)',
                 linewidth=thin_linewidth,markersize=markersize)
        
        plt.grid(True,which='major',axis='both')
        legend_styler()
#        plt.ylabel(self.y_test_series.name)
        if self.processing_mode=='raw':
            plt.ylabel('$')
        elif self.processing_mode=='% differences':
            plt.ylabel('%')
        
        ax2=plt.subplot(212,sharex=ax1)
        plt.plot(self.train_err_series,'c.-',label='train error',
                 linewidth=thin_linewidth,markersize=markersize)
        plt.plot(self.test_err_series,'r.-',label='test error',
                 linewidth=thin_linewidth,markersize=markersize)
                
        if rolling_window>0:
            train_err_rolling_std_series=self.train_err_series.rolling(window=rolling_window).std()
            test_err_rolling_std_series=self.test_err_series.rolling(window=rolling_window).std()
            plt.plot(train_err_rolling_std_series,'b-',linewidth=thick_linewidth,
                 label='train error rolling std (%d steps)'%rolling_window)
            plt.plot(test_err_rolling_std_series,'k-',linewidth=thick_linewidth,
                 label='train error rolling std (%d steps)'%rolling_window)
            
        legend_styler(loc='lower left')
        
        # plotting unlabaled:
        if rolling_window>0:
            plt.plot(-train_err_rolling_std_series,'b-',linewidth=thick_linewidth)
            plt.plot(-test_err_rolling_std_series,'k-',linewidth=thick_linewidth)
        
        plt.plot(0*pd.concat([self.train_err_series,self.test_err_series]),
                     'k--',linewidth=thin_linewidth)
        
        plt.grid(True,which='major',axis='both')
#        plt.ylabel('error (prediction-target)')
        if self.processing_mode=='raw':
            plt.ylabel('$')
        elif self.processing_mode=='% differences':
            plt.ylabel('%')
        
        return ax1,ax2
    
    def plot_error_hist(self,bins_number=200):
        plt.figure()
        plt.subplot(211)
        plt.hist(self.train_err_series,bins=bins_number)
        plt.grid(True,which='major',axis='both')
        plt.ylabel('counts')
        plt.title('TRAIN error: mean= %.4f, std= %.4f'%(self.train_err_mean,self.train_err_std))
        
        plt.subplot(212)
        plt.hist(self.test_err_series,bins=bins_number)
        plt.grid(True,which='major',axis='both')
        plt.xlabel('error (prediction-target)')
        plt.ylabel('counts')
        plt.title('TEST error: mean= %.4f, std= %.4f'%(self.test_err_mean,self.test_err_std))
        
    def decide_position(self,pred_confidence_level):
        """created on 2018/08/31 based on FinancialML 9.0
        
        creates and returns self.pred_test_position_series,
        translated from the prediction (y_test_pred_series):
            y_test_pred_series>long_entry_level -> +1 (long)
            y_test_pred_series<short_entry_level -> -1 (short)
            else: 0 (neutral),
        where:
            long_entry_level=pred_confidence_level
            short_entry_level=-pred_confidence_level
        """
        long_entry_level=pred_confidence_level
        short_entry_level=-pred_confidence_level
        if short_entry_level>long_entry_level:
            raise Exception('short_entry_level>long_entry_level')
        
        if self.processing_mode=='% differences': # predictions are in % differences
            self.y_train_pred_diff_percents_series=self.y_train_pred_series
            self.y_test_pred_diff_percents_series=self.y_test_pred_series
        else:
            (self.y_train_pred_diff_percents_series,
                 self.y_test_pred_diff_percents_series)=create_target_diff_percents(
                         self.y_train_pred_series,self.y_test_pred_series)
            
        self.pred_test_position_series=0*self.y_test_pred_diff_percents_series
        self.pred_test_position_series.name='predicted position'
        self.pred_test_position_series[
                self.y_test_pred_diff_percents_series>long_entry_level]=1
        self.pred_test_position_series[
                self.y_test_pred_diff_percents_series<short_entry_level]=-1
        
        return self.pred_test_position_series

class regression_shallow(regression):
    def train_skLinearRegression(self):     
        # http://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html
        self.regressor=linear_model.LinearRegression()
        logger.info('sklearn LinearRegression training started')
        tic=time()
        self.regressor.fit(self.X_train_df.values,self.y_train_series.values)
        toc=time()
        logger.info('training completed in %.3f seconds'%(toc-tic))
    
    def train_skElasticNet(self,l1_ratio,alpha):
        """created on 2018/09/04
        Minimizes the objective function:
            1/ (2 * n_samples) * ||y - Xw||^2_2
                + alpha * l1_ratio * ||w||_1
                + 0.5 * alpha * (1 - l1_ratio) * ||w||^2_2
        Notice: the regularization penalty (a*L1 + b*L2) is 
            equivalent to alpha=a+b and l1_ratio=a/(a + b)

        l1_ratio (= rho in the documentation):
            L2 penalty only (Ridge) = 0 <= l1_ratio <= 1 = L1 penalty only (Lasso)
        alpha>0 multiplies both penalties, 
            alpha=0 is theoretically equivalent to un-regularized linear regression, 
            but not numerically, it is advised to use LinearRegression instead of ElasticNet with alpha=0
        
        http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html
        http://scikit-learn.org/stable/modules/linear_model.html#elastic-net
        """
        self.regressor=linear_model.ElasticNet(l1_ratio=l1_ratio,alpha=alpha)
        logger.info('sklearn ElasticNet regression training started')
        tic=time()
        self.regressor.fit(self.X_train_df.values,self.y_train_series.values)
        toc=time()
        logger.info('training completed in %.3f seconds'%(toc-tic))
        if np.array_equal(self.regressor.coef_,np.zeros(np.shape(self.regressor.coef_))):
            logger.warn('over-regularization: coefficients are all zero!')
                
    def train_skElasticNetCV(self,CV_obj,l1_ratios,alphas,
            plotting='pcolormesh',plot_log_alpha=True,plot_log_l1_ratio=False):
        """created on 2018/09/04
        calculates the MSE on each combinarion of l1_ratios, alphas and cross-validation period,
            mse_path_array=self.regressor.mse_path_ # shape = (len(l1_ratio),len(alphas),cv_splits_number),
        then average over all cross validation periods (if any),
            mse_path_array_cv_averaged=mse_path_array.mean(axis=2) # shape = (len(l1_ratio),len(alphas)),
        
        plotting (mse_path_array_cv_averaged) -
            'none': no plotting
            'pcolormesh': plotting as image
            'contour areas': filled contours
            'contours': contour lines
        http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNetCV.html#sklearn.linear_model.ElasticNetCV
        """
        self.regressor=linear_model.ElasticNetCV(cv=CV_obj.iterator(),
                                            l1_ratio=l1_ratios,alphas=alphas)
        logger.info('sklearn ElasticNetCV regression training started')
        tic=time()
        self.regressor.fit(self.X_train_df.values,self.y_train_series.values)
        toc=time()
        logger.info('training completed in %.3f seconds'%(toc-tic))
        logger.info('optimal hyper-parameteres over cv and regularization path: l1_ratio=%.4f, alpha=%.4f'%(
                self.regressor.l1_ratio_,self.regressor.alpha_))
        
        mse_path_array=self.regressor.mse_path_ # shape = (len(l1_ratio),len(alphas),cv_splits_number)
        if CV_obj.splits_number>1:
            mse_path_array_cv_averaged=mse_path_array.mean(axis=2) # shape = (len(l1_ratio),len(alphas))
        else:
            mse_path_array_cv_averaged=mse_path_array
        regressor_alphas=self.regressor.alphas_ # ElasticNetCV sorts alphas input in descending order...
        
        """verifying that I understand how to make the minimization that 
            ElasticNetCV is doing to get to the optimal parameters,
            and that I plot it right:
            
        opt_mse_l1_ind=np.argmin(mse_path_array_cv_averaged,axis=1)
        opt_mse_alpha_ind=np.argmin(mse_path_array_cv_averaged,axis=0)
        logger.info('optimal hyper-parameteres over cv and regularization path: l1_ratio=%.4f, alpha=%.4f'%(
                l1_ratios[opt_mse_l1_ind[0]],regressor_alphas[opt_mse_alpha_ind[0]]))
        """


        if plotting!='none':
            fig = plt.figure()
            plot_2D_array(l1_ratios,regressor_alphas,mse_path_array_cv_averaged.T,
                          fig_handle=fig,mode=plotting)
            ax = fig.gca()
            plt.xlabel('l1_ratios')
            plt.ylabel('alphas')
            if plot_log_alpha:
                ax.set_yscale('log')
            if plot_log_l1_ratio:
                ax.set_xscale('log')
            plt.title('ElasticNetCV MSE on regularization path after cv averaging')
            
            
        if np.array_equal(self.regressor.coef_,np.zeros(np.shape(self.regressor.coef_))):
            logger.warn('over-regularization for the optimal coefficients, they are all zero!')
    
    def train_skDecisionTreeRegressor(self,plot_tree=True,max_depth=10):
        """created on 2018/09/04
        http://scikit-learn.org/stable/modules/tree.html
        http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html
        """
        from sklearn import tree
        
        self.regressor=tree.DecisionTreeRegressor(max_depth=max_depth,random_state=1)
        logger.info('sklearn DecisionTreeRegressor training started')
        tic=time()
        self.regressor.fit(self.X_train_df.values,self.y_train_series.values)
        toc=time()
        logger.info('training completed in %.3f seconds'%(toc-tic))
        
        if plot_tree:
            try:
                import graphviz
            except:
                raise RuntimeError("failed to import graphviz, must be installed to plot decision trees. On conda execute 'conda install python-graphviz' (NOT 'conda install graphviz')")
            
            dot_data=tree.export_graphviz(self.regressor,out_file=None,
                feature_names=self.X_train_df.columns,
                filled=True,rounded=True,proportion=True)
            graph = graphviz.Source(dot_data) 
            graph.render(datetime.strftime(datetime.now(),filenames_time_format)+' decision tree')
            graph.view()
            
    def visualize_coeff(self,max_coeff_to_show=0,attribute_names_list=0,
                        normalization=True,bottom_margin=0.3):
        """created on 2018/09/06 to visualize fitted model coefficients for each feature (plotted as bars)
        max_coeff_to_show -
            =0: showing all coefficients
            >0: showing only the max_coeff_to_show coefficients with the highest absolute value. 
        bottom_margin should be increased if xticks are out of the canvas
        normalization - by max(abs(coeff))
        attribute_names_list - optional, using this as x labels instead of 
            X_train_df.columns if possible (if the number of features is 
            the number of raw attributes)
        """
        if normalization:
            coeff_height=self.regressor.coef_/np.max(abs(self.regressor.coef_))
            ylabel='normalized coefficients'
        else:
            coeff_height=self.regressor.coef_
            ylabel='coefficients'
        
        if attribute_names_list==0:
            col_names=list(self.X_train_df.columns)
        elif len(attribute_names_list)==len(self.X_train_df.columns):
            col_names=attribute_names_list
        else:
            raise RuntimeError('cannot use attribute_names_list as x labels since the number of features does not match the number of raw attributes (set to default=0 to solve: use feature names)!')
            
        if max_coeff_to_show>0:
            coeff_height_sorted_ind=np.argsort(-abs(coeff_height))
            coeff_height_sorted_ind=coeff_height_sorted_ind[:max_coeff_to_show]
            coeff_height=coeff_height[coeff_height_sorted_ind]
            col_names=[col_names[col] for col in coeff_height_sorted_ind]
            title='%d coefficients with the highest absolute value'%max_coeff_to_show
        else:
            title='coefficient values'
        
        plt.figure()
        
#        plt.bar(col_names,coeff_height) # in Matplotlib 2.1 this line orders the bars by alphabetically, which is stupid and not my intention, therefore: 
        plt.bar(range(len(col_names)),coeff_height) # for compatibility with Matplotlib 2.1
        plt.xticks(range(len(col_names)),col_names) # for compatibility with Matplotlib 2.1
        
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True,which='major',axis='both')
        plt.xticks(rotation='vertical')
        plt.gcf().subplots_adjust(bottom=bottom_margin)

class regression_deep(regression):
    def train_NN(self,model=None,optimizer=None,learning_rate=0.01,
                 max_epochs=100,batch_size=100,shuffle=False,validation_split=0.25,
                 training_patience=10,min_dloss_to_stop=0.01,verbose=1):
        """created on 2018/09/07
        regression example: http://www.tensorflow.org/tutorials/keras/basic_regression
        
        Update Log:
            v2 (OzML_v4)- 2018/10/04: added parameters, changed to allow 
                building the model and setting the optimizer externally and 
                pass it to here.
        
        https://keras.io/layers/core/
            layers:
                keras.layers.Dropout(rate, noise_shape=None, seed=None)
                    rate: float between 0 and 1. Fraction of the input units to drop.
                    seed: A Python integer to use as random seed.
                    noise_shape: 1D integer tensor representing the shape of the binary dropout mask that will be multiplied with the input. For instance, if your inputs have shape  (batch_size, timesteps, features) and you want the dropout mask to be the same for all timesteps, you can use noise_shape=(batch_size, 1, features).
        
        http://keras.io/layers/core/#activation
            activations: 'tanh','relu','sigmoid','linear'
        
        http://keras.io/regularizers/
            available penalties:
                keras.regularizers.l1(0.)
                keras.regularizers.l2(0.)
                keras.regularizers.l1_l2(l1=0.01, l2=0.01)
        
        http://www.tensorflow.org/api_docs/python/tf/keras/Model#fit            
            batch_size: (integer, default: 32) number of samples per gradient update. If unspecified, batch_size will default to 32. Do not specify the batch_size if your data is in the form of symbolic tensors, datasets, or dataset iterators (since they generate batches).
            steps_per_epoch: (integer or None) total number of steps (batches of samples) before declaring one epoch finished and starting the next epoch. When training with input tensors such as TensorFlow data tensors, the default None is equal to the number of samples in your dataset divided by the batch size, or 1 if that cannot be determined.
            callbacks: stops training when a monitored quantity has stopped improving. 
                documentation: http://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping    
                inputs:
                    min_delta: minimum change in the monitored quantity to qualify as an improvement, i.e. an absolute change of less than min_delta, will count as no improvement.
                    patience: number of epochs with no improvement after which training will be stopped.
                    mode: one of {auto, min, max}. In min mode, training will stop when the quantity monitored has stopped decreasing; in max mode it will stop when the quantity monitored has stopped increasing; in auto mode, the direction is automatically inferred from the name of the monitored quantity.
            validation_split: Float between 0 and 1. 
                 Fraction of the training data to be used as validation data. 
                 The model will set apart this fraction of the training data, 
                 will not train on it, and will evaluate the loss and any model metrics 
                 on this data at the end of each epoch. 
                 The validation data is selected from the last samples in the x and y data provided, 
                 before shuffling. This argument is not supported when x is a dataset or a dataset iterator.   
            shuffle: Boolean (whether to shuffle the training data before each epoch) or str (for 'batch'). 'batch' is a special option for dealing with the limitations of HDF5 data; it shuffles in batch-sized chunks. Has no effect when steps_per_epoch is not None.
            verbose: Integer. 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
        """
        import tensorflow as tf
        from tensorflow import keras
        
        if model==None:
            model=keras.Sequential()
            model.add(keras.layers.Dense(50,input_shape=(len(self.X_train_df.columns),),activation='relu'))
    #        model.add(keras.layers.Dense(10, activation='tanh',
    #                                      kernel_regularizer=keras.regularizers.l1(0.1)))
            model.add(keras.layers.Dense(50, activation='tanh'))
            model.add(keras.layers.Dense(50, activation='relu'))
            model.add(keras.layers.Dense(50, activation='relu'))
            model.add(keras.layers.Dense(1))
    #        model.add(keras.layers.Dense(1,kernel_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01)))
        
        if optimizer==None:
            optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
            #        optimizer=tf.train.RMSPropOptimizer(learning_rate=learning_rate)
        
        model.compile(optimizer=optimizer,
#                loss=keras.losses.MSE,
#                metrics=[keras.metrics.MAE,keras.metrics.MSE])
                loss='mean_squared_error',
                metrics=['mean_squared_error'])
        callbk=keras.callbacks.EarlyStopping(monitor='val_loss', # can stop also on monitor='loss'
                    patience=training_patience,min_delta=min_dloss_to_stop)
        
        logger.info('NN training started on TF')
        tic=time()
        self.history=model.fit(x=self.X_train_df.values,y=self.y_train_series.values,
                          batch_size=batch_size,epochs=max_epochs,
                          validation_split=validation_split,shuffle=shuffle,
                          callbacks=[callbk],verbose=verbose)
        toc=time()
        logger.info('training completed in %.3f seconds, saved model into self.regressor (and history into self.history)'%(toc-tic))
        self.regressor=model
        
        logger.info('final sqrt(MSE): (train,validation) = (%.2f,%.2f)'%(
            self.history.history['mean_squared_error'][-1]**0.5,
            self.history.history['val_mean_squared_error'][-1]**0.5))
        
    def plot_training(self):
        plt.figure()
        plt.subplot(2,1,1)
        train_loss=np.array(self.history.history['loss'])
        val_loss=np.array(self.history.history['val_loss'])
        plt.plot(self.history.epoch,train_loss**0.5,label='train')
        plt.plot(self.history.epoch,val_loss**0.5,label='validation')
        legend_styler()
        plt.ylabel('sqrt(loss)')
        plt.title('loss=MSE+regularization')
        plt.grid(True,which='major',axis='both')
        
        plt.subplot(2,1,2)
        train_MSE=np.array(self.history.history['mean_squared_error']) # if metrics contains 'mean_squared_error'
        val_MSE=np.array(self.history.history['val_mean_squared_error']) # if using validation and metrics contains 'mean_squared_error'
        plt.plot(self.history.epoch,train_MSE**0.5,label='train')
        plt.plot(self.history.epoch,val_MSE**0.5,label='validation')
        legend_styler()
        plt.xlabel('epoch (completing batch updates on all training samples)')
        plt.ylabel('sqrt(MSE)')
        plt.title('MSE')
        plt.grid(True,which='major',axis='both')
        
        plt.show()
        
class classification:
    """v1 created on 2018/08/31 based on FinancialML 9.0
    
    Update log:
        v2 (OzML_v4)- 2018/09/17: takes into consideration processing_mode 
            that must be passed as input!
        v2 (OzML_v4)- 2018/09/30: improved analysis and plotting,
            took predict() method to the classification_shallow and 
            classification_deep subclasses (since they are a bit different...)
    """
    def __init__(self,split_data_obj=0,split_data_tuple=0,class_confidence_level=0):
        """can recieve data (y_train_series,X_train_df,
                y_test_series,X_test_df,processing_mode) in two ways:
            A) split_data_obj that contains the data as attributes, or
            B) split_data_tuple that contains the data in a tuple.
        """
        if split_data_tuple==0:
#            self.y_train_series=split_data_obj.y_train_series # not used in learning, instead using y_train_diff_percents_series
            self.X_train_df=split_data_obj.X_train_df
#            self.y_test_series=split_data_obj.y_test_series # not used in learning, instead using y_test_diff_percents_series
            self.X_test_df=split_data_obj.X_test_df
            
            self.processing_mode=split_data_obj.processing_mode
            self.y_train_diff_percents_series=split_data_obj.y_train_diff_percents_series
            self.y_test_diff_percents_series=split_data_obj.y_test_diff_percents_series
        elif split_data_obj==0:
            (y_train_series,self.X_train_df,y_test_series,
                 self.X_test_df,self.processing_mode)=split_data_tuple
            if self.processing_mode=='% differences':
                self.y_train_diff_percents_series=y_train_series
                self.y_test_diff_percents_series=y_test_series
            else:
                (self.y_train_diff_percents_series,
                 self.y_test_diff_percents_series)=create_target_diff_percents(
                         y_train_series,y_test_series)
        else:
            raise RuntimeError('both passed split_data_obj,split_data_tuple are zero, at least one should not be zero and contain data!')      
    
        if self.processing_mode!='% differences':
            self.X_train_df=self.X_train_df.iloc[1:,:]
            logger.warn('using X_train_df.iloc[1:,:] and not X_train_df, since predicting % differences, y_train_diff_percents_series=100*(y_train_series[1:]/y_train_series[0:-1]-1) starts from the second index of y_train_series and X_train_df')
        
    
        # defining target classes
        long_class_level=class_confidence_level
        short_class_level=-class_confidence_level
        
        self.y_train_classes_series=0*self.y_train_diff_percents_series
        self.y_train_classes_series.name=self.y_train_diff_percents_series.name+' classes'
        self.y_train_classes_series[self.y_train_diff_percents_series>long_class_level]=1
        self.y_train_classes_series[self.y_train_diff_percents_series<short_class_level]=-1
        
        self.y_test_classes_series=0*self.y_test_diff_percents_series
        self.y_test_classes_series.name=self.y_test_diff_percents_series.name+' classes'
        self.y_test_classes_series[self.y_test_diff_percents_series>long_class_level]=1
        self.y_test_classes_series[self.y_test_diff_percents_series<short_class_level]=-1
        
    def analyze(self):        
        train_accuracy=np.mean(self.y_train_classes_pred_series==self.y_train_classes_series)
        test_accuracy=np.mean(self.y_test_classes_pred_series==self.y_test_classes_series)
        logger.info('TRAIN accuracy, mean(predicted class == target class): %0.1f%%'%(100*train_accuracy))
        logger.info('TEST accuracy, mean(predicted class == target class): %0.1f%%'%(100*test_accuracy))  
        logger.info('compare with random accuracy = 1/class_number = %0.1f%%'%(100/len(self.classes_list)))
        
        # calculating train_positives_df here so it can be used in decide_position
        train_positives_array=np.zeros((len(self.y_train_classes_pred_series),len(self.classes_list)))
        train_positives_df_columns=[]
        test_positives_array=np.zeros((len(self.y_test_classes_pred_series),len(self.classes_list)))
        test_positives_df_columns=[]
        
        for cls in range(len(self.classes_list)):
            cls_value=self.classes_list[cls]
            train_positives_array[(self.y_train_classes_pred_series==cls_value) & (self.y_train_classes_series==cls_value),cls]=1 # true positive for class cls_value
            train_positives_array[(self.y_train_classes_pred_series==cls_value) & (self.y_train_classes_series!=cls_value),cls]=-1 # false positive for class cls_value
            train_positives_df_columns+=['train true/false pos: class %d'%cls_value]
            
            test_positives_array[(self.y_test_classes_pred_series==cls_value) & (self.y_test_classes_series==cls_value),cls]=1 # true positive for class cls_value
            test_positives_array[(self.y_test_classes_pred_series==cls_value) & (self.y_test_classes_series!=cls_value),cls]=-1 # false positive for class cls_value
            test_positives_df_columns+=['test true/false pos: class %d'%cls_value]
        
        self.train_positives_df=pd.DataFrame(train_positives_array,
                index=self.y_train_classes_pred_series.index,
                columns=train_positives_df_columns)
        self.test_positives_df=pd.DataFrame(test_positives_array,
                index=self.y_test_classes_pred_series.index,
                columns=test_positives_df_columns)
        
        train_precision=np.sum(train_positives_array==1,axis=0)/(np.sum(train_positives_array==1,axis=0)+np.sum(train_positives_array==-1,axis=0))
        test_precision=np.sum(test_positives_array==1,axis=0)/(np.sum(test_positives_array==1,axis=0)+np.sum(test_positives_array==-1,axis=0))
        
        self.precision_df=pd.DataFrame(np.array([train_precision,test_precision]).T,
                index=self.classes_list,
                columns=['train precision','test precision'])
        self.precision_df.index.name='class'

    def plot_prediction(self):
        train_positives_df_no_zeros=self.train_positives_df.copy()
        train_positives_df_no_zeros[train_positives_df_no_zeros==0]=np.nan
        test_positives_df_no_zeros=self.test_positives_df.copy()
        test_positives_df_no_zeros[test_positives_df_no_zeros==0]=np.nan
        
        plt.figure()
        ax=plt.subplot(411)
        plt.plot(self.y_train_classes_series,'bo',label='target (train)')
        plt.plot(self.y_train_classes_pred_series,'c.',label='predicted target (train)')
        plt.plot(self.y_test_classes_series,'ko',label='target (test)')
        plt.plot(self.y_test_classes_pred_series,'r.',label='predicted target (test)')
        
        plt.grid(True,which='major',axis='both')
        legend_styler()
#        plt.ylabel(self.y_train_classes_series.name)
        plt.ylabel('class')
        
        for cls in range(self.classes_number):
            plt.subplot(4,1,cls+2,sharex=ax)
            plt.plot(train_positives_df_no_zeros.iloc[:,cls],'c.',label='train')
            plt.plot(test_positives_df_no_zeros.iloc[:,cls],'r.',label='test')
        
            plt.grid(True,which='major',axis='both')
            plt.ylabel('class %d\npositives'%self.classes_list[cls])
            plt.yticks([1,-1], ('True','False'))
            legend_styler()
        
        return ax
        
    def plot_probabilities(self,bins_number=200):
        train_positives_array=self.train_positives_df.values
        test_positives_array=self.test_positives_df.values
        
        train_positives_prob_df=0*self.y_train_classes_pred_prob_df
        train_positives_prob_df[:]=np.nan
        train_positives_prob_df[train_positives_array==1]=self.y_train_classes_pred_prob_df[train_positives_array==1]
        train_positives_prob_df[train_positives_array==-1]=-self.y_train_classes_pred_prob_df[train_positives_array==-1] # wrong prediction probabilities are taken as negative
        train_positives_prob_df.columns=['class %d positives prob.'%cls for cls in self.classes_list]
        
        test_positives_prob_df=0*self.y_test_classes_pred_prob_df
        test_positives_prob_df[:]=np.nan
        test_positives_prob_df[test_positives_array==1]=self.y_test_classes_pred_prob_df[test_positives_array==1]
        test_positives_prob_df[test_positives_array==-1]=-self.y_test_classes_pred_prob_df[test_positives_array==-1] # wrong prediction probabilities are taken as negative
        test_positives_prob_df.columns=['class %d positives prob.'%cls for cls in self.classes_list]
            
        train_positives_prob_df.hist(bins=bins_number)
        plt.suptitle('train true(>0)/false(<0) positives prob. histogram',fontsize=12)
        
        test_positives_prob_df.hist(bins=bins_number)
        plt.suptitle('test true(>0)/false(<0) positives prob. histogram',fontsize=12)

    def decide_position(self,pred_mode='max pred_prob'):
        """created on 2018/08/31 based on FinancialML 9.0
        
        creates and returns self.pred_test_position_series translated from the prediction (y_test_pred_series).
        pred_mode=
            'debugging': checking that I know how to interpret the predicted probabilites (classifier.predict_proba output) into predicted resutls (classifier.predict output). returns my prediction
            'max pred_prob': pred_test_position_series=y_test_classes_pred_series, which is the output of classifier.predict - taken as the class with the max predicted probability
#           'max pred_prob/success_prob': the predicted classe is taken as the classe with the max predicted probability re-normalized by true_pos_df['train true pos prob']
        """
        if pred_mode=='debugging':
            y_test_numclasses_pred_prob_df=self.y_test_classes_pred_prob_df.copy()
            y_test_numclasses_pred_prob_df.columns=self.classes_list
            self.pred_test_position_series=y_test_numclasses_pred_prob_df.idxmax(axis=1)
            self.pred_test_position_series.name=self.y_test_classes_pred_series.name
            
            print('is my perdiction (pred_test_position_series) equivalent to classifier.predict output (y_test_classes_pred_series): ',
                  self.y_test_classes_pred_series.equals(self.pred_test_position_series))
        elif pred_mode=='max pred_prob':
            self.pred_test_position_series=self.y_test_classes_pred_series
        elif pred_mode=='max pred_prob/success_prob':
            if 0 in self.true_pos_df['train true pos prob'].values:
                raise RuntimeError("self.true_pos_df['train true pos prob'] contains zero, for this class max pred_prob/success_prob will be inf, avoid!")
            y_test_numclasses_pred_prob_renormalized_df=self.y_test_classes_pred_prob_df/self.true_pos_df['train true pos prob'].values.T    
            y_test_numclasses_pred_prob_renormalized_df.columns=[col+' / true positive probability' for col in self.y_test_classes_pred_prob_df.columns]
            self.pred_test_position_series=y_test_numclasses_pred_prob_renormalized_df.idxmax(axis=1)
            self.pred_test_position_series.name=self.y_test_classes_pred_series.name+' (renormalized by true pos prob)'
        else:
            raise RuntimeError('unsupported pred_mode!')
        
        return self.pred_test_position_series

class classification_shallow(classification):
    def predict(self):
        self.y_train_classes_pred_series=pd.Series(
                self.classifier.predict(self.X_train_df.values),
                name=self.y_train_classes_series.name+' predicted',
                index=self.y_train_classes_series.index)
        self.y_test_classes_pred_series=pd.Series(
                self.classifier.predict(self.X_test_df.values),
                name=self.y_test_classes_series.name+' predicted',
                index=self.y_test_classes_series.index)
        
        self.y_train_classes_pred_prob_df=pd.DataFrame(
                self.classifier.predict_proba(self.X_train_df.values),
                columns=['class %d predicted probability'%cls for cls in self.classes_list],
                index=self.y_train_classes_series.index)
        self.y_test_classes_pred_prob_df=pd.DataFrame(
                self.classifier.predict_proba(self.X_test_df.values),
                columns=['class %d predicted probability'%cls for cls in self.classes_list],
                index=self.y_test_classes_series.index)
    
    def train_skLogisticRegression(self,multi_class='multinomial',solver='lbfgs',
            penalty='l2',C=1e5,n_jobs=1,max_iter=1000):
        """
        penalty : str, ‘l1’ or ‘l2’, default: ‘l2’
            Used to specify the norm used in the penalization. 
            The ‘newton-cg’, ‘sag’ and ‘lbfgs’ solvers support only l2 penalties.

        solver : {‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’},
            default: ‘liblinear’ Algorithm to use in the optimization problem.
            
            For small datasets, ‘liblinear’ is a good choice, whereas ‘sag’ and
            ‘saga’ are faster for large ones.
            For multiclass problems, only ‘newton-cg’, ‘sag’, ‘saga’ and ‘lbfgs’
            handle multinomial loss; ‘liblinear’ is limited to one-versus-rest schemes.
            ‘newton-cg’, ‘lbfgs’ and ‘sag’ only handle L2 penalty, whereas
            ‘liblinear’ and ‘saga’ handle L1 penalty.
            Note that ‘sag’ and ‘saga’ fast convergence is only guaranteed on features
            with approximately the same scale. You can preprocess the data with a scaler from sklearn.preprocessing.
        
        http://scikit-learn.org/stable/auto_examples/linear_model/plot_iris_logistic.html
        """
        self.classifier = linear_model.LogisticRegression(C=C,
            multi_class=multi_class,solver=solver,penalty=penalty,
            random_state=1,n_jobs=n_jobs)
        logger.info('sklearn LogisticRegression training started')
        tic=time()
        self.classifier.fit(self.X_train_df.values,self.y_train_classes_series.values)
        toc=time()
        logger.info('training completed in %.3f seconds'%(toc-tic))
        self.classes_list=self.classifier.classes_
        self.classes_number=len(self.classes_list)
        
    def train_skDecisionTreeClassifier(self,plot_tree=True,max_depth=10,criterion='gini'):
        """created on 2018/09/04
        http://scikit-learn.org/stable/modules/tree.html
        http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        """
        from sklearn import tree
        
        self.classifier=tree.DecisionTreeClassifier(max_depth=max_depth,criterion=criterion,random_state=1)
        logger.info('sklearn DecisionTreeClassifier training started')
        tic=time()
        self.classifier.fit(self.X_train_df.values,self.y_train_classes_series.values)
        toc=time()
        logger.info('training completed in %.3f seconds'%(toc-tic))
        self.classes_list=self.classifier.classes_
        self.classes_number=len(self.classes_list)
        
        if plot_tree:
            try:
                import graphviz
            except:
                raise RuntimeError("failed to import graphviz, must be installed to plot decision trees. On conda execute 'conda install python-graphviz' (NOT 'conda install graphviz')")
            
            dot_data=tree.export_graphviz(self.classifier,out_file=None,
                feature_names=self.X_train_df.columns,
                class_names=['%d'%cls for cls in self.classifier.classes_],
                filled=True,rounded=True,proportion=True)
            graph = graphviz.Source(dot_data) 
            graph.render(datetime.strftime(datetime.now(),filenames_time_format)+' decision tree') 
            graph.view()

    def visualize_coeff(self,max_coeff_to_show=0,attribute_names_list=None,
                        normalization=True,mode='separate',bottom_margin=0.3):
        """created on 2018/09/07 to visualize fitted model coefficients for each feature (plotted as bars)
        mode -
            'separate': plots a figure for each class
            'together': plots all classes on the same figure
        max_coeff_to_show -
            =0: showing all coefficients
            >0: showing only the max_coeff_to_show coefficients with the highest absolute value. 
        bottom_margin should be increased if xticks are out of the canvas
        normalization - by max(abs(coeff))
        attribute_names_list - optional, using this as x labels instead of 
            X_train_df.columns if possible (if the number of features is 
            the number of raw attributes)
        """
        coeff_height_array=self.classifier.coef_
        if attribute_names_list==None:
            col_names=list(self.X_train_df.columns)
        elif len(attribute_names_list)==len(self.X_train_df.columns):
            col_names=attribute_names_list
        else:
            raise RuntimeError('cannot use attribute_names_list as x labels since the number of features does not match the number of raw attributes (set to default=0 to solve: use feature names)!')
            
        if mode=='together':
            plt.figure()
        
        if normalization:
            ylabel='normalized coefficients'
        else:
            ylabel='coefficients'
        if max_coeff_to_show>0:
            title='%d coefficients with the highest absolute value'%max_coeff_to_show
        else:
            title='coefficient values'
        
        for cls_idx in range(self.classes_number):
            if normalization:
                coeff_height=coeff_height_array[cls_idx,:]/np.max(abs(coeff_height_array[cls_idx,:]))
            else:
                coeff_height=coeff_height_array[cls_idx,:]
            
            if max_coeff_to_show>0:
                coeff_height_sorted_ind=np.argsort(-abs(coeff_height))
                coeff_height_sorted_ind=coeff_height_sorted_ind[:max_coeff_to_show]
                coeff_height=coeff_height[coeff_height_sorted_ind]
                col_names=[col_names[col] for col in coeff_height_sorted_ind]
            
            if mode=='together':
#                plt.bar(col_names,coeff_height) # # in Matplotlib 2.1 this line orders the bars by alphabetically, which is stupid and not my intention, therefore:
                plt.bar(range(len(col_names)),coeff_height) # for compatibility with Matplotlib 2.1
                plt.xticks(range(len(col_names)),col_names) # for compatibility with Matplotlib 2.1
            elif mode=='separate':
                plt.figure()
#                plt.bar(col_names,coeff_height) # # in Matplotlib 2.1 this line orders the bars by alphabetically, which is stupid and not my intention, therefore:
                plt.bar(range(len(col_names)),coeff_height) # for compatibility with Matplotlib 2.1
                plt.xticks(range(len(col_names)),col_names) # for compatibility with Matplotlib 2.1
                plt.ylabel(ylabel)
                plt.title('class %d '%(cls_idx)+title)
                plt.grid(True,which='major',axis='both')
                plt.xticks(rotation='vertical')
                plt.gcf().subplots_adjust(bottom=bottom_margin)
        
        if mode=='together':
            plt.ylabel(ylabel)
            plt.title(title)
            legend_styler(['class %d'%cls for cls in self.classes_list])
            plt.grid(True,which='major',axis='both')
            plt.xticks(rotation='vertical')
            plt.gcf().subplots_adjust(bottom=bottom_margin)

class trading:
    """v1 created on ~2018/08/23, defined as class to save all the trading 
        calculation results conveniently
    
    Update log:
        v2 (OzML_v3) - 2018/19/16: correcting: 
            each (non-unity) profit multiplication is not a trade,  merly a 
            change in the portfolio value. Continuous profit changes while 
            the position does not change are included in a single trade, which 
            ends only when the position changes, not on each (non-unity) 
            portfolio change...
        v3 (OzML_v4) - 2018/19/17: 
            * forcing to close realized portfolio on the last test date.
            * re-naming y_test_series to 
                y_test_diff_percents_series (since from OzML_v4 it is not 
                assumed that y_test_series is % differences)
            * moved the plotting outside of init, to its own method
    """
    def __init__(self,pred_test_position_series,y_test_diff_percents_series):
        y_test_multiplications_series=y_test_diff_percents_series/100+1
        
        portfolio_multiplications_series=0*y_test_multiplications_series+1
        portfolio_multiplications_series.name='portfolio multiplications'
        portfolio_multiplications_series[pred_test_position_series==1]=y_test_multiplications_series[pred_test_position_series==1] # profits for long positions
        portfolio_multiplications_series[pred_test_position_series==-1]=2-y_test_multiplications_series[pred_test_position_series==-1] # profits for short positions
        
        portfolio_changes_percents_series=100*(portfolio_multiplications_series-1)
        portfolio_changes_percents_series.name='portfolio % differences'
        self.portfolio_changes_percents_series=portfolio_changes_percents_series
        self.portfolio_changes_percents_std=portfolio_changes_percents_series.std()
        self.portfolio_changes_percents_mean=portfolio_changes_percents_series.mean()
        
        # calculating realtime and realized portfolio, and trades profit
        realtime_portfolio_array=0*portfolio_multiplications_series.values
        realtime_portfolio_array[0]=portfolio_multiplications_series.values[0]
        realized_portfolio_values_list=[realtime_portfolio_array[0]]
        realized_portfolio_dates_list=[portfolio_multiplications_series.index[0]]
        
        for idx in range(1,len(portfolio_multiplications_series)):
            realtime_portfolio_array[idx]=portfolio_multiplications_series.iloc[:idx+1].product()
            if pred_test_position_series.iloc[idx]!=pred_test_position_series.iloc[idx-1]:
                realized_portfolio_values_list.append(realtime_portfolio_array[idx])
                realized_portfolio_dates_list.append(portfolio_multiplications_series.index[idx])
        
        # forcing to close realized portfolio on the last test date
        if realized_portfolio_values_list[-1]!=realtime_portfolio_array[-1]: # meaning no trade closed on last date
            realized_portfolio_values_list.append(realtime_portfolio_array[-1])
            realized_portfolio_dates_list.append(portfolio_multiplications_series.index[-1])
        
        test_duration_years=timedelta.total_seconds(
                y_test_diff_percents_series.index[-1]-y_test_diff_percents_series.index[0])/(365*24*60*60)                
        self.total_portfolio_multiplication=realtime_portfolio_array[-1]
        self.total_portfolio_multiplication_annualized=self.total_portfolio_multiplication**(1/test_duration_years)
        
        self.realtime_portfolio_series=pd.Series(realtime_portfolio_array,
                                        name='realtime portfolio',
                                        index=portfolio_multiplications_series.index)
        realized_portfolio_array=np.array(realized_portfolio_values_list)
        self.realized_portfolio_series=pd.Series(realized_portfolio_array,
                name='realized portfolio',
                index=realized_portfolio_dates_list)
                
        self.trade_profits_percents_series=pd.Series(100*(
                realized_portfolio_array[1:]/realized_portfolio_array[0:-1]-1),
                name='trade profits %',
                index=realized_portfolio_dates_list[1:])

        logger.info('annualized portfolio multiplication on testing period: %.3f^(1/%.2f) = %.3f'% (
                self.total_portfolio_multiplication,test_duration_years,
                self.total_portfolio_multiplication_annualized))
        logger.info('real-time portfolio flactuations: (mean,std)=(%.2f,%.2f)%%'% (
                self.portfolio_changes_percents_mean,
                self.portfolio_changes_percents_std))
        logger.info('there were %d trades, with profit (mean,std)=(%.2f,%.2f)%%'%(
                len(self.trade_profits_percents_series),
                self.trade_profits_percents_series.mean(),
                self.trade_profits_percents_series.std()))
        
    def plot_portfolio(self):
            plt.figure()
            plt.plot(self.realtime_portfolio_series.index,
                     self.realtime_portfolio_series.values,label='real-time')
            plt.plot(self.realized_portfolio_series.index,
                     self.realized_portfolio_series.values,'k.',label='realized')
            plt.grid(True,which='major',axis='both')
            legend_styler()
            plt.ylabel('portfolio(t) / portfolio(t0)')

    def plot_trades(self):
            plt.figure()
            plt.hist(self.trade_profits_percents_series.values,bins=200)
            plt.grid(True,which='major',axis='both');
            plt.title('trades profit histogram\nmean= %.2f%%, std= %.2f%%'%(
                    self.trade_profits_percents_series.mean(),
                    self.trade_profits_percents_series.std()))
            plt.ylabel('counts')
            plt.xlabel('profit (%)')

class CV_split:
    """v1 created on 2018/09/04 (in OzML 1.0) 
        for my own timeseries cross validation splitting schemes, 
        avoiding the unuseful sklearn TimeSeriesSplit splitting scheme.
        
    Update log:
        v2 - 2018/09/08:
            +++ splitting original __init__ into different splitting methods, 
                each adding its splits to any existing splits - 
                therefore allowing methods to use others!
            ++ more efficient: instead of saving all indices arrays 
                (self.train_indices_arrays_list,self.validation_indices_arrays_list),
                now saving only train_start_indices_list,split_indices_list,validation_end_indices_list.
                The final train_indices,validation_indices arrays are made in the iterator:
                    train_indices_array=self.indices_array[self.train_start_indices_list[split]:self.split_indices_list[split]]
                    validation_indices_array=self.indices_array[self.split_indices_list[split]:self.validation_end_indices_list[split]]
            ++ moving plot out of __init__ to be a method using (and verifying)
                self.iterator directly.
        Tested v2 to reproduce v1 splits.
    """
    def __init__(self,y_series_to_split):
        """
        y_series_to_split is a pandas series, with a datetime index
        """
        self.y_series_to_split=y_series_to_split
        self.indices_array=np.arange(len(y_series_to_split))  
        self.train_start_indices_list=[]
        self.split_indices_list=[]
        self.validation_end_indices_list=[]
        self.splits_number=0
        
    def single_split(self,single_split_ratio):
        """created on 2018/09/08
        
        splitting self.y_series_to_split once to have validation_length/total_length=single_split_ratio
        """
        if single_split_ratio>=1 or single_split_ratio<=0:
            raise RuntimeError('invalid single_split_ratio=validation_length/total_length input! must be 0<single_split_ratio<1')
        else:
            train_length=round(len(self.y_series_to_split)*(1-single_split_ratio))
            
            self.train_start_indices_list.append(0)
            self.split_indices_list.append(train_length)
            self.validation_end_indices_list.append(len(self.y_series_to_split))
            
            self.splits_number=1
            logger.info('split into %d training-validation windows'%self.splits_number)
            
    def multi_split_constant_periods(self,train_duration_years,validation_duration_years,
                 shift_period_years):
        """created on 2018/09/08
        
        splitting self.y_series_to_split into training-validation periods with 
            constant lengths, then moving the training-validation window
            by a constant shift period until reaching the end 
            of self.y_series_to_split. 
        """
        # creating the first (zero) split
        train_start_datetime=self.y_series_to_split.index[0]
        splitting_datetime=train_start_datetime+timedelta(days=round(365*train_duration_years))
        validation_end_datetime=splitting_datetime+timedelta(days=round(365*validation_duration_years))
        if validation_end_datetime>self.y_series_to_split.index[-1]:
            raise RuntimeError('y_series_to_split is not long enough to allow a zeroth split, validation_end_datetime>y_series_to_split.index[-1]!')
        
        # creating all splits
        split=0
        while validation_end_datetime<=self.y_series_to_split.index[-1]:
            train_bool_array=(self.y_series_to_split.index>=train_start_datetime) & (self.y_series_to_split.index<splitting_datetime)
            validation_bool_array=(self.y_series_to_split.index>=splitting_datetime) & (self.y_series_to_split.index<validation_end_datetime)
            
            train_indices_array=self.indices_array[train_bool_array]
            validation_indices_array=self.indices_array[validation_bool_array]
            
            self.train_start_indices_list.append(train_indices_array[0])
            self.split_indices_list.append(validation_indices_array[0])
            self.validation_end_indices_list.append(validation_indices_array[-1]+1)
            
            train_start_datetime=train_start_datetime+timedelta(days=round(365*shift_period_years))
            splitting_datetime=train_start_datetime+timedelta(days=round(365*train_duration_years))
            validation_end_datetime=splitting_datetime+timedelta(days=round(365*validation_duration_years))
            split+=1
            
        self.splits_number=split        
    
    def iterator(self):
        for split in range(self.splits_number):
            train_indices_array=self.indices_array[self.train_start_indices_list[split]:self.split_indices_list[split]]
            validation_indices_array=self.indices_array[self.split_indices_list[split]:self.validation_end_indices_list[split]]
            yield (train_indices_array,validation_indices_array)

    def plot_cv_windows(self):
        plt.figure()
        split=0
        for train_indices_array,validation_indices_array in self.iterator():
            plt.plot(self.y_series_to_split.index[train_indices_array],0*train_indices_array+split,'g-')
            plt.plot(self.y_series_to_split.index[validation_indices_array],0*validation_indices_array+split,'r-')
            split+=1
        
        plt.title('training (green) and cross validation (red) periods')
        plt.ylabel('split number')
        plt.grid(True,which='major',axis='both')

def whiten(data_df,white_y_mean,white_y_std,white_X_mean,white_X_std,
    using_only_y_for_white_noise=False,frequency=None,seed=0,debugging=False):
    """v1 created on 2018/09/01 for white data generation to check causality breaking!
    
    to check causality: use this fucntion to convert the data to white noise,
        include target as features when building features - 
        only then, if the regression error std is not white_y_std - 
        causality was broken along the way!
    
    Update log:
        v2 (OzML_v4) - 2018/09/19: adding frequency to control the final 
            white data frequency while keeping the start and end dates.
    
    gets data_df, converts it to white noise: white_target and white_X, 
        each according to the inputs for mean and std of each.
    If frequency=None (default) - using the same size (and index) of the data, 
        otherwise changing to the supplied frequency, for example '1Hr','30Min'.
    If using_only_y_for_white_noise: white_df is made only from white_target.
    seed: the random seed (int) only for the target distribution.
    
    returns: (white_df,white_attribute_names_list,white_target_name)
    """

    logging.warning('to check causaility with white noise generation: include target as features when building features - only then, if the regression error std is not white_y_std - causality was broken along the way!')
    
    if frequency!=None:
        data_df=data_df.asfreq(frequency)
        
    white_noise_target_name='rand target with mu,std=%.1f,%.1f'%(white_y_mean,white_y_std)
    white_target_series=pd.Series(
            np.random.RandomState(seed=seed).normal(
                    loc=white_y_mean,scale=white_y_std,size=len(data_df)),
            index=data_df.index,
            name=white_noise_target_name)
    white_target_name=white_target_series.name
    white_target_df=white_target_series.to_frame() # since pandas does not support merging between series and dataframe
    logging.debug('target_df_mean/white_y_mean,target_df_std/white_y_std = %.3f,%.3f'%(np.mean(white_target_df)/white_y_mean,np.std(white_target_df)/white_y_std))
    
    if not using_only_y_for_white_noise:
        white_X_df=pd.DataFrame(np.random.normal(loc=white_X_mean,scale=white_X_std,size=data_df.shape),
                            index=data_df.index,
                            columns=['rand%d with mu,std=%.1f,%.1f'%(col,white_X_mean,white_X_std) for col in range(len(data_df.columns))])
        white_df=white_target_df.merge(white_X_df,how='inner',left_index=True,right_index=True)
        if debugging:
            print('X_df_mean/white_X_mean:',np.mean(white_X_df)/white_X_mean)
            print('X_df_std/white_X_std:',np.std(white_X_df)/white_X_std)
    else:
        white_df=white_target_df
    
    white_attribute_names_list=list(white_df.columns)
    white_df.columns=['x%d'%cl for cl in range(len(white_attribute_names_list))]
    
    logger.info('created white noise of length %d'%len(white_target_series))
    
    return (white_df,white_attribute_names_list,white_target_name)

def plot_2D_array(x_1D_array,y_1D_array,Z_2D_array,fig_handle,mode='pcolormesh',xscale_log=False,yscale_log=False):
    """created on 2018/09/09 for plotting a 2D array Z_2D_array by axes arrays x_1D_array,y_1D_array
    returns: plot_handle
    """   
    from matplotlib import cm
    
    X,Y = np.meshgrid(x_1D_array,y_1D_array)
    if mode=='contours':
        plot_handle=plt.contour(X,Y,Z_2D_array,cmap=cm.cool)
    elif mode=='contour areas':
        plot_handle=plt.contourf(X,Y,Z_2D_array,cmap=cm.cool)
    elif mode=='pcolormesh':
        plot_handle=plt.pcolormesh(X,Y,Z_2D_array,cmap=cm.cool,edgecolors='k',linewidths=0.25)
        logger.info("""notice: the color in a patch is the value of its lower-left corner, in a grid of MxN there are (M-1)x(N-1) patches!
                    (X[i+1, j], Y[i+1, j])          (X[i+1, j+1], Y[i+1, j+1])
                                          +--------+
                                          | C[i,j] |
                                          +--------+
                        (X[i, j], Y[i, j])          (X[i, j+1], Y[i, j+1]),""")
    elif mode=='pcolormesh + contours':
        plt.pcolormesh(X,Y,Z_2D_array,cmap=cm.hot)
        plot_handle=plt.contour(X,Y,Z_2D_array,cmap=cm.cool)
        logger.info("""notice: the color in a patch is the value of its lower-left corner, in a grid of MxN there are (M-1)x(N-1) patches!'
                    (X[i+1, j], Y[i+1, j])          (X[i+1, j+1], Y[i+1, j+1])
                                          +--------+
                                          | C[i,j] |
                                          +--------+
                        (X[i, j], Y[i, j])          (X[i, j+1], Y[i, j+1]),""")
    elif mode=='scatter':
        plot_handle=plt.scatter(X.flatten(),Y.flatten(),c=Z_2D_array.flatten(),cmap=cm.hot)
    elif mode=='scatter + contour areas':
        plt.contourf(X,Y,Z_2D_array,cmap=cm.cool)
        plot_handle=plt.scatter(X.flatten(),Y.flatten(),c=Z_2D_array.flatten(),cmap=cm.cool,edgecolors='k')
    else:
        raise RuntimeError('unsupported mode!')
    fig_handle.colorbar(plot_handle)
    ax=fig_handle.gca()
    if xscale_log:
        ax.set_xscale('log')
    if yscale_log:
        ax.set_yscale('log')
    
    return plot_handle
