
# coding: utf-8

# ## Load gmf

# In[1]:


## INPUT ##

gmf_file = './GMF_complete_IM0.1+MaxDist_10km_60arcsec.csv'
gmf_file_gmpe_rate = './GMF_complete_IM0.1+MaxDist_10km_60arcsec_gmpe_rate.csv'


# In[2]:


import pandas as pd

df_gmf = pd.read_csv(gmf_file, header=0)
df_gmf_gmpe_rate = pd.read_csv(gmf_file_gmpe_rate, header=0)


# In[3]:


gmfs_median = []

for i in range(len(df_gmf_gmpe_rate)):
    gmf_median = {}
    gmf_median['rate'] = df_gmf_gmpe_rate['rate'][i]
    gmv = df_gmf[df_gmf.keys()[1]].values
    gmf_median[df_gmf_gmpe_rate['gmpe'][i]] = [df_gmf[df_gmf.event_id == i][['gmv_PGA']].values,df_gmf[df_gmf.event_id == i][['gmv_SA(0.3)']].values]
    gmfs_median.append(gmf_median)


# ## Calculate Total Standard Deviation

# In[4]:


from openquake.hazardlib.imt import PGA, SA
from openquake.hazardlib.gsim.base import RuptureContext, SitesContext, DistancesContext
import numpy as np
from openquake.commonlib.readinput import get_oqparam, get_site_collection, get_gsim_lt, get_exposure, get_sitecol_assetcol


# In[5]:


## INPUT ##

imts = [PGA(), SA(0.3)]
#inter and intra values correspond to intra/inter ratio of 1.75
#These values are used only if the GMPE is defined for TOTAL st dev
inter = 0.496
intra = 0.868
vs30 = 180


# In[6]:


oq_param = get_oqparam("./job_eb_cr_60_SJ2.ini")

exposure = get_exposure(oq_param)
#Added this line
haz_sitecol = get_site_collection(oq_param)
sites, assets_by_site = get_sitecol_assetcol(oq_param, haz_sitecol)

gsimlt = get_gsim_lt(oq_param)
gsim_list = [br.uncertainty for br in gsimlt.branches]
gsim_list


# In[7]:


from openquake.hazardlib.gsim.zhao_2006 import ZhaoEtAl2006Asc
from openquake.hazardlib.gsim.zhao_2006 import ZhaoEtAl2006SInter
from openquake.hazardlib.gsim.zhao_2006 import ZhaoEtAl2006SSlab
from openquake.hazardlib import const


# In[8]:


#stddevs = [const.StdDev.TOTAL]
std_total = {}
std_inter = {}
std_intra = {}
for gsim in gsim_list:
    rctx = RuptureContext()
    #The calculator needs these inputs but they are not used in the std calculation
    rctx.mag = 5
    rctx.rake = 0
    rctx.hypo_depth = 0
    
    dctx = DistancesContext()
    dctx.rjb = np.copy(np.array([0]))
    dctx.rrup = np.copy(np.array([0]))
    #dctx.rhypo = np.copy(np.array([0]))
    
    sctx = SitesContext()
    sctx.vs30 = vs30 * np.ones_like(np.array([0]))
    for imt in imts:
        if gsim.DEFINED_FOR_STANDARD_DEVIATION_TYPES ==             set([const.StdDev.TOTAL]):
            gm_table, gm_stddevs = gsim.get_mean_and_stddevs(sctx, rctx, dctx, imt, stddevs)
            std_total[gsim,imt] = gm_stddevs[0][0]
            std_inter[gsim,imt] = gm_stddevs[0][0]*inter
            std_intra[gsim,imt] = gm_stddevs[0][0]*intra
        else:
            gm_table, [gm_stddev_inter, gm_stddev_intra] = gsim.get_mean_and_stddevs(sctx, rctx, dctx, imt, [const.StdDev.INTER_EVENT, const.StdDev.INTRA_EVENT])
            std_total[gsim,imt] = np.sqrt(gm_stddev_inter[0]**2+gm_stddev_intra[0]**2)
            std_inter[gsim,imt] = gm_stddev_inter[0]
            std_intra[gsim,imt] = gm_stddev_intra[0]
            
#print std_total
#print std_inter
#print std_intra


# ## Inter-event residuals

# In[9]:


## INPUT ##
realizations_inter = 5
import scipy


# In[10]:


#Importance Sampling
mean_shift = 0.75

rates_inter = np.array([1./realizations_inter]*realizations_inter)
cumulative_rates = np.cumsum(rates_inter)-rates_inter/2
distr_values = scipy.stats.norm.ppf(cumulative_rates,loc=mean_shift,scale=1)
p_distr_values = scipy.stats.norm.pdf(distr_values,loc=0,scale=1)
q_distr_values = scipy.stats.norm.pdf(distr_values,loc=mean_shift,scale=1)
weights = (p_distr_values/q_distr_values)
rates_inter = weights/sum(weights)
#print rates_inter
#print distr_values

#Calculate distribution mean - needs to be approximately zero
np.mean(distr_values*rates_inter)


# In[11]:


#get std_inter values from gmpe
gmpe_imt = list(std_inter.keys())
inter_residual = {}

#calculate inter_residual values
for i in range(len(gmpe_imt)):
    stddev_inter = [std_inter[gmpe_imt[i]]]
    inter_residual[str(gmpe_imt[i][0])+', '+str(gmpe_imt[i][1])] = stddev_inter * distr_values

inter_residual['rates_inter'] = rates_inter

#print inter_residual
#print rates_inter
#print distr_values


# ## Intra-event residuals: upload values from csv files

# In[12]:


## INPUT ##

realizations_intra = 5
#If No Correlation:
mu = 0.0
sigma = 1.0


# ### Intra-event residuals: No Correlation

# In[13]:


#df_coords = pd.DataFrame({'lons': sites.lons, 'lats': sites.lats})
#intra_residual_no_coords = {}
#intra_residual = {}
#intra_residual['rates_intra'] = rates_intra

#for x in range(len(gmpe_imt)):
#    df_part = np.random.normal(mu, sigma, len(sites)*realizations_intra).reshape((len(sites),realizations_intra))
#    
#    stddev_intra = [std_intra[gmpe_imt[x]]]

#    intra_residual_no_coords[str(gmpe_imt[x][0])+', '+str(gmpe_imt[x][1])] = stddev_intra * df_part
#    intra_residual[str(gmpe_imt[x][0])+', '+str(gmpe_imt[x][1])] = np.concatenate([df_coords.values,intra_residual_no_coords[str(gmpe_imt[x][0])+', '+str(gmpe_imt[x][1])]], axis=1)


# ### Intra-event residuals: Spatial and Cross Correlation

# In[14]:


## INPUT ##
intra_matrices_file = './Matrix_Intra_Res_SJ_60arcsec_10k_seed1_trunc3_withfilter_'


# In[15]:


from scipy.spatial.distance import cdist

df_0 = pd.read_csv(intra_matrices_file+str(imts[0])+'.csv', nrows = 2, header=None)
number_cols = len(df_0.columns)

#Find indeces of rows to extract from Matrices
df = pd.read_csv(intra_matrices_file+str(imts[0])+'.csv', usecols=[0,1], header=None)

coords_matrix = np.array(df)
exposure_coords = np.array(list(zip(*[sites.lons,sites.lats])))
rows_to_extract = np.argmin(cdist(coords_matrix,exposure_coords,'sqeuclidean'),axis=0)


# In[16]:


#Multiply entire table (1000matrices) by 
#Only works when rates are equal (Not applicable for IS)!!!

intra_residual = {}
#If rates are all equal
rates_intra = [1./realizations_intra]*realizations_intra
intra_residual['rates_intra'] = rates_intra

#np.random.seed(99)
#cols = np.random.choice(range(2,number_cols), realizations_intra, replace=False)
cols = np.array(range(2,number_cols))
#intra_residual['rates_intra']


# In[17]:


intra_residual_no_coords = {}

df_coords = pd.DataFrame({'lons': sites.lons, 'lats': sites.lats})

for x in range(len(gmpe_imt)):
    IMT = str(gmpe_imt[x][1])

    file_name = intra_matrices_file+str(IMT)+'.csv'
    df = pd.read_csv(file_name, usecols=cols, header = None)
    df_part = df.loc[rows_to_extract].reset_index(drop=True)
    
    #get std_intra values from gmpe
    stddev_intra = [std_intra[gmpe_imt[x]]]
    intra_residual_no_coords[str(gmpe_imt[x][0])+', '+str(gmpe_imt[x][1])] = stddev_intra * df_part.values
    
    intra_residual[str(gmpe_imt[x][0])+', '+str(gmpe_imt[x][1])] = np.concatenate([df_coords.values,intra_residual_no_coords[str(gmpe_imt[x][0])+', '+str(gmpe_imt[x][1])]], axis=1)

#intra_residual


# ## Sum median with residuals and save .csv file - For ruptures after filtering

# In[18]:


## INPUT ##

csv_gmf_file = './GMF_results_seed1.csv'
csv_rate_gmf_file = './GMF_results_seed1_rate.csv'
seed=1


# In[19]:


U16 = np.uint16
U32 = np.uint32
U64 = np.uint64
F32 = np.float32

num_coords = len(sites)
#num_gmfs = len(gmfs_median)*len(inter_residual['rates_inter'])*len(intra_residual['rates_intra'])
num_gmfs = 1*len(inter_residual['rates_inter'])*len(intra_residual['rates_intra'])

lst_=[]
for sid in range(num_coords):
    list_a = []
    for eid in range(int(num_gmfs)):
        list_a.append((sid+eid*num_coords, sid+eid*num_coords + 1))
    a = np.array(list_a, np.dtype([('start', U32), ('stop', U32)]))
    lst_.append(a)
lst = np.array(lst_)


# In[20]:


import openquake
ds = openquake.baselib.datastore.get_datadir()
last_id = openquake.baselib.datastore.get_last_calc_id()
ds
#last_id


# In[21]:


from openquake.commonlib import logs

logs.dbcmd('import_job', 1234, 'event_based',
               'eb_test_hdf5', 'ccosta', 'complete', 
               None, ds)


# In[23]:


import h5py
from openquake.baselib import hdf5
import numpy as np
from openquake.commonlib import source
from openquake.commonlib.readinput import get_oqparam, get_gsim_lt, get_exposure, get_sitecol_assetcol


f = hdf5.File('./mytestfile5.hdf5', 'a')

data = lst
dt = data[0].dtype
dtype = h5py.special_dtype(vlen=dt)
dset = f.create_dataset('gmf_data/indices', (len(sites),), dtype, compression=None)
for i, val in enumerate(lst):
    dset[i] = val
    
gmdata_dt = np.dtype([('PGA', F32), ('SA(0.3)', F32), ('events', U32), ('nbytes', U32)])
dset1 = f.create_dataset('gmdata', (1,), dtype=gmdata_dt)
dset1['events'] = int(num_gmfs)
dset1['PGA'] = 0.03
dset1['SA(0.3)'] = 0.031

cinfo = source.CompositionInfo.fake(gsimlt)

events_dt = np.dtype([('eid', U64), ('rup_id', U32), ('grp_id', U16), ('year', U32), ('ses', U32), ('sample', U32)])
dset2 = f.create_dataset('events', (num_gmfs,), dtype=events_dt)
dset2['eid'] = np.arange(0,num_gmfs)
dset2['ses'] = np.ones(num_gmfs)
dset2['rup_id'] = np.ones(num_gmfs)
dset2['year'] = np.ones(num_gmfs)

f['csm_info'] = cinfo
f['sitecol'] = sites
f['oqparam'] = oq_param

f.close()


# In[24]:


import csv
#row1 = 'rlzi,sid,eid,gmv_'+str(imts[0])+',gmv_'+str(imts[1])+ '\n'
#with open(csv_gmf_file, 'w') as text_fi:
#    text_fi.write(row1)
    
#row1_rate = 'event_id,rate' + '\n'
#with open(csv_rate_gmf_file, 'w') as text_fi_2:
#    text_fi_2.write(row1_rate)

num_coords = len(sites)
eid = -1
num_intra_matrices = len(df_0.columns)-2

zip_intra = {}
for al in range(len(gsim_list)):
    for c in range(len(imts)):
        zip_intra[str(gsim_list[al])+', '+str(imts[c])] = list(zip(*intra_residual[str(gsim_list[al])+', '+str(imts[c])]))

first_row = 0
I = len(imts)
shape_val = num_gmfs*num_coords
gmv_data_dt = np.dtype([('rlzi', U16), ('sid', U32), ('eid', U64), ('gmv', (F32, (I,)))])
with hdf5.File('./mytestfile5.hdf5', 'a') as f:
    dset3 = f.create_dataset('gmf_data/data', (shape_val,), dtype=gmv_data_dt, chunks=True)


#with open(csv_gmf_file, 'a') as text_fi:
#    aa = csv.writer(text_fi, delimiter=',')
    with open(csv_rate_gmf_file, 'a') as text_fi_2:
        ab = csv.writer(text_fi_2, delimiter=',')

        #for a in range(len(gmfs_median)):
        for a in range(0,1):
            index_gmf = a
            keys_gmfs = gmfs_median[index_gmf].keys()
            gmf_gmpe = [i for i in keys_gmfs if i in gsim_list][0]
            
            for d in range(len(inter_residual['rates_inter'])):
                for e in range(len(intra_residual['rates_intra'])):
                    np.random.seed(seed+a+d*1000+e*10000)
                    aleatoryIntraMatrices = np.random.choice(range(num_intra_matrices), realizations_intra, replace=False)
                    eid += 1
                    col1_txt = np.zeros((num_coords,1)).flatten()
                    col2_txt = np.arange(num_coords)
                    col3_txt = np.full((1,num_coords), eid, dtype=int)[0]
                    gmf_to_txt = np.transpose(np.array([col1_txt,col2_txt,col3_txt]))
                    rate =  gmfs_median[index_gmf]['rate'] * inter_residual['rates_inter'][d] * intra_residual['rates_intra'][e]
                    for c in range(len(imts)):
                        gmf_total_part = {}                
                        gmv = gmfs_median[index_gmf][gmf_gmpe][c]
                        gmf_total_part[gmf_gmpe,imts[c],d,e] = np.exp(np.log(gmv)
                                                        + inter_residual[str(gmf_gmpe)+', '+str(imts[c])][d]
                                                        + np.array(zip_intra[str(gsim_list[al])+', '+str(imts[c])][2+aleatoryIntraMatrices[e]]).reshape((-1,1)))

                        gmf_to_txt = np.c_[gmf_to_txt,gmf_total_part[gmf_gmpe,imts[c],d,e].flatten()]
                       
                    #aa.writerows(map(lambda t: ("%i" % t[0], "%i" % t[1], "%i" % t[2], "%.3f" % t[3], "%.3f" % t[4]), gmf_to_txt))
                    ab.writerow([eid,rate])
                    
                    #Save in hdf5
                    dset3['rlzi',first_row:first_row+num_coords] = gmf_to_txt[:,0]
                    dset3['sid',first_row:first_row+num_coords] = gmf_to_txt[:,1]
                    dset3['eid',first_row:first_row+num_coords] = gmf_to_txt[:,2]
                    dset3['gmv',first_row:first_row+num_coords] = gmf_to_txt[:,3:5]
                    first_row = first_row+num_coords
    f.close()  


# In[ ]:




