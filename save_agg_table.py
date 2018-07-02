from time import gmtime, strftime
import numpy as np
import h5py
#input_file (hdf5)
f = h5py.File('C:/Users/Catarina Costa/oqdata/calc_1765.hdf5','r')
#Output file
lossFile = 'C:/openquake/oq-cera/cr_120/agg_losses-all_1765.csv'

the_data_var = f["losses_by_event"]
the_data_array = the_data_var[:]
print(len(the_data_array))

# # For ebrisk
rup_index = list(map(int,np.array(list(zip(*the_data_array))[0])))

#Check which ids have loss=0 (if any)
original = set(rup_index)
full_set = set(range(int(min(original)), int(max(original)) + 1))
zero_loss_index = sorted(full_set - original)
print (len(zero_loss_index))

losses = np.array(list(zip(*the_data_array))[2])
loss_complete = np.zeros(rup_index[-1]+1)
for i in range(len(rup_index)):
    loss_complete[rup_index[i]] = losses[i]

event_id = np.arange(len(loss_complete))
rup_id = np.zeros(shape = (len(loss_complete)))
year = np.zeros(shape = (len(loss_complete)))
rlzi = np.zeros(shape = (len(loss_complete)))
structural = loss_complete

toSave = np.column_stack((event_id,rup_id,year,rlzi,structural))

np.savetxt(lossFile,toSave,delimiter=",",comments='',header="event_id,rup_id,year,rlzi,structural",fmt='%i,%i,%i,%i,%.10f')

# # For EB calculations

# inv_time = 1000000
# For EB - save just rates and losses
# losses = np.array(zip(*the_data_array)[2])
# losses_sort = np.sort(losses,axis=0)[::-1].flatten()
# rates = np.arange(1.0,len(losses_sort)+1)/inv_time
# toSave = np.column_stack((rates,losses_sort))

# np.savetxt(lossFile,toSave,delimiter=",",fmt='%.10f,%.10f')
