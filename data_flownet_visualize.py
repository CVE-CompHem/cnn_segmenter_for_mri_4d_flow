# import module and set paths
# ============================   
import numpy as np
from mr_io import FlowMRI
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20})

basepath_image = '/tmp/test.decrypt8/flownet/hpc_predict/v2/inference/'
basepath_image = basepath_image + '2021-03-19_15-46-05_daint102' # '2021-02-11_19-41-32_daint102'
R_values = [8, 10, 12, 14, 16, 18, 20, 22]
for subnum in [1,2,3,4,5,6,7]:

    plt.figure(figsize=[20,10])
    for r in range(len(R_values)):
        R = R_values[r]
        flowmripath = basepath_image + '_volN' + str(subnum) + '_R' + str(R) + '/output/kspc_R' + str(R) + '_volN' + str(subnum) + '_vn.mat.h5'
            
        # ============================   
        flow_mri = FlowMRI.read_hdf5(flowmripath)
        flowMRI_image = np.concatenate([np.expand_dims(flow_mri.intensity, -1), flow_mri.velocity_mean], axis=-1)  
        
        plt.subplot(2,4,r+1)
        plt.imshow(flowMRI_image[:,:,8,3,1],'gray')
        plt.title('R = ' + str(R))
        
    plt.show()