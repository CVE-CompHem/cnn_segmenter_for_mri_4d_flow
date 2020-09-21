# ==========================================
# import stuff
# ==========================================
import h5py
import numpy as np
import utils
import scipy.io

# ===============================================================
# reads the hdf5 file provided by Andreas and 
# converts it into a format that the cnn likes.
# ===============================================================
def read_image(path):

    data_flownet = h5py.File(path, 'r')
    # print(list(data_flownet.keys()))  --> returns ['flow-mri']
    
    data_flownet_ = data_flownet['flow-mri']
    # print(list(data_flownet_.keys()))  --> returns ['intensity', 't_coordinates', 'velocity_cov', 'velocity_mean', 'x_coordinates', 'y_coordinates', 'z_coordinates']
    
    # get array shape and create and empty array of this size
    image = np.zeros(list(data_flownet_['intensity'].shape) + [4], data_flownet_['intensity'].dtype)
    image[...,0] = data_flownet_['intensity'] # z, x, y, t
    image[...,1:4] = data_flownet_['velocity_mean'] # z, x, y, t
        
    # close the hdf5 file
    data_flownet.close()
    
    # transpose and flip lr in the x-y direction
    for n0 in range(image.shape[0]):
        for n3 in range(image.shape[3]):
            for n4 in range(image.shape[4]):
                image[n0, :, :, n3, n4] = np.fliplr(image[n0, :, :, n3, n4].T)
    
    # normalize the image
    image = utils.normalize_image(image)
    
    # crop or pad to make the same shape as the freiburg images
    image = utils.crop_or_pad_4dvol_along_0(image, 32)
    image = utils.crop_or_pad_4dvol_along_1(image, 144)
    image = utils.crop_or_pad_4dvol_along_2(image, 112)
    image = utils.crop_or_pad_4dvol_along_3(image, 48)

    return image    

# ===============================================================
# reads the mat files provided by Hannes (downloaded from https://codeocean.com/capsule/2587940/tree/v1)
# ===============================================================
def read_image_mat(filepath):
    
    data = scipy.io.loadmat(filepath)
    
    # =============
    # print the keys in this dict.
    # =============
    print(data.keys()) # dict_keys(['__header__', '__version__', '__globals__', 'masks', 'imrecon'])
        
    # =============
    # extract the recon data.
    # =============
    x = np.squeeze(data['imrecon']) # [velocity_encodings, t, z, y, x].   
    
    # =============
    # extract proton density image
    # =============
    x_pd = (np.abs(x[0,:,:,:,:]) + np.abs(x[1,:,:,:,:]) + np.abs(x[2,:,:,:,:]) + np.abs(x[3,:,:,:,:]) ) / 4
    
    # =============
    # extract velocity fields
    # =============    
    x_vx = np.angle(x[1,:,:,:,:]) - np.angle(x[0,:,:,:,:])  # [t, z, y, x]. # [-pi to pi]
    x_vy = np.angle(x[2,:,:,:,:]) - np.angle(x[0,:,:,:,:])  # [t, z, y, x]. # [-pi to pi]
    x_vz = np.angle(x[3,:,:,:,:]) - np.angle(x[0,:,:,:,:])  # [t, z, y, x]. # [-pi to pi]
    
    # =============   
    # concatenate and return image
    # =============   
    image = np.stack((x_pd, x_vx, x_vy, x_vz), axis=-1) # [t, z, y, x, 4].

    # =============   
    # swap axes
    # =============   
    image = np.swapaxes(np.swapaxes(np.swapaxes(image, 0, 1), 1, 2), 2, 3) # [z, y, x, t, 4].
    image = np.swapaxes(image, 1, 2) # [z, x, y, t, 4].
    
    # =============   
    # transpose and flip lr in the x-y direction
    # =============   
    for n0 in range(image.shape[0]):
        for n3 in range(image.shape[3]):
            for n4 in range(image.shape[4]):
                image[n0, :, :, n3, n4] = np.fliplr(image[n0, :, :, n3, n4])
    
    # =============   
    # The velocity data here is from -pi to pi. (float32).
    # While the velocity data in the freibrurg images is from 0 to 4096 (uint16).
    # I am assuming that in the freiburg data, for the velocity channels, 0 corresponds to -pi, 2048 to 0 and 4096 to pi.
    # =============   
    # LINEARLY RESCALING THE DATA TO THE SAME FORMAT AS THE FREIBURG IMAGES...
    # so, x becomes (x + pi) * 4096 / (2*pi)
    # =============   
    image[..., 1:4] = (image[..., 1:4] + np.pi) * (4096 / (2*np.pi))
    
    # =============   
    # normalize the image
    # =============   
    image = utils.normalize_image(image)
    
    # =============   
    # crop or pad to make the same shape as the freiburg images
    # =============   
    image = utils.crop_or_pad_4dvol_along_0(image, 32)
    image = utils.crop_or_pad_4dvol_along_1(image, 144)
    image = utils.crop_or_pad_4dvol_along_2(image, 112)
    image = utils.crop_or_pad_4dvol_along_3(image, 48)
    
    return image
    
# ===============================================================
# Main function that runs if this file is run directly
# ===============================================================
if __name__ == '__main__':

    # andreas hdf5
    basepath = '/usr/bmicnas01/data-biwi-01/nkarani/projects/hpc_predict/data/eth_ibt/flownet/andreas_fink/'
    dataset_filepath = basepath + 'flow_CStest_Volunteer_R4.h5'    
    image = read_image(dataset_filepath)    
    
    # hannes mat files
    basepath = '/usr/bmicnas01/data-biwi-01/nkarani/projects/hpc_predict/data/eth_ibt/flownet/hannes/'
    for n in range(1, 2):
        dataset_filepath = basepath + 'recon_R6_volN' + str(n) + '_vn.mat'    
        image = read_image_mat(dataset_filepath)    
        
        # visualize image
        utils.save_sample_image_and_labels_across_z(image,
                                                    np.zeros_like(image[..., 0]),
                                                    basepath + 'recon_R6_volN' + str(n) + '_vn')