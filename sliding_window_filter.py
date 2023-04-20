import cv2
import numpy as np
import copy
import scipy.signal as sci
import matrix as mt

"""refer to https://github.com/Beta-y/Side_Window_Filtering"""

'''将内核的某一区域保留,其余置0'''
def zeros_kernel(kernel,size=(1,1),loc=(0,0)):
    kernel_tmp = np.zeros(kernel.shape)
    kernel_tmp[loc[0]:(loc[0]+size[0]),loc[1]:(loc[1]+size[1])] = kernel[loc[0]:(loc[0]+size[0]),loc[1]:(loc[1]+size[1])]
    kernel_tmp = kernel_tmp/np.sum(kernel_tmp)
    return kernel_tmp

'''Side Gaussian Filter实现'''
def s_gausfilter(img,radius,sigma = 0,iteration = 1):
    r = radius
    gaus_kernel = cv2.getGaussianKernel(2*r+1,sigma) # sigma = ((n-1)*0.5 - 1)*0.3 + 0.8
    gaus_kernel = gaus_kernel.dot(gaus_kernel.T)
    gaus_kernel = gaus_kernel.astype(np.float)
    k_L = zeros_kernel(gaus_kernel,size= (2*r+1,r+1),loc= (0,0))
    k_R = zeros_kernel(gaus_kernel,size= (2*r+1,r+1),loc= (0,r))
    K_U = k_L.T
    k_D = K_U[::-1]
    k_NW = zeros_kernel(gaus_kernel,size= (r+1,r+1),loc= (0,0))
    k_NE = zeros_kernel(gaus_kernel,size= (r+1,r+1),loc= (0,r))
    k_SW = k_NW[::-1]
    k_SE = k_NE[::-1]
    kernels = [k_L,k_R,K_U,k_D,k_NW,k_NE,k_SW,k_SE]
    m = img.shape[0]+2*r
    n = img.shape[1]+2*r
    dis = np.zeros([8,m,n]);
    result = copy.deepcopy(img)
    # for ch in range(img.shape[2]):
    #     U = np.pad(img[:,:,ch],(r,r),'edge');
    #     for i in range(iteration):
    #         for id,kernel in enumerate(kernels):
    #             conv2 = sci.correlate2d(U,kernel,'same')
    #             dis[id] = conv2 - U
    #         U = U + mt.mat_absmin(dis)
    #     result[:,:,ch] = U[r:-r,r:-r]
    U = np.pad(img[:,:],(r,r),'edge');
    for i in range(iteration):
        for id,kernel in enumerate(kernels):
            conv2 = sci.correlate2d(U,kernel,'same')
            dis[id] = conv2 - U
        U = U + mt.mat_absmin(dis)
    result[:,:] = U[r:-r,r:-r]
    return result