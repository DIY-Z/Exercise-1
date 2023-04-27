import cv2
import numpy as np

"""refer to https://blog.csdn.net/coming_is_winter/article/details/92800150"""

def RLTikh_deconvolution(img, num_iterations):
    # Lucy-Richardson Deconvolution Function
    # input-1 img: NxM matrix image
    # input-2 num_iterations: number of iterations
    # input-3 sigma: sigma of point spread function (PSF)
    # output result: deconvolution result
    
    # Window size of PSF
    winSize = 11 # 10 * sigmaG + 1;
    
    # Initializations
    Y = img.copy()
    J1 = img.copy()
    J2 = img.copy()
    wI = img.copy()
    imR = img.copy()
    reBlurred = img.copy()
    
    T1 = np.zeros_like(img)
    T2 = np.zeros_like(img)
    
    # Lucy-Rich. Deconvolution CORE
    EPSILON = 0.01
    sigmaG = 4.0
    lambda_val = 0
    for j in range(num_iterations):
        if j > 1:
            # calculation of lambda
            tmpMat1 = T1 * T2
            tmpMat2 = T2 * T2
            lambda_val = np.sum(tmpMat1) / (np.sum(tmpMat2) + EPSILON)
            # calculation of lambda
            
        Y = J1 + lambda_val * (J1 - J2)
        Y[Y < 0] = 0
        
        # 1)
        reBlurred = cv2.GaussianBlur(Y, (winSize, winSize), sigmaG)
        reBlurred[reBlurred <= 0] = EPSILON
        
        # 2)
        imR = wI / reBlurred
        imR = imR + EPSILON
        
        # 3)
        imR = cv2.GaussianBlur(imR, (winSize, winSize), sigmaG)
        
        # 4)
        J2 = J1.copy()
        J1 = Y * imR
        
        T2 = T1.copy()
        T1 = J1 - Y
    
    # output
    result = J1
    return result

if __name__ == '__main__':
    guassianImage = cv2.imread("/data/HomeWork/DigitalImageProcess/Exercise/Exercise-1/lunaNoise2.png")
    guassianImageF = guassianImage.astype(np.float64) / 255.0
    
    cv2.imshow("guassianImageF", guassianImageF)
    
    resImage2 = RLTikh_deconvolution(guassianImageF, 2)
    resImage4 = RLTikh_deconvolution(guassianImageF, 4)
    resImage6 = RLTikh_deconvolution(guassianImageF, 6)
    # resImage21 = RLTikh_deconvolution(guassianImageF, 21)

    # resImage150 = RLTikh_deconvolution(guassianImageF, 150)
    
    cv2.imshow('restorationImage2', resImage2)
    cv2.imshow("restorationImage4", resImage4)
    cv2.imshow('restorationImage6', resImage6)
    # cv2.imshow("restorationImage21", resImage21)
    # cv2.imshow("restorationImage150", resImage150)
    
    cv2.waitKey(0)
