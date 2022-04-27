import numpy as np
from scipy.signal import convolve2d
import cv2

def FT(array, mode):

    if mode=='fft':
        
        array_FT = np.fft.ifftshift(np.fft.fft(np.fft.fftshift(array)))
    
    elif mode=='fft2':
        
        array_FT = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(array)))

    elif mode=='fftn':
        
        array_FT = np.fft.ifftshift(np.fft.fftn(np.fft.fftshift(array)))
        
    return array_FT

def get_img_from_fig(fig, dpi=180):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
    
def sampleImageGradient(array):
    
    #Define Sobel operators
    sobelX = np.array([
        [-1,-2,-1],
        [0,0,0],
        [1,2,1],])
    sobelY = np.array([
        [-1,0,1],
        [-2,0,2],
        [-1,0,1],])
    
    #Convolve to get the gradient
    gradientX = convolve2d(array, sobelX, boundary = "fill")
    gradientY = convolve2d(array, sobelY, boundary = "fill")
    
    #Remove the boundary conditions
    gradientX = gradientX[1:-1,1:-1]
    gradientY = gradientY[1:-1,1:-1]
    
    gradientX[:,0] = 0
    gradientX[:,-1] = 0
    gradientX[0,:] = 0
    gradientX[-1,:] = 0
    gradientY[:,0] = 0
    gradientY[:,-1] = 0
    gradientY[0,:] = 0
    gradientY[-1,:] = 0
    
    #Get scalar gradient
    gradient = (gradientX**2 + gradientY**2)**0.5
            
    return gradient