import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import matplotlib
import io
import cv2
import h5py
import time
from scipy.signal import convolve2d


#%% Typecasting

def rescale_to_uint8(array, max_value=[]):
    
    if not max_value:
        max_value = np.iinfo(array.dtype).max
    
    array = array.astype(np.float)
    array = array*255/max_value
    array = array.astype(np.uint8)
    
    return array

def convert_to_unsigned_integer(array, bit_number):
    
    max_value = 2*bit_number
    
    array[array>max_value] = max_value
    array[array<0] = 0
    
    if bit_number>0 & bit_number <= 8:
        return array.astype(np.uint8)

    elif bit_number>8 & bit_number <= 16:
        return array.astype(np.uint16)
    
    elif bit_number>16 & bit_number <= 32:
        return array.astype(np.uint32)
    
    else:
        print("Bit number too high or too low")

def convert_float_to_uint8_image(array, min_value=0, max_value=255):
    
    array = np.round(array*(max_value-min_value) + min_value)
    
    return array.astype(np.uint8)
 
def create_R(T, order = 'XYZ'):
    
    if order == 'XYZ':
        Tx = T[0]
        Ty = T[1]
        Tz = T[2]
        
        
        Rx = np.array([
            [1,0,0],
            [0,np.cos(Tx),-np.sin(Tx)],
            [0, np.sin(Tx), np.cos(Tx)]
            ])
        
        Ry = np.array([
            [np.cos(Ty),0,np.sin(Ty)],
            [0,1,0],
            [-np.sin(Ty),0,np.cos(Ty)]
            ])
        
        Rz = np.array([
            [np.cos(Tz), -np.sin(Tz),0],
            [np.sin(Tz),np.cos(Tz),0],
            [0,0,1]
            ])
        
        R = Rx@Ry@Rz
    
    else:
        print("UNWRITTEN")
    
    return R

def home_directory():
  
    current_directory = os.getcwd()
    
    cutoff = current_directory.find("ProjectyBoy2000")
    
    if cutoff == -1:
        print("HOME DIRECTORY NOT FOUND")
        
        return -1
    
    else:
    
        home_directory = current_directory[:cutoff+16]
    
        return home_directory

def load_arrays(measurement_directory, start, end, preffix="", suffix=""):
    
    stack = []
    for i in range(start,end):
        
        string = measurement_directory + preffix+ "{:02d}.npy".format(i) + suffix
                
        image = np.load(string)
        
        stack.append(image)
    
    if len(stack) == 1:
        stack = stack[0]
    
    return stack

def subsample(array, factor, mode='uniform'):
    
    if mode=='uniform':
        
        if array.ndim == 1:
            sub_sampled_array = array[::factor]
            
        elif array.ndim == 2:
            sub_sampled_array = array[::factor,::factor]
            
        elif array.ndim == 2:
            sub_sampled_array = array[::factor,::factor,::factor]
       
        else:
            print("INCORRECT ARRAY DIMENSION")
            
    elif mode=='random':
        
        print("UNWRITTEN")
        
    else:
        
        print("CHOOSE UNIFORM OR RANDOM")
        
    return sub_sampled_array
    
def FT(array, mode):
    
    if mode=='fft2':
        
        array_FT = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(array)))
        
    return array_FT

def find(array, x,y):
    l=2
    
    Xindex = np.logical_and(array[:,0]>x-l, array[:,0]<y+l)
    Yindex = np.logical_and(array[:,1]>y-l, array[:,1]<y+l)
    
    index = np.logical_and(Xindex, Yindex)
    
    possible_values = np.nonzero(index)
    
    return possible_values

def extract_theta_from_R(R, order = 'XYZ'):
    
    if order == 'XYZ':
        y = np.arcsin(R[0,2])
        
        z = -np.arcsin(R[0,1] / np.cos(y))
        
        x = -np.arcsin(R[1,2] / np.cos(y))
        
        return x,y,z
    
    if order == 'YZX':
                
        z = np.arcsin(R[1,0])
                              
        y = -np.arcsin(R[2,0] / np.cos(z))
        
        x = np.arccos(R[1,1] / np.cos(z))
        
        return x,y,z
    
def defineRfromVectors(A,B):
    
    v = np.cross(A,B,0,0)
    
    s = np.sum(v**2)**0.5
    
    c = A.T@B
    
    V = np.zeros((3,3))
    V[0,1] = -v[0,2]
    V[0,2] =  v[0,1]
    V[1,0] =  v[0,2]
    V[1,2] = -v[0,0]
    V[2,0] = -v[0,1]
    V[2,1] =  v[0,0]

    R = np.eye(3) + V + V@V * ((1-c)/s**2)
    
    return R

def rodriguesInv(R):
    
    angle = np.arccos(( R[0,0] + R[1,1] + R[2,2] - 1)/2)
    
    denom = ((R[2,1] - R[1,2])**2 + (R[0,2] - R[2,0])**2 + (R[1,0] - R[0,1])**2)**0.5
    
    x = (R[2,1] - R[1,2]) / denom
    y = (R[0,2] - R[2,0]) / denom
    z = (R[1,0] - R[0,1]) / denom
    
    r = np.empty(3)
    r[0],r[1],r[2] = x,y,z
    
    r = r*angle
    
    return r

def rodrigues(r1, r2 = None, r3 = None):
    
    if (r2==None) or (r3 == None):
        r2=r1[1]
        r3=r1[2]
        r1=r1[0]
    
    theta = (r1**2 + r2**2 + r3**2)**0.5
           
    K = np.zeros((3,3))
    K[0,1] = -r3
    K[0,2] = r2
    K[1,0] = r3
    K[1,2] = -r1
    K[2,0] = -r2
    K[2,1] = r1
    
    K = K/theta

    R = np.eye(3) + np.sin(theta)*K + (1-np.cos(theta))*K@K
    
    return R

def get_img_from_fig(fig, dpi=180):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img
#%% Sampling

def lineSegmentIndex(arraySize,start,end,thickness=1):
        
    x1 = int(start[0])
    x2 = int(end[0])
    
    y1 = int(start[1])
    y2 = int(end[1])
    
    #best fit line
    grad = (y2-y1)/(x2-x1)
    origin = -x1*grad+y1
    
    x = (np.linspace(x1, x2, np.abs(x2-x1) + 1)).astype(int)
    y = (origin + x*grad).astype(int)
        
    index = np.zeros(arraySize, dtype = bool)
    index[x,y] = True
    
    for i in range(1,thickness//2+1):
        
        index[x,y+i] = True
        index[x,y-i] = True
    
    return index

def interp1D(A, index):
        
    low = np.floor(index).astype(int)
    high = np.ceil(index).astype(int)
    
    low[low<0] = 0
    low[low>A.shape[0]-2] = A.shape[0]-2
    
    high[high<1] = 1
    high[high>A.shape[0]-1] = A.shape[0]-1
    
    r = index%1
    
    i = np.linspace(0,A.shape[1]-1, A.shape[1]).astype(int)
    
    B = A[low,i]*(1-r) + A[high,i]*r
    
    return B

def interp2D(z,xnew,ynew):
    
    x1 = np.floor(xnew).flatten().astype(int)
    x2 = np.ceil(xnew).flatten().astype(int)
    
    y1 = np.floor(ynew).flatten().astype(int)
    y2 = np.ceil(ynew).flatten().astype(int)
    
    v = xnew-x1
    u = ynew-y1
    
    A = z[x1,y2]
    B = z[x2,y2]
    C = z[x1,y1]
    D = z[x2,y1]
    
    out = A*(u-u*v) + B*(u*v) + C*(1-u-v+u*v) + D*(v-u*v)
    
    return out

def extractRegion(array, point, halfSize):
       
    a = int(point[0]-halfSize)
    b = int(point[0]+halfSize+1)
    c = int(point[1]-halfSize)
    d = int(point[1]+halfSize+1)
    
    assert (a>=0) & (c>=0) & (b<=array.shape[0]) & (c<=array.shape[1]),"Array extraction values outside array"
        
    return array[a:b, c:d], a, c  
#%% Image manipulation

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
#%% Plotting
def plot_array(array, fig_num = [], close = True):
    
    if close:
        matplotlib.pyplot.close(fig_num)
    
    if fig_num:
        matplotlib.pyplot.figure(fig_num)
        
    matplotlib.pyplot.imshow(array)
    matplotlib.pyplot.show()

def plot_scatter(X, Y, indexing = 'ij', fig_num = [], close = True):
    
    if close:
        matplotlib.pyplot.close(fig_num)
    
    if fig_num:
        matplotlib.pyplot.figure(fig_num)
    
    if indexing == 'ij':
        matplotlib.pyplot.scatter(X,Y,s=1, c='red')
    if indexing == 'xy':
        matplotlib.pyplot.scatter(Y,X,s=1, c='red')
        
    matplotlib.pyplot.show()

def bivariate_normal_distribution(x,y,meanX,meanY,stdX,stdY,corr):
    
        dist = (1/( stdX*stdY*2*np.pi*(1-corr**2)**0.5)) *\
        np.exp(- 1/(2*(1-corr**2)) * \
               ( \
               +((x - meanX)/stdX)**2 \
               -((x - meanX)/stdX)*((y - meanY)/stdY)*2*corr \
               +((y - meanY)/stdY)**2 \
               ))
            
        return dist
    
def plot_bivariate_normal_distribution(data, limits=(-1,1), N=1000):
    
    #Find mean
    meanX = np.mean(data[:,0])
    meanY = np.mean(data[:,1])
    
    #Find covariance matrix
    cov = np.cov(data[:,0], data[:,1])

    stdX = cov[0,0]**0.5
    stdY = cov[1,1]**0.5
        
    corr = cov[1,0]/(stdX*stdY)
    #Plot
    xmesh, ymesh = np.meshgrid(
        np.linspace(limits[0], limits[1], N),
        np.linspace(limits[0], limits[1], N))
    
    dist = bivariate_normal_distribution(xmesh,ymesh,meanX,meanY,stdX,stdY,corr)
    
    c1 = bivariate_normal_distribution(1*stdX,0,meanX,meanY,stdX,stdY,corr)
    c2 = bivariate_normal_distribution(2*stdX,0,meanX,meanY,stdX,stdY,corr)
    c3 = bivariate_normal_distribution(3*stdX,0,meanX,meanY,stdX,stdY,corr)
    
    contours = (c3, c2, c1)
        
    matplotlib.pyplot.contour(xmesh,ymesh,dist,contours, colors = 'r')
        
    matplotlib.pyplot.text(stdX*1, 0, '$\sigma_1$', color='black', 
        backgroundcolor='white', fontsize = 14)
    matplotlib.pyplot.text(stdX*2, 0, '$\sigma_2$', color='black', 
        backgroundcolor='white', fontsize = 14)
    matplotlib.pyplot.text(stdX*3, 0, '$\sigma_3$', color='black', 
        backgroundcolor='white', fontsize = 14)
    
def plot_normal_distribution(data, limits=(-1,1),N = 1000):
    
    mean = np.mean(data)

    std = np.std(data)
    
    x = np.linspace(limits[0], limits[1], N)
    
    normal = (1/(std * (2*np.pi)**0.5)) * np.exp(-0.5 * ((x - mean)/(std))**2)

    matplotlib.pyplot.plot(x,normal)
    
def plot_random_distribution(data, limits=(-1,1), N=1000):
    
    mean = np.mean(data)

    std = np.std(data)
    
    a = (std**2 * 3)**0.5
    boundary = (mean-a, mean+a)
    
    x = np.linspace(limits[0], limits[1], N)
    
    output = np.logical_and(x>=boundary[0],x<=boundary[1])
    output = output.astype(np.float)/(boundary[1]-boundary[0])
    
    matplotlib.pyplot.plot(x,output)

def plot_bivariate_random_distribution(data, limits=(-1,1), N=1000):
    
    #Find mean
    meanX = np.mean(data[:,0])
    meanY = np.mean(data[:,1])
        
    stdX = np.std(data[:,0])
    stdY = np.std(data[:,1])
    
    a = (stdX**2 * 3)**0.5
    b = (stdY**2 * 3)**0.5
    
    boundaryX = (meanX-a, meanX+a)
    boundaryY = (meanY-b, meanY+b)

    matplotlib.pyplot.plot(
        [boundaryX[1], boundaryX[0], boundaryX[0], boundaryX[1], boundaryX[1]],
        [boundaryY[1], boundaryY[1], boundaryY[0], boundaryY[0], boundaryY[1]])    

def show_array(array):
    
    array = array.astype(np.float)
    
    array_min = np.min(array)
    array_max = np.max(array)
    array_range = array_max - array_min
    
    #Scale
    array = (array - array_min)/array_range * 255
    
    array = array.astype("uint8")
            
    img = Image.fromarray(array, 'L')
    img.show()

def plot_line(array, axis="X", index=0):
    
    if axis == "X":
        line = array[index,:]
    
    elif axis == "Y":
        line = array[:,index]
        
    matplotlib.pyplot.plot(np.linspace(0,line.shape[0]-1, line.shape[0]), line)
    
def plot_lines_between_datasets(X, Y, ax=[]):
    
    if not ax:
        
        fig = matplotlib.pyplot.figure(10)
        ax = fig.add_subplot(111, projection='3d')
    
    for i in range(X.shape[1]):
        
        ax.plot(
            [X[0,i], Y[0,i]],
            [X[1,i], Y[1,i]],
            [X[2,i], Y[2,i]],
            'r')
           
def plotErrorEllipse(ax, point, V, k=1, lineArgs = 'r'):
    
    ev, evec = np.linalg.eig(V)

    # Get the index of the largest eigenvector
    i = np.flip(np.argsort(ev))
    
    ev = np.round(ev[i], 3)
    evec = np.round(evec[:,i], 3)
    
    # Calculate the angle between the x-axis and the largest eigenvector
    angle = np.arctan2(evec[1,0], evec[0,0])
        
    # Get the 95% confidence interval error ellipse
    chisquare_val = k#2.4477
    theta_grid = np.linspace(0,2*np.pi)
    phi = angle
    X0=point[0]
    Y0=point[1]
    a=chisquare_val*ev[0]**0.5
    b=chisquare_val*ev[1]**0.5
        
    # the ellipse in x and y coordinates
    ellipse = np.empty((50,2))
    ellipse[:,0]  = a*np.cos( theta_grid )
    ellipse[:,1]  = b*np.sin( theta_grid )
    
    # Define a rotation matrix
    R = np.array([ [np.cos(phi), -np.sin(phi)],[ np.sin(phi), np.cos(phi) ]])
    
    #let's rotate the ellipse to some angle phi
    ellipse = ellipse @ R.T
    
    # Draw the error ellipse
    ax.plot(ellipse[:,0] + X0,ellipse[:,1] + Y0,lineArgs)
    
    return

def plotErrorEllipsoid(ax, point, V, lineArgs = 'r'):
    
    Vi = np.linalg.inv(V)
        
    # find the rotation matrix and radii of the axes
    U, s, rotation = np.linalg.svd(Vi)
    radii = 1.0/np.sqrt(s)
    
    # now carry on with EOL's answer
    u = np.linspace(0.0, 2.0 * np.pi, 100)
    v = np.linspace(0.0, np.pi, 100)
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
    for i in range(len(x)):
        for j in range(len(x)):
            [x[i,j],y[i,j],z[i,j]] = np.dot([x[i,j],y[i,j],z[i,j]], rotation) + point
    
    # plot
    ax.plot_wireframe(x, y, z,  rstride=4, cstride=4, color='b', alpha=0.2)
    plt.show()    
    
    return 




#%% Directory   
def listdir(folder, condition):
    out = []
    for item in os.listdir(folder):
        if condition(item):
            out.append(item)
    
    return out

#%% Uncertainty

def corrMatrixFromCovMatrix(V):
    
    X = np.diag(V)**0.5
    X = X.reshape(-1,1)
    
    corr = V/(X@X.T)
    
    return corr

#%% polygon area

def sphericalPolygonArea(lats, lons, radius = None):
    """
    Computes area of spherical polygon, assuming spherical Earth. 
    Returns result in ratio of the sphere's area if the radius is specified.
    Otherwise, in the units of provided radius.
    lats and lons are in degrees.
    """
    # from numpy import arctan2, cos, sin, sqrt, pi, power, append, diff, deg2rad
    # lats = np.deg2rad(lats)
    # lons = np.deg2rad(lons)

    #close polygon
    if lats[0]!=lats[-1]:
        lats = np.append(lats, lats[0])
        lons = np.append(lons, lons[0])

    #colatitudes relative to (0,0)
    a = np.sin(lats/2)**2 + np.cos(lats)* np.sin(lons/2)**2
    colat = 2*np.arctan2( np.sqrt(a), np.sqrt(1-a) )

    #azimuths relative to (0,0)
    az = np.arctan2(np.cos(lats) * np.sin(lons), np.sin(lats)) % (2*np.pi)

    # Calculate diffs
    # daz = np.diff(az) % (2*np.pi)
    daz = np.diff(az)
    daz = (daz + np.pi) % (2 * np.pi) - np.pi

    deltas=np.diff(colat)/2
    colat=colat[0:-1]+deltas

    # Perform integral
    integrands = (1-np.cos(colat)) * daz

    # Integrate 
    area = abs(sum(integrands))/(4*np.pi)

    area = min(area,1-area)
    if radius is not None: #return in units of radius
        return area * 4*np.pi*radius**2
    else: #return in ratio of sphere total area
        return area