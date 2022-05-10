import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from PIL import Image

def plot_array(array, fig_num = [], close = True):
    
    if close:
        matplotlib.pyplot.close(fig_num)
    
    if fig_num:
        matplotlib.pyplot.figure(fig_num)
        
    matplotlib.pyplot.imshow(array)
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

def plotEllipseFromMat(self,x0,y0,a,b,T,L=20,N=100):
    
    x,y = np.meshgrid(np.linspace(-L,L,N), np.linspace(-L,L,N), indexing = 'ij')
    
    vec = np.empty((3,N**2))
    vec[0,:] = x.flatten() + x0
    vec[1,:] = y.flatten() + y0
    vec[2,:] = 1
    
    M = self.defineMatrix(self.defineCoeffs(x0,y0,a,b,T))

    test = np.sum(vec * (M @ vec), axis=0)
    test = test.reshape(N,N)

    plt.imshow(np.abs(test<0).T, extent = [L+x0,-L+x0,-L+y0,L+y0], cmap = 'gray')
    
def plotEllipseContour(x0,y0,a,b,T,L=20,N=100):
    
    x,y = np.meshgrid(np.linspace(x0-L,x0+L,N), np.linspace(y0-L,y0+L,N), indexing = 'ij')
    
    vec = np.empty((3,N**2))
    vec[0,:] = x.flatten()
    vec[1,:] = y.flatten()
    vec[2,:] = 1
    
    M = defineMatrix(defineCoeffs(x0,y0,a,b,T))

    test = np.sum(vec * (M @ vec), axis=0)
    test = test.reshape(N,N)

    plt.contour(x,y,test, [0], colors = 'red') 

def defineMatrix(coeffs):

    A,B,C,D,E,F = coeffs    

    M = np.empty((3,3))
    M[0,0] = A
    M[0,1] = B/2
    M[0,2] = D/2
    M[1,0] = B/2
    M[1,1] = C
    M[1,2] = E/2
    M[2,0] = D/2
    M[2,1] = E/2
    M[2,2] = F

    return M

def defineCoeffs(x,y,a,b,T):
    A = a**2*np.sin(T)**2 + b**2*np.cos(T)**2
    B = 2*(b**2 - a**2)*np.sin(T)*np.cos(T)
    C = a**2*np.cos(T)**2 + b**2*np.sin(T)**2
    D = -2*A*x - B*y
    E = -B*x - 2*C*y
    F = A*x**2 + B*x*y + C*y**2 - a**2*b**2
        
    return A,B,C,D,E,F

def plotEllipseLine(x0,y0,a,b,T,N=100,c=''):
    
    theta = np.linspace(0,2*np.pi,N)
    
    x = a*np.cos(theta - T) + x0
    y = b*np.sin(theta - T) + y0
    
    plt.plot(x,y,c)
    
