import numpy as np
import matplotlib.pyplot as plt
   
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
    
def plotEllipseContour(self,x0,y0,a,b,T,L=20,N=100):
    
    x,y = np.meshgrid(np.linspace(x0-L,x0+L,N), np.linspace(y0-L,y0+L,N), indexing = 'ij')
    
    vec = np.empty((3,N**2))
    vec[0,:] = x.flatten()
    vec[1,:] = y.flatten()
    vec[2,:] = 1
    
    M = self.defineMatrix(self.defineCoeffs(x0,y0,a,b,T))

    test = np.sum(vec * (M @ vec), axis=0)
    test = test.reshape(N,N)

    plt.contour(x,y,test, [0], colors = 'red') 

def defineMatrix(self,coeffs):

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

def defineCoeffs(self,x,y,a,b,T):
    A = a**2*np.sin(T)**2 + b**2*np.cos(T)**2
    B = 2*(b**2 - a**2)*np.sin(T)*np.cos(T)
    C = a**2*np.cos(T)**2 + b**2*np.sin(T)**2
    D = -2*A*x - B*y
    E = -B*x - 2*C*y
    F = A*x**2 + B*x*y + C*y**2 - a**2*b**2
        
    return A,B,C,D,E,F

def plotEllipseLine(self,x0,y0,a,b,T,N=100,c=''):
    
    theta = np.linspace(0,2*np.pi,N)
    
    x = a*np.cos(theta - T) + x0
    y = b*np.sin(theta - T) + y0
    
    plt.plot(x,y,c)