import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

def plot_pointcloud(xyz, colourScheme = None):
    pointCloud = []
    
    assert colourScheme in ('z','z-flat', None)
    
    if colourScheme == 'x':
        colours = getAxisColours(xyz,0)
    elif colourScheme == 'y':
        colours = getAxisColours(xyz,1)
    elif colourScheme == 'z':
        colours = getAxisColours(xyz,2)
    elif colourScheme == 'z-flat':
        colours = getFlatColours(xyz)
    
    
    for i in range(len(xyz)):
        pointCloud.append(o3d.geometry.PointCloud())
        pointCloud[i].points = o3d.utility.Vector3dVector(xyz[i])
        
        if colourScheme is not None:
            pointCloud[i].colors = o3d.utility.Vector3dVector(colours[i])
    
    o3d.visualization.draw_geometries(pointCloud)

def getAxisColours(x,k):
    
    minimum = np.inf
    maximum = -np.inf
    
    colours = []
    
    for i in range(len(x)):
        
        if x[i][:,k].min()<minimum:minimum=x[i][:,k].min()
        if x[i][:,k].max()>maximum:maximum=x[i][:,k].max()
        
    for i in range(len(x)):
        temp = (x[i][:,2] - minimum)/(maximum - minimum)
        temp = 1-temp
        temp = np.round(temp*255).astype(int)
        temp = plt.get_cmap("inferno")(temp)[:,:3]
        
        colours += [temp]
    
    return colours

def getFlatColours(x):
       
    colours = []
    for i in range(len(x)):
        X = np.empty((x[i].shape[0], 3))
        X[:,0] = 1
        X[:,1:] = x[i][:,:2]
          
        P = np.linalg.lstsq(X.T@X, X.T@x[i][:,2:3], rcond=None)[0]
        
        R = x[i][:,2:3] - X@P
        
        s = R.std()
        
        R = (R - R.mean() + 1*s)*255/(2*s)
        R[R>255] = 255
        R[R<0] = 0
        
        colours += [plt.get_cmap("seismic")(R.flatten())[:,:3]]
     
    return colours
