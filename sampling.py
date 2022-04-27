import numpy as np

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

def subsample(array, factor, mode='uniform'):
    
    if mode=='uniform':
        
        if array.ndim == 1:
            sub_sampled_array = array[::factor]
            
        elif array.ndim == 2:
            sub_sampled_array = array[::factor,::factor]
            
        elif array.ndim == 3:
            sub_sampled_array = array[::factor,::factor,::factor]
       
        else:
            print("INCORRECT ARRAY DIMENSION")
            
    elif mode=='random':
        
        print("UNWRITTEN")
        
    else:
        
        print("CHOOSE UNIFORM OR RANDOM")
        
    return sub_sampled_array

def find(array,x,y):
    l=2
    
    Xindex = np.logical_and(array[:,0]>x-l, array[:,0]<y+l)
    Yindex = np.logical_and(array[:,1]>y-l, array[:,1]<y+l)
    
    index = np.logical_and(Xindex, Yindex)
    
    possible_values = np.nonzero(index)
    
    return possible_values