import numpy as np

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