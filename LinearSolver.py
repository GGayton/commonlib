import numpy as np

def sqDiff(a,b):
    
    return np.sum((a - b)**2)**0.5

class LinearSolver():
    
    def __init__(self):
                
        self.options = {
            "verbosity":1
            }

    def setOptions(self,optionsIn):
        self.options = optionsIn

    def getOptions(self):
        return self.options

    def weightedTotalLeastSquares(self,A,y,VA,Vy):
        
        '''
        A = 
        [A11,A12,A13 ...]
        [A21,A22,A23 ...]
        [A31,A32,A33 ...]
        
        VA = 
        [A11,A12,A13,...A21,A22,A23]
        '''
        
        verbosity = self.options["verbosity"]
        
        nObs = y.shape[0]
        nParams = A.shape[1]
        
        #Obtain first estimate
        x = np.linalg.inv(A.T@A) @ A.T @ y
    
        running = True
        countMax = 100
        count = 0
        
        while running: 
            
            e = y - A@x
            
            T = np.kron(np.eye(nObs),x)
                        
            Vu = Vy + T.T @ VA @ T
                
            Vui = np.linalg.inv(Vu)
            
            EA = -VA @ T @ Vui @ e
            EA = EA.reshape(nObs,nParams)
        
            Au = A - EA
            yu = y - EA@x
            
            xnew = np.linalg.inv(Au.T@Vui@Au) @ Au.T @ Vui @ yu
            
            test = sqDiff(x,xnew)
            
            if verbosity==1:print(count, " : ", test)
            
            running = (test > 1e-10) & (count<countMax)
            
            count+=1
            
            x = xnew
        
        T = np.kron(np.eye(nObs),x)
                        
        Vu = Vy + T.T @ VA @ T
                
        Vui = np.linalg.inv(Vu)

        l = Vui@(y-A@x)
        
        sigma2 = l.T @ (y-A@x) / (nObs-nParams)    
    
        Vx = sigma2*np.linalg.inv(Au.T@Vui@Au)
     
        return x,Vx
    
    def indexedWeightedTotalLeastSquares(self,AIn,yIn,VAIn,VyIn,I):
        
        '''
        A = 
        [A11,A12,A13 ...]
        [A21,A22,A23 ...]
        [A31,A32,A33 ...]
        
        VA = 
        [A11,A12,A13,...A21,A22,A23]
        
        Estimate x with only the indexed points
        Estimate covariance with al points
        '''
        
        verbosity = self.options["verbosity"]
                
        nParams = AIn.shape[1]
        nObs = np.sum(I)
        
        A = AIn[I,:]
        y = yIn[I,:]
        VA = VAIn[np.repeat(I,nParams),:][:,np.repeat(I,nParams)]
        Vy = VyIn[I,:][:,I]
        
        
        #Obtain first estimate
        x = np.linalg.inv(A.T@A) @ A.T @ y
    
        running = True
        countMax = 100
        count = 0
        
        while running: 
            
            e = y - A@x
            
            T = np.kron(np.eye(nObs),x)
                        
            Vu = Vy + T.T @ VA @ T
                
            Vui = np.linalg.inv(Vu)
            
            EA = -VA @ T @ Vui @ e
            EA = EA.reshape(nObs,nParams)
        
            Au = A - EA
            yu = y - EA@x
            
            xnew = np.linalg.inv(Au.T@Vui@Au) @ Au.T @ Vui @ yu
            
            test = sqDiff(x,xnew)
            
            if verbosity==1:print(count, " : ", test)
            
            running = (test > 1e-10) & (count<countMax)
            
            count+=1
            
            x = xnew
        
        #Estimate sigma
        A = AIn[:,:]
        y = yIn[:,:]
        VA = VAIn[:,:]
        Vy = VyIn[:,:]
        nObs = y.shape[0]
        nParams = A.shape[1]
            
        T = np.kron(np.eye(nObs),x)
                        
        Vu = Vy + T.T @ VA @ T
                
        Vui2 = np.linalg.inv(Vu)
        
        sigma2 = (y-A@x).T @ Vui2 @ (y-A@x) / (nObs-nParams)
        
        if sigma2<1:sigma2 = 1
        # print(sigma2)
        Vx = sigma2*np.linalg.inv(Au.T@Vui@Au)
        
        # sigma3 = (y-A@x).T @ (y-A@x) / (nObs-nParams)

        # Vx1 = sigma3*np.linalg.inv(Au.T@Au)
        
        return x,Vx
    
    def simpleLeastSquares(self,A,y):
        
        return  np.linalg.inv(A.T@A) @ A.T @ y