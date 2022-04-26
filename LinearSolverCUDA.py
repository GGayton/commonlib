import cupy as xp

def sqDiff(a,b):
    
    return xp.sum((a - b)**2)**0.5

class LinearSolver():
    
    def __init__(self):
                
        self.options = {
            "verbosity":1,
            "TLSlimit": 1e-10
            }

    def setOptions(self,optionsIn):
        self.options = optionsIn

    def getOptions(self):
        return self.options

    def totalLeastSquares(self,A,y,VA,Vy):
        
        verbosity = self.options["verbosity"]
        TLSlimit = self.options["TLSlimit"]
        
        nObs = y.shape[0]
        xparams = A.shape[1]
        
        #Obtain first estimate
        x = xp.linalg.inv(A.T@A) @ A.T @ y
    
        running = True
        countMax = 100000
        count = 0
        
        while running: 
            
            e = y - A@x
            
            T = xp.kron(xp.eye(nObs),x)
                        
            Vu = Vy + T.T @ VA @ T
                
            Vui = xp.linalg.inv(Vu)
            
            EA = -VA @ T @ Vui @ e
            EA = EA.reshape(nObs,xparams)
        
            Au = A - EA
            yu = y - EA@x
            
            xnew = xp.linalg.inv(Au.T@Vui@Au) @ Au.T @ Vui @ yu
            
            test = sqDiff(x,xnew)
            
            if verbosity==1:print(count, " : ", test.get())
            
            running = (test > TLSlimit) & (count<countMax)
            
            count+=1
            
            x = xnew
            
            Vx = xp.linalg.inv(Au.T@Vui@Au)
     
        return x,Vx