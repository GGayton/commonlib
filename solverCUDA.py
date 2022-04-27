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

class NonLinearSolver():
    
    def __init__(self, jacobianFunction, transformFunction):
        
        self.transformFunction = transformFunction
        self.jacobianFunction = jacobianFunction
        
        self.options = {
            "iterationMax": 2000,
            "failureCountMax": 10,
            "minimumChange": 1e-5,
            "verbosity":1
            }

    def setOptions(self,optionsIn):
        self.options = optionsIn

    def getOptions(self):
        return self.options
               
    def solve(self,x,y,params):
        
        ITERATION_MAX = self.options["iterationMax"]
        FAILURE_COUNT_MAX = self.options["failureCountMax"]
        CHANGE_MIN = self.options["minimumChange"]
        VERBOSITY = self.options["verbosity"]
        
        optimised = False
        failureCount = 0
        epoch = 0
        
        dampingFactor = xp.array(10)
        
        #Initialise first loss and jacobian
        loss = y - self.transformFunction(x,params)
        lossSum = xp.sum(loss**2)
        J = self.jacobianFunction(x,params)
         
        while not optimised:
            
            JtJ = J.T @ J
                
            JtJ  = JtJ + dampingFactor * xp.diag(JtJ)*xp.eye(J.shape[1])
            
            # update,_,_,_ = xp.linalg.lstsq(JtJ, J.T@loss, rcond=None)
            update = xp.linalg.inv(JtJ)@ J.T@loss

            #Assign the update
            paramsUpdate = params + update
                
            #Calculate a new loss
            lossUpdate = y - self.transformFunction(x,paramsUpdate)
            
            #Has this improved the loss?
            lossSumUpdate = xp.sum(lossUpdate**2)
                                 
            lossSumChange = lossSumUpdate - lossSum
            
            condition = lossSumChange<0

            #If condition is True
            if condition:
                 
                #Decrease the damping
                dampingFactor = self.iterateDamping(dampingFactor, 0.5)
                
                #Accept new value of loss, loss sum and parameters
                loss = lossUpdate
                lossSum = lossSumUpdate
                params = paramsUpdate

                #Calculate new jacobian
                J = self.jacobianFunction(x,params)
                              
                #Reset consecutive failure count
                if lossSumChange>-CHANGE_MIN:
                    failureCount += 1
                else:
                    failureCount = 0
                                                                
            #If condition2 fails    
            else:
                
                #Increase the damping
                dampingFactor = self.iterateDamping(dampingFactor, 5)
                failureCount += 1
                                            
            #Optimisation Check
            optimised = (epoch>ITERATION_MAX) | (failureCount>FAILURE_COUNT_MAX)
            if VERBOSITY == 1:self.printUpdate(epoch, dampingFactor, lossSum)
            
            epoch += 1

        if VERBOSITY != 0: 
            self.printUpdate(epoch, dampingFactor, lossSum)            
            print("\n===", "FINISHED", "===\n")
        
        loss = y - self.transformFunction(x,params)

        return params, J, loss
    
    def iterateDamping(self,dampingFactor, change):
        
        dampingFactor = dampingFactor*change
        
        if dampingFactor<1e-5: dampingFactor = dampingFactor*0 + 1e-5
        
        return dampingFactor
    
    def printUpdate(self, Epoch, damping_factor, loss_sum):

        print("Epoch: {:02}".format(Epoch),", lambda: {:02f}".format(damping_factor.get()),", loss: {:02f}".format(loss_sum.get()))
    