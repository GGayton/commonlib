#%%
import cv2
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
import numpy as np

from commonlib.typecasting import rescale_to_uint8

def plot_coords(out, coords,down_scale):
    for i in range(len(coords)):
        cv2.circle(out, (coords[i][1], coords[i][0]), 1, (0,0,255),2)

    if len(coords)>=3:
        lines = np.array(coords, np.int32)//down_scale
        lines = lines.reshape((-1, 1, 2))
        lines=np.flip(lines, 2)
        cv2.polylines(out, [lines], True, (0, 0, 255), 2)
    cv2.imshow('Cropping tool', out)

#define an area on image using opencv
def define_area_ocv(image, down_scale=6):
    # function to display the coordinates of 
    # of the points clicked on the image  
    res = image.shape

    u,v = np.meshgrid(np.linspace(1,res[0],res[0]), np.linspace(1,res[1],res[1]), indexing = 'ij')
    vec = np.concatenate((u.astype(np.uint16).reshape(-1,1), v.astype(np.uint16).reshape(-1,1)), axis=1)
    # function to display the coordinates of 
    # of the points clicked on the image  
    coords = []
    def click_event(event, x, y, flags, params): 
        nonlocal coords,test_image
        # checking for left mouse clicks
        out = test_image.copy()

        if event == cv2.EVENT_LBUTTONDOWN: 
            coords.append((down_scale*y,down_scale*x))
            plot_coords(out,coords,down_scale)

        elif event == cv2.EVENT_RBUTTONDOWN:
            coords = coords[:-1]
            plot_coords(out,coords,down_scale)
    
    test_image = rescale_to_uint8(image, max_value=1023)
    
    width = int(test_image.shape[0]/down_scale)
    height = int(test_image.shape[1]/down_scale)
    test_image = np.repeat(test_image.reshape(width,height,1),3,2)

    test_image = cv2.resize(test_image, (width, height))
    
    cv2.imshow('Cropping tool', test_image)
    # setting mouse hadler for the image 
    # and calling the click_event() function 
    cv2.setMouseCallback('Cropping tool', click_event) 
    cv2.waitKey(0)  
      
    #closing all open windows  
    cv2.destroyAllWindows()
    
    p = matplotlib.path.Path(coords)
    mask = p.contains_points(vec)
    
    mask = mask.reshape(res[0], res[1])
    
    return mask

#Define an area on image using matplotlib
def define_area(image):
    
    coords = []
    index = None
    stopping = True
    
    res = image.shape
    
    u,v = np.meshgrid(np.linspace(1,res[0],res[0]), np.linspace(1,res[1],res[1]), indexing = 'ij')
    vec = np.concatenate((u.astype(np.uint16).reshape(-1,1), v.astype(np.uint16).reshape(-1,1)), axis=1)
    
    def onclick(event):
        
        nonlocal coords
        
        fig = plt.gcf()
                        
        if event.button is MouseButton.RIGHT:
            coords = coords[:-1]

        elif event.button is MouseButton.LEFT:
            coords.append((event.ydata, event.xdata))
            
        a = np.array(coords)

        plt.plot(a[:,1],a[:,0],'r')
        plt.plot(a[:,1],a[:,0],'r.')
        
        fig.canvas.draw()
     
    def onclose(event):

        nonlocal coords,index,stopping
                                            
        p = matplotlib.path.Path(coords)
        index = p.contains_points(vec)
        
        fig = plt.gcf()

        fig.canvas.stop_event_loop()
        plt.close(fig)
        stopping = False
    
    fig = plt.figure()
    
    plt.imshow(image)
    
    plt.title('Please enclose region of interest')
    fig.canvas.mpl_connect('button_press_event', onclick)
    fig.canvas.mpl_connect('close_event', onclose)
    plt.show()
    
    while stopping:
        
        plt.pause(0.5)
        
    return index.reshape(res[0],res[1])

#define a set of points on image using matplolib
def define_centres(image, downScale = 1):
        
    coords = []
    stopping = True
    
    downScaledImage = image[::downScale, ::downScale]
    
    def onclick(event):
        
        nonlocal coords, scatt
        
        fig = plt.gcf()
                                
        if event.button is MouseButton.RIGHT:
            coords = coords[:-1]

        elif event.button is MouseButton.LEFT:
            coords.append((event.ydata * downScale, event.xdata * downScale))
            
        a = np.array(coords)
        
        plot(a)
        
        fig.canvas.draw()
  
    def onclose(event):

        nonlocal stopping,fig
                    
        fig.canvas.stop_event_loop()
        plt.close(fig)
        stopping = False
    
    def plot(a):
        
        nonlocal scatt
        
        scatt.remove()
        scatt = plt.scatter(a[:,1] / downScale,a[:,0] / downScale, s=5, c='r')
                
    fig = plt.figure()
    plt.imshow(downScaledImage)
    scatt = plt.scatter([],[], s=1, c='r')
    
    plt.title('Please identify centres of features')
    fig.canvas.mpl_connect('button_press_event', onclick)
    fig.canvas.mpl_connect('close_event', onclose)
    plt.show()
    
    while stopping:
        
        plt.pause(0.5)
        
    return coords


if __name__ == "__main__":
    import numpy as np
    image = np.random.rand(1000,1000)*255//1
    
    define_area_ocv(image, 1)
# %%
