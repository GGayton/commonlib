import numpy as np

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
 