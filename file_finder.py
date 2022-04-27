import os

def recursive_file_finder(directory, fileType):
    
    dirList = os.listdir(directory)
    
    print(r"[0] - \\")
    
    for i in range(0, len(dirList)):
        print("[{}] - ".format(i+1) + dirList[i])
        
    choice = int(input("Choose file:"))
    
    if choice == 0:
        
        if directory[-2:] == "\\":directory = directory[:-2]
        
        cap = directory[::-1].find("\\") + 1
        
        print(cap, directory)
        
        directory = directory[:-cap]
        
        return recursive_file_finder(directory, fileType)
    
    elif dirList[choice-1][-len(fileType):] == fileType:
        
        return directory + "\\" + dirList[choice-1]
    
    else:
        
        return recursive_file_finder(directory + "\\" + dirList[choice-1], fileType)

def recursive_dir_finder(directory):
    
    dirList = os.listdir(directory)
    
    print(r"[0] - \\")
    
    for i in range(0, len(dirList)):
        print("[{}] - ".format(i+1) + dirList[i])
        
    choice = input("Choose file:")

    #Check for completion
    if choice[-1]=='!':
        choice = int(choice[:-1])
        return directory + "\\" + dirList[choice-1]
    else:
        choice = int(choice)
    
    #Check for up level
    if choice == 0:
        
        if directory[-2:] == "\\":directory = directory[:-2]
        
        cap = directory[::-1].find("\\") + 1
        
        print(cap, directory)
        
        directory = directory[:-cap]
        
        return recursive_dir_finder(directory)

    #Go down one level
    else:
        
        return recursive_dir_finder(directory + "\\" + dirList[choice-1])

def recursive_dir_list_finder(init_dir):

    dir_list = [recursive_dir_finder(init_dir)]
    
    finished = False
    while not finished:
        dir_list += [recursive_dir_finder(init_dir)]

        choice = input("continue? y/n")

        if choice=='n':
            finished=True
        
    return dir_list

def list_dir(folder, condition):
    out = []
    for item in os.listdir(folder):
        if condition(item):
            out.append(item)
    
    return out
