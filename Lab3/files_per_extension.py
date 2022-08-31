"""
files_per_extension(new_path, extension='*.*', FLAG=True, VERBOSE=False)

This function:
 1. Changes the cwd to the one specified as argument: "new_path"
 2. Reads all the files of the type specified in the extension in that directory.
 By default, extension contains '*.*', reading all the files
 3. Collects the different types of extensions (only if needed)
 4. Says how many files of each type are in that directory
 5. And, if FLAG is True, returns a list where each element is a two
 elements list with the extension and the number of files having that
 extension
"""

def files_per_extension(new_path, extension='*.*', FLAG=True, VERBOSE=False):
    import os
    import glob
    #1
    print("1:")
    os.chdir(new_path)
    cwd=os.getcwd()
    print("Path changed.")
    print()
    
    #2
    print("2:")
    targetExtension = new_path+extension
    print(glob.glob(targetExtension))
    print("Files read.")
    print()
    
    #3
    if(VERBOSE):
        print("3:")
        print("Different types of extension: ")
        
        extension_list=set()
        
        for i in os.listdir(cwd):
            split_tup = os.path.splitext(i)
            
            if(len(split_tup[1])>1):
                extension_list.add(split_tup[1])

        print(extension_list)
    print("Extensions printed.")
    print()
    
    #4
    print("4:")
    extensions={}

    print("Numbers of files of each type: ")
    
    for i in os.listdir(cwd):
        split_tup = os.path.splitext(i)
        
        if(len(split_tup[1])>1):
            if(split_tup[1] in extensions):
                extensions[split_tup[1]]+=1
                
            else:
                extensions[split_tup[1]]=1
                
    for i in extensions:
        print("There are", extensions[i], "files of extension type", i)
    
    print("Extensions listed.")
    print()
    
    #5
    if(FLAG):
        print("5:")
        flag_list=[]
        ext_set=[]
        for i in extensions:
            ext_set.append(i)
            ext_set.append(extensions[i])
            
            flag_list.append(ext_set)
            ext_set=[]
        print(flag_list)
        return "FLAG list returned."
        

print(files_per_extension("../", extension='*.*', FLAG=True, VERBOSE=True))
