import os
import glob

#%% 1. Obtains the current working directory (cwd)
cwd=os.getcwd()

print("The current working dir: ", cwd)
print()

#%% 2. Changes the cwd to another specified 
os.chdir("../")

cwd=os.getcwd()
print("The new current working dir: ", cwd)
print()


#%% 3. Reads all the files in that directory
print("List of files in cwd: ")

print(os.listdir(cwd))
print()

#%% 4. Collects the different types of extensions
extensions={}

print("Different types of extension: ")
for i in os.listdir(cwd):
    split_tup = os.path.splitext(i)
    
    if(len(split_tup[1])>1):
        if(split_tup[1] in extensions):
            extensions[split_tup[1]]+=1
            
        else:
            extensions[split_tup[1]]=1
            
        print(split_tup[1])

print()

#%% 5. And says how many files of each type are in that directory    
print("Numbers of each extension: ")

for i in extensions:
    print("There are", extensions[i], "extensions of file type", i)
