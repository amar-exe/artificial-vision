# -*- coding: utf-8 -*-

import ktools18 as kt
import os


datasets = ('Im_Cars', 'Im_Charcoal', 'Im_People',
'Im_Leaves', 'Im_Shops')

print(datasets)

wdir = os.getcwd()

abs_paths = []

for ds in datasets:
    abs_paths.append(wdir + "\\" + ds)
    
var1 = abs_paths[0]
var2 = var1
