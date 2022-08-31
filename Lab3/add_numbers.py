"""
This function adds up to 3 numbers introduced as arguments.
 Realise that the second and third are optional arguments, with values 1
 and 0, respectively, as default.
@author: afazlic Created on February 2020
"""

def add_numbers(num1, num2=1, num3=0):
    return(num1+num2+num3)

print(add_numbers(5)) # 5+1+0
print(add_numbers(5, num2=2)) # 5+2+0
print(add_numbers(5, num3=8)) # 5+1+8
print(add_numbers(5, num2=4, num3=8)) # 5+4+8
print(add_numbers(5, num3=4, num2=8)) # 5+8+4

"""
 The function works on the principle of adding
3 numbers together, but 2 of the numbers(second
and third do not have to be put in and default values of 1
0 will be put in their place.)

 When we define which variables hold which values, their 
position in function definition does not matter. 

For example: 
    print(add_numbers(5, num2=4, num3=8)) 
    print(add_numbers(5, num3=4, num2=8)) 
    
 These two lines are not the same, as the numbers
 num3 and num2 switched their values. The result of
 this particular function will be the same but the 
 variables values are not.
"""
