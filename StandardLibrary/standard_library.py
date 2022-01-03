# standard_library.py
"""Python Essentials: The Standard Library.
Patricia D. Bata
BUDS Program
July 15, 2019
"""

import calculator as calc
from itertools import combinations, chain, permutations
import random

"""
Calculator script contents:
def product(a,b):
    return a*b

def add(a,b):
    return a+b

def sqrt(a,b):
    return math.sqrt(a,b)
"""

# Problem 1
def prob1(L):
    """Returns the minimum, maximum, andand average of the entries of L (in that order).
    
    Parameters:
        L (list): the input list for the function. An example of an input list in function: prob1((range(0,9))) or prob1((1,2,3,4,5,6,7,8,9)
    Returns:
        The minimum value, maximum value, and average value of the list L, in that order. Return output is a tuple.
    """
    return min(L), max(L), sum(L)/len(L)
    raise NotImplementedError("Problem 1 Incomplete")
    
def prob2():
    """Determine which Python objects are mutable and which are immutable.
    Test numbers, strings, lists, tuples, and sets. Print your results.
     Parameters:
       None.
    Returns:
        A print on whether the 5 different objects are mutable or immutable.
        
    Object order in code: integer, string, list, tuple, set.
    """
    mut=[]
    i1 = 1
    i2 = i1
    i1 = i1+1
    if i1 == i2:
        mut.append('Mutable')
    else: mut.append('Immutable')

    str1 = 'hello'
    str2 = str1
    str1 = str1 + str1
    if str1 == str2:
        mut.append('Mutable')
    else: mut.append('Immutable')
    
    l1 = [1,2,3]
    l2=l1
    l1[0:2] = [4,5,6]
    if l1 == l2:
        mut.append('Mutable')
    else: mut.append('Immutable')
    
    my_tuple = (1,2,3,4,5)
    my_tuple2 = my_tuple
    my_tuple2 += (1,) 
    if my_tuple == my_tuple2:
        mut.append('Mutable')
    else: mut.append('Immutable')
    
    set1 = {1,2,3,4,5,6,7,8,9}
    set2 = set1
    set2.pop()
    if set1 == set2:
        mut.append('Mutable')
    else: mut.append('Immutable')

    return print('Are the object types mutable? \n Integer: {} \n String: {} \n List: {} \n Tuple: {} \n Set: {}'.format (mut[0],mut[1],mut[2],mut[3],mut[4]))
    raise NotImplementedError("Problem 2 Incomplete")


# Problem 3
def hypot(a, b):
    """Calculate and return the length of the hypotenuse of a right triangle.
    Do not use any functions other than those that are imported from your
    'calculator' module.

    Parameters:
        a(int or float): the length one of the sides of the triangle.
        b(int or float): the length the other non-hypotenuse side of the triangle.
    Returns:
        The length of the triangle's hypotenuse as a float.
    """
    a2 = calc.product(a,a)
    b2 = calc.product(b,b)
    return calc.sqrt(calc.add(a2,b2))
    raise NotImplementedError("Problem 3 Incomplete")


# Problem 4
def power_set(A):
    """Use itertools to compute the power set of A.

    Parameters:
        A (iterable): a str, list, set, tuple, or other iterable collection.

    Returns:
        (list(sets)): The power set of A as a list of sets.
    """
    powlist = list(A)
    powset = []
    for i  in range(0,(len(A)+1)):
        for j in list(combinations(powlist,i)):
            powset.append(set(j))
    return powset
    raise NotImplementedError("Problem 4 Incomplete")


# Problem 5: Implement shut the box.
def shutthebox():
    """Plays the game shut the box"""
    import box as box
    import random, time
    nom = input('Enter your name: ')
    tlimit = float(input("Set a time limit (in seconds): "))
    print("\n\n")
    die = list(range(1,7))
    posnum = list(range(1,10))
    start = time.time()
    turn = 1
    while time.time()-start<=tlimit: 
        if sum(posnum) > 6 :
            rand = random.choice(die) + random.choice(die)
        elif sum(posnum) <=6:
            rand = random.choice(die)
        trem = round(tlimit - (time.time()-start),3)
        val = box.isvalid(rand,posnum)
        print("--------------------------")
        print(" Round {}\n".format(turn))
        print(" Remaining Numbers: {}".format(posnum)) 
        print(" Time Remaining: {} seconds".format(trem))
        print("\n Rolled dice: {}".format(rand))
        if val == True:
            x = input(" Numbers you want to remove (separate numbers with a space): ")
            print("--------------------------")
            rem = box.parse_input(x, posnum)
            while sum(rem) != rand:
                print("Invalid input. Try again.")
                print("--------------------------")
                print("\n Round {}\n".format(turn))
                print(" Remaining Numbers: {}".format(posnum)) 
                print(" Time Remaining: {} seconds".format(trem))
                print("\n Rolled dice: {}".format(rand))
                x = input(" Numbers you want to remove (separate numbers with a space): ")
                print("--------------------------")
                rem = box.parse_input(x, posnum)
                val = box.isvalid(rand,posnum)
            if sum(rem) == rand:
                posnum = [x for x in posnum if x not in rem]
                turn += 1
                continue
        elif val == False:
            print("--------------------------")
            print("You lose. You don't have numbers you can remove, {}!".format(nom))
            print("Score: {} points".format(sum(posnum)))
            print("Time played: {} seconds".format((round((time.time()-start),3))))
            print("--------------------------")
            break
        if posnum == []:
            print("--------------------------")
            print("Congratulations, {}!! You shut the box!".format(nom))
            print("Score: {} points".format(sum(posnum)))
            print("Time played: {} seconds".format(round((time.time()-start),3)))
            print("--------------------------")
            
    if (time.time()-start)>tlimit:
        print("--------------------------")
        print("You ran out of time. Try again, {}.".format(nom))
        print("Score: {} points".format(sum(posnum)))
        print("Time played: {} seconds".format(round((time.time()-start),3)))
        print("--------------------------")
    
    
    
    
    
    
    
    
    
    
    
    
    
    