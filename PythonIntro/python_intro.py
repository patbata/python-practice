# python_intro.py
"""Python Essentials: Introduction to Python
Patricia D. Bata
BUDS Program
10 July 2019
"""

#Problem 1
"""Not graded; print 'Hello world!' to console"""
print ("Hello, world!")

#Problem 2
"""Define pi = 3.14159"""
pi = 3.14159
def sphere_volume(r):
    """Calculate and return the volume of a sphere with radius r.

    Parameters:
        r (int): the radius of the sphere.
    Returns:
        The volume of the sphere.
    """
    return (4/3)*pi*r**3

#Problem 3
def isolate(a, b, c, d, e):
    """Accept 5 arguments: first 3 separated by 5 spaces and the last 2 are separated with 1 space
    
    Parameters:
        a = string 1
        b = string 2
        c = string 3
        d = string 4
        e = string 5
        
    Returns:
        Doesn't actually return; prints the first 3 strings with 5 spaces in between and the last two strings with 1 space.
    """
    print(a, b, c, sep='     ', end=' ')
    print(d, e)
    
#Problem 4
def first_half(z):
    '''Accepts a string z (should be in quoatation) and returns half the string
    x is the half of legnth of string, rounded down (ex: with length 5, x is 2)
    
    Parameters:
        z (str): string that user wishes to cut in half
        
    Returns:
        Half the string, when the length of the string is odd, the quotient is rounded up.
    '''
    x = (len(z)//2)
    return z[:x]
def backward(y):
    '''Accepts string (should be in quotation) and reverses the order
    
    Parameters:
        y (str): string that the user wishes to spell backwards
        
    Returns:
        A string that is the reverse spelling of y.
    
    '''
    return y[::-1]

#Problem 5

def list_ops():
    mylist = ['bear','ant','cat','dog']
    """For a specific list of animals, perform list functions to return a specific order of animals:
    [fox, hawk, dog, bearhunter]. The final list is returned
    
    Steps: append 'eagle' at end; replace entry at index 2 with 'fox'; pop entry at index 1;
    sort in reverse alphabetical order; replace wherever the element 'eagle' is with 'hawk';
    append/add 'hunter' at the end item of the list.
    """
    mylist.append('eagle') #append eagle at end
    mylist[2] = "fox" #replace entry at index 2 with "fox"
    mylist.pop(1) #pop the entry at index 1
    mylist.sort(reverse=True) #sort in reverse alphabetical order
    mylist[mylist.index('eagle')]='hawk' #replace wherever the "eagle" element is with "hawk"
    mylist[len(mylist)-1]=mylist[len(mylist)-1]+'hunter' #append to last elemnt in list with -"hunter"
    return (mylist)
      
#Problem 6
my_vowels = "aeiou" #string of vowels
def pig_latin(f):
    """Returns whatever string inputted (f) to pig-latin. Depending on starting letter:
    
    Starts with vowel: returns word + 'hay' at the end (allow = allowhay)
    Starts with consonant: returns word with first letter moved to the end then adding 'ay' (hello = ellohay)
    
    Place string in code in quotation
    
    Parameter:
        f(str): word user wants to translate to pig latin (in quotation)
    
    Returns:
        Pig latin translated word, as a string, dependent on starting letter (vowel or consonant)
        """
    if f[0] in my_vowels:
        return f+'hay'
    else:
        return f[1:len(f)]+f[0]+'ay'
    
#Problem 7    
def palindrome():
    """Calculates and return the largest palindrome and its factors given a certain range.

    Parameters:
        i: the first list of factors (that are integers) in the given range.
        b: the second list of factors (that are integers) in the given range.
    Returns:
        The largest palindrome produced from the factors i and j
    """
    pal = []                        #list of palindromes
    fact = []                       #list of factors
    for i in range (100,999):       #range of factor 1. In this case, 100-999
        for j in range (100,999):   #range of factor 2. In this case, 100-999
           prod = str(i*j)          #Converts the product of i and j as a string
           if prod == prod[::-1]:   #Checks if the string (product) is the same forward and backward (palindrome property)
              pal_int = int(prod)   #If satisfies if statement, string is converted back to integer 
              pal.append(pal_int)   #Adds to list of palindromes found
              fact.append({i,j})    #Adds to list of palindrome factors
           else:
                continue
    y = max(pal)                    #Largest palindrome in list pal
    maxfact = fact[pal.index(y)]
    return y # y is the palindrome while maxfactor are its factors (not included in return)

#Problem 8
def alt_harmonic(n):
    """Calculates and returns the sum of the alternating harmonic series up to the first n terms.

    Parameters:
        n (int): number of terms in the series
    Returns:
        The sum of the alternating harmonic series up to the nth term.
    """
    return sum([((-1)**(n+1))/n for n in (range(1,n+1))])
    