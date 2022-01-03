#0 exceptions_fileIO.py
"""Python Essentials: Exceptions and File Input/Output.
Patricia D. Bata
BUDS Program 2019
July 28, 2019
"""

from random import choice
import numpy as np
#%%
# Problem 1
def arithmagic():
    """
    Asks 4 inputs from user:
        step_1: Enter a 3-digit number where the first and last digits differ
        by 2 or more.
            ERRORS: (1) not a 3 digit number (2) first and last digits must
            differ by 2 or more
        step_2: Enter the reverse of step_1
            ERRORS: (1) If step_2 is not the reverse of step_1
        step_3: Enter the positive difference of step_2 and step_1
            ERRORS: (1) If the number inputted is not the positive difference
            of the first two numbers
        step_4: Enter the reverse of step_3
            ERRORS: (1) Inputted step_4 does not equal backwards of step_3
    
    The leading zeroes in the input are parsed.
    """
    #Asks for 3-digit number where last two digits differ by 2 or more
    step_1 = input("Enter a 3-digit number where the first and last "
                                           "digits differ by 2 or more: ")
    # Errors arise if not 3 digit number
    if int(step_1) < 100 or int(step_1) >= 999:
        raise ValueError("Please enter a 3-digit number.")
    # Error if first and last don't differ by 2 or more
    # (checks absolute of difference)
    if abs(int(str(int(step_1))[2])-int(str(int(step_1))[0]))<2:
        raise ValueError("The first and last numbers must differ by 2 or more.")
    #Asks for the reverse of the first number
    step_2 = input("Enter the reverse of the first number, obtained "
                                              "by reading it backwards: ")
    # Error if step_2 input is not the reverse of step_1 input
    if int(step_2) != int(str(int(step_1))[::-1]):
        raise ValueError("That's not the reverse of the first number.")
    
    # Asks for the positive difference of step_1 and step_2
    step_3 = input("Enter the positive difference of these numbers: ")
    # Checks if input at step_3 is not the positive difference of previous steps.
    if int(step_3) != abs(int(step_2)-int(step_1)):
        raise ValueError("That's not the positive difference of the first two numbers.")
  
    # Asks for the reverse of step_3
    step_4 = input("Enter the reverse of the previous result: ")
    # Error if step_4 input is not the reverse of step_3 input
    if int(step_4) != int(str(int(step_3))[::-1]):
        raise ValueError("That's not the reverse of the third number.")    
    
    # Prints the sum of step_3 and step_4 as 1089 (ta-da!)
    print(str(int(step_3)), "+", str(int(step_4)), "= 1089 (ta-da!)")

#%%
# Problem 2
def random_walk(max_iters=1e12):
    """
    Random walk is a path created by a sequence of steps and max_iters is
    default 1e12. Function tries to talk a walk. When a KeyboardInterrupt
    error is triggered, the function catches it and 'stops the walk' and
    prints: 'Process interrupted at iteration: i'. 
    
    If no KeyboardInterrupt occurs, the walk is completed and prints 'Process
    completed.'
    
    In both cases, the function returns the walk.
    """
    # Function tries to take a random walk starting at 0
    try:
        print("Taking a random walk...",end='')
        walk = 0
        directions = [1, -1]
        for i in range(int(max_iters)):
                walk += choice(directions)
    # When a KeyboardInterrupt occurs, the walk stops and prints
    # 'process interrupted at iteration: i'
    except KeyboardInterrupt:
        print("process interupted at iteration:",i)
    # If the function finishes iterating, it finishes the walk and prints:
    # 'process completed.'
    else:
        print("process completed.")
    return walk

#%%
# Problems 3 and 4: Write a 'ContentFilter' class.
class ContentFilter:
    """A ContentFilter object class. Implement the constructor so that
    it accepts the name of a file to be read.
    
    Attributes:
        name (str): name of the file to be opened
        contents (str): content of file in one line string
    """
    def __init__(self, name):
        self.name = name
        self.contents = ''
        """ Input the name of file constructor wants to open and
        empty list of contents.
        
        Parameters:
            name(str): name of the file to be read/opened
            contents(str): single-line string of contents of file
        """
        # While the errors: FileNotFoundError, TypeError, and OSError exists,
        # ask for an input of a file to read from user.
        # When error is not found, read file and store contents into the variable
        while True:
            try:
                myfile = open(self.name,'r')
            except (FileNotFoundError, TypeError, OSError):
                self.name = input("Please enter a valid file name: ")
            else:
                contents = myfile.read()
                myfile.close()
                self.contents = contents
                break
    
    def uniform(self, new, case = "upper",mode = 'w'):
        """Writes a new output file with uniform case. If case = upper,
        data is written in uppercase. If case = lower, write the data in
        lowercase. Default case is "upper" and default mode is "w".
        
        ValueError arises when mode specified is not 'w', 'x', or 'a' OR
        if case specified is neither 'upper' nor 'lower'
        
        Parameters:
            new (str): the name of the new output file
            case (str): the case of the characters in new file; default 'upper'
                        (can only be 'upper' or 'lower').
            mode (str): the mode for writing the text file; default 'w'
                        (can only be 'w', 'x', 'a')
        """
        if mode not in ['w','x','a']:
            raise ValueError("That's not a valid file access mode. Only 'w', 'x', and 'a' are valid.")
        if case not in ['upper','lower']:
            raise ValueError("That's not valid letter case. Only 'upper' and 'lower' are valid.")
        with open(new, mode) as outfile:
            if case == "upper":
                if mode == "a":
                    outfile.write('\n'+self.contents.upper())
                else:
                    outfile.write(self.contents.upper())    
            elif case == "lower":
                if mode == "a":
                    outfile.write('\n'+self.contents.lower())
                else:
                    outfile.write(self.contents.lower()) 

    def reverse(self, new, unit = 'line', mode = 'w'):
        """Writes a new output file with the contents of source file reversed.
        If unit = 'line', then the line order of the contents is reversed but
            not the words
        If unit = 'word', then the words in a line are presented in reverse
            order
        
        Default unit is "line" and default mode is "w".
        
        ValueError arises when mode specified is not 'w', 'x', or 'a' OR
        if unit specified is neither 'line' nor 'word'
        
        Parameters:
            new (str): the name of the new output file
            unit (str): the unit of reversal of the contents; default 'line'
                        (can only be 'line' or 'word').
            mode (str): the mode for writing the text file; default 'w'
                        (can only be 'w', 'x', 'a')
        """
        # ValueError if mode specified is not in list of possible modes
        if mode not in ['w','x','a']:
            raise ValueError("That's not a valid file access mode. Only 'w', 'x', and 'a' are valid.")
        # ValueError if unit specified is not in list of possible unit
        if unit not in ['line','word']:
            raise ValueError("That's not valid reverse unit. Only 'line' and 'word' are valid.")
        # Creates a new file specified as "new" with mode specified (default = 'w')
        with open(new, mode) as outfile:
            # If specified unit of reversal is "word"
            if unit == "word":
                #If the mode is append, content starts the output in a new line from the existing content of the file.
                if mode == "a":
                    #Stripped and split list of the contents and this list was reversed per element (word)
                    con = [word[::-1] for word in self.contents.strip().split('\n')]
                    # List was joined into string with "\n"
                    out = '\n'.join(con)
                    # File is written as "new" with out as its contents + a new line is added at the end
                    outfile.write(('\n'+out))
                else:
                    #Stripped and split list of the contents and this list was reversed per element (word)
                    con = [word[::-1] for word in self.contents.strip().split('\n')]
                    # List was joined into string with "\n"
                    out = '\n'.join(con)
                    # File is written as "new" with out as its contents + a new line is added at the end
                    outfile.write((out))
                    
            if unit == "line":
                #If the mode is append, content starts the output in a new line from the existing content of the file.
                if mode == "a":
                    #Stripped and split list of the contents and this list was reversed per line
                    lines = self.contents.strip().split('\n')[::-1]
                    # List was joined into string with "\n"
                    out = '\n'.join(lines)
                    # File is written as "new" with out as its contents + a new line is added at the end
                    outfile.write(('\n'+out))
                else:
                    #Stripped and split list of the contents and this list was reversed per line
                    lines = self.contents.strip().split('\n')[::-1]
                    # List was joined into string with "\n"
                    out = '\n'.join(lines)
                    # File is written as "new" with out as its contents + a new line is added at the end
                    outfile.write((out))
                

            
    def transpose(self, new, mode = 'w'):
        """Writes a new output file with the contents of source file "transposed".
        
        Default mode is "w".
        
        Stored in trans: the content of the source file was stripped and split
        for every \n and every line was furter split into elements of a list.
        The list was then converted to an np.array the array was transposed
        The individual elements per row were then joined using list comprehension
        Finally the list was converted to a string by joining each list with '\n'
        
        ValueError arises when mode specified is not 'w', 'x', or 'a'.
        
        Parameters:
            new (str): the name of the new output file
            mode (str): the mode for writing the text file; default 'w'
                        (can only be 'w', 'x', 'a')
        """
        # ValueError if mode specified is not in list of possible modes
        if mode not in ['w','x','a']:
            raise ValueError("That's not a valid file access mode. Only 'w', 'x', and 'a' are valid.")
        # Creates a new file specified as "new" with mode specified (default = 'w')
        with open(new, mode) as outfile:
            # The content of the source file was stripped and split by '\n'
            cont = self.contents.strip().split('\n')
            # Every line was further split into elements of a list.
            firstlist = [element.split(' ') for element in cont]
            # List was then converted to an np.array the array was transposed
            array = np.array(firstlist).T
            # The individual elements per row were then joined using list comprehension
            lastlist = [' '.join(listy) for listy in array]
            # The list elements were joined for every '\n'
            trans = '\n'.join(lastlist)
            #Output file is written with trans as contents
            outfile.write((trans))


    def __str__(self):
        """When you type print(<ContentFilter variable>), it shows
        Source file, total characters, # alphabetic characters, # of digits,
        #of white spaces, # of lines
            
        Format is organized with <attribute>:  <value>}"""
        return ("Source File:\t\t"+str(self.name)+"\nTotal Characters:\t"+
                str(len(self.contents))+"\nAlphabetic characters:\t"+
                str(sum([s.isalpha() for s in self.contents]))+"\nNumerical characters:\t"
                +str(sum([s.isdigit() for s in self.contents]))+"\nWhitespace Characters:\t"
                +str(sum([s.isspace() for s in self.contents]))+ "\nNumber of Lines:\t"
                +str(len(self.contents.split("\n"))))

#%%
"""One line code for prob4(transpose):
    
    trans = '\n'.join([' '.join(listy) for listy in np.array([element.split(' ') for element in self.contents.strip().split('\n')]).T])
"""








