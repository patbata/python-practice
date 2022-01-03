# object_oriented.py
"""Python Essentials: Object Oriented Programming.
Patricia D. Bata
BUDS Program
July 22,2018
"""
import math

class Backpack:
    """A Backpack object class. Has a name and a list of contents.
    Attributes:
        name (str): the name of the backpack's owner.
        contents (list): the contents of the backpack.
        color (str): color of the backpack
        max_size (int): max number of items in backpack (default=5)
    """

    # Problem 1: Modify __init__() and put(), and write dump().
    def __init__(self, name,color, max_size = 5):
        """Set the name and initialize an empty list of contents.
        Parameters:
            name (str): the name of the backpack's owner.
            color (str): color of the backpack
            max_size (int): max number of items in backpack (default = 5)
        """
        self.name = name
        self.color = color
        self.contents = []
        self.max_size = int(max_size)
        
    def put(self, item):
        """Add an item to the backpack's list of contents.
        If number of items (len(self.contents)) is more than max_size
        an error appears: No room!
        
        To add an item, type the variable of the backpack (ex. x = Backpack("Pat","black")) then put an item
            ex: x.put('item')
        
        Parameters:
            item (str): Item you want placed in bag
        Returns:
            self.contents list is appended with the item in 'put' function
        """
        if len(self.contents) >= self.max_size: #Limits contents to max_size
            print("No room!")
        else:
            self.contents.append(item)
            #Use 'self.contents', not just 'contents'.
            
    def take(self, item):
        """Remove an item from the backpack's list of contents.
        
        To remove an item, type the variable of the backpack (ex. x = Backpack("Pat","black")) then take an item
            ex: x.remove('item')
        
        Parameters:
            item (str): Item you want removed from bag
        Returns:
            self.contents list is edited by removing item.
        """
        self.contents.remove(item)

    def dump(self):
        """Removes all items from backpack by restoring to empty list.
        
        Parameters:
            self: backpack stored in a variable
            
        Returns:
            self.contents is cleared to an empty list
        """
        self.contents.clear()
    
    # Magic Methods -----------------------------------------------------------

    # Problem 3: Write __eq__() and __str__().
    def __add__(self, other):
        """Add the number of contents of each Backpack.
        
        Parameters:
            self: one Backpack object with a name, color, contents, and max_size
            other: another Backpack object with a name, color, contents, and max_size
        Returns:
            An integer with the added number of items (contents) of the two bags.
        """
        return len(self.contents) + len(other.contents)

    def __lt__(self, other):
        """Compare two backpacks. If 'self' has fewer contents
        than 'other', return True. Otherwise, return False.
        
        Parameters:
            self: one Backpack object with a name, color, contents, and max_size
            other: another Backpack object with a name, color, contents, and max_size
        Returns:
            Boolean. If critera stated above is met, returns TRUE. Otherwise, returns FALSE.
        """
        return len(self.contents) < len(other.contents)
    
    def __eq__(self,other):
        """Checks if two Backpack objects are equal if they have the same: name, color, and number of contents.
        
        If one attribute is not equal, self == other returns False.
        
        Parameters:
            self: one Backpack object with a name, color, contents, and max_size
            other: another Backpack object with a name, color, contents, and max_size
        Returns:
            Boolean. If critera stated above is met, returns TRUE. Otherwise, returns FALSE.
        """
        return self.name == other.name and self.color == other.color and len(self.contents) == len(other.contents)
    
    def __str__(self):
        """When you type print(<backpack variable>), it shows owner, color, size, max size, and contents of the backpack
        Format is organized with <attribute>:  <value>}"""
        return "Owner:\t\t"+str(self.name)+"\nColor:\t\t"+str(self.color)+"\nSize:\t\t"+str(len(self.contents))+"\nMax Size:\t"+str(self.max_size)+"\nContents:\t"+str(self.contents)

# An example of inheritance. You are not required to modify this class.
class Knapsack(Backpack):
    """A Knapsack object class. Inherits from the Backpack class.
    A knapsack is smaller than a backpack and can be tied closed.
    Attributes:
        name (str): the name of the knapsack's owner.
        color (str): the color of the knapsack.
        max_size (int): the maximum number of items that can fit inside.
        contents (list): the contents of the backpack.
        closed (bool): whether or not the knapsack is tied shut.
    """
    def __init__(self, name, color):
        """Use the Backpack constructor to initialize the name, color,
        and max_size attributes. A knapsack only holds 3 item by default.
        Parameters:
            name (str): the name of the knapsack's owner.
            color (str): the color of the knapsack.
            max_size (int): the maximum number of items that can fit inside.
        """
        Backpack.__init__(self, name, color, max_size=3)
        self.closed = True

    def put(self, item):
        """If the knapsack is untied, use the Backpack.put() method."""
        if self.closed:
            print("I'm closed!")
        else:
            Backpack.put(self, item)

    def take(self, item):
        """If the knapsack is untied, use the Backpack.take() method."""
        if self.closed:
            print("I'm closed!")
        else:
            Backpack.take(self, item)

    def weight(self):
        """Calculate the weight of the knapsack by counting the length of the
        string representations of each item in the contents list.
        """
        return sum(len(str(item)) for item in self.contents)


# Problem 2: Write a 'Jetpack' class that inherits from the 'Backpack' class.

class Jetpack(Backpack):
    """A Jetpack object class. Inherits from the Backpack class.
    A Jetpack is like a backpack but has fuel and can fly.
    
    Attributes:
        name (str): the name of the jetpack's owner
        color (str): the color of the jetpack
        max_size (int): the maximum number of items that can fit inside.
        contents (list): the contents of the jetpack.
        fuel (int): amount of fuel in jetpack
    """
    
    def __init__(self, name, color, max_size = 2, fuel = 10):
        """Use the Backpack contructor to initialize the name, color,
        and max_size attributes. A knapsack only holds 3 items by default.
        
        Parameters:
            name (str): the name of the jetpack's owner
            color (str): the color of the jetpack
            max_size (int): the maximum number of items that can fit inside.
            contents (list): the contents of the jetpack.
            fuel (int or float): amount of fuel in jetpack (this attribute is added)
        """
        Backpack.__init__(self, name, color, max_size)
        self.fuel = fuel
        
    def fly(self, amt):
        """Checks if amount of fuel (amt) user wants to burn is enough in tank.
        If not, it'll show an error message without depleting the fuel.
        If there's enough fuel, the new amount of fuel will be
        the initial fuel - burned fuel (amt)
        """
        if self.fuel < amt:
            print("Not enough fuel!")
        else:
            self.fuel = self.fuel - amt
    
    def dump(self):
        """Overrides backpack dump function to dump all contents and fuel
        
        Empty fuel tank is self.jetpack.fuel = 0 (integer 0)
        
        Empty content (dumped) jetpack is
        self.jetpack.contents = [] (empty list)"""
        
        self.contents.clear()
        self.fuel = 0


# Problem 4: Write a 'ComplexNumber' class.
        
class ComplexNumber:
    """A Complex Number object class. Has two components: real and imaginary.
    Form: a + bj where a and b are real numbers'
    
    Attributes:
        real (int or float): the real component of complex.
        imag (int or float): the imag component of complex.
    
    To display the format of a complex number, use the print() function on the ComplexNumber (or variable you stored it to).
    """
    def __init__(self, real, imag):
        """Set the real and imaginary components where real and imag are
        in the set of real numbers R.
        
        Parameters:
            real (int or float): the real component of complex.
            img (int or float): the imag component of complex.
        """
        self.real = real
        self.imag = imag
        
    def conjugate(self):
        """Returns the conjugate of the inputted complex number; negating the imag term"""
        return ComplexNumber(self.real,-self.imag)
    
    def __str__(self):
        """(a + bj) is printed for inputs of real and imaginary as a string (b >= 0)
        (a - bj) is printed when b < 0  for inputs of real and imaginary as a string"""
        if self.imag >= 0:
            return "("+str(self.real) + "+" + str(self.imag)+'j'+")"
        else:
            return "("+str(self.real) + str(self.imag)+'j'+")"
        
    def __abs__(self):
        """Returns the magnitude of the ComplexNumber with the abs() function."""
        return math.sqrt((self.real)**2+(self.imag)**2)
    
    def __eq__(self,other):
        """Checks if two ComplexNumbers are equal if and only if their real and
        imaginary components are equal."""
        return self.real == other.real and self.imag == other.imag
    
    def __add__(self,other):
        """Adds two ComplexNumbers Together"""
        return ComplexNumber((self.real+other.real),(self.imag+other.imag))
    
    def __sub__(self,other):
        """Subtracts two ComplexNumbers from each other"""
        return ComplexNumber((self.real-other.real),(self.imag-other.imag))
    
    def __mul__(self,other):
        """Multiplies two Complex numbers"""
        return ComplexNumber( ((self.real*other.real)-(self.imag*other.imag)) , ((self.real*other.imag)+(self.imag*other.real)) )
    
    def __truediv__(self,other):
        """Divides two complex numbers"""
        return ComplexNumber((((self*other.conjugate()).real)/(other*other.conjugate()).real), (((self*other.conjugate()).imag)/(other*other.conjugate()).real))