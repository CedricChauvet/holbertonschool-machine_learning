#!/usr/bin/env python3
"""
probability project
"""


class Poisson():
    """
    
    """

    def __init__(self, data=None, lambtha=1.):
        self.lambtha = lambtha  # lambtha is the expected number of occurences in a given time frame
        self.data = data          # data is a list of the data to be used to estimate the distribution
 
        if  type(self.data) == list :
            
            if len(data)>=2:
                s=0
                for i in data:
                    s+=i
                mean = s /len(data)
                self.lambtha = mean




            else:
                raise ValueError("data must contain multiple values")

        elif self.data == None:
            pass

        else:
            raise TypeError("data must be a list")

        if self.lambtha <= 0:
            raise ValueError("lambtha must be a positive value")
        
        