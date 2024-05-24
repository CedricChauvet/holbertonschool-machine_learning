#!/usr/bin/env python3
"""
Regularization project
by Ced
"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """
    doc
    """
    if (opt_cost - cost) > threshold:
        return (False, 0)
    
    elif (opt_cost - cost) <= threshold:
        if count < patience -1:
            return (False, count +1)
        if count == patience -1:
            return(True, patience)
        
        