import numpy as np
class ContinuationVariable(object):
    def __init__(self,name,target):
        self.name = name
        self.target = target
        self.value = np.nan
        self.steps = []
