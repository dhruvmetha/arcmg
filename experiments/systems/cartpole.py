import numpy as np
from arcmg.system import BaseSystem

class Cartpole(BaseSystem):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "cartpole"

    
        
