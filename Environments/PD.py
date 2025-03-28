
import numpy as np

class PD(object):


    def __init__(self, **kwargs):
        """
        Initialize the IRP model with given or default parameters.

        """
        self.tstable = kwargs.get('tstable', 1e2)
        self.tmax = kwargs.get('tmax', 1e4)

        self.T = kwargs.get('T', 1.5)  # D/C
        self.R = kwargs.get('R', 1) # C/C
        self.P = kwargs.get('P', 0) # D/D
        self.S = kwargs.get('S', -0.2) # C/D

    
    def compute_profits(self, p):
        """
        Compute profits for each firm given their prices.

        Parameters
        ----------
        p : ndarray
            Array of prices set by each firm.

        Returns
        -------
        ndarray
            Array of profits for each firm.
        """ 
        # 0 = D
        # 1 = C

        p1, p2 = p[0], p[1]

        if p1 == p2:
            if p1 == 0:
                return np.array([self.P, self.P])
            else:
                return np.array([self.R, self.R])
        elif p1 == 0:
            return np.array([self.T, self.S])
        else:
            return np.array([self.S, self.T]) 
        

    def compute_p_competitive_monopoly(self):
        """
        Compute competitive and monopoly prices.

        Returns
        -------
        tuple
            A tuple containing competitive and monopoly prices.
        """
        p_competitive = np.array([self.P, self.P])
        p_monopoly = np.array([self.R, self.R])
        return p_competitive, p_monopoly
