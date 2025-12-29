# -*- coding: utf-8 -*-
"""
Created on Mon Dec 29 13:27:19 2025

@author: alber
"""



class VariableUncertainty ():

    r"""Strategy of Active Learning to select instances more significative based on uncertainty.

    The variable uncertainty sampler selects samples for labeling based on the uncertainty of the prediction.
    The higher the uncertainty, the more likely the sample will be selected for labeling. The uncertainty
    measure is compared with a random variable uncertainty limit.


    References
    ----------
    [^1]: I. Zliobaite, A. Bifet, B.Pfahringer, G. Holmes. “Active Learning with Drifting Streaming Data”, IEEE Transactions on Neural Netowrks and Learning Systems, Vol.25 (1), pp.27-39, 2014.

"""

    def __init__(self, theta: float = 0.95, s=0.5):
        super().__init__()

        self.theta = theta
        self.s = s


    def isSignificative(self, x, y_pred) -> bool:
        """Ask for the label of a current instance.

        Based on the uncertainty of the base classifier, it checks whether the current instance should be labeled.

        Parameters
        ----------
        x
            Instance

        y_pred

           Arrays of predicted labels


        Returns
        -------
        selected
            A boolean indicating whether a label is needed.
            True for selected instance.
            False for not selecte instance.


        """
        maximum_posteriori = max(y_pred.values())
        selected = False


        if maximum_posteriori < self.theta:
            self.theta = self.theta*(1-self.s)
            selected = True
        else:
            self.theta = self.theta*(1+self.s)
            selected = False

        return selected