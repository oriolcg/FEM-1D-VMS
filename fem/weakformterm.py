import sys
import numpy as np


class WeakFormTerm(object):
    """Weak form term."""

    def __init__(self, weak_form_type, constant=1):
        self.type = weak_form_type
        self.constant = constant

    @classmethod
    def w_u(cls, constant=1):
        return cls("W_u", constant)

    @classmethod
    def w_du(cls, constant=1):
        return cls("W_du", constant)

    @classmethod
    def dw_u(cls, constant=1):
        return cls("dW_u", constant)

    @classmethod
    def dw_du(cls, constant=1):
        return cls("dW_du", constant)
