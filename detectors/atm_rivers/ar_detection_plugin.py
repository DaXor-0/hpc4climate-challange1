import comin
import sys
import numpy as np
from datetime import datetime

jg = 1
domain = comin.descrdata_get_domain(jg)

def reshape_icon2d(field):
    domain = comin.descrdata_get_domain(jg)
    ncells = domain.cells.ncells
    return np.reshape(field, (-1,), order="F")[:ncells]


@comin.EP_SECONDARY_CONSTRUCTOR
def secondary_constrector():
 # access ICON vars

@comin.EP_ATM_WRITE_OUTPUT_BEFORE
def callback_fct():
# writting callbacks

