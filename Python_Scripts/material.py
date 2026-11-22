import numpy as np

def plane_stress_matrix(E, nu):
    c=E/(1.0-nu**2)
    return c*np.array([[1.0,nu,0.0],[nu,1.0,0.0],[0.0,0.0,(1.0-nu)/2.0]])

MATERIALS={
    1:{'name':'Steel clamp plate','E_MPa':200000.0,'nu':0.30},
    2:{'name':'Aluminium bracket/gusset','E_MPa':70000.0,'nu':0.33},
    3:{'name':'Steel wall','E_MPa':200000.0,'nu':0.30},
}
