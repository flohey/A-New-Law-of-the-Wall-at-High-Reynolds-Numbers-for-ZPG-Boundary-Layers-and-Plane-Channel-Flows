import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd


def g0(yplus,C_plus=6.17,kappa=0.37):
    """ 
    Panton's Reynolds stress function g0(y⁺)

    INPUT:
        yplus  - y⁺-cooridnate
        C_plus - C⁺ in Panton's equation
        kappa  - \kappa in Panton's equation

    RETURN:
        g0 - Reynolds stress function
    """

    return 2/np.pi * np.arctan(2*kappa*yplus/np.pi)*(1-np.exp(-yplus/C_plus))**2

def dudy(yplus,Uplus):
    """ 
     dU⁺/dy⁺= 1 -  g0(y⁺)  -> U+(y⁺) = y⁺ - \\int g0(y⁺) dy⁺
    """

    # Definition of ODE Panton parameters
    #---------------------------------------
    C_plus = 6.17        # C⁺ in Panton's equation
    kappa  = 0.37        # \kappa in Panton's equation

    return 1 - g0(yplus,C_plus=C_plus,kappa=kappa)


# Parameters numerical integration
# #--------------------------------- 
yp0, Up0 = 0, 0   # boundary conditions y⁺, U⁺
ypmax    = 1e4    # max. y⁺-value to which to perform integration
nyp      = 1e5    # no. y⁺-values to use for integration
yp       = np.linspace(.1, ypmax, int(nyp))
method   = "RK45" # integration method, here Runge-Kutta 4,5

# Solve dU⁺/dy⁺ from y+=0 to y+=1, with U⁺(y⁺=0) = 0
# #-----------------------------------------------------
sol   = solve_ivp(dudy, t_span=[yp0, ypmax],y0=[Up0], t_eval=yp,method=method)

###########################
# Plot New Law of the Wall
###########################
fig,axs = plt.subplots(figsize=(10,5))

axs.semilogx(sol.t,sol.y[0],color="black",linewidth=2,label="Extended Law of the Wall")
axs.set_xlabel(r'Dimensionless Distance from the Wall $y^{+}$')
axs.set_ylabel(r'Dimensionless Velocity $\overline{U}^{+}$')
axs.grid()
axs.legend()
plt.show()
###########################
# Save to Excel
###########################

#df_data = pd.DataFrame({"y+":sol.t,"U+":sol.y[0]})
#df_data.to_excel("NLW.xlsx")
