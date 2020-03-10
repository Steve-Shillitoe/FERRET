"""This module contains functions that calculate the variation 
of concentration or MR signal with time according to a tracer kinetic model.
"""
import MathsTools as tools
import ExceptionHandling as exceptionHandler
import numpy as np
from scipy.optimize import fsolve
from joblib import Parallel, delayed
import logging
logger = logging.getLogger(__name__)

def TRISTAN_Rat_Model_v2_0_4_7T(xData2DArray, Kbh, Khe, 
                                 constantsString):
        """This function contains the algorithm for calculating 
           how the MR signal from a 3D scan varies with time using the 
           TRISTAN Rat Model v2.0 at 4.7T
        
                Input Parameters
                ----------------
                    xData2DArray - time (sec) and spleen signal 1D arrays 
                        stacked into one 2D array.
              
                    Khe - Hepatocyte Uptake Rate (mL/min/mL)
                    Kbh - Biliary Efflux Rate (mL/min/mL) 
                    constantsString - String representation of a dictionary 
                    of constant name:value pairs used to convert concentrations 
                    predicted by this model to MR signal values.

                Returns
                -------
                St_rel - list of calculated MR signals at each of the 
                    time points in array 'time'.
                """ 
        try:
            exceptionHandler.modelFunctionInfoLogger()
            t = xData2DArray[:,0]
            Ss = xData2DArray[:,1]

            TR = 0.0058
            baseline = 4
            FA = 20
            r1p = 6.4
            r1h = 7.6
            R10_s = 0.7458
            R10_l = 1.3203
            ve_s = 0.314
            ve_l = 0.230

            # Convert to concentrations
            # n_jobs set to 1 to turn off parallel processing
            # because parallel processing caused a segmentation
            # fault in the compiled version of this application.
            # This is not a problem in the uncompiled script
            R1_s = [Parallel(n_jobs=1)(delayed(fsolve)
              (tools.spgr3d_func, x0=0, 
               args = (FA, TR, R10_s, baseline, Ss[p])) 
               for p in np.arange(0,len(t)))]
            R1_s = np.squeeze(R1_s)
        
            DR1_s = R1_s - R10_s
      
            Th = (1-ve_l)/(Kbh/60)
            DR1_l = (ve_l/ve_s)*DR1_s + (r1h/r1p)*((Khe/60)/ve_s)*Th*tools.expconv(Th,t,DR1_s,'TRISTAN_Rat_Model_v2_0_4_7T')
        
            # Convert to signal
            c = np.cos(FA*np.pi/180)
            R1_l = R10_l + DR1_l
            E1 = np.exp(-TR*R1_l)
            Sl = (1-E1)/(1-c*E1)
            Sl0 = sum(Sl[0:baseline-1])/baseline
            Sl_rel = Sl/Sl0
      
        
            return(Sl_rel) #Returns tissue signal relative to the baseline St/St_baseline
        
        except ZeroDivisionError as zde:
            exceptionHandler.handleDivByZeroException(zde)
        except Exception as e:
            exceptionHandler.handleGeneralException(e)
