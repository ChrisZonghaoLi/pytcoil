# This script is used to parse the data from s-parameter

import math
import skrf as rf
import numpy as np
import scipy
from sklearn.linear_model import LinearRegression



def sub_para_ext(network, slicer_sw = False, slicer = 1):
    """
    This function intakes a network model and finds the input and output shunt branch
    admittance (substrate parasitic) consists of Cox, Rsub and Csub by using linear
    regression technique; details can be found here:
    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1703682
    
    ----------
    Parameters
    ----------
    network : network object by skrf
        network.Network type, it is the network model of a circuit
    slicer_sw (optional): boolean
        by defult it is false, so the function will automatically neglect
        the first lower half of the frequency point data used for linear
        regression analysis for parasitic extraction; in case you would like
        to specify how many data point from the low frequency to skip, set this
        to True
    slicer (optional): int
        when slicer_sw is set to True, this input controls how many data point
        from the low frequency to skip
    y_params : array
        Y parameters of the network
    y_shunt_in : list
        admittanc of the input shunt branch represents substrate parasitics
    y_shunt_out : list
        admittanc of the out shunt branch represents substrate parasitics 
    model1_in/out, model2_in/out: 
        models used for linear regression
    k1_1/2_in/out, k2_1/2_in/out:
        linear regression coefficient after LS is done. k1 is the intercept point, 
        k2 is the slope of the line

    Returns
    -------
    ((Cox_in, Rsub_in, Csub_in),(Cox_out, Rsub_out, Csub_out)) : tuple
        shunt branch at input and output represent substrate parasitic

    """

    f = network.f
    w = 2 * math.pi * f/1e9
    y_params = network.y
    # pick the point from linear region to do the regression to avoid numerial
    # purtabation in ASITIC
    
    if slicer_sw == True:
        w = w[int(slicer):]
        y_params = y_params[int(slicer):]
    else:
        w = w[int(len(y_params)/2):]
        y_params = y_params[int(len(y_params)/2):]
    
    y11_list = [y_params[i][0][0] for i in range(len(y_params))]
    y12_list = [y_params[i][0][1] for i in range(len(y_params))]
    y22_list = [y_params[i][1][1] for i in range(len(y_params))]

    y_shunt_in = [y11_list[i] + y12_list[i] for i in range(len(y_params))]
    y_shunt_out = [y22_list[i] + y12_list[i] for i in range(len(y_params))]

    f1_in = [1/(np.real(y_shunt_in[i])) * w[i]**2 for i in range(len(y_params))] 
    f2_in = [np.imag(y_shunt_in[i])/(np.real(y_shunt_in[i])) * w[i] for i in range(len(y_params))] 
    
    f1_out = [1/(np.real(y_shunt_out[i])) * w[i]**2 for i in range(len(y_params))] 
    f2_out = [np.imag(y_shunt_out[i])/(np.real(y_shunt_out[i])) * w[i] for i in range(len(y_params))] 
    
    #plt.plot(np.array(w)**2, np.array(f1_out))
    #plt.plot(np.array(w)**2, np.array(f2_out))
    
    model1_in = LinearRegression().fit(np.array(w**2).reshape(-1,1), np.array(f1_in).reshape(-1,1))
    model2_in = LinearRegression().fit(np.array(w**2).reshape(-1,1), np.array(f2_in).reshape(-1,1))

    model1_out = LinearRegression().fit(np.array(w**2).reshape(-1,1), np.array(f1_out).reshape(-1,1))
    model2_out = LinearRegression().fit(np.array(w**2).reshape(-1,1), np.array(f2_out).reshape(-1,1))

    k1_1_in = model1_in.intercept_[0]
    k1_2_in = model1_in.coef_[0][0]
    k2_1_in = model2_in.intercept_[0]
    k2_2_in = model2_in.coef_[0][0]
    
    k1_1_out = model1_out.intercept_[0]
    k1_2_out = model1_out.coef_[0][0]
    k2_1_out = model2_out.intercept_[0]
    k2_2_out = model2_out.coef_[0][0]

    a1_in = 1 / k1_1_in / (1e9**2)
    b1_in = k1_2_in * a1_in
    c1_in = k2_1_in * 1e9 * a1_in
    
    a1_out = 1 / k1_1_out / (1e9**2)
    b1_out = k1_2_out * a1_out
    c1_out = k2_1_out * 1e9 * a1_out

    
    counter = 1
    flag = False
    
    while k1_1_in < 0 or k1_2_in < 0:

        print('** Input subsrate parasitic exibits inductivity at low frequency, data skipping from the low frequency is running now for {}. **'.format(network.name))
        # This happens when the subsrate parasitic is very inductive at low frequency
        # and in fact, still using capacitor in the shunt branch is not quite right,
        # so here the appraoch is to keep pushing the data to high frequency by the
        # "counter" until the data above this can be used to run linear regression

        if counter < len(y_params):

            y11_list = [y_params[i][0][0] for i in range(counter,len(y_params))]
            y12_list = [y_params[i][0][1] for i in range(counter,len(y_params))]
            y22_list = [y_params[i][1][1] for i in range(counter,len(y_params))]
            
            y_shunt_in = np.array(y11_list) + np.array(y12_list)
            f1_in = np.array(1/(np.real(y_shunt_in))) * np.array(w[counter:])**2 

            f2_in = np.array(np.imag(y_shunt_in)/(np.real(y_shunt_in))) * np.array(w[counter:]) 
            
            model1_in = LinearRegression().fit(np.array(w[counter:]**2).reshape(-1,1), f1_in.reshape(-1,1))
            model2_in = LinearRegression().fit(np.array(w[counter:]**2).reshape(-1,1), f2_in.reshape(-1,1))
            
            k1_1_in = model1_in.intercept_[0]
            
            k1_2_in = model1_in.coef_[0][0]
            k2_1_in = model2_in.intercept_[0]
            k2_2_in = model2_in.coef_[0][0]
                    
            a1_in = 1 / k1_1_in / (1e9**2)
            b1_in = k1_2_in * a1_in
            c1_in = k2_1_in * 1e9 * a1_in
            
            counter = counter + 1
            print(counter)
 
        else:
            print('** No good linear regression is found. Possible ASITIC numerical error. Flag is raised for {}. **'.format(network.name))
            
            flag = True
            #print(flag)
            
            break
    
    counter = 1
    flag = False
    
    while k1_1_out < 0 or k1_2_out < 0:

        print('** Output subsrate parasitic exibits inductivity at low frequency, data skipping from the low frequency is running now for {}. **'.format(network.name))
        # This happens when the subsrate parasitic is very inductive at low frequency
        # and in fact, still using capacitor in the shunt branch is not quite right,
        # so here the appraoch is to keep pushing the data to high frequency by the
        # "counter" until the data above this can be used to run linear regression

        if counter < len(y_params):

            y11_list = [y_params[i][0][0] for i in range(counter,len(y_params))]
            y12_list = [y_params[i][0][1] for i in range(counter,len(y_params))]
            y22_list = [y_params[i][1][1] for i in range(counter,len(y_params))]
            
            y_shunt_out = np.array(y22_list) + np.array(y12_list)
            f1_out = np.array(1/(np.real(y_shunt_out))) * np.array(w[counter:])**2 

            f2_out = np.array(np.imag(y_shunt_out)/(np.real(y_shunt_out))) * np.array(w[counter:]) 
            
            model1_out = LinearRegression().fit(np.array(w[counter:]**2).reshape(-1,1), f1_out.reshape(-1,1))
            model2_out = LinearRegression().fit(np.array(w[counter:]**2).reshape(-1,1), f2_out.reshape(-1,1))
            
            k1_1_out = model1_out.intercept_[0]
            
            k1_2_out = model1_out.coef_[0][0]
            k2_1_out = model2_out.intercept_[0]
            k2_2_out = model2_out.coef_[0][0]
                    
            a1_out = 1 / k1_1_out / (1e9**2)
            b1_out = k1_2_out * a1_out
            c1_out = k2_1_out * 1e9 * a1_out
            
            counter = counter + 1
            print(counter)
 
        else:
            print('** No good linear regression is found. Possible ASITIC numerical error. Flag is raised for {}. **'.format(network.name))
            
            flag = True
            #print(flag)
            
            break
    
    Cox_in = c1_in
    Rsub_in = a1_in / (c1_in)**2
    Csub_in = abs(Cox_in - (b1_in / Rsub_in**2)**0.5)

    Cox_out = c1_out
    Rsub_out = a1_out / (c1_out)**2
    Csub_out = abs(Cox_out - (b1_out / Rsub_out**2)**0.5)    
    

    return ((Cox_in, Rsub_in, Csub_in),(Cox_out, Rsub_out, Csub_out), flag)


def series_ext(network, guess_sw = False, guess = [1,1]):
    """
    This function intakes a network model and finds series component of
    Ls and Rs that represent inductor's inductance and loss using 
    Levenberg-Maqut method, and Lskin and Rskin that model skin effect'; 
    details can be found here:
    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1703682
    
    ----------
    Parameters
    ----------
    network : network object by skrf
        network.Network type, it is the network model of a circuit
    guess_sw (optional): boolean
        Initial guess switch, default is False, if it is set to True, 
        the list value of guess input will be used for the intial guess
        value of Lskin and Rskin
    guess (optional): list [Lskin, Rskin]
        list of intial guess value for Lskin (in nH) and Rskin (in Ohm)
    Ls : float
        series inductance of inductor
    Rs : float
        series resistance of inductor
    Lskin : float
        model skin effect
    Rskin: 
        model skin effect
    k1_1/2_in/out, k2_1/2_in/out:
        linear regression coefficient after LS is done. k1 is the intercept point, 
        k2 is the slope of the line

    Returns
    -------
    Ls/1e9, Rs, Lskin/1e9, Rskin : tuple
        series branch component values

    """
    
    f = network.f
    w = 2 * math.pi * f/1e9
    y_params = network.y
    
    t_train = np.array(w)
    y_train = np.array([np.imag(-1/y_params[i][0][1])/w[i] for i in range(len(y_params))])
    
    Ls = np.min([np.imag(-1/y_params[i][0][1])/w[i] for i in range(len(y_params))]) # in nH
    Rs = np.min([np.real(-1/y_params[i][0][1]) for i in range(len(y_params))])
    
    fun = lambda x, t, y: Ls + Rs**2 * x[0] /((Rs + x[1])**2 + t**2 * x[0]**2) - y    
    
    # first is inductance in nH, second is resistance in Ohm
    if guess_sw == True:
        x0 = np.array(guess)
    else:
        x0 = np.array([1, 1]) 
    
    results = scipy.optimize.least_squares(fun, x0, bounds = ([0,0],[np.inf,np.inf]), args=(t_train, y_train))
    
    Lskin = results.x[0] # in nH
    Rskin = results.x[1] # in Ohm
    
    return Ls/1e9, Rs, Lskin/1e9, Rskin


def pi_eq_ckt(network, slicer_sw = False, slicer = 1, guess_sw = False, guess = [1,1]):
    """
    This function is a wrapper of two functions sub_para_ext() and
    series_ext() that extract the substrate parasitic and series component
    of a pi-equivalent circuit (without winding cap)
    
    ----------
    Parameters
    ----------
    network : network object by skrf
        network.Network type, it is the network model of a circuit
    slicer_sw (optional): boolean
        by defult it is false, so the function will automatically neglect
        the first lower half of the frequency point data used for linear
        regression analysis for parasitic extraction; in case you would like
        to specify how many data point from the low frequency to skip, set this
        to True
    slicer (optional): int
        when slicer_sw is set to True, this input controls how many data point
        from the low frequency to skip
    guess_sw (optional): boolean
        Initial guess switch, default is False, if it is set to True, 
        the list value of guess input will be used for the intial guess
        value of Lskin and Rskin
    guess (optional): list [Lskin, Rskin]
        list of intial guess value for Lskin (in nH) and Rskin (in Ohm)
    
    Returns
    -------
    series (Ls, Rs, Lskin, Rskin), sub_para (Cox, Rsub, Csub): tuple
        all components of a pi equivalent circuit except the winding capacitor

    """
    
    sub_para = sub_para_ext(network, slicer_sw, slicer)
    series = series_ext(network, guess_sw = False, guess = [1,1])
    return series, sub_para


def two_pi_eq_ckt(network1, network2, network3, middle_branch = True,
                  slicer_sw1 = False, slicer1 = 1, 
                  guess_sw1 = False, guess1 = [1,1], 
                  slicer_sw2 = False, slicer2 = 1, 
                  guess_sw2 = False, guess2 = [1,1]):
    
    """
    Solve for eq. ckt. of 3-tap inductor/t-coil
    
    This function is a wrapper of pi_eq_ckt(). It can combine pi eq. ckt.
    of inductor "a" and "b" (network1 and newtork2, respectively), and combine
    them into a 2-pi eq. ckt.; details can be found here:
    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1703682
    
    Alternatively, since the middle branch will have an artificial impact
    on the parasitic seen from the middle tap, this branch can be removed. In 
    this case, the input and output parasitic is extracted from network3 that
    is the network obtaine by leaving the middle tap floating; see page 288 
    on Sorin's book "High-Frequency Integrated Circuits".
    
    ----------
    Parameters
    ----------
    network1 : network object by skrf
        network.Network type, it is the network model of inductor a
    network2 : network object by skrf
        network.Network type, it is the network model of inductor b
    network3 : network object by skrf
        network.Network type, it is the network model of cascading inductor
        a and b while middle tap is floating
    middle_branch (optional): boolean
        set to True by default; if False, then there will be no shunt branch
        at middle tap.
    slicer_sw (optional): boolean
        by defult it is false, so the function will automatically neglect
        the first lower half of the frequency point data used for linear
        regression analysis for parasitic extraction; in case you would like
        to specify how many data point from the low frequency to skip, set this
        to True
    slicer (optional): int
        when slicer_sw is set to True, this input controls how many data point
        from the low frequency to skip
    guess_sw (optional): boolean
        Initial guess switch, default is False, if it is set to True, 
        the list value of guess input will be used for the intial guess
        value of Lskin and Rskin
    guess (optional): list [Lskin, Rskin]
        list of intial guess value for Lskin (in nH) and Rskin (in Ohm)
    
    Returns
    -------
    ((Ls1, Rs1, Lskin1, Rskin1),(Ls2, Rs2, Lskin2, Rskin2),
                (Cox_in, Rsub_in, Csub_in),(Cox_out, Rsub_out, Csub_out)) : tuple
        all components of a 2-pi equivalent circuit except the winding capacitor
        and coupling factor if middle shunt branch is omitted
        
        or 
    ((Ls1, Rs1, Lskin1, Rskin1),(Ls2, Rs2, Lskin2, Rskin2),
                (Cox_in1, Rsub_in1, Csub_in1),
                (Cox_mid, Rsub_mid, Csub_mid),
                (Cox_out2, Rsub_out2, Csub_out2)) : tuple
        all components of a 2-pi equivalent circuit except the winding capacitor
        and coupling factor if middle shunt branch is included
        
    """
    
    
    pi_eq_ckt1 = pi_eq_ckt(network1, slicer_sw1, slicer1, 
                  guess_sw1, guess1)
    
    pi_eq_ckt2 = pi_eq_ckt(network2, slicer_sw2, slicer2, 
                  guess_sw2, guess2)
    
    Ls1, Rs1, Lskin1, Rskin1 = pi_eq_ckt1[0]
    Ls2, Rs2, Lskin2, Rskin2 = pi_eq_ckt2[0]
    
    flag1 = pi_eq_ckt1[1][2]
    flag2 = pi_eq_ckt2[1][2]
    
    if middle_branch == True:
        if flag1 == False and flag2 == False:
            Cox_in1, Rsub_in1, Csub_in1 = pi_eq_ckt1[1][0]
            Cox_out1, Rsub_out1, Csub_out1 = pi_eq_ckt1[1][1]
            Cox_in2, Rsub_in2, Csub_in2 = pi_eq_ckt2[1][0]
            Cox_out2, Rsub_out2, Csub_out2 = pi_eq_ckt2[1][1]
        elif flag1 == False and flag2 == True: 
            print('** Numerical errors found for ind b, its input and output parasitic will be estimated to be the same as ind a. **')  
            Cox_in1, Rsub_in1, Csub_in1 = pi_eq_ckt1[1][0]
            Cox_out1, Rsub_out1, Csub_out1 = pi_eq_ckt1[1][1]
            Cox_in2, Rsub_in2, Csub_in2 = Cox_in1, Rsub_in1, Csub_in1
            Cox_out2, Rsub_out2, Csub_out2 = Cox_out1, Rsub_out1, Csub_out1
        elif flag1 == True and flag2 == False:
            print('** Numerical errors found for ind a, its input and output parasitic will be estimated to be the same as ind b. **')  
            Cox_in2, Rsub_in2, Csub_in2 = pi_eq_ckt2[1][0]
            Cox_out2, Rsub_out2, Csub_out2 = pi_eq_ckt2[1][1]
            Cox_in1, Rsub_in1, Csub_in1 = Cox_in2, Rsub_in2, Csub_in2
            Cox_out1, Rsub_out1, Csub_out1 = Cox_out2, Rsub_out2, Csub_out2
        else:
            print('** Bad numerical results. Please use pi-model for the tcoil instead. ** \n')    
        
        Cox_mid = Cox_out1 + Cox_in2
        Csub_mid = Csub_out1 + Csub_in2
        Rsub_mid = Rsub_out1 * Rsub_in2 / (Rsub_out1 + Rsub_in2)
        return ((Ls1, Rs1, Lskin1, Rskin1),(Ls2, Rs2, Lskin2, Rskin2),
                (Cox_in1, Rsub_in1, Csub_in1),
                (Cox_mid, Rsub_mid, Csub_mid),
                (Cox_out2, Rsub_out2, Csub_out2))
    
    else:
        print('** Middle branch is excluded; pi-model is used for the t-coil. ** \n')
        pi_eq_ckt3 = pi_eq_ckt(network3)
        Cox_in, Rsub_in, Csub_in = pi_eq_ckt3[1][0]
        Cox_out, Rsub_out, Csub_out = pi_eq_ckt3[1][1]
        return ((Ls1, Rs1, Lskin1, Rskin1),(Ls2, Rs2, Lskin2, Rskin2),
                (Cox_in, Rsub_in, Csub_in),(Cox_out, Rsub_out, Csub_out))
    
def input_ind_res_asitic(network):
    """
    This function extracts the input inductance and resistance 
    from a single-pi model, which is the one used by ASITIC
    
    ----------
    Parameters
    ----------
    network : network object by skrf
        network.Network type, it is the network model of a circuit

    Returns
    -------
    Lin, Rin: list
        input inductance and resistance of the pi-model
        as a function of frequency

    """
    

    f = network.f
    w = 2 * math.pi * f
    y_params = network.y
    
    y11_list = [y_params[i][0][0] for i in range(len(y_params))]

    Rin = np.real(1/np.array(y11_list))
    Lin = np.imag(1/np.array(y11_list))/np.array(w)
    
    return {
            'Lin':Lin, 
            'Rin':Rin
            }


def s3p2s2p(network, port_num, termination):
    """
    This function converts a 3-port S-parameter to a 2-port S-parameter
    by selecting either port "1", "2", or "3" (defined by "port_num") 
    to be either "open", "short", or "match" (defined by "termination");
    see R. E. Collins "Foundations for Microwave Engineering" page 846 for
    more details.
    
    ----------
    Parameters
    ----------
    network : network object by skrf
        network.Network type, it is the network model of a circuit
    port_num: int
        port that one wants to either "open", "short", or "match"
        entries can be "1", "2", or "3"
    termination: string
        termination method, supports "open", "short", or "match"
        
        1 --[ckt] --2
              |
              |
              3
                       
    
    Returns
    -------
    s_params_pX******: list
        list of 2-port S-parameter

    """
    
    s_params = network.s

    s11_list = [s_params[i][0][0] for i in range(len(s_params))]
    s12_list = [s_params[i][0][1] for i in range(len(s_params))]
    s13_list = [s_params[i][0][2] for i in range(len(s_params))]
    s21_list = [s_params[i][1][0] for i in range(len(s_params))]
    s22_list = [s_params[i][1][1] for i in range(len(s_params))]
    s23_list = [s_params[i][1][2] for i in range(len(s_params))]
    s31_list = [s_params[i][2][0] for i in range(len(s_params))]
    s32_list = [s_params[i][2][1] for i in range(len(s_params))]
    s33_list = [s_params[i][2][2] for i in range(len(s_params))]

    if port_num == 1:
        if termination == 'short':
            s11_p1short = np.array(s22_list) - np.array(s12_list)*np.array(s21_list)/(1+np.array(s11_list))
            s12_p1short = np.array(s23_list) - np.array(s13_list)*np.array(s21_list)/(1+np.array(s11_list))
            s21_p1short = np.array(s32_list) - np.array(s31_list)*np.array(s12_list)/(1+np.array(s11_list))
            s22_p1short = np.array(s33_list) - np.array(s31_list)*np.array(s13_list)/(1+np.array(s11_list))
            
            s_params_p1short = [ [[s11_p1short[i], s12_p1short[i]],
                     [s21_p1short[i], s22_p1short[i]]] 
                    for i in range(len(s_params))
                    ]
            
            return s_params_p1short
        
        elif termination == 'open':
            s11_p1open = np.array(s22_list) + np.array(s12_list)*np.array(s21_list)/(1-np.array(s11_list))
            s12_p1open = np.array(s23_list) + np.array(s13_list)*np.array(s21_list)/(1-np.array(s11_list))
            s21_p1open = np.array(s32_list) + np.array(s31_list)*np.array(s12_list)/(1-np.array(s11_list))
            s22_p1open = np.array(s33_list) + np.array(s31_list)*np.array(s13_list)/(1-np.array(s11_list))
            
            s_params_p1open = [ [[s11_p1open[i], s12_p1open[i]],
                     [s21_p1open[i], s22_p1open[i]]] 
                    for i in range(len(s_params))
                    ]
            
            return s_params_p1open
        
        elif termination == 'match':
            s11_p1match = np.array(s22_list) - np.array(s12_list)*np.array(s21_list)/(np.array(s11_list))
            s12_p1match = np.array(s23_list) - np.array(s13_list)*np.array(s21_list)/(np.array(s11_list))
            s21_p1match = np.array(s32_list) - np.array(s31_list)*np.array(s12_list)/(np.array(s11_list))
            s22_p1match = np.array(s33_list) - np.array(s31_list)*np.array(s13_list)/(np.array(s11_list))
            
            s_params_p1match = [ [[s11_p1match[i], s12_p1match[i]],
                     [s21_p1match[i], s22_p1match[i]]] 
                    for i in range(len(s_params))
                    ]
            
            return s_params_p1match
        
        else:
            print('Invalid termination type; enter either "open", "short", or "match".')
            
    elif port_num == 2:
        if termination == 'short':
            s11_p2short = np.array(s11_list) - np.array(s21_list)*np.array(s12_list)/(1+np.array(s22_list))
            s12_p2short = np.array(s13_list) - np.array(s23_list)*np.array(s12_list)/(1+np.array(s22_list))
            s21_p2short = np.array(s31_list) - np.array(s21_list)*np.array(s32_list)/(1+np.array(s22_list))
            s22_p2short = np.array(s33_list) - np.array(s23_list)*np.array(s32_list)/(1+np.array(s22_list))
            
            s_params_p2short = [ [[s11_p2short[i], s12_p2short[i]],
                     [s21_p2short[i], s22_p2short[i]]] 
                    for i in range(len(s_params))
                    ]
            
            return s_params_p2short
            
        elif termination == 'open':
            s11_p2open = np.array(s11_list) + np.array(s21_list)*np.array(s12_list)/(1-np.array(s22_list))
            s12_p2open = np.array(s13_list) + np.array(s23_list)*np.array(s12_list)/(1-np.array(s22_list))
            s21_p2open = np.array(s31_list) + np.array(s21_list)*np.array(s32_list)/(1-np.array(s22_list))
            s22_p2open = np.array(s33_list) + np.array(s23_list)*np.array(s32_list)/(1-np.array(s22_list))
            
            s_params_p2open = [ [[s11_p2open[i], s12_p2open[i]],
                     [s21_p2open[i], s22_p2open[i]]] 
                    for i in range(len(s_params))
                    ]
            
            return s_params_p2open
        
        elif termination == 'match':
            s11_p2match = np.array(s11_list) - np.array(s21_list)*np.array(s12_list)/(np.array(s22_list))
            s12_p2match = np.array(s13_list) - np.array(s23_list)*np.array(s12_list)/(np.array(s22_list))
            s21_p2match = np.array(s31_list) - np.array(s21_list)*np.array(s32_list)/(np.array(s22_list))
            s22_p2match = np.array(s33_list) - np.array(s23_list)*np.array(s32_list)/(np.array(s22_list))

            s_params_p2match = [ [[s11_p2match[i], s12_p2match[i]],
                     [s21_p2match[i], s22_p2match[i]]] 
                    for i in range(len(s_params))
                    ]
            
            return s_params_p2match
        
        else:
            print('Invalid termination type; enter either "open", "short", or "match".')
            
            
    elif port_num == 3:
        
        if termination == 'short':
            
            s11_p3short = np.array(s11_list) - np.array(s31_list)*np.array(s13_list)/(1+np.array(s33_list))
            s12_p3short = np.array(s12_list) - np.array(s32_list)*np.array(s13_list)/(1+np.array(s33_list))
            s21_p3short = np.array(s21_list) - np.array(s31_list)*np.array(s23_list)/(1+np.array(s33_list))
            s22_p3short = np.array(s22_list) - np.array(s32_list)*np.array(s23_list)/(1+np.array(s33_list))

            s_params_p3short = [ [[s11_p3short[i], s12_p3short[i]],
                     [s21_p3short[i], s22_p3short[i]]] 
                    for i in range(len(s_params))
                    ]

            return s_params_p3short

        elif termination == 'open':

            s11_p3open = np.array(s11_list) + np.array(s31_list)*np.array(s13_list)/(1-np.array(s33_list))
            s12_p3open = np.array(s12_list) + np.array(s32_list)*np.array(s13_list)/(1-np.array(s33_list))
            s21_p3open = np.array(s21_list) + np.array(s31_list)*np.array(s23_list)/(1-np.array(s33_list))
            s22_p3open = np.array(s22_list) + np.array(s32_list)*np.array(s23_list)/(1-np.array(s33_list))

            s_params_p3open = [ [[s11_p3open[i], s12_p3open[i]],
                     [s21_p3open[i], s22_p3open[i]]] 
                    for i in range(len(s_params))
                    ]

            return s_params_p3open

        elif termination == 'match':

            s11_p3match = np.array(s11_list) - np.array(s31_list)*np.array(s13_list)/(np.array(s33_list))
            s12_p3match = np.array(s12_list) - np.array(s32_list)*np.array(s13_list)/(np.array(s33_list))
            s21_p3match = np.array(s21_list) - np.array(s31_list)*np.array(s23_list)/(np.array(s33_list))
            s22_p3match = np.array(s22_list) - np.array(s32_list)*np.array(s23_list)/(np.array(s33_list))

            s_params_p3match = [ [[s11_p3match[i], s12_p3match[i]],
                     [s21_p3match[i], s22_p3match[i]]] 
                    for i in range(len(s_params))
                    ]

            return s_params_p3match
        
        else:
            print('Invalid termination type; enter either "open", "short", or "match".')
              
    else:
        print('Port decleration is wrong; enter either "1", "2", or "3" \
              for the port number you want to "open", "short", or "match".')


def t_network_eq_ckt(network, port_num = 3):
    """
    This function converts a 3-port t-coil to a 2-port device by 
    grounding the middle tap; then conver the 2-port S-parameter
    to Z-parameter and extract La, Ra, Lb, Rb and k
    
    ----------
    Parameters
    ----------
    network : network object by skrf
        network.Network type, it is the network model of a circuit
    port_num: int
        port of the t-coil that is grounded (short)
        
        1 --[tcoil] --2
               |
               | 3
              ---
               -        
        
    
    Returns
    -------
    La, Lb, Ra, Rb, k: tuple
        inductance and resistance of inductor "a" and "b", 
        and their coupling factor

    """
    
    
    s2p_p2short = s3p2s2p(network, port_num, 'short')
    z2p_p2short = rf.network.s2z(np.array(s2p_p2short))
    
    f = network.f
    w = 2 * math.pi * f
    
    z11_p2short = [z2p_p2short[i][0][0] for i in range(len(z2p_p2short))]
    z12_p2short = [z2p_p2short[i][0][1] for i in range(len(z2p_p2short))]
    z21_p2short = [z2p_p2short[i][1][0] for i in range(len(z2p_p2short))]
    z22_p2short = [z2p_p2short[i][1][1] for i in range(len(z2p_p2short))]
    
    z3 = 0.5 * (np.array(z12_p2short) + np.array(z21_p2short))
    z1 = np.array(z11_p2short) - np.array(z3)
    z2 = np.array(z22_p2short) - np.array(z3)
    
    #print(z3)
    
    M = -np.imag(z12_p2short)/np.array(w)
    La = np.imag(z1)/np.array(w) - M
    Lb = np.imag(z2)/np.array(w) - M
    Ra = np.real(z1) + np.real(z3) 
    Rb = np.real(z2) + np.real(z3)
    k = np.array(M)/np.sqrt(np.array(La)*np.array(Lb))
    Qa = np.array(w)*La/Ra
    Qb = np.array(w)*Lb/Rb
    
    # in case any negative inductance appears
    # if any(i<0 for i in La) or any(i<0 for i in Lb):
    #     print(network)
      
    return {
            'La': La, 
            'Lb': Lb, 
            'Ra': Ra, 
            'Rb': Rb, 
            'Qa': Qa,
            'Qb': Qb,
            'k': k
            }
        
        

    
    




