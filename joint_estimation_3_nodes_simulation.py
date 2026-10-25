import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize
from mealpy import FloatVar,DE
from joblib import Parallel, delayed
import os
import time
import sdeint
# Sigmoid函数
# Sigmoid函数
def S(v,e0,r,v0):
    v0 = v0
    return 2 * e0 / (1 + np.exp(r * (v0 - v)))

# ODE subsystem
def jansen_rit_ode_part(y, parameters, n_nodes):
    """
    Generalized Jansen–Rit linear part for n nodes
    
    Parameters
    ----------
    y : ndarray, shape (6*n_nodes,)
        State vector
    parameters : tuple
        (a, b)
    n_nodes : int
        Number of nodes
    """
    a, b = parameters
    
    dy = np.zeros_like(y)

    for i in range(n_nodes):
        idx = 6 * i

        y0, y1, y2 = y[idx:idx+3]
        y3, y4, y5 = y[idx+3:idx+6]

        # first-order equations
        dy[idx + 0] = y3
        dy[idx + 1] = y4
        dy[idx + 2] = y5

        # second-order equations
        dy[idx + 3] = -2 * a * y3 - a**2 * y0
        dy[idx + 4] = -2 * a * y4 - a**2 * y1
        dy[idx + 5] = -2 * b * y5 - b**2 * y2

    return dy


# SDE subsystem
def jansen_rit_sde_drift(y, params):
    """
    n-node Jansen–Rit SDE drift with node-specific C_i

    params = {
        'A' : array(n),
        'B' : array(n),
        'C' : array(n),        # <-- node-specific
        'K' : array(n, n),
        'a' : float,
        'b' : float,
        'e0': float,
        'r' : float,
        'v0': array(n)
    }
    """
    A  = params['A']
    B  = params['B']
    C  = params['C']          # shape (n,)
    K  = params['K']
    a  = params['a']
    b  = params['b']
    e0 = params['e0']
    r  = params['r']
    v0 = params['v0']

    n_nodes = len(A)
    dy = np.zeros_like(y)

    for i in range(n_nodes):
        idx = 6 * i

        y0, y1, y2 = y[idx:idx+3]
        y3, y4, y5 = y[idx+3:idx+6]

        # node-specific intra-cortical gains
        C1 = C[i]
        C2 = 0.8  * C[i]
        C3 = 0.25 * C[i]
        C4 = 0.25 * C[i]

        # no drift for membrane potentials
        dy[idx:idx+3] = 0.0

        # pyramidal population
        dy[idx+3] = A[i] * a * S(y1 - y2, e0, r, v0[i])

        # excitatory interneurons + inter-node coupling
        coupling = 0.0
        for j in range(n_nodes):
            if K[i, j] != 0:
                jdx = 6 * j
                coupling += (
                    K[i, j] *
                    S(y[jdx+1] - y[jdx+2], e0, r, v0[i])
                )

        dy[idx+4] = (
            A[i] * a * C2 * S(C1 * y0, e0, r, v0[i])
            + A[i] * a * coupling
        )

        # inhibitory interneurons
        dy[idx+5] = B[i] * b * C4 * S(C3 * y0, e0, r, v0[i])

    return dy
# SDE系统的扩散
def jansen_rit_diffusion(diff):
    """
    n-node diffusion term (diagonal noise)

    Parameters
    ----------
    diff : array-like, shape (n,)
        diffusion strength for each node
    """
    n_nodes = len(diff)
    g = np.zeros(6 * n_nodes)

    for i in range(n_nodes):
        g[6 * i + 4] = diff[i]

    return g




def jansen_rit_deterministic(P):
    """
    n-node deterministic input

    Parameters
    ----------
    P : array-like, shape (n,)
        external input to each node
    """
    n_nodes = len(P)
    f = np.zeros(6 * n_nodes)

    for i in range(n_nodes):
        f[6 * i + 4] = P[i]

    return f




import numpy as np

def strang_step(y, h, params):
    """
    One Strang splitting step for n-node Jansen–Rit SDE

    params = {
        'A' : array(n),
        'B' : array(n),
        'C' : array(n),
        'K' : array(n,n),
        'a' : float,
        'b' : float,
        'e0': float,
        'r' : float,
        'v0': array(n),
        'diff': array(n),
        'P'   : array(n)
    }
    """

    A  = params['A']
    B  = params['B']
    a  = params['a']
    b  = params['b']
    diff = params['diff']
    P    = params['P']

    n_nodes = len(A)

    # ---------- Half step: linear ODE part ----------
    y_half_A = y + 0.5 * h * jansen_rit_ode_part(
        y, (a, b), n_nodes
    )

    # ---------- Full step: nonlinear drift + noise ----------
    drift = jansen_rit_sde_drift(y_half_A, params)
    g     = jansen_rit_diffusion(diff)
    l     = jansen_rit_deterministic(P)

    dW = np.sqrt(h) * np.random.randn(6 * n_nodes)

    # node-wise gain term (与你原来的 A*a, B*b 结构一致)
    gain = np.zeros(6 * n_nodes)
    for i in range(n_nodes):
        idx = 6 * i
        gain[idx + 3] = A[i] * a
        gain[idx + 4] = A[i] * a
        gain[idx + 5] = B[i] * b

    y_full_B = (
        y_half_A
        + h * drift
        + g * dW
        + h * gain * l
    )

    # ---------- Half step: linear ODE part ----------
    y_full = y_full_B + 0.5 * h * jansen_rit_ode_part(
        y_full_B, (a, b), n_nodes
    )

    return y_full


def fun_model_x2_generate(t1,t2,v0,q0,q2,q3,q5,t0,t):
    '''Generate x2 using Chen-Fliess series expansion'''
    e0 = 5
    r = 0.56
    b = 50
    Sig = np.exp(r*(-0.25*t2*q0+v0))
    L = q2 + q5*(t-t0) + (0.25*(1/t1)*e0*b*t2/(1+Sig)-b**2*q2-2*b*q5)*((t-t0)**2/2)+(0.0625*(1/t1)*e0*b*t2**2*r*q3*Sig/(1+Sig)**2-b**2*q5-2*b*(0.25*(1/t1)*e0*b*t2/(1+Sig)-b**2*q2-2*b*q5))*((t-t0)**3/6)
    return L


def fun_model_x2_generate_EM(t1,t2,v0,q0,q2,q3,q5,t0,t):
    '''Generate x2 using Chen-Fliess series expansion'''
    e0 = 5
    r = 0.56
    b = 50
    Sig = np.exp(r*(-0.25*t2*q0+v0))
    L = q2 + q5*(t-t0) 
    return L

def fun_model_x5_generate(t1,t2,v0,q0,q2,q3,q5,t0,t):
    '''Generate x5 using Chen-Fliess series expansion'''
    e0 = 5
    r = 0.56
    b = 50
    Sig = np.exp(r*(-0.25*t2*q0+v0))
    L = q5 + (0.25*(1/t1)*e0*b*t2/(1+Sig)-b**2*q2-2*b*q5)*(t-t0)+(0.0625*(1/t1)*e0*b*t2**2*r*q3*Sig/(1+Sig)**2-b**2*q5-2*b*(0.25*(1/t1)*e0*b*t2/(1+Sig)-b**2*q2-2*b*q5))*((t-t0)**2/2)
    return L

def fun_model_x5_generate_EM(t1,t2,v0,q0,q2,q3,q5,t0,t):
    '''Generate x5 using Chen-Fliess series expansion'''
    e0 = 5
    r = 0.56
    b = 50
    Sig = np.exp(r*(-0.25*t2*q0+v0))
    L = q5 + (0.25*(1/t1)*e0*b*t2/(1+Sig)-b**2*q2-2*b*q5)*(t-t0)
    return L





def fun_model_x0_generate(v0,q0,q3,y1,y2,t0,t):
    '''Generate x0 using Chen-Fliess series expansion'''
    r = 0.56
    e0 = 5
    a = 100
    Sig = np.exp(r*(v0-y1))
    L = q0 + q3*(t-t0) + ((e0*a)/(1+Sig)-a**2*q0-2*a*q3)*((t-t0)**2/2)+(e0*a*r*Sig*(y2)/(Sig+1)**2-a**2*q3-2*a*((e0*a)/(1+Sig)-a**2*q0-2*a*q3))*((t-t0)**3/6)
    return L

def fun_model_x0_generate_EM(v0,q0,q3,y1,y2,t0,t):
    '''Generate x0 using Chen-Fliess series expansion'''
    r = 0.56
    e0 = 5
    a = 100
    Sig = np.exp(r*(v0-y1))
    L = q0 + q3*(t-t0) 
    return L

def fun_model_x3_generate(v0,q0,q3,y1,y2,t0,t):
    '''Generate x3 using Chen-Fliess series expansion'''
    r = 0.56
    e0 = 5
    a = 100
    Sig = np.exp(r*(v0-y1))
    L = q3 + ((e0*a)/(1+Sig)-a**2*q0-2*a*q3)*(t-t0)+(e0*a*r*Sig*(y2)/(Sig+1)**2-a**2*q3-2*a*((e0*a)/(1+Sig)-a**2*q0-2*a*q3))*((t-t0)**2/2)
    return L

def fun_model_x3_generate_EM(A,v0,q0,q3,y1,y2,t0,t):
    '''Generate x3 using Chen-Fliess series expansion'''
    r = 0.56
    e0 = 5
    a = 100
    Sig = np.exp(r*(v0-y1))
    L = q3 + ((A*e0*a)/(1+Sig)-a**2*q0-2*a*q3)*(t-t0)
    return L

#############################################################################
#####Objective   function

def fun_model_output(t1,t2,t3,v0, q,y_1,y_2,t0,t):
    '''Model output using Chen-Fliess series expansion--second order'''
    a = 100
    b = 50
    e0 = 5
    r = 0.56
    y1,y2 = y_1; y3,y4 = y_2; q0, q1, q2, q3, q4, q5= q
    Sig1 = np.exp(r*(v0-y3));Sig_x2 = np.exp(r*(-t2*q0+v0));  Sig_x3 = np.exp(r*(-0.25*t2*q0+v0))
    L = y1 + y2*(t-t0)+(0.8*t2*e0*a/(1+Sig_x2)+t3*e0*a/(1+Sig1)-a**2*q1-2*a*q4-(0.25*(1/t1)*e0*b*t2/(1+Sig_x3)-b**2*q2-2*b*q5))*((t-t0)**2/2)
    #+(0.8*A*e0*a*C**2*r*q3*Sig_x2/(1+Sig_x2)**2+r*A*e0*a*K*y4*Sig1/(1+Sig1)**2-a**2*q4-2*a*(0.8*A*e0*a*C/(1+Sig_x2)-a**2*q1-2*a*q4)
    #  -(0.0625*B*e0*b*C**2*r*q3*Sig_x3/(1+Sig_x3)**2-b**2*q5-2*b*(0.25*B*e0*b*C/(1+Sig_x3)-b**2*q2-2*b*q5)))*((t-t0)**3/6))
    return L




### Re-estimate method
def sigma_time_integrate(t_last,t_0):
   a = 100
   d = (t_last-t_0)**3/3
   return d
def objective_function_re_minimize(params,N_interval,y_iniii_node1,y_iniii_node2,t_spans, mean_sol_data_true):
    t1,t2,v0,t3,t4 = params
   
    L = t4*100
    sol_fitted = []
    qq = []
    q_ini =  np.random.normal(0,1,6)
    #q_ini = [0,0,0,0,0,0]
    #q_ini = [0,0.5,1,1.5,2,2.5]
    raw_data = []
    predict = []
    for i in range(N_interval):
        y_135 = mean_sol_data_true[:,i]
        q = q_ini
        q[1] = y_iniii_node1[i,0]+q[2];q[4] = y_iniii_node1[i,1]+q[5] 
        qq.append(q)
        y12 = y_iniii_node1[i]
        y_12 = y_iniii_node2[i]
        t_span31 = t_spans[i,]
        raw = y_135[-1]
        raw_data.append(raw)
        pre = fun_model_output(t1,t2,t3,v0, q,y12, y_12,t_span31[0], t_span31[-1])
        predict.append(pre)
        
        sigma = L**2*sigma_time_integrate(t_span31[-1],t_span31[0])
        fun =  (raw-pre)**2/sigma +np.log(L**2)

       
        sol_fitted.append(fun)   
        q0 = fun_model_x0_generate(v0,q[0],q[3],y12[0],y12[1],t_span31[0], t_span31[-1])
        q2 = fun_model_x2_generate(t1,t2,v0,q[0],q[2],q[3],q[5],t_span31[0], t_span31[-1])
        q3 = fun_model_x3_generate(v0,q[0],q[3],y12[0],y12[1],t_span31[0], t_span31[-1])
        q5 = fun_model_x5_generate(t1,t2,v0,q[0],q[2],q[3],q[5],t_span31[0], t_span31[-1])
        q_ini = [q0,0,q2,q3,0,q5]
    fun_epoch = np.array(sol_fitted)
    
    return (np.sqrt(sum(fun_epoch[255:])))



###Bias-correct using bootstrap

# ODE subsystem
def jansen_rit_ode_part_boot(y,paramers):
    '''Deterministic part for the Jansen-Rit model'''
    a, b= paramers
    y0, y1, y2, y3, y4, y5 = y
    dy0 = y3
    dy1 = y4
    dy2 = y5
    dy3 = - 2 * a * y3 - a**2 * y0
    dy4 = - 2 * a * y4 - a**2 * y1 
    dy5 = - 2 * b * y5 - b**2 * y2
    return np.array([dy0, dy1, dy2, dy3, dy4, dy5])

# SDE subsystem
def jansen_rit_sde_part_drift_boot(y,p,paramers):
    '''Stochastic part for the Jansen-Rit model'''
    A,a,B,b,C,K,e0,r,v0= paramers
    C1 = C          
    C2 = 0.8 * C  
    C3 = 0.25 * C
    C4 = 0.25 * C

    y0, y1, y2, y3, y4, y5 = y
    dy0 = 0
    dy1 = 0
    dy2 = 0
    dy3 = A * a * S(y1 - y2,e0,r,v0)
    dy4 = A * a *  C2 * S(C1 * y0,e0,r,v0) + A * a * K * S(p,e0,r,v0)
    dy5 = B * b * C4 * S(C3 * y0,e0,r,v0)
    return np.array([dy0, dy1, dy2, dy3, dy4, dy5])

# SDE系统的扩散项
def jansen_rit_diffusion_boot(diff):
    '''Diffusion term for the Jansen-Rit model SDEs'''
    return np.array([0,0,0,0, diff, 0])  # 简单起见，我们假设扩散系数是常数


def strang_step_boot(y, h, p, params,diff):
    '''Strang splitting step for the Jansen-Rit model with SDEs and ODEs'''
    diff = diff

    A,a,B,b,C,K,e0,r,v0 = params
    PARAMER = np.array([a, b])
    parameter = np.array([A,a,B,b,C,K,e0,r,v0])
    # Half step for A
    y_half_A = y + (h/2) * jansen_rit_ode_part_boot(y,PARAMER)
    # Full step for B
    g = jansen_rit_diffusion_boot(diff)
    dW_node1 = np.sqrt(h) * np.random.randn(6)
    y_full_B = y_half_A + h * jansen_rit_sde_part_drift_boot(y_half_A,p,parameter)  + g * dW_node1 
    # Half step for A again
    y_full_A = y_full_B + (h/2) * jansen_rit_ode_part_boot(y_full_B,PARAMER)
    return y_full_A








def create_subarrays(data,split_size = 2,step = 1):
    '''segment data into subarrays'''
    return [data[i:i + split_size] for i in range(0, len(data) - split_size + 1, step)]

def  jansen_rit_bootstrap_bias_correction_minimize(x_best,n_boot,T,dt,input_data,N_interval,t_spans):
    t1,t2,v0,t3,t4 = x_best
    A = 3.25
    B = A/t1; C = t2/A; K = t3/A; sigma = t4/A
    B = B
    C = C
    K = K
    a = 100
    b = 50
    e0 = 2.5
    r = 0.56
    v0 = v0
    diff = A*a*sigma
    estimate_value_boot = np.zeros((n_boot,len(x_best)))
    for kl in range(n_boot):   
       paramers = [A,a,B,b,C,K,e0,r,v0]
       x = np.random.normal(0,1,6)
       true_states = np.zeros((T, 6))
       for t in range(T):
           true_states[t,:] = x
           x = strang_step_boot(x,dt,input_data[t],paramers,diff)
       observation_output_boot = (true_states[:, 1] - true_states[:, 2])
       observation_input_boot = input_data
    
       subarrays_output_boot = create_subarrays(observation_output_boot)
       subarrays_input_boot = create_subarrays(observation_input_boot)
       sol_output_boot = np.stack(subarrays_output_boot, axis=1)        # shape: (split_size, n_parts)

       sol_input_boot = np.stack(subarrays_input_boot, axis=1)    # shape: (split_size, n_parts)
       
       y_iniii_input_boot = np.zeros((len(subarrays_output_boot),2))
       y_iniii_input_boot[:,0] = sol_input_boot[0, :]
       y_iniii_input_boot[:,1] = (sol_input_boot[1,:]-sol_input_boot[0,:])/dt

       y_iniii_output_boot = np.zeros((len(subarrays_output_boot),2))
       y_iniii_output_boot[:,0] = sol_output_boot[0, :]
       y_iniii_output_boot[:,1] = (sol_output_boot[1,:]-sol_output_boot[0,:])/dt
       initial_guess = x_best
       # 使用 minimize 函数来优化参数
       bounds = [(0.05,1),(0,x_best[1]+200),(0,15),(0,x_best[3]+200),(5,100)]
       result = minimize(objective_function_re_minimize, initial_guess, args=(N_interval,y_iniii_output_boot,y_iniii_input_boot, t_spans, sol_output_boot),method='L-BFGS-B',bounds=bounds)
       # 输出优化后的参数值
       optimized_params_C = result.x
       estimate_value_boot[kl] =  optimized_params_C
    return estimate_value_boot





###Runge-Kutta approximation for the Jansen-Rit model SDEs
def jansen_rit_sde_part_RK(y,t,p,paramers):
    '''Stochastic part for the Jansen-Rit model'''
    t1,t2,t3,v0= paramers
    a = 100
    b = 50  
    e0 = 2.5
    r = 0.56
  

    y0, y1, y2, y3, y4, y5 = y
    dy0 = y3
    dy1 = y4
    dy2 = y5
    dy3 = a * S(y1 - y2,e0,r,v0) - 2 * a * y3 - a**2 * y0
    dy4 = t2 * a *  0.8 * S(t2 * y0,e0,r,v0) +  t3 * a  * S(p,e0,r,v0)- 2 * a * y4 - a**2 * y1 
    dy5 = (0.25/t1) * b * t2 * S(0.25*t2 * y0,e0,r,v0) - 2 * b * y5 - b**2 * y2
    return np.array([dy0, dy1, dy2, dy3, dy4, dy5])    

def jr_diffusion(x, t, t4):

    a = 100

    G = np.zeros((6,1))
    G[4,0] = t4*a

    return G




def objective_function_re_RK(params, p, observation_data, h):

    t1, t2, v0, t3, t4 = params
 
    a = 100
    paramers = (t1,t2,t3, v0)
    
    x =  np.random.normal(0,1,6)
    #x = [0,0,0,0,0,0]
    #x = [0,0.5,1,1.5,2,2.5]
    trajectory = []
    
    for i in range(len(observation_data)):
        p_i = p[i]
        f = lambda x,t: jansen_rit_sde_part_RK(x,t,p_i,paramers)
        g = lambda x,t: jr_diffusion(x,t,t4)
        sol = sdeint.stratSRS2(f, g, x, [0,h])
        x = sol[-1]
        
        
        trajectory.append(x[1] - x[2])  # 模拟的输出（y1 - y2）
    
    trajectory = np.array(trajectory)
    
    loss = np.sqrt(
        np.sum((trajectory[255:] - observation_data[255:])**2)
    )
    
    return loss

def objective_function_re_Euler(params, p, observation_data, h):
    t1, t2, v0, t3, t4 = params

    a = 100
    paramers = (t1,t2,t3, v0)

    x =  np.random.normal(0,1,6)
    #x = [0,0,0,0,0,0]
    #x = [0,0.5,1,1.5,2,2.5]
    trajectory = []

    for i in range(len(observation_data)):
        
        dx = jansen_rit_sde_part_RK(x, 0, p[i], paramers)
        x = x + h * dx 
        # ---------- 加 Wiener 噪声 ----------
        dW = np.sqrt(h) * np.random.randn()
        # 假设噪声作用在输入方程 (常见设置)
        x[4] += t4*a * dW
        
        eeg_signal = x[1] - x[2]
        trajectory.append(eeg_signal)

    trajectory = np.array(trajectory)

    loss = np.sqrt(
        np.sum((trajectory[255:] - observation_data[255:])**2)
    )
    return loss








def denormalize(z):
    log_idx = [1, 3]
    lb = np.array([0.05, 0,    0,  0,     5])
    ub = np.array([1,    1500, 15, 1500, 100])
    x = lb + z * (ub - lb)
    
    x[log_idx] = np.exp(
        np.log(lb[log_idx] + 1) + 
        z[log_idx] * (np.log(ub[log_idx] + 1) - np.log(lb[log_idx] + 1))
    ) - 1
    
    return x



def bias_correction(theta_hat, theta_bootstrap):
    eps = 1e-8
    lb = np.array([0.05, 0,    0,  0,     5])
    ub = np.array([1,    1500, 15, 1500, 100])
    theta_hat = np.clip(theta_hat, lb + eps, ub - eps)
    theta_bootstrap = np.clip(theta_bootstrap, lb + eps, ub - eps)

    # forward
    def forward(theta):
        z = (theta - lb) / (ub - lb)
        return np.log(z) - np.log(1 - z)

    # inverse
    def inverse(phi):
        sigmoid = 1 / (1 + np.exp(-phi))
        return lb + (ub - lb) * sigmoid

    phi_hat = forward(theta_hat)
    phi_boot = forward(theta_bootstrap)

    phi_bc = 2 * phi_hat - np.mean(phi_boot, axis=0)

    return inverse(phi_bc)






start = 0.0
stim_time =20
dt = 1/256
time_array = np.arange(start=start, stop=stim_time, step=dt)
vec_len = len(time_array)

################



T = vec_len


print(T)
import numpy as np
A = np.array([3.25, 4, 3])
a = 100
sigma = 10
params = {
    # node-specific gains
    'A':  A,
    'B':  np.array([22.0, 14, 15]),

    # node-specific intra-cortical connectivity
    'C':  np.array([135, 135, 135]),

    # time constants
    'a':  a,
    'b':  50.0,

    # sigmoid parameters
    'e0': 2.5,
    'r':  0.56,
    'v0': np.array([6.0, 4, 5]),

    # directed coupling matrix (node j → node i)
    'K': np.array([
        [0.0, 0.0,50],   # node 2 → node 0
        [70, 0.0,  0.0],   # node 0 → node 1
        [0.0, 50,  0.0]    # node 1 → node 2
    ]),

    # diffusion (noise strength)
    'diff': A*a*sigma,

    # deterministic external input
    'P': np.array([0, 0, 0])
}
split_size = 2
n_interval = split_size
n_reptation = 100




n_boot = 1

A = 3.25; A1 = 4; A2 = 3
B = 22;   B1 = 14; B2 = 15
a = 100
b = 50
C = 135;  C1 = 135; C2 = 135
K = 50;   K1 = 70; K2 = 50
e0 = 2.5
r = 0.56
v0 = 6;   v01 = 4; v02 = 5
sig = 10


def run_one_mm(mm):
    ARE_VALUE_combine_Q = np.zeros((15))
    ARE_VALUE_combine_Q_bia = np.zeros((15))
    ARE_VALUE_combine_EU_Q = np.zeros((15))
    ARE_VALUE_combine_RK_Q = np.zeros((15))
    x = np.random.normal(0,1,18)
    true_states = np.zeros((T, 18))
    for t in range(T):
       true_states[t,:] = x
       x = strang_step(x,dt,params)
    observation = ((true_states[:, 1] - true_states[:, 2]))
    observations2 = ((true_states[:, 7] - true_states[:, 8]))
    observations3 = ((true_states[:, 13] - true_states[:, 14]))
    step = split_size - 1
    def create_subarrays(data):
        return [data[i:i + split_size] for i in range(0, len(data) - split_size + 1, step)]
    subarrays_node1 = create_subarrays(observation)
    subarrays_node2 = create_subarrays(observations2)
    subarrays_node3 = create_subarrays(observations3) 
    sol_node1 = np.stack(subarrays_node1, axis=1)        # shape: (split_size, n_parts)
  
    sol_node2 = np.stack(subarrays_node2, axis=1)    # shape: (split_size, n_parts)
    sol_node3 = np.stack(subarrays_node3, axis=1)
    h = dt #discretization step
    y_iniii_node1 = np.zeros((len(subarrays_node1),2))
    y_iniii_node1[:,0] = sol_node1[0, :]
    y_iniii_node1[:,1] = (sol_node1[1,:]-sol_node1[0,:])/h

    y_iniii_node2 = np.zeros((len(subarrays_node1),2))
    y_iniii_node2[:,0] = sol_node2[0, :]
    y_iniii_node2[:,1] = (sol_node2[1,:]-sol_node2[0,:])/h

    y_iniii_node3 = np.zeros((len(subarrays_node1),2))
    y_iniii_node3[:,0] = sol_node3[0, :]
    y_iniii_node3[:,1] = (sol_node3[1,:]-sol_node3[0,:])/h




    t_last = len(observation)*h
    t_value = np.arange(0, t_last, h)
    aa = sol_node1.shape
    N_interval = aa[1]
    t_spans = np.zeros((N_interval, n_interval))
    t_spans[0] = t_value[:n_interval]
    for i in range(1, N_interval):
        t_spans[i, 0] = t_spans[i-1, -1]
        t_spans[i, 1:] = t_value[n_interval + (i-1)*(n_interval-1) : n_interval + i*(n_interval-1)]
    
    
###Node 1
    ###EU method
    def wrapped_function_EU(params):
        return objective_function_re_Euler(params,  observations3, observation, h)
    problem_dict = {
    "obj_func": wrapped_function_EU,
    "bounds": FloatVar(lb=(0.05,0,0,0,5), ub=(1,1500,15,1500,100),name='delta'),
    "minmax": "min",
    "verbose": False,
}

    start_EU = time.time()
    model = DE.JADE(epoch=100, pop_size = 15,  verbose = False)
    g_best = model.solve(problem_dict,n_workers=-1)
    Time_EU = (time.time() - start_EU)
    ARE_VALUE_combine_EU_Q[:5] = np.abs((g_best.solution-[A/B,A*C,v0,A*K,A*sigma])/[A/B,A*C,v0,A*K,A*sigma])
    
    ###Chen-Fliess 
    def wrapped_function(z):
        params = denormalize(z)
        return objective_function_re_minimize(params,N_interval,y_iniii_node1,y_iniii_node3, t_spans, sol_node1)
    problem_dict = {
    "obj_func": wrapped_function,
    "bounds": FloatVar(lb=(0,0,0,0,0), ub=(1,1,1,1,1),name='delta'),
    "minmax": "min",
    "verbose": False,
}

    start_CH = time.time()
    model = DE.JADE(epoch=100, pop_size = 15,  verbose = False)
    g_best = model.solve(problem_dict,n_workers=-1)
    
    z_best = g_best.solution
    x_best = denormalize(z_best)
    estimate_value_boot = jansen_rit_bootstrap_bias_correction_minimize(x_best,n_boot,T,dt,observations3,N_interval,t_spans)
    estimate_value_bias = bias_correction(x_best, estimate_value_boot)
    #estimate_value_bias[0] = x_best[0]
    #initial_guess = estimate_value_bias
    #bounds = [(2.6,9.75),(0.05,1),(0,estimate_value_bias[2]+200),(0,15),(0,estimate_value_bias[4]+200),(5,100)]
    #result = minimize(objective_function_re_minimize, initial_guess, args=(N_interval,y_iniii_node1,y_iniii_node3, t_spans, sol_node1),method='L-BFGS-B',bounds=bounds)
    #estimate_value_boot_RE = jansen_rit_bootstrap_bias_correction_minimize(result.x,n_boot,T,dt,observations3,N_interval,t_spans)
    #estimate_value_bias_RE = bias_correction(result.x, estimate_value_boot_RE)
    #estimate_value_bias_RE[0] = result.x[0]
    Time_CH = (time.time() - start_CH)
    ARE_VALUE_combine_Q[:5] = np.abs((x_best-[A/B,A*C,v0,A*K,A*sigma])/[A/B,A*C,v0,A*K,A*sigma])
    ARE_VALUE_combine_Q_bia[:5] = np.abs((estimate_value_bias-[A/B,A*C,v0,A*K,A*sigma])/[A/B,A*C,v0,A*K,A*sigma])
    ####RK method
    def wrapped_function_RK(params):
        return objective_function_re_RK(params,  observations3, observation, h)
    problem_dict = {
    "obj_func": wrapped_function_RK,
    "bounds": FloatVar(lb=(0.05,0,0,0,5), ub=(1,1500,15,1500,100),name='delta'),
    "minmax": "min",
    "verbose": False,
}

    start_RK = time.time()
    model = DE.JADE(epoch=100, pop_size = 15,  verbose = False)
    g_best = model.solve(problem_dict,n_workers=-1)
    Time_RK = (time.time() - start_RK)
    ARE_VALUE_combine_RK_Q[:5] = np.abs((g_best.solution-[A/B,A*C,v0,A*K,A*sigma])/[A/B,A*C,v0,A*K,A*sigma])

###Node2
    ###EU method
    def wrapped_function_EU(params):
        return objective_function_re_Euler(params,  observation, observations2, h)
    problem_dict = {
    "obj_func": wrapped_function_EU,
    "bounds": FloatVar(lb=(0.05,0,0,0,5), ub=(1,1500,15,1500,100),name='delta'),
    "minmax": "min",
    "verbose": False,
}

     
    model = DE.JADE(epoch=100, pop_size = 15,  verbose = False)
    g_best = model.solve(problem_dict,n_workers=-1)
    ARE_VALUE_combine_EU_Q[5:10] = np.abs((g_best.solution-[A1/B1,A1*C1,v01,A1*K1,A1*sigma])/[A1/B1,A1*C1,v01,A1*K1,A1*sigma])
    
    ###Chen-Fliess
    def wrapped_function(z):
        params = denormalize(z)
        return objective_function_re_minimize(params,N_interval,y_iniii_node2,y_iniii_node1, t_spans, sol_node2)
    problem_dict = {
    "obj_func": wrapped_function,
    "bounds": FloatVar(lb=(0,0,0,0,0), ub=(1,1,1,1,1),name='delta'),
    "minmax": "min",
    "verbose": False,
}

    
    model = DE.JADE(epoch=100, pop_size = 15,  verbose = False)
    g_best = model.solve(problem_dict,n_workers=-1)
    z_best = g_best.solution
    x_best = denormalize(z_best)
    estimate_value_boot = jansen_rit_bootstrap_bias_correction_minimize(x_best,n_boot,T,dt,observation,N_interval,t_spans)
    estimate_value_bias = bias_correction(x_best, estimate_value_boot)
    #estimate_value_bias[0] = x_best[0]
    #initial_guess = estimate_value_bias
    #bounds = [(2.6,9.75),(0.05,1),(0,estimate_value_bias[2]+200),(0,15),(0,estimate_value_bias[4]+200),(5,100)]
    #result = minimize(objective_function_re_minimize, initial_guess, args=(N_interval,y_iniii_node2,y_iniii_node1, t_spans, sol_node2),method='L-BFGS-B',bounds=bounds)
    #estimate_value_boot_RE = jansen_rit_bootstrap_bias_correction_minimize(result.x,n_boot,T,dt,observation,N_interval,t_spans)
    #estimate_value_bias_RE = bias_correction(result.x, estimate_value_boot_RE)
    #estimate_value_bias_RE[0] = result.x[0]
    ARE_VALUE_combine_Q[5:10] = np.abs((x_best-[A1/B1,A1*C1,v01,A1*K1,A1*sigma])/[A1/B1,A1*C1,v01,A1*K1,A1*sigma])
    ARE_VALUE_combine_Q_bia[5:10] = np.abs((estimate_value_bias-[A1/B1,A1*C1,v01,A1*K1,A1*sigma])/[A1/B1,A1*C1,v01,A1*K1,A1*sigma])
    #A_reestimate_SIGMA[1] = function_bias_correct_estimate_A(estimate_value_bias[1:],N_interval,y_iniii_node2,y_iniii_node1, t_spans, sol_node2)
    #A_reestimate_Embeding[1] = function_EMBBDIN_estimate_A(observations2,estimate_value_bias[1:],dt,observation,T,20)
    #estimate_value_combine[mm,5:10] = estimate_value_bias
    #ARE_VALUE_combine[5:10] = np.abs((estimate_value_bias-[A1,B1,C1,v01,K1])/[A1,B1,C1,v01,K1])
    
    ###RK method
    def wrapped_function_RK(params):
        return objective_function_re_RK(params,  observation, observations2, h)
    problem_dict = {
    "obj_func": wrapped_function_RK,
    "bounds": FloatVar(lb=(0.05,0,0,0,5), ub=(1,1500,15,1500,100),name='delta'),
    "minmax": "min",
    "verbose": False,
}

    
    model = DE.JADE(epoch=100, pop_size = 15,  verbose = False)
    g_best = model.solve(problem_dict,n_workers=-1)
    ARE_VALUE_combine_RK_Q[5:10] = np.abs((g_best.solution-[A1/B1,A1*C1,v01,A1*K1,A1*sigma])/[A1/B1,A1*C1,v01,A1*K1,A1*sigma])


###Node3
    ###EU method
    def wrapped_function_EU(params):
        return objective_function_re_Euler(params,  observations2, observations3, h)
    problem_dict = {
    "obj_func": wrapped_function_EU,
    "bounds": FloatVar(lb=(0.05,0,0,0,5), ub=(1,1500,15,1500,100),name='delta'),
    "minmax": "min",
    "verbose": False,
}

    
    model = DE.JADE(epoch=100, pop_size = 15,  verbose = False)
    g_best = model.solve(problem_dict,n_workers=-1)
    ARE_VALUE_combine_EU_Q[10:] = np.abs((g_best.solution-[A2/B2,A2*C2,v02,A2*K2,A2*sigma])/[A2/B2,A2*C2,v02,A2*K2,A2*sigma])  
    
    ###Chen-Fliess
    def wrapped_function(z):
        params = denormalize(z)
        return objective_function_re_minimize(params,N_interval,y_iniii_node3,y_iniii_node2, t_spans, sol_node3)
    problem_dict = {
    "obj_func": wrapped_function,
    "bounds": FloatVar(lb=(0,0,0,0,0), ub=(1,1,1,1,1),name='delta'),
    "minmax": "min",
    "verbose": False,
}

    model = DE.JADE(epoch=100, pop_size = 15,  verbose = False)
    g_best = model.solve(problem_dict,n_workers=-1)
    z_best = g_best.solution
    x_best = denormalize(z_best)
    estimate_value_boot = jansen_rit_bootstrap_bias_correction_minimize(x_best,n_boot,T,dt,observations2,N_interval,t_spans)
    estimate_value_bias = bias_correction(x_best, estimate_value_boot)
    #estimate_value_bias[0] = x_best[0]  
    #initial_guess = estimate_value_bias
    #bounds = [(2.6,9.75),(0.05,1),(0,estimate_value_bias[2]+200),(0,15),(0,estimate_value_bias[4]+200),(5,100)]
    #result = minimize(objective_function_re_minimize, initial_guess, args=(N_interval,y_iniii_node3,y_iniii_node2, t_spans, sol_node3),method='L-BFGS-B',bounds=bounds)
    #estimate_value_boot_RE = jansen_rit_bootstrap_bias_correction_minimize(result.x,n_boot,T,dt,observations2,N_interval,t_spans)
    #estimate_value_bias_RE = bias_correction(result.x, estimate_value_boot_RE)
    #estimate_value_bias_RE[0] = result.x[0]
    #estimate_value_combine_RE[mm,10:] = estimate_value_bias_RE
    ARE_VALUE_combine_Q[10:] = np.abs((x_best-[A2/B2,A2*C2,v02,A2*K2,A2*sigma])/[A2/B2,A2*C2,v02,A2*K2,A2*sigma])
    ARE_VALUE_combine_Q_bia[10:] = np.abs((estimate_value_bias-[A2/B2,A2*C2,v02,A2*K2,A2*sigma])/[A2/B2,A2*C2,v02,A2*K2,A2*sigma])
    #A_reestimate_SIGMA[2] = function_bias_correct_estimate_A(estimate_value_bias[1:],N_interval,y_iniii_node3,y_iniii_node2, t_spans, sol_node3)
    #A_reestimate_Embeding[2] = function_EMBBDIN_estimate_A(observations3,estimate_value_bias[1:],dt,observations2,T,20)
    #estimate_value_combine[mm,10:] = estimate_value_bias
    #ARE_VALUE_combine_Q[10:] = np.abs((estimate_value_bias-[A2,A2/B2,A2*C2,v02,A2*K2])/[A2,A2/B2,A2*C2,v02,A2*K2])
    
    ###RK method
    def wrapped_function_RK(params):
        return objective_function_re_RK(params,  observations2, observations3, h)
    problem_dict = {
    "obj_func": wrapped_function_RK,
    "bounds": FloatVar(lb=(0.05,0,0,0,5), ub=(1,1500,15,1500,100),name='delta'),
    "minmax": "min",
    "verbose": False,
}

    
    model = DE.JADE(epoch=100, pop_size = 15,  verbose = False)
    g_best = model.solve(problem_dict,n_workers=-1)
    ARE_VALUE_combine_RK_Q[10:] = np.abs((g_best.solution-[A2/B2,A2*C2,v02,A2*K2,A2*sigma])/[A2/B2,A2*C2,v02,A2*K2,A2*sigma])
    
    
    return Time_CH, Time_EU, Time_RK, ARE_VALUE_combine_Q,ARE_VALUE_combine_Q_bia, ARE_VALUE_combine_EU_Q, ARE_VALUE_combine_RK_Q







# ===== 并行运行 =====
results = Parallel(n_jobs=-1, verbose=10)(
    delayed(run_one_mm)(mm) for mm in range(n_reptation)
)


ARE_VALUE_combine = np.zeros((n_reptation,15))
ARE_VALUE_combine_bia = np.zeros((n_reptation,15))
ARE_VALUE_combine_EU = np.zeros((n_reptation,15))
ARE_VALUE_combine_RK = np.zeros((n_reptation,15))
Time_EU = np.zeros((n_reptation))
Time_CH = np.zeros((n_reptation))
Time_RK = np.zeros((n_reptation))


for mm, (time_ch, time_eu, time_rk, ARE_combine, ARE_combine_BIA, ARE_combine_EU, ARE_combine_RK) in enumerate(results):
    Time_RK[mm] = time_rk
    Time_EU[mm] = time_eu
    Time_CH[mm] = time_ch
    ARE_VALUE_combine[mm] = ARE_combine
    ARE_VALUE_combine_bia[mm] = ARE_combine_BIA
    ARE_VALUE_combine_EU[mm] = ARE_combine_EU
    ARE_VALUE_combine_RK[mm] = ARE_combine_RK    
import os   
folder_path = "/home/ky209/results_old/final/sigma/3_channel/no_bias_correctiton_3_channels"
ESTIMATES = ARE_VALUE_combine
data_are = pd.DataFrame(ESTIMATES, columns=['A0B0', 'A0C0', 'v00', 'A0K0','A0sigma', 'A1B1', 'A1C1', 'v01', 'A1K1','A1sigma', 'A2B2', 'A2C2', 'v02', 'A2K2','A2sigma'])
data_are.to_csv(os.path.join(folder_path, 'NewPLUS_ARE_Jansen_random_init_add_noise.csv'), index=False)

ESTIMATES = ARE_VALUE_combine_bia
data_are = pd.DataFrame(ESTIMATES, columns=['A0B0', 'A0C0', 'v00', 'A0K0','A0sigma', 'A1B1', 'A1C1', 'v01', 'A1K1','A1sigma', 'A2B2', 'A2C2', 'v02', 'A2K2','A2sigma'])
data_are.to_csv(os.path.join(folder_path, 'NewPLUS_ARE_Jansen_random_init_add_noise_bias_corrected.csv'), index=False)



ESTIMATES = ARE_VALUE_combine_EU
data_are = pd.DataFrame(ESTIMATES, columns=['A0B0', 'A0C0', 'v00', 'A0K0','A0sigma', 'A1B1', 'A1C1', 'v01', 'A1K1','A1sigma', 'A2B2', 'A2C2', 'v02', 'A2K2','A2sigma'])
data_are.to_csv(os.path.join(folder_path, 'NewPLUS_ARE_Jansen_EU_random_init_add_noise.csv'), index=False)

ESTIMATES = ARE_VALUE_combine_RK
data_are = pd.DataFrame(ESTIMATES, columns=['A0B0', 'A0C0', 'v00', 'A0K0','A0sigma', 'A1B1', 'A1C1', 'v01', 'A1K1','A1sigma', 'A2B2', 'A2C2', 'v02', 'A2K2','A2sigma'])
data_are.to_csv(os.path.join(folder_path, 'NewPLUS_ARE_Jansen_RK_random_init_add_noise.csv'), index=False)


#ESTIMATES = Time_CH
#data_are = pd.DataFrame(ESTIMATES, columns=['Time_CH'])
#data_are.to_csv(os.path.join(folder_path, 'Time_CH_add_noise_NEW.csv'), index=False)

#ESTIMATES = Time_RK
#data_are = pd.DataFrame(ESTIMATES, columns=['Time_RK'])
#data_are.to_csv(os.path.join(folder_path, 'Time_RK_add_noise_NEW.csv'), index=False)


#ESTIMATES = Time_EU
#data_are = pd.DataFrame(ESTIMATES, columns=['Time_EU'])
#data_are.to_csv(os.path.join(folder_path, 'Time_EU_add_noise_NEW.csv'), index=False)