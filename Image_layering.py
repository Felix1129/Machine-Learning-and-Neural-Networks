import numpy as np
import matplotlib.pyplot as plt

a = []
b = []
c = open('ICA_2_mess.dat', 'r')
for line in c:
    exec("b = np.array("+line+")") 
    a.append(b) 
c.close()
data = np.array(a) 

plt.figure('Figure Object 1',       # 图形对象名称  窗口左上角显示
           figsize = (8, 6),        # 窗口大小
           dpi = 130,               # 分辨率
           facecolor = 'white',     # 背景色
           )
m = data.shape[0]   
n = np.arange( m )    
plt.scatter(n, data[:, 0], color='red', s=1)  
plt.scatter(n, data[:, 1], color='blue', s=1) 
ax = plt.gca()                     #生成座標
ax.spines['top'].set_color('none')
ax.spines['bottom'].set_position(('data', 0))
ax.spines['left'].set_position(('data', 0))
ax.spines['right'].set_color('none')

data = data.T



def covariance(x):  
    mean = np.mean(x, axis=1, keepdims=True)   
    n = np.shape(x)[1] - 1                      
    m = x - mean                               
    return (m.dot(m.T))/n     

def whiten():
    coVarM = covariance(data)   
    U, S, V = np.linalg.svd(coVarM)                                     
    d = np.diag(1.0 / np.sqrt(S))       #計算白化矩陣D^(-0.5)
                                        #公式v(w,t) = D^(−0.5)V'
                                        #    v(fin,t) = O(α) · v(w,t)
    whiteM = np.dot(U, np.dot(d, U.T)) #得到白化矩陣    
    Xw = np.dot(whiteM, data)          #得到白化資料
    return Xw,whiteM

Xw, whiteM = whiten() 

data1=data
def likelihoodIca(signals,  alpha = 1):
    m, n = signals.shape        
    W_n =  np.random.rand(m, m) 
    W_m = np.dot(whiteM,0)    
    it=1                        
    while it < 10000:       
        for c in range(m):
            
            W_n = W_n.reshape(m, -1)           #將W_n矩陣之列存進w並轉成(m*n)陣列
            w = W_n[c, :]
            W_n[c, :]= W_n[c, :] / np.sqrt((w ** 2).sum())
            z = np.dot(W_n[c, :].T, signals) 
            g_z = 1 / (1+np.exp(-z))           #g(z)=1/(1+e^(-z))
            X_G = 1 - 2*np.dot(g_z,data1.T)    #X_G=1-2g(w^Tx)
            W_m= W_n + alpha*(X_G.dot(W_m.T))+np.linalg.inv(W_n).T           
            it=it+1     
    return W_n

W = likelihoodIca(data,alpha=1)   
unMixed = np.dot(W,data).T         

plt.figure('Figure Object 2',       # 图形对象名称  窗口左上角显示
           figsize = (8, 6),        # 窗口大小
           dpi = 130,               # 分辨率
           facecolor = 'white',     # 背景色
           )
m = unMixed.shape[0]
n = np.arange(1, m + 1)
plt.scatter(n, unMixed[:, 0], color='red', s=1)
ax = plt.gca()                     #生成座標
ax.spines['top'].set_color('none')
ax.spines['bottom'].set_position(('data', 0))
ax.spines['left'].set_position(('data', 0))
ax.spines['right'].set_color('none')

plt.figure('Figure Object 3',       # 图形对象名称  窗口左上角显示
           figsize = (8, 6),        # 窗口大小
           dpi = 130,               # 分辨率
           facecolor = 'white',     # 背景色
           )
m = unMixed.shape[0]
n = np.arange(1, m + 1)
plt.scatter(n, unMixed[:, 1], color='blue', s=1)
plt.xticks([-5,0,5])
ax = plt.gca()                     #生成座標
ax.spines['top'].set_color('none')
ax.spines['bottom'].set_position(('data', 0))
ax.spines['left'].set_position(('data', 0))
ax.spines['right'].set_color('none')


