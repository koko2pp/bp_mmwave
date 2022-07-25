import numpy as np
import math
import matplotlib.pyplot as plt


def calculate_ED(Ncp,Ncf,Vcp,Vcf,Angel):
    Ecp = 0
    Ecf = 0
    Dcp = 0
    Dcf = 0
    Amax = -10000
    Amin = 10000
    Dabs = 0
    Dabs_temp = 0
    for x in range(0,np.size(Ncp)):
        Ecp = Ecp + 0.5 * Ncp[x] * math.pow(Vcp[x],2)
        Ecf = Ecf + 0.5 * Ncf[x] * math.pow(Vcf[x],2)
        Dcp = Dcp + Vcp[x]
        Dcf = Dcf - Vcf[x]
        Dabs_temp = Vcp[x] - Vcf[x]
        if Dabs_temp > Dabs:
            Dabs = Dabs_temp
        if Angel[x] < 50:  # 原值22
            if Angel[x] > Amax:
                Amax = Angel[x]
            if Angel[x] < Amin:
                Amin = Angel[x]
    A_range = Amax - Amin
    Etotal = Ecp + Ecf
    Dcp = Dcp / np.mean(Ncp)
    Dcf = Dcf / np.mean(Ncf)
    Dtotal = (Dcp - Dcf) / A_range
    return Etotal, Dtotal, Dabs, A_range

def judge(E, D):
    if E >= 10 and E <= 400 and D <= 5 and D >= -5:
#    if E > 10 and E < 150:
#    if D < 5 and D > -5:
        mark = 1
    else:
        mark = 0
    return mark

def f(mydir, action, file_range, cnt):
    x0 = np.zeros((cnt, 20))  # Ncp
    x1 = np.zeros((cnt, 20))  # Ncf
    x2 = np.zeros((cnt, 20))  # Vcp
    x3 = np.zeros((cnt, 20))  # Vcf
    x4 = np.zeros((cnt, 20))  # Angle
    
    i = 0
    for a in action:
        for f in file_range:
            file = mydir + a + f
            x = np.loadtxt(file)
            x = x.reshape(20, 12)
            
            x0[i] = x[:, 4]
            x1[i] = x[:, 7]
            x2[i] = x[:, 5]
            x3[i] = x[:, 8]
            x4[i] = x[:, 10]
            
            i += 1
            
    print(i)
    
    x0 = np.transpose(x0)
    x1 = np.transpose(x1)
    x2 = np.transpose(x2)
    x3 = np.transpose(x3)
    x4 = np.transpose(x4)
    
    Amin = np.zeros(cnt)
    Amax = np.zeros(cnt)
    
    for i in range(cnt):
        Amin[i] = np.min(x4[:, i])
        Amax[i] = np.max(x4[:, i])
    
    print(np.max(Amax))  # 确定那个if Angel[x] < 22
    print(np.min(Amin))
    
    Etotals = np.zeros(cnt)
    Dtotals = np.zeros(cnt)
    
    for i in range(cnt):
        Ncp = x0[:, i]
        Ncf = x1[:, i]
        Vcp = x2[:, i]
        Vcf = x3[:, i]
        Angle = x4[:, i]
        
        Etotal, Dtotal, Dabs, A_range = calculate_ED(Ncp, Ncf, Vcp, Vcf, Angle)
        
        Etotals[i] = Etotal / 1e6
        
        Dtotals[i] = Dtotal * 1000
    
    ee = sorted(Etotals)
    dd = sorted(Dtotals)
    
#    for i in range(cnt):
#        print((Etotals[i], Dtotals[i]))
    
    return Etotals, Dtotals


if __name__ == "__main__":
    
    mydir = "C:/Users/dzh/Desktop/Gestures_ZTE/dzh_"
    action0 = ["knock", "lswipe", "rswipe", "lcircle", "rcircle"]
    action1 = ["redund"]
    
    cnt0 = 5 * 500
    cnt1 = 1 * 100
    
    file_range0 = []
    for i in range(500):
        if i < 10:
            file_range0.append("_00"+str(i)+".txt")
        elif i >= 10 and i < 100:
            file_range0.append("_0"+str(i)+".txt")
        elif i >= 100:
            file_range0.append("_"+str(i)+".txt")
    
    file_range1 = []
    for i in range(100):
        if i < 10:
            file_range1.append("_00"+str(i)+".txt")
        elif i >= 10 and i < 100:
            file_range1.append("_0"+str(i)+".txt")
        elif i >= 100:
            file_range1.append("_"+str(i)+".txt")
    
    
    e0, d0 = f(mydir, action0, file_range0, cnt0)
#    e1, d1 = f(mydir, action1, file_range1, cnt1)
    
    fig = plt.figure()
    plt.scatter(e0, d0, marker = '+',color = 'red', s = 40 ,label = 'Valid')
#    plt.scatter(e1, d1, marker = 'x', color = 'blue', s = 40, label = 'Redund')
    plt.ylim(-2000, 2000)
    plt.xlim(0, 7)
    plt.legend(loc="best",markerscale=1.,numpoints=2,scatterpoints=1,fontsize=12)
    plt.title('Et-Dt',fontsize=16)
    plt.xlabel('Et',fontsize=10)
    plt.ylabel('Dt',fontsize=10)
    plt.show()
    
    print("有效手势：")
    print("Et范围", (np.min(e0), np.max(e0)))
    print("Dt范围", (np.min(d0), np.max(d0)))
    
    
    