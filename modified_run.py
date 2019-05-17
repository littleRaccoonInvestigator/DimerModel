
def run(S,sizeOfSample,shuffle_timestep,loop_timestep,mid_term,temperature,J,k_B,mcstp):
    
    energy_array = []
    energy2_array = []
    energy3_array = []
    energy4_array = []
    energy5_array = []
    
    capacity_array = []
    capacity2_array = []
    capacity3_array = []
    capacity4_array = []
    capacity5_array = []
    
    order_parm_array = []
    order_parm2_array = []
    order_parm3_array = []
    order_parm4_array = []
    order_parm5_array = []
   
    refresh_accept = 0
    refresh_reject = 0
    length_stat = []

    percent = 0
    for k in range(shuffle_timestep):
        rand_i2 = random.randrange(sizeOfSample)
        rand_j2 = random.randrange(sizeOfSample)
        path2 = findcycle(S[rand_i2][rand_j2],S,sizeOfSample)
        S = updateLoop(S,path2)
        if(k*100/shuffle_timestep > percent):
            print("%d%%"%percent, end = ' ',flush = True)
            if percent%10 == 0:
                print()
            percent += 1
           

    percent = 0
    count = 0
    flag = False
    count2 = 0
##    for k in range(loop_timestep):
    while True:
        count2 += 1
##        if(k*100/loop_timestep > percent):
##            print("%d%%"%percent, end = ' ',flush = True)
##            if percent%10 == 0:
##                print()
##            percent += 1
        
        for i in range(mcstp):
            rand_i = random.randrange(sizeOfSample)
            rand_j = random.randrange(sizeOfSample)
            path = findcycle(S[rand_i][rand_j],S,sizeOfSample)
            E_previous  = totalNumOfStar(S,sizeOfSample)*J
            S = updateLoop(S,path)
            E_after  = totalNumOfStar(S,sizeOfSample)*J
            E1 = E_after - E_previous

            if flag:
                aa = random.random()
                aa = np.log(aa)
                
                if (E1/k_B/temperature+aa<0): 
                    E = E_after
                    refresh_accept += 1
                else:
                    E = E_previous
                    S = updateLoop(S,path)
                    refresh_reject += 1
                    
                length = len(path)
                length_stat.append(length)
            else:
                if (E1<=1): 
                    E = E_after
                    refresh_accept += 1
                else:
                    E = E_previous
                    S = updateLoop(S,path)
                    refresh_reject += 1
                if(E==-192 and (flag==False)):
                    flag = True
                    print('T='+str(temperature)+'时,触底次数:'+str(count2))
                    print(orderparameter(S,sizeOfSample))
                if(count2>=30):
                    count2 = 0
                    for k in range(shuffle_timestep):
                        rand_i2 = random.randrange(sizeOfSample)
                        rand_j2 = random.randrange(sizeOfSample)
                        path2 = findcycle(S[rand_i2][rand_j2],S,sizeOfSample)
                        S = updateLoop(S,path2)


        if flag:
            if(count%(5000/100) == 0):
                print("%d%%"%percent, end = ' ',flush = True)
                percent += 1
                print()
            count += 1
            if(count<=1):
                ordr = orderparameter(S,sizeOfSample)# calculate orderparameter
                energy_array.append(E)
                capacity_array.append(E**2)
                order_parm_array.append(ordr)
            elif(count<=2):
                ordr = orderparameter(S,sizeOfSample)
                energy2_array.append(E)
                capacity2_array.append(E**2)
                order_parm2_array.append(ordr)
            elif(count<=3):
                ordr = orderparameter(S,sizeOfSample)
                energy3_array.append(E)
                capacity3_array.append(E**2)
                order_parm3_array.append(ordr)
            elif(count<=4):
                ordr = orderparameter(S,sizeOfSample)
                energy4_array.append(E)
                capacity4_array.append(E**2)
                order_parm4_array.append(ordr)
            elif(count<=5):
                ordr = orderparameter(S,sizeOfSample)
                energy5_array.append(E)
                capacity5_array.append(E**2)
                order_parm5_array.append(ordr)
##            if k >= mid_term:
##                energy2 += E
##                capacity2 += E**2
            if(count==5):
                break
        else:
            print(str(count2)+'次:'+str(E),end = ' ')
        

##        if k >= mid_term:
##            energy2_array.append(E)
##            capacity2_array.append(E**2)
##            order_parm_array2.append(ordr)
        
       
    energy = np.mean(np.array(energy_array))
    energy2 = np.mean(np.array(energy2_array))
    energy3 = np.mean(np.array(energy3_array))
    energy4 = np.mean(np.array(energy4_array))
    energy5 = np.mean(np.array(energy5_array))
    
    capacity = np.mean(np.array(capacity_array))
    capacity2 = np.mean(np.array(capacity2_array))
    capacity3 = np.mean(np.array(capacity3_array))
    capacity4 = np.mean(np.array(capacity4_array))
    capacity5 = np.mean(np.array(capacity5_array))
    
    capacity = (capacity-energy**2)/((k_B*temperature)**2)/sizeOfSample**2
    capacity2 =(capacity2-energy2**2)/((k_B*temperature)**2)/sizeOfSample**2
    capacity3 =(capacity3-energy3**2)/((k_B*temperature)**2)/sizeOfSample**2
    capacity4 =(capacity4-energy4**2)/((k_B*temperature)**2)/sizeOfSample**2
    capacity5 =(capacity5-energy5**2)/((k_B*temperature)**2)/sizeOfSample**2
    
    energy = energy/sizeOfSample**2
    energy2 = energy2/sizeOfSample**2
    energy3 = energy3/sizeOfSample**2
    energy4 = energy4/sizeOfSample**2
    energy5 = energy5/sizeOfSample**2

    orderParm = np.mean(np.array(order_parm_array))
    orderParm2 = np.mean(np.array(order_parm2_array))
    orderParm3 = np.mean(np.array(order_parm3_array))
    orderParm4 = np.mean(np.array(order_parm4_array))
    orderParm5 = np.mean(np.array(order_parm5_array))
    
    print('\nparameters:(energy,capacity,temperature,loop_timesteps)\n',[energy,energy2,energy3,energy4,energy5,capacity,capacity2,capacity3,capacity4,capacity5,temperature,loop_timestep])
    print()
    print()
    
    accept_ratio = refresh_accept/(refresh_accept + refresh_reject)
    avg_length = np.mean(np.array(length_stat))
        
    return [temperature,energy,energy2,energy3,energy4,energy5,capacity,capacity2,capacity3,capacity4,capacity5,orderParm,orderParm2,orderParm3,orderParm4,orderParm5,accept_ratio,avg_length,
    sizeOfSample,shuffle_timestep,mcstp,loop_timestep,mid_term,J,k_B]
