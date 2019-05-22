from multiprocessing import Pool
import random
import matplotlib.pyplot as plt
import numpy as np
import datetime
import pickle

def go_right(node, sz): #typically sz = 6
    global S
    x0 = node.location[1]
    y0 = node.location[0]
    x,y = 0,0
    short = sz*2/3
    if x0 == sz-1:
        if y0 >= short/2:
            x = 0
            y = y0 - short/2
        else:
            x = 0
            y = y0 + short/2
    else:
        x = x0 + 1
        y = y0
    x = int(x)
    y = int(y)
    return S[y][x]

def go_left(node, sz):
    global S
    x0 = node.location[1]
    y0 = node.location[0]
    x,y = 0,0
    short = sz*2/3
    if x0 == 0:
        if y0 < short/2:
            x = sz-1
            y = y0 + short//2
        else:
            x = sz-1
            y = y0 -short//2
    else:
        x = x0 - 1
        y = y0
    x = int(x)
    y = int(y)
    
    return S[y][x]
    
def go_updown(node, sz):
    global S
    x0 = node.location[1]
    y0 = node.location[0]
    x,y = 0,0
    short = sz*2/3
    x = x0
    if (x0 + y0)%2 == 1: #go up
        y = (y0 + 1)%short
    else:   #go down
        y = (y0 - 1)%short
    x = int(x)
    y = int(y)
    return S[y][x]
    
def retrive(S, location): # node.linkedto = retrive(S,go_updown(node, sizeOfSample))
    return S[location[0]][location[1]]

class Node(object):
    x = 0
    y = 0
    location = None
    direction = True
    linkedto = None
    def __init__(self, location):
        self.location = location       
        self.direction = True
        if (sum(location)%2 == 0):
            self.direction = False
        self.x = location[1]*np.sqrt(3)/2
        self.y = location[0]*1.5+0.5 if(self.direction) else location[0]*1.5
    
    def plot(self):
        plt.plot(self.x, self.y,marker = 'o',markersize = 5.0,mec = [0.5,0,0],mfc = [0.5,0,0])    

def initialGraph(sizeOfSample):
    global S
    S = np.array([[Node([i,j]) for j in range(sizeOfSample)]for i in range(sizeOfSample*2//3)])    
    for i in range(sizeOfSample*2//3):
        for j in range(sizeOfSample):
            node = S[i][j]
            location = node.location
            direction = node.direction
            if (direction):
                S[i][j].linkedto = S[(location[0]+1)%(sizeOfSample*2//3)][location[1]]
            else:
                S[i][j].linkedto = S[(location[0]-1)%(sizeOfSample*2//3)][location[1]]
    seq = [complex(-0.5,np.sqrt(3)/2),1,complex(-0.5,-np.sqrt(3)/2)]
    global plaquetteType
    plaquetteType = np.array([[seq[j%3-(i%2)] for j in range(sizeOfSample//2)]for i in range(sizeOfSample*2//3)])
    for i in range(sizeOfSample*2//3):
        for j in range(sizeOfSample//2):
            k = j%3-(i%2)
            if(k==0):
                S[i][j*2+i%2].linkedto = go_left(S[i][j*2+i%2],sizeOfSample)
                go_right(S[i][j*2+i%2],sizeOfSample).linkedto = go_right(S[(i+1)%(sizeOfSample*2//3)][j*2+i%2],sizeOfSample)
            elif(k==2 or k==-1):
                go_right(S[i][j*2+i%2],sizeOfSample).linkedto = go_right(go_right(S[i][j*2+i%2],sizeOfSample),sizeOfSample)
                S[i][(j*2+i%2)%sizeOfSample].linkedto = S[(i-1)%(sizeOfSample*2//3)][(j*2+i%2)%sizeOfSample]
            elif(k==1):
                S[i][j*2+i%2].linkedto = go_right(S[i][j*2+i%2],sizeOfSample)
                go_right(S[i][j*2+i%2],sizeOfSample).linkedto = S[i][(j*2+i%2)%sizeOfSample]
   # plot(S,sizeOfSample)
    print(orderparameternew(S,sizeOfSample))
    
    return S


def findcycle(node,S,sizeOfSample):
    path = [node]
    mark = True
    count = 0
    while True:
        direction = node.direction
        if(direction):
            updown = 1
        else:
            updown = -1

        all_choice = [-1,0,1]
        linked_index = node.linkedto.location
        now_index = node.location
        if(linked_index[1]==now_index[1]):
            linked_type = 0
        elif(linked_index[1]-now_index[1] == 1):
            linked_type = 1
        else:
            linked_type = -1
        
        all_choice.remove(linked_type)
        choice = random.choice(all_choice)

        if(mark):
            nextnode = node.linkedto
            mark = False
            
        else:
            if(choice == 0):
                nextnode = S[(now_index[0]+updown)%(sizeOfSample*2//3)][now_index[1]]
            elif(choice==1):
                nextnode = go_right(S[now_index[0]][(now_index[1])],sizeOfSample)
            elif(choice==-1):
                nextnode = go_left(S[now_index[0]][(now_index[1])],sizeOfSample)
            mark = True

        count += 1
        node = nextnode
        if(nextnode in path):
            break
        path.append(nextnode)
        
    end = path.index(nextnode)
    path = path[end:]
    return path
        
def updateLoop(S,path):
    
    flag = 0
    if(path[0].linkedto == path[1]):
        flag = 1
    for i in range(len(path)):
        index = path[i].location
        index2 = path[(i+(-1)**(i+flag))%len(path)].location
        S[index[0]][index[1]].linkedto = S[index2[0]][index2[1]]
        
    return S

def starType(S,node,sizeOfSample): 
    location1 = node.location
    location1_x = location1[0]
    location1_y = location1[1]
    right = go_right(node,sizeOfSample)
    left = go_left(node,sizeOfSample)
    node_list = [node,
                 right,
                 go_updown(right,sizeOfSample),
                 S[(location1_x+1)%(sizeOfSample*2//3)][location1_y],
                 go_updown(left,sizeOfSample),
                 left
                 ]
    count = 0
    for item in node_list:
        if(item.linkedto in node_list):
            count += 1
   
    if count == 6:
        if node_list[0].linkedto == node_list[1]:
            return 1 #up type star
        else:
            return -1 #down type star
    elif count == 0:
        return 0 #radiative star
    
    return "other kind of star"
    
def orderParameter_obsolete(S,sizeOfSample):
    countOfStar = [0,0,0]
    
    for i in range(sizeOfSample*2//3):
        for j in range(sizeOfSample//2):
            if(starType(S,S[i][j*2+i%2],sizeOfSample)==1):
                countOfStar[0] += 1
            elif(starType(S,S[i][j*2+i%2],sizeOfSample)==-1):
                countOfStar[1] += 1
            elif(starType(S,S[i][j*2+i%2],sizeOfSample)==0):
                countOfStar[2] += 1
                
    unit_vector_1 = complex(1,0) # radiative
    unit_vector_2 = complex(-1/2, np.sqrt(3)/2) #up
    unit_vector_3 = complex(-1/2, -np.sqrt(3)/2) #down
    return countOfStar[0]*unit_vector_2 + countOfStar[1]*unit_vector_3 + countOfStar[2]*unit_vector_1

def orderparameterold(S,sizeOfSample):
    return sum(sum(np.multiply(plaquetteType,finalType(S,sizeOfSample))))*6/(sizeOfSample)**2

def orderparameternew(S,sizeOfSample):
    seq = [1,2,0]
    a = complex(-0.5,np.sqrt(3)/2)
    b = 1
    c = complex(-0.5,-np.sqrt(3)/2)
    order = 0
    for i in range(sizeOfSample*2//3):
        for j in range(sizeOfSample//2):
            k = seq[j%3-(i%2)]
            y = S[i][j*2+i%2].linkedto.location[1]
            if k==0:
                #print(k)
                if(y==j*2+i%2):
                    order += c
                    #print('c')
                elif(y-(j*2+i%2) == 1 or y-(j*2+i%2)<-1):
                    order += a
                    #print('a')
                elif((j*2+i%2)-y == 1 or y-(j*2+i%2)>1):
                    order += b
                   # print('b')
                else:
                    print('A Warning')
            elif k==1:
                #print(k)
                if(y==j*2+i%2):
                    order += a
                    #print('a')
                elif(y-(j*2+i%2) == 1 or y-(j*2+i%2)<-1):
                    order += b
                   # print('b')
                elif((j*2+i%2)-y == 1 or y-(j*2+i%2)>1):
                    order += c
                    #print('c')
                else:
                    print('B Warning')
            elif k==2:
                #print(k)
                if(y==j*2+i%2):
                    order += b
                    #print('b')
                elif(y-(j*2+i%2) == 1 or y-(j*2+i%2)<-1):
                    order += c
                    #print('c')
                elif(j*2+i%2-y == 1 or y-(j*2+i%2)>1):
                    order += a
                    #print('a')
    return order*3/(sizeOfSample**2)

def finalType(S,sizeOfSample):
    final = [[0 for j in range(sizeOfSample//2)]for i in range(sizeOfSample*2//3)]
    for i in range(sizeOfSample*2//3):
        for j in range(sizeOfSample//2):
            if(isStar(S,S[i][j*2+i%2],sizeOfSample)==3):
                if(S[i][j*2+i%2].linkedto==go_left(S[i][j*2+i%2],sizeOfSample)):
                    final[i][j]=1
    final = np.array(final)
    return final

def isStar(S,node,sizeOfSample): 
    location1 = node.location
    location1_x = location1[0]
    location1_y = location1[1]
    right = go_right(node,sizeOfSample)
    left = go_left(node,sizeOfSample)
    node_list = [node,
                 right,
                 go_updown(right,sizeOfSample),
                 S[(location1_x+1)%(sizeOfSample*2//3)][location1_y],
                 go_updown(left,sizeOfSample),
                 left
                 ]
    count = 0
    for item in node_list:
        if(item.linkedto in node_list):
            count += 1
    return count/2

def totalNumOfStar(S,sizeOfSample):
    countOfStar = 0
    for y in range(sizeOfSample*2//3):
        for x in range(sizeOfSample//2):
            if(isStar(S,S[y][x*2+y%2],sizeOfSample)==3):
                countOfStar += 1
    return countOfStar

def adjacent_plat(S, node1, node2, sizeOfSample):
    bottom_node_list = []
    x1,y1,x2,y2 = node1.location[1], node1.location[0],node2.location[1], node2.location[0]
    if x1 == x2:
        if y1 == 0 and y2 == sizeOfSample - 1:
            bottom_node_list.append(go_right(node2,sizeOfSample))
            bottom_node_list.append(go_left(node2,sizeOfSample))
        elif y2 == 0 and y1 == sizeOfSample - 1:
            bottom_node_list.append(go_right(node1,sizeOfSample))
            bottom_node_list.append(go_left(node1,sizeOfSample))
        elif y1 < y2:
            bottom_node_list.append(go_right(node1,sizeOfSample))
            bottom_node_list.append(go_left(node1,sizeOfSample))
        else:
            bottom_node_list.append(go_right(node2,sizeOfSample))
            bottom_node_list.append(go_left(node2,sizeOfSample))
    else:
        if (x1 + y1)%2 == 0:
            bottom_node_list.append(S[y1][x1])
            bottom_node_list.append(S[(y2-1)%(sizeOfSample*2//3)][x2])
        elif (x2 + y2)%2 == 0:
            bottom_node_list.append(S[y2][x2])
            bottom_node_list.append(S[(y1-1)%(sizeOfSample*2//3)][x1])
        else:
            return None
    return bottom_node_list

def delta_star(lattice, path, sizeOfSample):
    previous_count = 0
    next_count = 0
    bnl = []
    for index in range(len(path)):
        bnl.extend(adjacent_plat(lattice,path[index],path[(index+1)%len(path)],sizeOfSample))  
    bnl = set(bnl) 
    for node in bnl:
        if isStar(lattice, node, sizeOfSample) == 3:
            previous_count += 1
    updateLoop(lattice, path)
    for node in bnl:
        if isStar(lattice, node, sizeOfSample) == 3:
            next_count += 1
    delta_star =  next_count - previous_count
    return delta_star
    
def run(S,sizeOfSample,shuffle_timestep,loop_timestep,mid_term,temperature,J,k_B,mcstp):
    
    energy1_array = []
    energy2_array = []
    energy3_array = []
    energy4_array = []
    energy5_array = []
    energy6_array = []
    energy7_array = []
    energy8_array = []
    energy9_array = []
    energy10_array = []
    
    capacity1_array = []
    capacity2_array = []
    capacity3_array = []
    capacity4_array = []
    capacity5_array = []
    capacity6_array = []
    capacity7_array = []
    capacity8_array = []
    capacity9_array = []
    capacity10_array = []
    
    
    order_parm1_array = []
    order_parm2_array = []
    order_parm3_array = []
    order_parm4_array = []
    order_parm5_array = []
    order_parm6_array = []
    order_parm7_array = []
    order_parm8_array = []
    order_parm9_array = []
    order_parm10_array = []

    OldOrder_parm1_array = []
    OldOrder_parm2_array = []
    OldOrder_parm3_array = []
    OldOrder_parm4_array = []
    OldOrder_parm5_array = []
    OldOrder_parm6_array = []
    OldOrder_parm7_array = []
    OldOrder_parm8_array = []
    OldOrder_parm9_array = []
    OldOrder_parm10_array = []
    
    order_square1_array = []
    order_square2_array = []
    order_square3_array = []
    order_square4_array = []
    order_square5_array = []
    order_square6_array = []
    order_square7_array = []
    order_square8_array = []
    order_square9_array = []
    order_square10_array = []
   
    order_fourthpower1_array = []
    
    order_fourthpower2_array = []
    order_fourthpower3_array = []
    order_fourthpower4_array = []
    order_fourthpower5_array = []
    order_fourthpower6_array = []
    order_fourthpower7_array = []
    order_fourthpower8_array = []
    order_fourthpower9_array = []
    order_fourthpower10_array = []  
   
    refresh_accept = 0
    refresh_reject = 0
    length_stat = []

    percent = 0
    for k in range(shuffle_timestep):
        rand_i2 = random.randrange(sizeOfSample*2//3)
        rand_j2 = random.randrange(sizeOfSample)
        path2 = findcycle(S[rand_i2][rand_j2],S,sizeOfSample)
        S = updateLoop(S,path2)
        if(k*100/shuffle_timestep > percent):
            print("%d%%"%percent, end = ' ',flush = True)
            if percent%10 == 0:
                print()
            percent += 1
           
    gsop = 0
    percent = 0
    
    S = initialGraph(sizeOfSample)
  #  S = load_lattice(sizeOfSample, "ground_state_36")
    gsop = orderparameternew(S,sizeOfSample)
    count = 0
    percent = 0
    while True:   
        accept_list = [] #if you do not want to print step information delete marked line 281,282,292,296,305,306,307
        sample_list = []
        for i in range(mcstp):
            rand_i = random.randrange(sizeOfSample*2//3)
            rand_j = random.randrange(sizeOfSample)
            path = findcycle(S[rand_i][rand_j],S,sizeOfSample)
            E1 = delta_star(S, path, sizeOfSample)*J

            aa = random.random()
            aa = np.log(aa)
            
            sample_list.extend(path) #
            if (E1/k_B/temperature+aa<0): 
                
                refresh_accept += 1
                accept_list.extend(path) #
            else:
    
                S = updateLoop(S,path)
                refresh_reject += 1
                
            length = len(path)
            length_stat.append(length)
        
        sample_list = list(set(sample_list)) #
        accept_list = list(set(accept_list)) #
        #print("step %d: acpt_rat: %f, smp_rat: %f"%(count, len(accept_list)/sizeOfSample**2, len(sample_list)/sizeOfSample**2)) #
        
        if(count*100/loop_timestep > percent):
            print("%d%%"%percent, end = ' ',flush = True)
            if percent%10 == 0:
                print()
            percent += 1
        count += 1
        
        
        E = totalNumOfStar(S,sizeOfSample)*J
        ordr = orderparameternew(S,sizeOfSample)
        ordrold = orderparameterold(S,sizeOfSample)
        ordra = abs(ordr)
        if(count<=loop_timestep/10):
            energy1_array.append(E)
            capacity1_array.append(E**2)
            order_parm1_array.append(ordr)
            OldOrder_parm1_array.append(ordrold)
            order_square1_array.append(ordra**2)
            order_fourthpower1_array.append(ordra**4)
        elif(count<=loop_timestep/10*2):
            energy2_array.append(E)
            capacity2_array.append(E**2)
            order_parm2_array.append(ordr)
            OldOrder_parm2_array.append(ordrold)
            order_square2_array.append(ordra**2)
            order_fourthpower2_array.append(ordra**4)
        elif(count<=loop_timestep/10*3):
            energy3_array.append(E)
            capacity3_array.append(E**2)
            order_parm3_array.append(ordr)
            OldOrder_parm3_array.append(ordrold)
            order_square3_array.append(ordra**2)
            order_fourthpower3_array.append(ordra**4)
        elif(count<=loop_timestep/10*4):
            energy4_array.append(E)
            capacity4_array.append(E**2)
            order_parm4_array.append(ordr)
            OldOrder_parm4_array.append(ordrold)
            order_square4_array.append(ordra**2)
            order_fourthpower4_array.append(ordra**4)
        elif(count<=loop_timestep/10*5):
            energy5_array.append(E)
            capacity5_array.append(E**2)
            order_parm5_array.append(ordr)
            OldOrder_parm5_array.append(ordrold)
            order_square5_array.append(ordra**2)
            order_fourthpower5_array.append(ordra**4)
        elif(count<=loop_timestep/10*6):
            energy6_array.append(E)
            capacity6_array.append(E**2)
            order_parm6_array.append(ordr)
            OldOrder_parm6_array.append(ordrold)
            order_square6_array.append(ordra**2)
            order_fourthpower6_array.append(ordra**4)
        elif(count<=loop_timestep/10*7):
            energy7_array.append(E)
            capacity7_array.append(E**2)
            order_parm7_array.append(ordr)
            OldOrder_parm7_array.append(ordrold)
            order_square7_array.append(ordra**2)
            order_fourthpower7_array.append(ordra**4)
        elif(count<=loop_timestep/10*8):
            energy8_array.append(E)
            capacity8_array.append(E**2)
            order_parm8_array.append(ordr)
            OldOrder_parm8_array.append(ordrold)
            order_square8_array.append(ordra**2)
            order_fourthpower8_array.append(ordra**4)
        elif(count<=loop_timestep/10*9):
            energy9_array.append(E)
            capacity9_array.append(E**2)
            order_parm9_array.append(ordr)
            OldOrder_parm9_array.append(ordrold)
            order_square9_array.append(ordra**2)
            order_fourthpower9_array.append(ordra**4)
        elif(count<=loop_timestep):
            energy10_array.append(E)
            capacity10_array.append(E**2)
            order_parm10_array.append(ordr)
            OldOrder_parm10_array.append(ordrold)
            order_square10_array.append(ordra**2)
            order_fourthpower10_array.append(ordra**4)
            
        if(count==loop_timestep):
            break
        
    energy1 = np.mean(np.array(energy1_array))
    energy2 = np.mean(np.array(energy2_array))
    energy3 = np.mean(np.array(energy3_array))
    energy4 = np.mean(np.array(energy4_array))
    energy5 = np.mean(np.array(energy5_array))
    energy6 = np.mean(np.array(energy6_array))
    energy7 = np.mean(np.array(energy7_array))
    energy8 = np.mean(np.array(energy8_array))
    energy9 = np.mean(np.array(energy9_array))
    energy10 = np.mean(np.array(energy10_array))
    
    capacity1 = np.mean(np.array(capacity1_array))
    capacity2 = np.mean(np.array(capacity2_array))
    capacity3 = np.mean(np.array(capacity3_array))
    capacity4 = np.mean(np.array(capacity4_array))
    capacity5 = np.mean(np.array(capacity5_array))
    capacity6 = np.mean(np.array(capacity6_array))
    capacity7 = np.mean(np.array(capacity7_array))
    capacity8 = np.mean(np.array(capacity8_array))
    capacity9 = np.mean(np.array(capacity9_array))
    capacity10 = np.mean(np.array(capacity10_array))
    
    capacity1 =(capacity1-energy1**2)/((k_B*temperature)**2)/sizeOfSample**2*3/2
    capacity2 =(capacity2-energy2**2)/((k_B*temperature)**2)/sizeOfSample**2*3/2
    capacity3 =(capacity3-energy3**2)/((k_B*temperature)**2)/sizeOfSample**2*3/2
    capacity4 =(capacity4-energy4**2)/((k_B*temperature)**2)/sizeOfSample**2*3/2
    capacity5 =(capacity5-energy5**2)/((k_B*temperature)**2)/sizeOfSample**2*3/2
    capacity6 =(capacity6-energy6**2)/((k_B*temperature)**2)/sizeOfSample**2*3/2
    capacity7 =(capacity7-energy7**2)/((k_B*temperature)**2)/sizeOfSample**2*3/2
    capacity8 =(capacity8-energy8**2)/((k_B*temperature)**2)/sizeOfSample**2*3/2
    capacity9 =(capacity9-energy9**2)/((k_B*temperature)**2)/sizeOfSample**2*3/2
    capacity10 =(capacity10-energy10**2)/((k_B*temperature)**2)/sizeOfSample**2**3/2
    
    
    energy1 = energy1/sizeOfSample**2*3/2
    energy2 = energy2/sizeOfSample**2*3/2
    energy3 = energy3/sizeOfSample**2*3/2
    energy4 = energy4/sizeOfSample**2*3/2
    energy5 = energy5/sizeOfSample**2*3/2
    energy6 = energy6/sizeOfSample**2*3/2
    energy7 = energy7/sizeOfSample**2*3/2
    energy8 = energy8/sizeOfSample**2*3/2
    energy9 = energy9/sizeOfSample**2*3/2
    energy10 = energy10/sizeOfSample**2*3/2

    orderParm1 = np.mean(np.array(order_parm1_array))
    orderParm2 = np.mean(np.array(order_parm2_array))
    orderParm3 = np.mean(np.array(order_parm3_array))
    orderParm4 = np.mean(np.array(order_parm4_array))
    orderParm5 = np.mean(np.array(order_parm5_array))
    orderParm6 = np.mean(np.array(order_parm6_array))
    orderParm7 = np.mean(np.array(order_parm7_array))
    orderParm8 = np.mean(np.array(order_parm8_array))
    orderParm9 = np.mean(np.array(order_parm9_array))
    orderParm10 = np.mean(np.array(order_parm10_array))
    

    real1 = orderParm1.real
    real2 = orderParm2.real
    real3 = orderParm3.real
    real4 = orderParm4.real
    real5 = orderParm5.real
    real6 = orderParm6.real
    real7 = orderParm7.real
    real8 = orderParm8.real
    real9 = orderParm9.real
    real10 = orderParm10.real

    imag1 = orderParm1.imag
    imag2 = orderParm2.imag
    imag3 = orderParm3.imag
    imag4 = orderParm4.imag
    imag5 = orderParm5.imag
    imag6 = orderParm6.imag
    imag7 = orderParm7.imag
    imag8 = orderParm8.imag
    imag9 = orderParm9.imag
    imag10 = orderParm10.imag

    abs1 = abs(orderParm1)
    abs2 = abs(orderParm2)
    abs3 = abs(orderParm3)
    abs4 = abs(orderParm4)
    abs5 = abs(orderParm5)
    abs6 = abs(orderParm6)
    abs7 = abs(orderParm7)
    abs8 = abs(orderParm8)
    abs9 = abs(orderParm9)
    abs10 = abs(orderParm10)


    OldOrderParm1 = np.mean(np.array(OldOrder_parm1_array))
    OldOrderParm2 = np.mean(np.array(OldOrder_parm2_array))
    OldOrderParm3 = np.mean(np.array(OldOrder_parm3_array))
    OldOrderParm4 = np.mean(np.array(OldOrder_parm4_array))
    OldOrderParm5 = np.mean(np.array(OldOrder_parm5_array))
    OldOrderParm6 = np.mean(np.array(OldOrder_parm6_array))
    OldOrderParm7 = np.mean(np.array(OldOrder_parm7_array))
    OldOrderParm8 = np.mean(np.array(OldOrder_parm8_array))
    OldOrderParm9 = np.mean(np.array(OldOrder_parm9_array))
    OldOrderParm10 = np.mean(np.array(OldOrder_parm10_array))

    Oldreal1 = OldOrderParm1.real
    Oldreal2 = OldOrderParm2.real
    Oldreal3 = OldOrderParm3.real
    Oldreal4 = OldOrderParm4.real
    Oldreal5 = OldOrderParm5.real
    Oldreal6 = OldOrderParm6.real
    Oldreal7 = OldOrderParm7.real
    Oldreal8 = OldOrderParm8.real
    Oldreal9 = OldOrderParm9.real
    Oldreal10 = OldOrderParm10.real

    Oldimag1 = OldOrderParm1.imag
    Oldimag2 = OldOrderParm2.imag
    Oldimag3 = OldOrderParm3.imag
    Oldimag4 = OldOrderParm4.imag
    Oldimag5 = OldOrderParm5.imag
    Oldimag6 = OldOrderParm6.imag
    Oldimag7 = OldOrderParm7.imag
    Oldimag8 = OldOrderParm8.imag
    Oldimag9 = OldOrderParm9.imag
    Oldimag10 = OldOrderParm10.imag

    Oldabs1 = abs(OldOrderParm1)
    Oldabs2 = abs(OldOrderParm2)
    Oldabs3 = abs(OldOrderParm3)
    Oldabs4 = abs(OldOrderParm4)
    Oldabs5 = abs(OldOrderParm5)
    Oldabs6 = abs(OldOrderParm6)
    Oldabs7 = abs(OldOrderParm7)
    Oldabs8 = abs(OldOrderParm8)
    Oldabs9 = abs(OldOrderParm9)
    Oldabs10 = abs(OldOrderParm10)
    
    order_square_mean1 = np.mean(np.array(order_square1_array))
    order_square_mean2 = np.mean(np.array(order_square2_array))
    order_square_mean3 = np.mean(np.array(order_square3_array))
    order_square_mean4 = np.mean(np.array(order_square4_array))
    order_square_mean5 = np.mean(np.array(order_square5_array))
    order_square_mean6 = np.mean(np.array(order_square6_array))
    order_square_mean7 = np.mean(np.array(order_square7_array))
    order_square_mean8 = np.mean(np.array(order_square8_array))
    order_square_mean9 = np.mean(np.array(order_square9_array))
    order_square_mean10 = np.mean(np.array(order_square10_array))
    
    order_fourthpower_mean1 = np.mean(np.array(order_fourthpower1_array))
    order_fourthpower_mean2 = np.mean(np.array(order_fourthpower2_array))
    order_fourthpower_mean3 = np.mean(np.array(order_fourthpower3_array))
    order_fourthpower_mean4 = np.mean(np.array(order_fourthpower4_array))
    order_fourthpower_mean5 = np.mean(np.array(order_fourthpower5_array))
    order_fourthpower_mean6 = np.mean(np.array(order_fourthpower6_array))
    order_fourthpower_mean7 = np.mean(np.array(order_fourthpower7_array))
    order_fourthpower_mean8 = np.mean(np.array(order_fourthpower8_array))
    order_fourthpower_mean9 = np.mean(np.array(order_fourthpower9_array))
    order_fourthpower_mean10 = np.mean(np.array(order_fourthpower10_array))
    
    binder_ratio1 = 1 - order_fourthpower_mean1/(3*order_square_mean1**2)
    binder_ratio2 = 1 - order_fourthpower_mean2/(3*order_square_mean2**2)
    binder_ratio3 = 1 - order_fourthpower_mean3/(3*order_square_mean3**2)
    binder_ratio4 = 1 - order_fourthpower_mean4/(3*order_square_mean4**2)
    binder_ratio5 = 1 - order_fourthpower_mean5/(3*order_square_mean5**2)
    binder_ratio6 = 1 - order_fourthpower_mean6/(3*order_square_mean6**2)
    binder_ratio7 = 1 - order_fourthpower_mean7/(3*order_square_mean7**2)
    binder_ratio8 = 1 - order_fourthpower_mean8/(3*order_square_mean8**2)
    binder_ratio9 = 1 - order_fourthpower_mean9/(3*order_square_mean9**2)
    binder_ratio10 = 1 - order_fourthpower_mean10/(3*order_square_mean10**2)
    print('\nparameters:(energy,capacity,temperature,loop_timesteps)\n',[energy1,energy2,energy3,energy4,energy5,capacity1,capacity2,capacity3,capacity4,capacity5,temperature,loop_timestep])
    print()
    print()
    
    accept_ratio = refresh_accept/(refresh_accept + refresh_reject)
    avg_length = np.mean(np.array(length_stat))
    
    save_lattice(S, sizeOfSample, "lattice\\lattice_" + str(sizeOfSample) + '_' + str(temperature) + 'K') 
    return [temperature,energy1,energy2,energy3,energy4,energy5,energy6,energy7,energy8,energy9,energy10,capacity1,capacity2,capacity3,capacity4,capacity5,capacity6,capacity7,capacity8,capacity9,capacity10,
            real1,imag1,abs1,real2,imag2,abs2,real3,imag3,abs3,real4,imag4,abs4,real5,imag5,abs5,real6,imag6,abs6,real7,imag7,abs7,real8,imag8,abs8,real9,imag9,abs9,real10,imag10,abs10,
            Oldreal1,Oldimag1,Oldabs1,Oldreal2,Oldimag2,Oldabs2,Oldreal3,Oldimag3,Oldabs3,Oldreal4,Oldimag4,Oldabs4,Oldreal5,Oldimag5,Oldabs5,Oldreal6,Oldimag6,Oldabs6,Oldreal7,Oldimag7,Oldabs7,Oldreal8,Oldimag8,Oldabs8,Oldreal9,Oldimag9,Oldabs9,Oldreal10,Oldimag10,Oldabs10,binder_ratio1,binder_ratio2,binder_ratio3,binder_ratio4,binder_ratio5,
            gsop,accept_ratio,avg_length,
    sizeOfSample,shuffle_timestep,mcstp,loop_timestep,mid_term,J,k_B]

def plot(lattice,sizeOfSample):
    a,b = lattice.shape
    plt.axis('equal')
    for y in range(a):
        for x in range(b):
            node = lattice[y][x]
            plt.scatter(node.x, node.y, color = 'b', s = 2)
            if node.location[1] == 0 : #left boundary
                if node.linkedto.location[1] == 0: # linked node in the same column
                    if sorted([node.linkedto.location[0],node.location[0]]) == [0,sizeOfSample -1] : #left down and left up
                        if node.location[0] == 0:
                            plt.plot([node.x, node.x], [node.y,node.y-1/2] ,color = 'g', linewidth = 0.5)
                            plt.plot([node.linkedto.x, node.linkedto.x], [node.linkedto.y,node.linkedto.y+1/2] ,color = 'g', linewidth = 0.5)
                        else:
                            plt.plot([node.x, node.x], [node.y,node.y+1/2] ,color = 'g', linewidth = 0.5)
                            plt.plot([node.linkedto.x, node.linkedto.x], [node.linkedto.y,node.linkedto.y-1/2] ,color = 'g', linewidth = 0.5)
                    else:
                        plt.plot([node.x, node.linkedto.x], [node.y,node.linkedto.y] ,color = 'r', linewidth = 0.5)
                elif node.linkedto.location[1] == sizeOfSample - 1: # periodic boundary condition
                    if node.direction:
                        plt.plot([node.x, node.x-np.sqrt(3)/4], [node.y,node.y-1/4] ,color = 'g', linewidth = 0.5)
                        plt.plot([node.linkedto.x, node.linkedto.x+np.sqrt(3)/4], [node.linkedto.y,node.linkedto.y+1/4] ,color = 'g', linewidth = 0.5)
                    else:
                        plt.plot([node.x, node.x-np.sqrt(3)/4], [node.y,node.y+1/4] ,color = 'g', linewidth = 0.5)
                        plt.plot([node.linkedto.x, node.linkedto.x+np.sqrt(3)/4], [node.linkedto.y,node.linkedto.y-1/4] ,color = 'g', linewidth = 0.5)
                else:
                    pass
            elif node.location[1] == sizeOfSample - 1:   #right boundary
                if node.linkedto.location[1] == sizeOfSample - 1: #same column
                    plt.plot([node.x, node.linkedto.x], [node.y,node.linkedto.y] ,color = 'r', linewidth = 0.5)
                else:
                    pass 
                        
            elif node.location[0] == 0: #bottom boundary
                if node.linkedto.location[0] == 0: #same row
                    plt.plot([node.x, node.linkedto.x], [node.y,node.linkedto.y] ,color = 'r', linewidth = 0.5)
                elif node.linkedto.location[0] == sizeOfSample - 1: #cross
                    plt.plot([node.x, node.x], [node.y,node.y-1/2] ,color = 'g', linewidth = 0.5)
                    plt.plot([node.linkedto.x, node.linkedto.x], [node.linkedto.y,node.linkedto.y+1/2] ,color = 'g', linewidth = 0.5)
                else:
                    pass
            
            elif node.location[0] == sizeOfSample - 1: #top boundary
                if node.linkedto.location[0] == sizeOfSample - 1: #same row
                    plt.plot([node.x, node.linkedto.x], [node.y,node.linkedto.y] ,color = 'r', linewidth = 0.5)
                else:
                    pass
            
            else:              
                plt.plot([node.x, node.linkedto.x], [node.y,node.linkedto.y] ,color = 'r', linewidth = 0.5)

    for y in range(sizeOfSample*2//3):
        for x in range(sizeOfSample//2):
            if (starType(lattice,lattice[y][x*2+y%2],sizeOfSample) == 1):
                plt.plot((x*2+y%2)*np.sqrt(3)/2,y*1.5+1.5 if(lattice[y][x*2+y%2].direction) else y*1.5 + 1.0,
                marker = '>',markersize = 8.0,mec = [0.7,0,0.8],mfc = [1,1,0])
                
            elif (starType(lattice,lattice[y][x*2+y%2],sizeOfSample) == -1):
                plt.plot((x*2+y%2)*np.sqrt(3)/2,y*1.5+1.5 if(lattice[y][x*2+y%2].direction) else y*1.5 + 1.0,
                marker = '<',markersize = 8.0,mec = [0.7,0,0.8],mfc = [1,1,0])
            
            elif (starType(lattice,lattice[y][x*2+y%2],sizeOfSample) == 0):
                plt.plot((x*2+y%2)*np.sqrt(3)/2,y*1.5+1.5 if(lattice[y][x*2+y%2].direction) else y*1.5 + 1.0,
                marker = 'o',markersize = 8.0,mec = [0.7,0,0.8],mfc = [1,1,0])

    plt.show()
 

def process(c):
    lattice = initialGraph(c[6])
    ret = run(lattice,c[6],c[0],c[1],c[2],c[3],c[4],c[5],c[7])
    return ret
    
def write_data(parameters):
    try:
        f = open("results.csv","r")
        created = True
    except FileNotFoundError:
        created = False
            
    now = datetime.datetime.now()
    with open("results.csv","a+") as f:
        f.write("temperature,energy1,energy2,energy3,energy4,energy5,energy6,energy7,energy8,energy9,energy10,\
                capacity1,capacity2,capacity3,capacity4,capacity5,capacity6,capacity7,capacity8,capacity9,capacity10,\
                real1,imag1,abs1,real2,imag2,abs2,real3,imag3,abs3,real4,imag4,abs4,real5,imag5,abs5,real6,imag6,abs6,real7,imag7,abs7,real8,imag8,abs8,real9,imag9,abs9,real10,imag10,abs10,\
                Oldreal1,Oldimag1,Oldabs1,Oldreal2,Oldimag2,Oldabs2,Oldreal3,Oldimag3,Oldabs3,Oldreal4,Oldimag4,Oldabs4,Oldreal5,Oldimag5,Oldabs5,Oldreal6,Oldimag6,Oldabs6,Oldreal7,Oldimag7,Oldabs7,Oldreal8,Oldimag8,Oldabs8,Oldreal9,Oldimag9,Oldabs9,Oldreal10,Oldimag10,Oldabs10,\
                binder_ratio1,binder_ratio2,binder_ratio3,binder_ratio4,binder_ratio5,binder_ratio6,binder_ratio7,binder_ratio8,binder_ratio9,binder_ratio10,\
                gsop,accept_ratio,avg_length,sizeOfSample,shuffle_timestep,mcstp,loop_timestep,mid_term,J,k_B\n")
        f.write(str(parameters)[1:-1]+','+now.strftime("%m-%d_%H_%M") + '\n')
        f.close()

def save_lattice(lattice, sizeOfSample, filename):
    try:
        filepointer = open(str(filename) + ".dat", "wb")
        pickle.dump(lattice,filepointer)
        return
    except Exception as inst:
        print("An error occurred when saving data")
        print(inst.args)
        return 
        
def load_lattice(sizeOfSample, filename):
    try:
        filepointer = open(str(filename) + ".dat", "rb")
        lat = pickle.load(filepointer)
    except Exception as inst:
        print("An error occurred when loading data")
        print(inst.args)
        return None
    return lat

def show_path(path):
    for node in path:
        node.plot()
if __name__ == "__main__":
    mcstp = 12*12
    sizeOfSample = 12
    shuffle_timestep = 100
    loop_timestep = 2000
    mid_term = 1
    temperature = [0.96]
    k_B = 1 
    E_per_star = -1

    for i in temperature:
        configuration = (shuffle_timestep,loop_timestep,mid_term,i,E_per_star,k_B,sizeOfSample,mcstp)
       
        parm = process(configuration)
        write_data(parm)

def manual_flip(lattice,sizeOfSample):
    plot(lattice,sizeOfSample)
    
    vertices = eval(input("input coordinate(y1,x1,y2,x2,...),(y1,x1).linkedto == (y2,x2) \n"))
    path = []
    y = vertices[0::2]
    x = vertices[1::2]
    for i in range(len(x)):
        node = lattice[y[i]][x[i]]
        node.plot()
#    
#        
#        node.plot()
#        path.append(node)
        
 #   updateLoop(lattice,path)
    plot(lattice,sizeOfSample)
