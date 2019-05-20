from multiprocessing import Pool
import random
import matplotlib.pyplot as plt
import numpy as np
import datetime
import pickle


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
        plt.plot(self.x, self.y,marker = 'o',markersize = 3.0,mec = [0,1,0],mfc = [0,1,0])    

def initialGraph(sizeOfSample):
    S = np.array([[Node([i,j]) for j in range(sizeOfSample)]for i in range(sizeOfSample)])    
    for i in range(sizeOfSample):
        for j in range(sizeOfSample):
            node = S[i][j]
            location = node.location
            direction = node.direction
            if (direction):
                S[i][j].linkedto = S[(location[0]+1)%sizeOfSample][location[1]]
            else:
                S[i][j].linkedto = S[(location[0]-1)%sizeOfSample][location[1]]
    seq = [complex(-0.5,np.sqrt(3)/2),1,complex(-0.5,-np.sqrt(3)/2)]
    global plaquetteType
    plaquetteType = np.array([[seq[j%3-(i%2)] for j in range(sizeOfSample//2)]for i in range(sizeOfSample)])
    for i in range(sizeOfSample):
        for j in range(sizeOfSample//2):
            k = j%3-(i%2)
            if(k==1):
                S[i][j*2+i%2].linkedto = S[i][(j*2+i%2-1)%sizeOfSample]
                S[i][(j*2+i%2+1)%sizeOfSample].linkedto = S[(i+1)%sizeOfSample][(j*2+i%2+1)%sizeOfSample]
            elif(k==0):
                S[i][(j*2+i%2+1)%sizeOfSample].linkedto = S[i][(j*2+i%2+2)%sizeOfSample]
                S[i][(j*2+i%2)%sizeOfSample].linkedto = S[(i-1)%sizeOfSample][(j*2+i%2)%sizeOfSample]
            elif(k==2 or k==-1):
                S[i][j*2+i%2].linkedto = S[i][(j*2+i%2+1)%sizeOfSample]
                S[i][(j*2+i%2+1)%sizeOfSample].linkedto = S[i][(j*2+i%2)%sizeOfSample]
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
                nextnode = S[(now_index[0]+updown)%sizeOfSample][now_index[1]]
            else:
                nextnode = S[now_index[0]][(now_index[1]+choice)%sizeOfSample]
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
    node_list = [node,
                 S[location1_x][(location1_y+1)%sizeOfSample],
                 S[(location1_x+1)%sizeOfSample][(location1_y+1)%sizeOfSample],
                 S[(location1_x+1)%sizeOfSample][location1_y],
                 S[(location1_x+1)%sizeOfSample][(location1_y-1)%sizeOfSample],
                 S[location1_x][(location1_y-1)%sizeOfSample]
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
    
    for i in range(sizeOfSample):
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

def orderparameter(S,sizeOfSample):
    return sum(sum(np.multiply(plaquetteType,finalType(S,sizeOfSample))))*6/(sizeOfSample)**2

def finalType(S,sizeOfSample):
    final = [[0 for j in range(sizeOfSample//2)]for i in range(sizeOfSample)]
    for i in range(sizeOfSample):
        for j in range(sizeOfSample//2):
            if(isStar(S,S[i][j*2+i%2],sizeOfSample)==3):
                if(S[i][j*2+i%2].linkedto.location==[i,(j*2+i%2-1)%sizeOfSample]):
                    final[i][j]=1
    final = np.array(final)
    return final

def isStar(S,node,sizeOfSample): 
    location1 = node.location
    location1_x = location1[0]
    location1_y = location1[1]
    node_list = [node,
                 S[location1_x][(location1_y+1)%sizeOfSample],
                 S[(location1_x+1)%sizeOfSample][(location1_y+1)%sizeOfSample],
                 S[(location1_x+1)%sizeOfSample][location1_y],
                 S[(location1_x+1)%sizeOfSample][(location1_y-1)%sizeOfSample],
                 S[location1_x][(location1_y-1)%sizeOfSample]
                 ]
    count = 0
    for item in node_list:
        if(item.linkedto in node_list):
            count += 1
    return count/2

def totalNumOfStar(S,sizeOfSample):
    countOfStar = 0
    for y in range(sizeOfSample):
        for x in range(sizeOfSample//2):
            if(isStar(S,S[y][x*2+y%2],sizeOfSample)==3):
                countOfStar += 1
    return countOfStar

def adjacent_plat(S, node1, node2, sizeOfSample):
    bottom_node_list = []
    x1,y1,x2,y2 = node1.location[1], node1.location[0],node2.location[1], node2.location[0]
    if x1 == x2:
        if y1 == 0 and y2 == sizeOfSample - 1:
            bottom_node_list.append(S[y2][(x2+1)%sizeOfSample])
            bottom_node_list.append(S[y2][(x2-1)%sizeOfSample])
        elif y2 == 0 and y1 == sizeOfSample - 1:
            bottom_node_list.append(S[y1][(x1+1)%sizeOfSample])
            bottom_node_list.append(S[y1][(x1-1)%sizeOfSample])
        elif y1 < y2:
            bottom_node_list.append(S[y1][(x1+1)%sizeOfSample])
            bottom_node_list.append(S[y1][(x1-1)%sizeOfSample])
        else:
            bottom_node_list.append(S[y2][(x2+1)%sizeOfSample])
            bottom_node_list.append(S[y2][(x2-1)%sizeOfSample])
    else:
        if (x1 + y1)%2 == 0:
            bottom_node_list.append(S[y1][x1])
            bottom_node_list.append(S[(y2-1)%sizeOfSample][x2])
        elif (x2 + y2)%2 == 0:
            bottom_node_list.append(S[y2][x2])
            bottom_node_list.append(S[(y1-1)%sizeOfSample][x1])
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
    
    capacity1_array = []
    capacity2_array = []
    capacity3_array = []
    capacity4_array = []
    capacity5_array = []
    
    order_parm1_array = []
    order_parm2_array = []
    order_parm3_array = []
    order_parm4_array = []
    order_parm5_array = []
    
    order_square1_array = []
    order_square2_array = []
    order_square3_array = []
    order_square4_array = []
    order_square5_array = []    
   
    order_fourthpower1_array = []
    order_fourthpower2_array = []
    order_fourthpower3_array = []
    order_fourthpower4_array = []
    order_fourthpower5_array = []    
   
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
           
    gsop = 0
    percent = 0
    
    S = initialGraph(sizeOfSample)
    print(orderparameter(S,sizeOfSample))
  #  S = load_lattice(sizeOfSample, "ground_state_36")
    gsop = orderparameter(S,sizeOfSample)
    count = 0
    percent = 0
    while True:   
        accept_list = [] #if you do not want to print step information delete marked line 281,282,292,296,305,306,307
        sample_list = []
        for i in range(mcstp):
            rand_i = random.randrange(sizeOfSample)
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
        ordr = orderparameter(S,sizeOfSample)
        ordra = abs(ordr)
        if(count<=loop_timestep/5):
            energy1_array.append(E)
            capacity1_array.append(E**2)
            order_parm1_array.append(ordr)
            order_square1_array.append(ordra**2)
            order_fourthpower1_array.append(ordra**4)
        elif(count<=loop_timestep/5*2):
            energy2_array.append(E)
            capacity2_array.append(E**2)
            order_parm2_array.append(ordr)
            order_square2_array.append(ordra**2)
            order_fourthpower2_array.append(ordra**4)
        elif(count<=loop_timestep/5*3):
            energy3_array.append(E)
            capacity3_array.append(E**2)
            order_parm3_array.append(ordr)
            order_square3_array.append(ordra**2)
            order_fourthpower3_array.append(ordra**4)
        elif(count<=loop_timestep/5*4):
            energy4_array.append(E)
            capacity4_array.append(E**2)
            order_parm4_array.append(ordr)
            order_square4_array.append(ordra**2)
            order_fourthpower4_array.append(ordra**4)
        elif(count<=loop_timestep):
            energy5_array.append(E)
            capacity5_array.append(E**2)
            order_parm5_array.append(ordr)
            order_square5_array.append(ordra**2)
            order_fourthpower5_array.append(ordra**4)
            
        if(count==loop_timestep):
            break
        
    energy1 = np.mean(np.array(energy1_array))
    energy2 = np.mean(np.array(energy2_array))
    energy3 = np.mean(np.array(energy3_array))
    energy4 = np.mean(np.array(energy4_array))
    energy5 = np.mean(np.array(energy5_array))
    
    capacity1 = np.mean(np.array(capacity1_array))
    capacity2 = np.mean(np.array(capacity2_array))
    capacity3 = np.mean(np.array(capacity3_array))
    capacity4 = np.mean(np.array(capacity4_array))
    capacity5 = np.mean(np.array(capacity5_array))
    
    capacity1 =(capacity1-energy1**2)/((k_B*temperature)**2)/sizeOfSample**2
    capacity2 =(capacity2-energy2**2)/((k_B*temperature)**2)/sizeOfSample**2
    capacity3 =(capacity3-energy3**2)/((k_B*temperature)**2)/sizeOfSample**2
    capacity4 =(capacity4-energy4**2)/((k_B*temperature)**2)/sizeOfSample**2
    capacity5 =(capacity5-energy5**2)/((k_B*temperature)**2)/sizeOfSample**2
    
    energy1 = energy1/sizeOfSample**2
    energy2 = energy2/sizeOfSample**2
    energy3 = energy3/sizeOfSample**2
    energy4 = energy4/sizeOfSample**2
    energy5 = energy5/sizeOfSample**2

    orderParm1 = np.mean(np.array(order_parm1_array))
    orderParm2 = np.mean(np.array(order_parm2_array))
    orderParm3 = np.mean(np.array(order_parm3_array))
    orderParm4 = np.mean(np.array(order_parm4_array))
    orderParm5 = np.mean(np.array(order_parm5_array))
    
    order_square_mean1 = np.mean(np.array(order_square1_array))
    order_square_mean2 = np.mean(np.array(order_square2_array))
    order_square_mean3 = np.mean(np.array(order_square3_array))
    order_square_mean4 = np.mean(np.array(order_square4_array))
    order_square_mean5 = np.mean(np.array(order_square5_array))
    
    order_fourthpower_mean1 = np.mean(np.array(order_fourthpower1_array))
    order_fourthpower_mean2 = np.mean(np.array(order_fourthpower2_array))
    order_fourthpower_mean3 = np.mean(np.array(order_fourthpower3_array))
    order_fourthpower_mean4 = np.mean(np.array(order_fourthpower4_array))
    order_fourthpower_mean5 = np.mean(np.array(order_fourthpower5_array))
    
    binder_ratio1 = 1 - order_fourthpower_mean1/(3*order_square_mean1**2)
    binder_ratio2 = 1 - order_fourthpower_mean2/(3*order_square_mean2**2)
    binder_ratio3 = 1 - order_fourthpower_mean3/(3*order_square_mean3**2)
    binder_ratio4 = 1 - order_fourthpower_mean4/(3*order_square_mean4**2)
    binder_ratio5 = 1 - order_fourthpower_mean5/(3*order_square_mean5**2)
    
    print('\nparameters:(energy,capacity,temperature,loop_timesteps)\n',[energy1,energy2,energy3,energy4,energy5,capacity1,capacity2,capacity3,capacity4,capacity5,temperature,loop_timestep])
    print()
    print()
    
    accept_ratio = refresh_accept/(refresh_accept + refresh_reject)
    avg_length = np.mean(np.array(length_stat))
    
    save_lattice(S, sizeOfSample, "lattice_" + str(sizeOfSample) + '_' + str(temperature) + 'K') 
    return [temperature,energy1,energy2,energy3,energy4,energy5,capacity1,capacity2,capacity3,capacity4,capacity5,
            orderParm1,orderParm2,orderParm3,orderParm4,orderParm5,binder_ratio1,binder_ratio2,binder_ratio3,binder_ratio4,binder_ratio5,
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

    for y in range(sizeOfSample):
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
        if not created:
            f.write("temperature,energy1,energy2,energy3,energy4,energy5,capacity1,capacity2,capacity3,capacity4,capacity5," +
                    "orderParm1,orderParm2,orderParm3,orderParm4,orderParm5,binderratio1,binderratio2,binderratio3,binderratio4,binderratio5,gsop,acpt_rat,avg_len," +
            "sizeOfSample,shuffle_timestep,mcstp,loop_timestep,mid_term,Energy_per_star,k_B,date\n")
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
    mcstp = 48*48
    sizeOfSample = 48
    shuffle_timestep = 5000
    loop_timestep = 5000
    mid_term = 1
    temperature = [0.2]
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
        path.append(node)
        
    updateLoop(lattice,path)
    plot(lattice,sizeOfSample)
