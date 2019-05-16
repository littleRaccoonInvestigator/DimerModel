from multiprocessing import Pool
import random
import matplotlib.pyplot as plt
import numpy as np
import datetime



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
        plt.plot(self.x, self.y,marker = 'o',markersize = 3.0,mec = [0.8,0,0.8],mfc = [0.8,0,0.8])    

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
    return S


def linkedlist(S): 
    a,b = S.shape
    X = [[S[i][j].x for i in range(b)]for j in range(a)]
    Y = [[S[i][j].y for i in range(b)]for j in range(a)]  
    link_x = [[S[i][j].linkedto.x for i in range(b)]for j in range(a)]
    link_y = [[S[i][j].linkedto.y for i in range(b)]for j in range(a)]
    return link_x,link_y

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
    
def orderParameter(S,sizeOfSample):
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
    for i in range(sizeOfSample):
        for j in range(sizeOfSample//2):
            if(isStar(S,S[i][j*2+i%2],sizeOfSample)==3):
                countOfStar += 1
    return countOfStar

def run(S,sizeOfSample,shuffle_timestep,loop_timestep,mid_term,temperature,J,k_B,mcstp):

    energy_array = []
    capacity_array = []
    energy2_array = []
    capacity2_array = []
    order_parm_array = []
    order_parm_array2 = []
   
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
    for k in range(loop_timestep):
        if(k*100/loop_timestep > percent):
            print("%d%%"%percent, end = ' ',flush = True)
            if percent%10 == 0:
                print()
            percent += 1
        
        for i in range(mcstp):
            rand_i = random.randrange(sizeOfSample)
            rand_j = random.randrange(sizeOfSample)
            path = findcycle(S[rand_i][rand_j],S,sizeOfSample)
            E_previous  = totalNumOfStar(S,sizeOfSample)*J
            S = updateLoop(S,path)
            E_after  = totalNumOfStar(S,sizeOfSample)*J
            E1 = E_after - E_previous
            
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
        
        ordr = orderParameter(S,sizeOfSample)
            
        energy_array.append(E)
        capacity_array.append(E**2)
        order_parm_array.append(ordr)
        if k >= mid_term:
            energy2_array.append(E)
            capacity2_array.append(E**2)
            order_parm_array2.append(ordr)
        
       
    energy = np.mean(np.array(energy_array))
    energy2 = np.mean(np.array(energy2_array))
    capacity = np.mean(np.array(capacity_array))
    capacity2 = np.mean(np.array(capacity2_array))
    
    capacity = (capacity-energy**2)/((k_B*temperature)**2)/sizeOfSample**2
    capacity2 =(capacity2-energy2**2)/((k_B*temperature)**2)/sizeOfSample**2
    energy = energy/sizeOfSample**2
    energy2 = energy2/sizeOfSample**2
    
    print('\nparameters:(energy,capacity,temperature,loop_timesteps)\n',[energy,capacity,temperature,loop_timestep])
    print([energy2,capacity2,temperature,loop_timestep])
    print()
    print()
    
    orderParm = np.mean(np.array(order_parm_array2))
    accept_ratio = refresh_accept/(refresh_accept + refresh_reject)
    avg_length = np.mean(np.array(length_stat))
        
    return [temperature,energy,capacity,energy2,capacity2,orderParm,accept_ratio,avg_length,
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
            if(isStar(lattice,lattice[y][x*2+y%2],sizeOfSample)==3):
                plt.plot((x*2+y%2)*np.sqrt(3)/2,y*1.5+1.5 if(lattice[y][x*2+y%2].direction) else y*1.5 + 1.0,
                marker = 'o',markersize = 8.0,mec = [0,1,0],mfc = [1,1,0])

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
            f.write("temperature,energy,capacity,energy2,capacity2,order_parameter,acpt_rat,avg_len," + 
            "sizeOfSample,shuffle_timestep,mcstp,loop_timestep,mid_term,Energy_per_star,k_B,date\n")
        f.write(str(parameters)[1:-1]+','+now.strftime("%m-%d_%H_%M") + '\n')
        f.close()

    
if __name__ == "__main__":
    mcstp = 100
    sizeOfSample = 12
    shuffle_timestep = 10
    loop_timestep = 100
    mid_term = 1
    temperature = [5.0]
    k_B = 1 
    E_per_star = -1
    
    configuration = (shuffle_timestep,loop_timestep,mid_term,temperature[0],E_per_star,k_B,sizeOfSample,mcstp)
   
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
    