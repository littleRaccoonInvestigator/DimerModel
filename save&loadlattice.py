
import pickle

def save_lattice(lattice, sizeOfSample, filename):
    try:
        filepointer = open(str(filename) + ".dat", "wb")
        pickle.dump(lattice,filepointer)
        filepointer.close()
    except Exception as inst:
        print("An error occurred when saving data")
        print(inst.args)
        return 
   
        
def load_lattice(sizeOfSample, filename):
    try:
        filepointer = open("ground_state_24.dat", "rb")
        lat = pickle.load(filepointer)
        filepointer.close()
    except Exception as inst:
        print("An error occurred when loading data")
        print(inst.args)
        return None
    return lat
