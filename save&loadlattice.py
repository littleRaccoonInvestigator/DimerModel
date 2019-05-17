
import pickle

def save_lattice(lattice, sizeOfSample, filename):
    try:
        f = open(str(filename) + ".dat", "wb")
        pickle.dump(lattice,f)
    except:
        print("An error occurred when saving data")
    finally:
        f.close() 
        
def load_lattice(sizeOfSample, filename):
    try:
        f = open(str(filename) + ".dat", "rb")
        lattice = pickle.load(f)
    except:
        print("An error occurred when loading data")
    finally:
        f.close()
    return lattice
