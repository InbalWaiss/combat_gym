
import os
import pickle
import numpy as np




PREPROCESSING_PATH = os.path.dirname(os.path.realpath(__file__))
#path = 'C:/Users/eranice/Documents/FPS_Inbal/combat_gym/gym_combat/gym_combat/envs/Common/Preprocessing'

def list_dict_2_tuple_dict(in_dict):
    out_dict = {}
    for k,v in in_dict.items():
        if v:
            p = np.array(v)
            out_dict[k] = (p[:,0],p[:,1])
        else:
            out_dict[k] = ([],[])
    return out_dict

for filename in os.listdir(PREPROCESSING_PATH):
    if "position_los" in filename and "Baqa" in filename:
        outfile = filename[:-4] + "_tuple.pkl"
        print (outfile)
        with open(os.path.join(PREPROCESSING_PATH, filename), 'rb') as f:
            in_dict = pickle.load(f)
        out_dict = list_dict_2_tuple_dict(in_dict)
        with open(os.path.join(PREPROCESSING_PATH, outfile), 'wb') as f_out:
            pickle.dump(out_dict,f_out)