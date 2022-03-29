def ReadVarsFile(path2base):    
    import json
    import numpy as np
    with open(path2base + 'Beh_class/vars_file.json') as fi:
        vars_session=json.load(fi)

    vars, num_time_steps, var_names = list(), None, list()
    for name, data in vars_session.items():
        data = np.array(data) 
        if len(data.shape)>1:
            continue
        vars.append(data)
        var_names.append(name)

    vars = np.stack(vars, 1)
    return vars,var_names

def MyLabelEncoder(string_labels,dict2transfer=None):
    if dict2transfer is None:
        dict2transfer={}
        dict2transfer['Unknown']=int(-1)
        dict2transfer['Locomotion']=int(0)
        dict2transfer['Cleaning']=int(1)
        dict2transfer['Rearing']=int(2)
        dict2transfer['Sitting']=int(3)
        dict2transfer['Exploration']=int(4)  
        dict2transfer['Sniffing']=int(5)
    encoded_labels = [dict2transfer[label] for label in string_labels]
    return encoded_labels

def GetDict4GAMencoding():
    dict2transfer={}
    dict2transfer['Unknown']=int(-1)
    dict2transfer['Locomotion']=int(0)
    dict2transfer['Rearing']=int(1)
    dict2transfer['Exploration']=int(2)
    dict2transfer['Sitting']=int(3)
    dict2transfer['Cleaning']=int(4)
    return dict2transfer

def ReadMuviAnno(path2base,lenof_sess,framesper_slice = 30,verbose = 0):  
    import json,glob
    import numpy as np
    try:
        path2anno = glob.glob(path2base+'Beh_class/my_labels2*.json')[0]
    except:
        print('No Muvi anno file found')
        return None
    with open(path2anno,'r') as fi:
        anno=json.load(fi)
    time_keys=[int(an['video'].split('_')[-1].split('.')[0]) for an in anno]
    vid_labels=[an['label'] for an in anno]    
    vid_labels =MyLabelEncoder(vid_labels)
    labels  = np.ones((lenof_sess,))*-1 #initialize with unclear labels
    for t, la in zip(time_keys,vid_labels):        
        labels[t*framesper_slice:t*framesper_slice+framesper_slice] = la
    labels = labels[:lenof_sess] #cut overhang
    labels = [int(lab) for lab in labels]
    return labels
