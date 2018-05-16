'''
'''

import os
import numpy as np

from sklearn.model_selection import StratifiedKFold
from datasets.wrapper import Dataset
from datasets import DATA_DIR


LOG_DIR = os.path.join(DATA_DIR, 'ui', 'raw')


class UIData(Dataset):
    
    NUM_ACTIONS = 14
    NUM_TASKS = 2
    
    def __init__(self, num_test):
        self._num_test = num_test
        super(UIData, self).__init__('ui')
        
        
    def tostr(self):
        return '{:s}_nt{:d}'.format(self.name(), 2 * self._num_test)
    
    def create_data(self):
        self._data = load_ui_data()
        
    def make_adjacency_list(self):
        res = dict()
        task_states = range(UIData.NUM_ACTIONS, UIData.NUM_ACTIONS + UIData.NUM_TASKS)
        all_actions = range(UIData.NUM_ACTIONS) + task_states
        for i in range(UIData.NUM_ACTIONS):
            res[i] = all_actions
        for i in task_states:
            res[i] = [i]
        self._adjacency_list = res
        
        
    def get_num_states(self):
        return UIData.NUM_ACTIONS + UIData.NUM_TASKS
    
    
    def get_data(self):
        return self._data
    
    
    def get_labels(self):
        return [s[-1] - UIData.NUM_ACTIONS  for s in self._data]
    
    
    def get_data_generator(self, num_folds, seed):
        # first permute the data
        train_data = []
        test_data = []
        labels = self.get_labels()
        rs = np.random.RandomState(seed)
        for task in range(UIData.NUM_TASKS):
            td = [s  for i, s in enumerate(self._data) if labels[i] == task]
            
            idx = rs.permutation(len(td))
            td = [td[i] for i in idx]
         
            train_data += td[:-self._num_test]
            test_data += td[-self._num_test:]
            
        data = train_data + test_data
        labels = [s[-1] - UIData.NUM_ACTIONS  for s in data]
        
        kf = StratifiedKFold(n_splits=num_folds, random_state=seed)
        kf.get_n_splits(data, labels)
        
        for train_idx, test_idx in kf.split(data, labels):
            train = [data[idx]  for idx in train_idx]      
            test  = [data[idx]  for idx in test_idx]
            yield train, test
            

def readdir(dirname=LOG_DIR):
    d = []
    p = next(os.walk(dirname))
    
    for ddir in p[1]:
        d += readdir(os.path.join(p[0], ddir))
    
    for fn in p[2]:
        d += readfile(os.path.join(p[0], fn))
    
    return d
    
    
    
def readfile(filename):
    wordlist=[]
    highmatrix=[]
    with open(filename, "r") as f:
        try:
            data = f.readlines()
        except UnicodeDecodeError:
            pass
        for line in data:
            word=line.split()
            if len(word)>3:
                if "RCL"in word[3]:
                    #state2
                    wordlist.append(word[3]+word[4]+word[5])
                if "Non-object" in word[3]:
                    #state1
                    wordlist.append(word[3]+word[4])
                if "Table" in word[3]:
                    #state2
                    wordlist.append(word[3]+word[4])
                if "mouse" in word[3]:
                    #state1
                    wordlist.append(word[3]+word[4])
                if "Column" in word[3]:
                    #state2
                    wordlist.append(word[3]+word[4]+word[5])
                if "Button" in word[3]:
                    #state4
                    wordlist.append(word[3]+word[4]+word[5])
                if "Minimap" in word[3]:
                    #state5
                    if "box" in word[6]:
                        wordlist.append(word[3]+word[4]+word[5]+word[6]+word[7])
                    else:
                        wordlist.append(word[3]+word[4]+word[5]+word[6])
        highmatrix.append(wordlist)
        wordlist=[]
    return highmatrix


def readdata(mylist, index): 
    '''
    Read all the data throughout the files
    '''
    
    def checkindex(local1, local2):
        '''check if two viarables are equal'''
        if local1 == local2:
            return 1
        else:
            return 0
    
    confidential=1
    dataleak=2
    embezzlement=3
    passwords=4
    data=5
    temp=''
    if checkindex(confidential,index)==1:
        temp='confidential'
    if checkindex(dataleak,index)==1:
        temp='dataleak'
    if checkindex(embezzlement,index)==1:
        temp='embezzlement'
    if checkindex(passwords,index)==1:
        temp='passwords'
    if checkindex(data,index)==1:
        temp='data'
        
    wordlist=[]
    highmatrix=[]
    logDir = mylist[0]
    logFiles = next(os.walk(logDir))[2]
   
    p = os.path.join(LOG_DIR, 'task1', temp)
   
    for i in range(len(logFiles)):
        with open(os.path.join(p, logFiles[i]), "r") as f:
            try:
                data = f.readlines()
            except UnicodeDecodeError:
                pass
            for line in data:
                word=line.split()
                if len(word)>3:
                    if "RCL"in word[3]:
                        #state2
                        wordlist.append(word[3]+word[4]+word[5])
                    if "Non-object" in word[3]:
                        #state1
                        wordlist.append(word[3]+word[4])
                    if "Table" in word[3]:
                        #state2
                        wordlist.append(word[3]+word[4])
                    if "mouse" in word[3]:
                        #state1
                        wordlist.append(word[3]+word[4])
                    if "Column" in word[3]:
                        #state2
                        wordlist.append(word[3]+word[4]+word[5])
                    if "Button" in word[3]:
                        #state4
                        wordlist.append(word[3]+word[4]+word[5])
                    if "Minimap" in word[3]:
                        #state5
                        if "box" in word[6]:
                            wordlist.append(word[3]+word[4]+word[5]+word[6]+word[7])
                        else:
                            wordlist.append(word[3]+word[4]+word[5]+word[6])
            highmatrix.append(wordlist)
            wordlist=[]
    return highmatrix



def transferindex(mylist):
    '''
    This function transfers all the text to states
    #highmatrix2 is a matrix containing all the actions in each sequence
    '''
    highmatrix2=[]
    temp=[]
    for i in range(len(mylist)):
        for j in range (len(mylist[i])):
            if "Non-object" in mylist[i][j] or "mouse" in mylist[i][j] or "Buttonpush" in mylist[i][j]:
                temp.append(1)
            if "RCL" in mylist[i][j] or "Table" in mylist[i][j] or "column" in mylist[i][j]:
                temp.append(2)
            if "Buttonclick" in mylist[i][j]:
                temp.append(4)
            if "Minimap" in mylist[i][j]:
                temp.append(5)
        highmatrix2.append(temp)
        temp=[]
    return highmatrix2



def transferaction(mylist): 
    '''
    This function transfers all the text to actions
    #highmatrix2 is a matrix containing all the actions in each sequence
    '''
    highmatrix2=[]
    temp=[]
    for i in range(len(mylist)):
        for j in range (len(mylist[i])):
            if "Non-object" in mylist[i][j]:
                temp.append(0)
            if  "mouse" in mylist[i][j]:
                temp.append(1)
 
            if "Buttonpushed" in mylist[i][j]:
                temp.append(2)  
            if "RCLScroll" in mylist[i][j]:
                temp.append(3)

            if "Table" in mylist[i][j]:
                temp.append(4) 

            if "Columnclicked" in mylist[i][j]:
                temp.append(5) 
            if "Buttonclick:Global" in mylist[i][j]:
                temp.append(6)
            if "Buttonclick:Normal" in mylist[i][j]:
                temp.append(7)
            if "Buttonclick:Malicious" in mylist[i][j]:
                temp.append(8)
            if "Buttonclick:Suspect" in mylist[i][j]:
                temp.append(9)
            if "Minimap" in mylist[i][j]:
                if "Minimapclickonbox" not in mylist[i][j]:
                    temp.append(10)
                else:
                    temp.append(11)
        highmatrix2.append(temp)
        temp=[]
    return highmatrix2


def get_adjacent_state_matrix_UI():
    return np.ones((14,14))



def add_goal_state_to_UI_interface(data, goal_state):
    for i in range(len(data)):
        data[i].append(goal_state)
    return data


def get_adjacency_list(num_actions=14, num_tasks=4):
    res = dict()
#     task_states = [100*v  for v in range(1, num_tasks + 1)]
    task_states = range(num_actions, num_actions + num_tasks)
    all_actions = range(num_actions) + task_states
    for i in range(num_actions):
        res[i] = all_actions
    for i in task_states:
        res[i] = [i]
    return res


def merge_sequence(mylist):
    '''
    Merge all the consecutive 'mouse XXX' and 'RCL' actions
    '''
    output=[]
    temp=[]
    last=-1
    for j in mylist:
        for i in j:
            if i==last and i==1: #mousexxx  actions
                last=i
                continue
            if i==last and i==3: #RCL actions
                last=i
                continue
            last=i
            temp.append(i)
        output.append(temp)
        temp=[]
    return output



def load_ui_data():
    tasks = [1, 2]
    data = []
    task_count = -1
    for task_id in tasks:
        td = readdir(os.path.join(LOG_DIR, 'task' + str(task_id)))
        td = transferaction(td)
        td = merge_sequence(td)
        task_count += 1
        td = [s + [UIData.NUM_ACTIONS + task_count]  for s in td]
        data += td
            
    return data

