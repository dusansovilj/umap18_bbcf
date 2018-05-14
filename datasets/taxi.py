'''
Taxi tracking data set.
https://www.kaggle.com/c/pkdd-15-predict-taxi-service-trajectory-i
'''

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

from datasets.utils import remove_selftransitions, is_consistent

from datasets.wrapper import Dataset
from datasets import DATA_DIR


# TAXI_DATA_RAW = './data/taxi/raw/train.csv'
TAXI_DATA_RAW = os.path.join(DATA_DIR, 'taxi', 'raw', 'train.csv') 


class TaxiData(Dataset):
    
    def __init__(self, num_samples, grid=[20, 20]):
        self._grid = grid
        self._num_samples = num_samples
        super(TaxiData, self).__init__('taxi')
        
        
    def create_data(self):
        loaded_data = load_and_clean_taxi_data(None, self._num_samples, self._grid)
        self._data = remove_selftransitions(loaded_data)
        
    def tostr(self):
        return  '{:s}_ns{:d}_grid{:d}-{:d}'.format(
            self.name(),
            self._num_samples,
            self._grid[0],
            self._grid[1])
        
    def make_adjacency_list(self):
        self._adjacency_list = grid_adjacency_list(self._grid)
    
    
    def get_num_states(self):
        return self._grid[0] * self._grid[1]
    
    
    def get_data(self):
        return self._data
    
    
    def get_labels(self):
        return [s[-1]  for s in self._data]
    
    
    def get_data_generator(self, num_folds, seed):
        kf = KFold(n_splits=num_folds, random_state=seed)
        kf.get_n_splits(range(len(self._data)))
        
        for train_idx, test_idx in kf.split(self._data):
            train = [self._data[idx]  for idx in train_idx]      
            test  = [self._data[idx]  for idx in test_idx]
            yield train, test
            


#######################################
#######     UTILITY FUNCTIONS    ######
#######################################

def load_and_clean_taxi_data(filename, num_of_data, grid=[20, 20]):
    '''
    Input: Filename about Taxi Data you want to load ***in the same directory*** and num_of_data you want to import
    Output: A list whose elements are taxi trajectories in a specified grid
    '''
    if not filename:
        filename = TAXI_DATA_RAW
    data = pd.read_csv(filename, nrows=num_of_data, header=None) # In total : 1710670 data point
    # Delete useless information other than trajectories
    res = data[8]
    del res[0]
    
    # Data cleaining: convert pair-wise trajectories to separate row and column representation
    trajectory = []
    temp = []
    for i in range(len(res)):
        #get the string version of each trajectory
        trajectory.append(res[i+1].split(',')) 
    
    for i in range(len(trajectory)):
            # delete redundent data
            if len(trajectory[i]) > 1 :
                for j in range(len(trajectory[i])):
                    #remove  '[' and ']'
                    temp1 = trajectory[i][j].replace(']','')
                    temp2 = temp1.replace('[','')
                    trajectory[i][j] = float(temp2)

    # Delete empty element
    unwanted = []
    for i in range(len(trajectory)):
        if trajectory[i] == ['[]']:
            unwanted.append(i)
    trajectory = [j for i,j in enumerate(trajectory) if i not in unwanted]
    
    # Separate longitude and latitude for graph
    longitude = [1]*(len(trajectory))
    latitude = [1]*(len(trajectory))
    for i in range(len(trajectory)):
        longitude[i] = trajectory[i][::2]
        latitude[i] = trajectory[i][1::2]

    ######################################################
    # Limit the range of the data
    bound_left = -8.692
    bound_right = -8.56
    bound_up = 41.200
    bound_down = 41.125
    
    # Mark the unwanted element/trajectories from the original list
    unwanted = []
    for i in range(len(trajectory)):
        for j in range(len(longitude[i])):
            ##if its not in the range
            if (longitude[i][j]>bound_right) or (longitude[i][j]<bound_left):
                unwanted.append(i)
                break
            elif (latitude[i][j]>bound_up) or (latitude[i][j]<bound_down):
                unwanted.append(i)
                break

    # Delete unwanted element from the list 
    longitude_bound = [j for i,j in enumerate(longitude) if i not in unwanted]
    latitude_bound = [j for i,j in enumerate(latitude) if i not in unwanted]

    # Refresh the trajecotory  
    final_res_new = [j for i,j in enumerate(trajectory) if i not in unwanted]

    ###################################################
    ##set grid
    num_grid_row = grid[0]
    num_grid_col = grid[1]

    len_grid_row = (bound_right - bound_left) / num_grid_row
    len_grid_col = (bound_up - bound_down) / num_grid_col
    
    row_index = []
    col_index = []
    row_index_temp1 = []
    col_index_temp1 = []
    
    for i in range(len(final_res_new)):
        for j in range(len(latitude_bound[i])):
            # Store every grid-trajectory into temp variable
            row_index_temp = int((longitude_bound[i][j] - bound_left)//len_grid_row)
            col_index_temp = int((latitude_bound[i][j] - bound_down)//len_grid_col)
            row_index_temp1.append(row_index_temp)
            col_index_temp1.append(col_index_temp)
            
        # Store every grid-trajectory into the nested list containing all trajectories
        row_index.append(row_index_temp1)
        col_index.append(col_index_temp1)
        row_index_temp1 = []
        col_index_temp1 = []
        
    # Convert from row and column index to state number between 0 - 399
    output_data = []
    adjacency_list = grid_adjacency_list(grid)
    
    len_rem = 0
    con_rem = 0
    
    for i in range(len(row_index)):
        temp = []
        for j in range(len(row_index[i])):
            state_index  = row_index[i][j] * num_grid_col + col_index[i][j]
            temp.append(state_index)
        if (2 < len(temp) < 100) and is_consistent(temp, adjacency_list): # Remove all trajectories with length > 100
#         if (2 < len(temp) < 100):
            output_data.append(temp)
            
        if len(temp) <= 2 or len(temp) >= 100:
            len_rem += 1
            
        if not is_consistent(temp, adjacency_list):
            con_rem += 1
            
#     print('removed {:d} points with invalid length'.format(len_rem))
#     print('removed {:d} points with inconsistency'.format(con_rem))
    return output_data



def grid_adjacency_list(grid=[20, 20]):
    num_states = grid[0] * grid[1]
    adj_list = []
    for state in range(num_states):
        state_adj_list = []
        for direction in range(9): 
            next_state = adjacent_point(state, direction, num_states, grid)
            if next_state != None:
                state_adj_list.append(next_state)
                
        adj_list.append(state_adj_list)
        
    return adj_list
    
    
def adjacent_point(start_state, direction, num_states, grid=[20, 20]):
    '''
    Associate the direction of the points in map with states-number. 
    '''
    num_of_grid = grid[1]
    
    start_point_row = start_state // num_of_grid
    start_point_col = start_state % num_of_grid
    if (direction == 0):
        next_state = start_state
    elif (direction == 1):
        if start_point_col == 0 or start_point_row == 0:
            next_state = None
        else:
            next_state = (start_point_row - 1) * num_of_grid + start_point_col - 1
        
    elif (direction == 2):
        if start_point_col == 0:
            next_state = None
        else:  
            next_state = start_point_row * num_of_grid + start_point_col - 1
            
    elif (direction == 3):
        if start_point_col == 0 or start_point_row == grid[0]-1:
            next_state = None
        else:
            next_state = (start_point_row + 1) * num_of_grid + start_point_col - 1
            
    elif (direction == 4):
        if start_point_row == 0:
            next_state = None
        else:
            next_state = (start_point_row - 1) * num_of_grid + start_point_col
            
    elif (direction == 5):
        if start_point_row == grid[0]-1:
            next_state = None
        else:
            next_state = (start_point_row + 1) * num_of_grid + start_point_col
            
    elif (direction == 6):
        if start_point_row == 0 or start_point_col == grid[1]-1:
            next_state = None
        else:
            next_state = (start_point_row - 1) * num_of_grid + start_point_col + 1
            
    elif (direction == 7):
        if start_point_col == grid[1]-1:
            next_state = None
        else:
            next_state = start_point_row * num_of_grid + (start_point_col + 1)
            
    elif (direction == 8):
        if start_point_row == grid[0]-1 or start_point_col == grid[1]-1:
            next_state = None
        else:
            next_state = (start_point_row + 1) * num_of_grid + start_point_col + 1
    
    if next_state < 0 or next_state >= num_states:
        next_state = None
    
    return next_state


def draw_trajectory(trajectory, grid=[20, 20]):
    '''This function is specifically used for LxK grid Taxi data to show the heat map of the weight P(T_g|C)'''
    plt.figure(figsize=(8, 8))
    plt.gca().set_xlim(-0.5, grid[1]-0.5)
    plt.gca().set_ylim(-0.5, grid[0]-0.5)
    
    plt.xticks(np.linspace(-0.5, grid[1] - 0.5, grid[1] + 1), [])
    plt.yticks(np.linspace(-0.5, grid[0] - 0.5, grid[0] + 1), [])
    plt.grid(True)
#     plt.gca().grid(color='k', linestyle=':', linewidth=0.5)
    
#     num_rows = grid[0]
    num_cols = grid[1]
    
    # Draw trajectory
    x=[]
    y=[]
    
    for i in range(len(trajectory)):
        x.append(trajectory[i] // num_cols )
        y.append(trajectory[i] % num_cols)
    plt.plot(x,y,label ='trajectory', c='b', marker='o', linewidth=6)

    # Draw Start and End point of the prior-trajectory
    plt.plot(trajectory[0] // num_cols, trajectory[0] % num_cols, marker='d', markersize=10, label = 'start_Point', c='g')
    plt.plot(trajectory[-1] // num_cols, trajectory[-1] % num_cols, marker='x', markersize=14,  label = 'last_point', c='r')
    
    plt.legend()
    plt.show()
    
    

def draw_heat_map(prior_trajectory, grid=[20, 20], weight_record=None, goal_state_label=None, next_state=None, prediction=None, savefig=None):
    '''
    This function is specifically used for 20 by 20 grid Taxi data to show the heat map of the weight P(T_g|C).
    Grid specification must be NxN !!
    '''
    plt.figure(figsize=(8,8))
    
    # Get heat map record 
#     _, heat_map_record = P_a_c(prior_trajectory,train_data,400,goal_state_label,0.1,T_res,initial_state_probability,frequency)
    # Draw trajectory
    num_grid = grid[1]
    x=[]
    y=[]
    for i in range(len(prior_trajectory)):
        x.append(prior_trajectory[i] // num_grid)
        y.append(prior_trajectory[i] % num_grid)
    plt.plot(x, y, label = 'trajectory', c='b', linewidth = 6)

    # Draw Heat Map (converting from the recorded heat map)
    heat_map = np.zeros((grid[1], grid[0]))
    if weight_record != None:
        for i, s in enumerate(goal_state_label):
            row = s % grid[1]
            col = s // grid[1]
            heat_map[row][col] = weight_record[i]

    if next_state != None:
        x = [prior_trajectory[-1] // num_grid, next_state // num_grid]
        y = [prior_trajectory[-1] % num_grid, next_state % num_grid]
        plt.plot(x, y, label = 'gtruth', c = 'w', linewidth = 6)
        
    if prediction != None:
        x = [prior_trajectory[-1] // num_grid, prediction // num_grid]
        y = [prior_trajectory[-1] % num_grid, prediction % num_grid]
        plt.plot(x, y, label = 'prediction', c = 'm', linewidth = 4)
        
    # start and end point of the prior-trajectory
    plt.scatter(prior_trajectory[0] // num_grid, prior_trajectory[0] % num_grid, s = 60,label = 'start', c='k', marker='o')
    plt.scatter(prior_trajectory[-1] // num_grid, prior_trajectory[-1] % num_grid, s = 60,label = 'end', c='k', marker='x')
    
    plt.imshow(heat_map, cmap='summer', interpolation='nearest')
    plt.colorbar()
    
    # Set other features
    axes = plt.gca()
    axes.set_xlim([0, grid[1] - 1])
    axes.set_ylim([0, grid[0] - 1])
    plt.xticks((), ())
    plt.yticks((), ())
    plt.legend()
    plt.title('Heat Map of goal-state weight and prior trajectory')
    if savefig != None:
        plt.savefig(savefig)
    else:
        plt.show()
    

