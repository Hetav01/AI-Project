
'''
Project - Elevator Optimazation

BY :
1. 19dcs001 - Shwet Khatri
2. 19ucs186 - Akshat Airan
3. 19ucs034 - Harsh Chauhan
4. 19ucs171 - Hetav Patel
5. 19ucs245 - Poojan Gadhiya

Using Reinforcement Learning (Maximizing Rewards in Order to Find Minimum Time)

'''

# Importing Required Libraries
import argparse
import numpy as np
import os
os.makedirs('./modelweights', exist_ok=True)

import torch
import torch.nn as nn
import torch.nn.functional as Func
import torch.optim as optim
from torch.distributions import Categorical
from tensorboardX import SummaryWriter

from utils import is_finish, state_preprocessing
from environment.Building import Building
from agent.agent import Agent

import time

#defining action dimensions
act_dim = 4 # up , down , load , unload

#setting device as cpu usage

device = 'cpu'

# Defining Main Function

def main():
    #Initiating All the global arguments with parser
    global args 
    parser = argparse.ArgumentParser('parameters')
    parser.add_argument('--test', type=bool, default=False, help="True if test, False if train (default: False)") 
    parser.add_argument('--epochs', type=int, default=40, help='number of epochs, (default: 1001)') 
    parser.add_argument('--lr_rate', type=float, default=0.0003, help='learning rate (default : 0.0003)') 
    parser.add_argument('--lift_num', type=int, default=1, help='number of elevators') 
    parser.add_argument('--T_horizon', type=int, default=2048, help='number of steps at once')
    parser.add_argument('--gamma', type=float, default=0.99, help='training gamma') 
    parser.add_argument('--K_epoch', type=int, default=10, help='number of train at once')
    parser.add_argument('--minibatch_size', type=int, default=64, help='batch size')
    parser.add_argument('--lmbda', type=float, default=0.95, help='training lmbda')
    parser.add_argument('--eps_clip', type=float, default=0.2, help='training eps_clip')
    parser.add_argument('--critic_coef', type=float, default=0.5, help='training ciritic_coef')
    parser.add_argument('--building_height', type=int, default=8, help='building height ')
    parser.add_argument('--max_people_in_floor', type=int, default=8, help='maximum people in one floor')
    parser.add_argument('--max_people_in_elevator', type=int, default=8, help='maximum people in one elevator')
    parser.add_argument('--add_people_prob', type=float, default=0.8, help='add people probability')
    parser.add_argument("--load_file", type=str, default = 'no', help = 'load initial parameters')
    parser.add_argument("--save_interval", type=int, default = 500, help = 'save interval')
    parser.add_argument("--print_interval", type=int, default = 20, help = 'print interval')
    args = parser.parse_args()

    print('args.test : ',args.test)
    print('args.epochs : ', args.epochs)
    print('args.lr_rate : ',args.lr_rate)
    print('args.lift_num : ', args.lift_num)
    print('args.building_height :',args.building_height)
    print('args.max_people_in_floor : ',args.max_people_in_floor )
    print('args.max_people_in_elevator :', args.max_people_in_elevator)
    print('args.load_file : ',args.load_file)
    print('args.save_interval :',args.save_interval)
    print('args.print_interval :',args.print_interval)

    #Lets Setup Building in the which elevator will run
    building = Building(args.lift_num, args.building_height, args.max_people_in_floor,\
                        args.max_people_in_elevator)
    #Initiating Agent
    agent = Agent(device,args.lift_num, args.building_height, args.max_people_in_floor,\
                        args.max_people_in_elevator,act_dim,args.K_epoch,\
                          args.gamma,args.lmbda,args.lr_rate,args.eps_clip,args.critic_coef,args.minibatch_size)



    #Interface representing a stateful summary writer object
    sumry = SummaryWriter()

    #Start position of the model empty building
    building.empty_building()

    #generating requests of passengers
    while building.remail_passengers_num == 0:
        building.generate_passengers(args.add_people)

    #setting floor and elevator states
    floor_state,elv_state,elv_place_state = building.get_state()
    floor_state,elv_state,elv_place_state = state_preprocessing(args,device,floor_state,elv_state,elv_place_state)

    done = False
    global_step = 0
    score_lst = []
    score = 0.0

    #running epoch to train further

    for epoch in range(args.epochs):
        for t in range(args.T_horizon):
            global_step += 1
            action_prob = agent.get_action(floor_state,elv_state,elv_place_state)[0]
            m = Categorical(action_prob)
            action = m.sample().tolist()
            reward = building.perform_action(action)
            next_floor_state,next_elv_state,next_elv_place_state = building.get_state()
            
            next_floor_state,next_elv_state,next_elv_place_state = state_preprocessing(args,device,next_floor_state,next_elv_state,next_elv_place_state)
            done = is_finish((next_floor_state,next_elv_state))
            # condition for reward increment 
            if done == True:
                reward = 100.
            else:
                reward = -10.

            #recording all the data to agent    
            agent.put_data((floor_state.cpu().tolist(),\
                            elv_state.cpu().tolist(),\
                            elv_place_state.cpu().tolist(),\
                            action, reward/100.0, \
                            next_floor_state.cpu().tolist(),\
                            next_elv_state.cpu().tolist(), \
                            next_elv_place_state.cpu().tolist(),\
                            [action_prob[idx][action[idx]] for idx in range(len(action_prob))],\
                            done))
            score += reward


            if args.test:
                os.system("cls")
                building.print_building(global_step)
                print(action)
                print('now reward : ',reward)

                time.sleep(1.5)
            if (global_step > 300):
                done = True
            if done : 
                score_lst.append(score)
                sumry.add_scalar('reward', score, epoch)
                score = 0
                global_step = 0
                building.empty_building()
                while building.remain_passengers_num == 0 :
                    building.generate_passengers(args.add_people_prob)
                floor_state,elv_state,elv_place_state = building.get_state()
                floor_state,elv_state,elv_place_state = state_preprocessing(args,device,floor_state,elv_state,elv_place_state)
            else:
                floor_state = next_floor_state
                elv_state = next_elv_state
                elv_place_state = next_elv_place_state

        # After getting all the data we will train the agent for further implementation

        agent.train(sumry,epoch)

         #printing data retrived
        if epoch%args.print_interval==0 and epoch!=0:
            print("# of episode :{}, avg score : {:.1f}".format(epoch, sum(score_lst)/len(score_lst)))
            score_lst = []

        if (epoch % args.save_interval == 0 )& (epoch != 0):
            torch.save(agent.state_dict(), './model_weights/model_'+str(epoch)) 
            

if __name__ == '__main__':
    main()    






