import wandb
from torch.utils.tensorboard import SummaryWriter
import torch
import time

import utils
import numpy
import math
import pygame
from enum import Enum 

wandb.init(
    project="cacla_rl",
    config={
        "learning_rate": 0.01,
    }
)


EVENT_EVALUATION_CRITIC = pygame.event.custom_type()
EVENT_EVALUATION_ACTOR_X = pygame.event.custom_type()
EVENT_EVALUATION_ACTOR_Y = pygame.event.custom_type()
EVENT_EVALUATION_ACTIONS = pygame.event.custom_type()

class States(Enum):
    NO_GUI = 0
    TRAIN = 1
    EVALUATE_ACTIONS = 2

STATE = States.NO_GUI

def set_state(state:States):
    global STATE
    STATE = state

class World:
    size_x = 10 # Interval from 0 to 10
    size_y = 10 # Interval from 0 to 10
    gaussian_feature_resolution = 2 # How many gaussian values over the size
    x_features = int(size_x*gaussian_feature_resolution)
    y_features = int(size_y*gaussian_feature_resolution)
    gaussian_var = 1.0
    goal = (5,5)
    goal_width= 1
    line_obstacles = []
    user_start = (0,0)

    def __init__(self):
        # init input position
        self.x = numpy.random.uniform(0, self.size_x - 1.0)
        self.y = numpy.random.uniform(0, self.size_y - 1.0)
        self.states = self.gaussian_feature()
        """This map holds values representative how good the states are known by the model normed to 1.
            With this information the start position of the actor in the world is not at random but at 
                positions with a low valuefunction or where a obstacle has just been placed. 
            """
        self.map_of_knowledge = numpy.ones((self.x_features,self.y_features))
        self.map_of_knowledge /= numpy.sum(self.map_of_knowledge)

    def new_init(self):

        if STATE == States.TRAIN:
            # with precentatge of 33% start at worst location
            if numpy.random.randint(0,3) == 1:
                index_min_flatten = self.map_of_knowledge.argmin() # find min value in flatten array
                index_min = numpy.unravel_index(index_min_flatten,self.map_of_knowledge.shape) # get index according to matrix coordinates
                index_min_x,index_min_y = (index_min[0]/self.gaussian_feature_resolution,index_min[1]/self.gaussian_feature_resolution) # map the index of the feature interval to the real coordinates of the world
                # Init coordinates normal distributed around those min values
                sigma_init = 1.0 
                self.x = numpy.random.normal(loc=index_min_x,scale=sigma_init)
                self.y = numpy.random.normal(loc=index_min_y,scale=sigma_init)
            # start randomly
            else:
                self.x = numpy.random.uniform(0, self.size_x-1.0)
                self.y = numpy.random.uniform(0, self.size_y-1.0)            
        else:
            self.x,self.y = self.user_start


        self.states = self.gaussian_feature()

    def set_map_of_knowledge(self,value_function_values):
        self.map_of_knowledge += numpy.reshape(value_function_values,(self.x_features,self.y_features))
        self.map_of_knowledge /= numpy.sum(self.map_of_knowledge)

    def gaussian_feature(self):
        """This function implements the feature of the state as being a visual input.
            Meaning it gives a probabelistic representation of the states 2 real values 
            for several intervals deviding the world size. """
        states = numpy.zeros(self.x_features*self.y_features)
        for a in range(0, self.x_features):
            for b in range(0, self.y_features):
                distance = ((a + 0.5) - self.x) ** 2 + \
                    ((b + 0.5) - self.y) ** 2
                states[a * self.size_y +
                       b] = math.exp(-distance / (2 * self.gaussian_var ** 2))
        states /= numpy.sum(states)
        return torch.tensor(states, dtype=torch.float32)

    def act(self, act):
        # negative reward for making an action
        reward = -1.0  # recommend 0 or -1
        found_goal = False
        new_x = self.x + act[0]
        new_y = self.y + act[1]

        # position boundary conditions
        collided_boundaries = False
        if new_x <= 0.0:
            new_x = 0.0
            collided_boundaries = True
        elif new_x >= self.size_x - 1.0:
            new_x = self.size_x - 1.0
            collided_boundaries = True
        if new_y <= 0.0:
            new_y = 0.0
            collided_boundaries = True
        elif new_y >= self.size_y - 1.0:
            new_y = self.size_y - 1.0
            collided_boundaries = True
        if collided_boundaries:
            reward -= 1

        # # obstacles crossing conditions
        # old_coordinate = (self.x,self.y)
        # new_coordinate = (new_x,new_y)
        # step_over_no_obstacle = True
        # for line in self.line_obstacles:
        #     # line coordinates
        #     line_coordiante_0 = line[0]
        #     line_coordiante_1 = line[1]
        #     # intersection exists
        #     intersects = utils.line_intersect(old_coordinate,new_coordinate,line_coordiante_0,line_coordiante_1)
        #     if intersects != None:
        #         # intersection is in range of given line
        #         step_line_direction = utils.is_in_rect(intersects,line_coordiante_0,line_coordiante_1)
        #         # intersection is in range of given step by actor
        #         step_over_line = utils.is_in_rect(intersects,old_coordinate,new_coordinate)
        #         step_over_intersection = step_over_line and step_line_direction
        #         step_over_no_obstacle = step_over_no_obstacle and not step_over_intersection
        # if step_over_no_obstacle:
        #     self.x = new_x
        #     self.y = new_y
        # else:
        #     reward -= 1
        
        # negative reward for length of action in relation to a square-function normed to half of the diagonal
        action_length = numpy.sqrt(int(act[0])**2 + int(act[1])**2)  # corrected. This was wrongly numpy.sqrt(int(self.x)**2 + int(self.y)**2
        reward -= numpy.sqrt(action_length/(numpy.sqrt(self.size_x**2+self.size_y**2)/2))
               
        # applying visual feature with gaussian
        self.states = self.gaussian_feature()

        # check goal boundaries if it was found
        x_goal,y_goal= self.goal
        goal_width = self.goal_width
        goal_height = self.goal_width
        if self.x >= x_goal - goal_width / 2 and self.x <= x_goal + goal_width / 2 and self.y >= y_goal - goal_height / 2 and self.y <= y_goal + goal_height / 2:
            reward += 100.0   # recommended 100 (if smaller, then downscale above "negative reward for length of action")
            found_goal = True

        return reward, self.states, found_goal

class ActorModel(torch.nn.Module):
    def __init__(self, lr,input_size):
        super(ActorModel, self).__init__()
        self.lr = lr
        self.h = 50
        self.linear1 = torch.nn.Linear(input_size, self.h)
        self.relu1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(self.h,self.h)
        self.relu2 = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(self.h,2)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.loss_fnc = torch.nn.MSELoss()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        return x

    def update_weights(self, prediction, target):
        # Update the weights of the function approximator. Backprogate to compute gradients.
        self.zero_grad()
        loss = self.loss_fnc(prediction, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class CriticModel(torch.nn.Module):
    def __init__(self, lr,input_size):
        super(CriticModel, self).__init__()
        self.lr = lr
        self.h = 50
        self.linear1 = torch.nn.Linear(input_size, 1)
        # self.relu1 = torch.nn.ReLU()
        # self.linear2 = torch.nn.Linear(self.h,self.h)
        # self.relu2 = torch.nn.ReLU()
        # self.linear3 = torch.nn.Linear(self.h,1)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        self.loss_fnc = torch.nn.MSELoss()

    def forward(self, x):
        x = self.linear1(x)
        # x = self.relu1(x)
        # x = self.linear2(x)
        # x = self.relu2(x)
        # x = self.linear3(x)
        return x

    def update_weights(self, prediction, target):
        # Update the weights of the function approximator. Backprogate to compute gradients.
        self.zero_grad()
        loss = self.loss_fnc(prediction, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

def evaluate_actor_critic(world_model,actor_model,critic_model):
    """Inputs states for all of x and y states according to gaussian feature resolution"""

    critic_model.eval()
    actor_model.eval()

    with torch.no_grad():
        critic_outputs = [
            [x for x in range(World.x_features)] for _ in range(World.y_features)]
        actor_outputs = [
            [x for x in range(World.x_features)] for _ in range(World.y_features)]
        for index_y,y in enumerate(numpy.arange(0,World.size_y,1/World.gaussian_feature_resolution)):
            world_model.y = y
            for index_x,x in enumerate(numpy.arange(0,World.size_x,1/World.gaussian_feature_resolution)):
                world_model.x = x
                states = world_model.gaussian_feature()
                critic_value = critic_model(states)
                critic_outputs[index_x][index_y] = critic_value.numpy()
                actor_value = actor_model(states)
                actor_outputs[index_x][index_y] = actor_value.numpy()
        
        world_model.set_map_of_knowledge(numpy.array(critic_outputs))
        
        actor_x_outputs : dict = utils.map_output_color(numpy.array(actor_outputs)[:, :, 0])
        actor_y_outputs : dict = utils.map_output_color(numpy.array(actor_outputs)[:, :, 1])
        critic_outputs : dict = utils.map_output_color(numpy.array(critic_outputs))
        # post outputs to gui event handler
        if pygame.get_init():
            pygame.event.post(pygame.event.Event(EVENT_EVALUATION_ACTOR_X,actor_x_outputs))
            pygame.event.post(pygame.event.Event(EVENT_EVALUATION_ACTOR_Y,actor_y_outputs))
            pygame.event.post(pygame.event.Event(EVENT_EVALUATION_CRITIC,critic_outputs))

    critic_model.train()
    actor_model.train()


class CACLA():
    sigma = 1.5
    gamma = 0.9   # recommend 0.9

    def __init__(self):
        self.world_model = World()
        features = int((World.size_x * World.size_y)*World.gaussian_feature_resolution**2)
        self.critic_model = CriticModel(0.01,features)  # recommend 0.01
        self.actor_model = ActorModel(0.01,features)    # recommend 0.01
    
    def train(self,episode_number):
        self.world_model.new_init()
        state = self.world_model.states
        reward = 0.0
        t = 0
        found_goal = False
        updated_actor = 0

        reward_per_episode = 0.0  # wandb logging
        while not found_goal and t < 500:
            approximated_action = self.actor_model(
                state)  # getting the movement angle from the population code state via the weights

            action = torch.distributions.Normal(approximated_action,
                                                    self.sigma).sample()  # Exploration  with Gaussian policy
            
            reward, state_new, found_goal = self.world_model.act(action)
            # TODO reward also visual feature
            reward_per_episode += reward  # wandb logging

            with torch.no_grad():
                value_new = self.critic_model(state_new)
            value = self.critic_model(state)

            prediction = value
            target = reward + self.gamma * value_new
            self.critic_model.update_weights(prediction, target)

            if target-prediction > 0.0:
                # update actor towards action
                updated_actor += 1
                self.actor_model.update_weights(
                    target=action, prediction=approximated_action)

            state = state_new
            t += 1

        print(episode_number, f"{updated_actor}", f"{self.world_model.x:.3}|{self.world_model.y:.3}",
            f"found goal after steps: {t}" if found_goal else "did not find goal")

        wandb.log({"steps_taken_training": t, "total_reward_training": reward_per_episode}) # wandb logging

        if STATE != States.NO_GUI:
           evaluate_actor_critic(self.world_model,self.actor_model,self.critic_model)

    def evaluate(self):
        self.world_model.new_init()
        state = self.world_model.states
        start = (self.world_model.x,self.world_model.y)
        total_reward = 0
        reward = 0
        t = 0
        found_goal = False
        self.actor_model.eval()
        self.critic_model.eval()
        points = [[self.world_model.x,self.world_model.y]]
        predicted_value = self.critic_model(state)
        while not found_goal and t < 50:
            action = self.actor_model(
                state)  # getting the movement angle from the population code state via the weights
            reward, state_new, found_goal = self.world_model.act(action)
            state = state_new
            t += 1
            total_reward += reward
            points.append([float(self.world_model.x),float(self.world_model.y)])

        points.append([float(self.world_model.x),float(self.world_model.y)])
        pygame.event.post(pygame.event.Event(EVENT_EVALUATION_ACTIONS,{"actions":points}))
        print(f"Evaluating from start: {start} Reward: {float(total_reward)} Predicted Reward: {float(predicted_value)}",
             f"found goal after steps: {t}" if found_goal else "did not find goal")

        wandb.log({"steps_taken_evaluation": t, "total_reward_evaluation": total_reward})  # wandb logging

def main():
    model = CACLA()
    for iter in range(10000000):
        if STATE == States.EVALUATE_ACTIONS:
            model.evaluate()
        else:
            model.train(iter)

if __name__ == "__main__":
    main()
