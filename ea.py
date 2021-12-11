import copy
import random
import numpy as np
import time
import os
from tqdm import tqdm
import sys
import torch
import torch.nn.functional as F
import scipy.stats
from code2sound import code2sound
from pickle import NONE
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import copy
import scipy
from fitness_ANN import fitness_ANN
from fitness_tranditional import fitness_tranditional

class EvolutionController:
    def __init__(self, mutate_prob=0.1, population_size=100, n_evolution=100, parent_fraction=0.5, mutation_fraction=0.5, crossover_fraction=0, log_path='output/'):
        self.mutate_prob = mutate_prob
        self.population_size = population_size
        self.n_generations = n_evolution
        self.parent_num = int(self.population_size*parent_fraction)
        self.mutation_num = int(self.population_size*mutation_fraction)
        self.crossover_num = int(self.population_size*crossover_fraction)
        self.log_path = log_path
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        self.log_file = open(os.path.join(self.log_path,'ea_accuracy.txt'), 'w+')

        self.population=[]
        self.scores=[]

    def add_individual(self,individual,score=None):
        self.population.append(individual)
        if score is None:
            score=individual.evaluate()
        self.scores.append(score)

    def init_population(self,individual_generator,allow_repeat=False,max_sample_times=1000):
        self.population.clear()
        print(f"Generate {self.population_size} individuals")
        if allow_repeat:
            n=1
        else:
            n=max_sample_times
        for i in tqdm(range(self.population_size),desc="Generate individuals"):
            repeat_flag=True
            for i in range(n):
                individual=individual_generator()
                if individual not in self.population:
                    self.add_individual(individual)
                    repeat_flag=False
                    break
            if repeat_flag==True:
                self.add_individual(individual)
                print(f"WARNING: sample {n} times but all the sampled individuals are repeatted in population")

    def mutation(self,parents):
        for _ in range(self.mutation_num):
            selected_parent = parents[np.random.randint(self.parent_num)]
            child = selected_parent.mutate()
            self.add_individual(child)
    
    def crossover(self,parents):
        for _ in range(self.crossover_num):
            selected_parent1=parents[np.random.randint(self.parent_num)]
            selected_parent2=parents[np.random.randint(self.parent_num)]
            child=selected_parent1.crossover(selected_parent1,selected_parent2)
            self.add_individual(child)

    def run_evolution_search(self, verbose=False):

        best_score_history = []
        history_change=''
        t=tqdm(range(self.n_generations),desc="Evolutionary Search")
        for generation in t: 
            sorted_inds=np.argsort(self.scores)[::-1]
            parents = [self.population[_] for _ in sorted_inds[:self.parent_num]]

            now_best_score = self.scores[sorted_inds[0]]
            t.set_postfix({'new_best_score': now_best_score,'history_change': history_change})
            if generation and now_best_score>best_score_history[-1]:
                history_change+='+'
            else:
                history_change+='-'
            self.log_file.write(
                f"==={generation}/{self.n_generations}===\n")
            for i in sorted_inds[:3]:
                self.log_file.write(f"{self.scores[i]} {self.population[i]}\n")
            self.log_file.flush()
            
            best_score_history.append(now_best_score)

            # remove individuals
            self.population=parents
            self.scores=[self.scores[_] for _ in sorted_inds[:self.parent_num]]

            # mutation and crossover
            #print("Start Mutation")
            self.mutation(parents)
            #print("Start Crossover")
            self.crossover(parents)

        #print('Finish Evolution Search')
        ind=np.argmax(self.scores)
        return self.population[ind],self.scores[ind]


class MusicIndividual:
    def __init__(self, music = None, length = 8, note_num=27) -> None:
        self.length = length
        self.music = []
        self.note_num = note_num
        if music == None:
            for i in range(length):
                self.music.append(random.randint(0, note_num))
        else:
            self.music = music
    
    
    def evaluate(self):
        # 在这里修改fitness function！
        # return fitness_ANN(self.music)
        return -fitness_tranditional(self.music)
    
    def mutate(self):
        child_music=copy.deepcopy(self.music)
        p = random.random()
        if p < 0.7: #变异就是随机替换一个音符为随机值
            index = random.randint(0, self.length - 1)
            child_music[index] = random.randint(0, self.note_num)
        elif p < 0.8: #随机移调
            move = random.randint(-8, 8)
            for i in range(self.length):
                child_music[i] = child_music[i] + move
                if child_music[i] < 0:
                    child_music[i] = child_music[i] + self.note_num
                if child_music[i] > self.note_num:
                    child_music[i] = child_music[i] - self.note_num
        elif p < 0.9: #倒影变换
            for i in range(self.length):
                child_music[i] = self.note_num - child_music[i]
        else:         #逆行变换
            child_music = child_music.reverse()
        child = MusicIndividual(music=child_music, length=self.length)
        return child
    
    def crossover(self, parent1, parent2):
        child_music = []
        cut_index = random.randint(0, self.length)
        for i in range(self.length):
            if i < cut_index:
                child_music.append(parent1.music[i])
            else:
                child_music.append(parent2.music[i])
        child = MusicIndividual(music=child_music, length=self.length)
        return child
            
    def __repr__(self):
        s = ''
        for i in self.music:
            s += " " + str(i)
        return s



controller=EvolutionController(population_size=1000, crossover_fraction=0.2)
print(controller.population_size)
                
for individual_i in range(controller.population_size):
    individual=MusicIndividual(length=32)
    controller.add_individual(individual)
selected_individual,best_similarity=controller.run_evolution_search()
print(selected_individual.music)
code2sound(selected_individual.music)