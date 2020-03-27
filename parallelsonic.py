import retro        
import numpy as np  
import cv2         
import neat         
import pickle       

resume = True
restore_file = "neat-checkpoint-800"

class Worker(object):
    def __init__(self, genome, config):
        self.genome = genome
        self.config = config
        
    def work(self):
        self.env = retro.make('SonicTheHedgehog-Genesis', 'GreenHillZone.Act1')
        
        self.env.reset()

        ob = self.env.reset()
        #ac = env.action_space.sample()
        inx, iny, inc = self.env.observation_space.shape
        inx = 45
        iny = 25
        net = neat.nn.recurrent.RecurrentNetwork.create(self.genome, self.config)
        fitness = 0
        xpos = 0
        xpos_max = 0
        counter = 0
        imgarray = []
        done = False

        while not done:
            self.env.render()
            ob = ob[60:-20, 20:-30]
            ob = cv2.resize(ob, (inx, iny))
            ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
            ob = np.reshape(ob, (inx,iny))
            imgarray = np.ndarray.flatten(ob)
            nnOutput = net.activate(imgarray)
            ob, rew, done, info = self.env.step(nnOutput)
            xpos = info['x']

            
            if xpos > xpos_max:
                xpos_max = xpos
                counter = 0
                fitness += 10
            else:
                counter += 1
                
            if counter > 250:
                done = True
                
        print(fitness)
        return fitness

def eval_genomes(genome, config):
    
    worky = Worker(genome, config)
    return worky.work()

if __name__=="__main__":
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, 
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         'config-feedforward')

    if resume == True:
        p = neat.Checkpointer.restore_checkpoint(restore_file)
    else:
        p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(10))

    pe = neat.ParallelEvaluator(2, eval_genomes)

    winner = p.run(pe.evaluate)

    with open('winner.pkl', 'wb') as output:
        pickle.dump(winner, output, 1)
