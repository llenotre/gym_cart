from model import GymModel
import genetic_optimization
import sys

def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--train':
        genetic_optimization.train(1000, 1, 0.1)
    else:
        print('Starting simulation from training data...')
        GymModel().play()

main()
