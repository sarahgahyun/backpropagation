"""
PSYCH 420: Backpropagation

To customize number of epochs (training cycles) and/or the learning rate (alpha), please add arguments when running. 
The below are using the default values: 
    -e=10000 --epochs=10000
    -l=0.1 --learning_rate=0.1
You may also specify the training set. The default is the xor logic gate.
Please note that the shapes is buggy!
    -tx --train_xor
    -ta --train_and
    -ts --train_shapes
"""
import argparse
import sys
from train_shapes import train_shapes
from train_logic import train_and, train_xor

def initialize_network(n_inputs, n_hidden, n_outputs):
    network = list()

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--learning_rate", type=float, default=0.1)
    parser.add_argument("-e", "--epochs", type=int, default=10000)
    parser.add_argument("-tx", "--train_xor", action="store_true")
    parser.add_argument("-ta", "--train_and", action="store_true")
    parser.add_argument("-ts", "--train_shapes", action="store_true")
    return parser.parse_args()

def main():
    if len(sys.argv) > 1:
        if sys.argv[1] == "--help" or sys.argv[1] == "-h":
            print(__doc__)
            sys.exit(1)
    args = parse_arguments()
    
    if args.train_and:
        train_and(args.epochs, args.learning_rate)
    if args.train_xor:
        train_xor(args.epochs, args.learning_rate)
    if args.train_shapes:
        train_shapes(args.epochs, args.learning_rate)
    # default
    if not args.train_and and not args.train_xor and not args.train_shapes:
        train_xor(args.epochs, args.learning_rate)

if __name__ == "__main__":
    main()
