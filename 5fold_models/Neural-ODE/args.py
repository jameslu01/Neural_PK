import argparse

parser = argparse.ArgumentParser("nerual ODE model")

parser.add_argument("--data", type=str, help="data file for processing")
parser.add_argument("--fold", type=int, help="current fold number")
parser.add_argument("--model", type=int, help="current model number")
parser.add_argument("--save", type=str, help="save dirs for the results")
parser.add_argument("--continue-train", action="store_true", help="continue training")
parser.add_argument("--random-seed", type=int, default=1000, help="random seed")

parser.add_argument("--layer", type=int, default=2, help="hidden layer of the ODE Function")
parser.add_argument("--lr", type=float, help="learning rate")
parser.add_argument("--l2", type=float, help="l2 regularization")
parser.add_argument("--hidden", type=int, help="hidden dim in ODE Function")
parser.add_argument("--tol", type=float, help="control the precision in ODE solver")
parser.add_argument("--epochs", type=int, help="epochs for training")


args = parser.parse_args()
