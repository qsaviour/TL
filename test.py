import argparse
import pickle

parser_ = argparse.ArgumentParser()
parser_.add_argument("-n", "--name", help="inferring", type=str)
args = parser_.parse_args()

print(args)
