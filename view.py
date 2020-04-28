import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--work_dir', help='the dir to save logs and models')
parser.add_argument('config', help='the config file')
args = parser.parse_args()
print(args)
