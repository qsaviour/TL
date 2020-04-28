import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-train", "--train", help="training", action="store_true")
    parser.add_argument("-test", "--test", help="testing", action="store_true")
    parser.add_argument("-inf", "--inference", help="inferring", type=str)
    args = parser.parse_args()
    if not (args.train or args.test or args.inference):
        raise ValueError("main.py need --train or --test")

    if args.train:
        print('training.....')

    if args.test:
        print('testing.....')

    if args.inference:
        print('inferring.....')


if __name__ == '__main__':
    main()
