import argparse
from tool.path_parser import cvt_abs_path
from tool.others import print_

DATA_FOLDER = cvt_abs_path('../data')


def parse_argument():
    parser_ = argparse.ArgumentParser()
    parser_.add_argument("--project", help="project name", type=str, default="TL")
    parser_.add_argument("--base", help="base path", type=str, default="../")

    parser_.add_argument("-train", "--train", help="training", action="store_true")
    parser_.add_argument("--batch", help="training batch_size", type=int, default=32)
    parser_.add_argument("--epoch", help="training epochs", type=int, default=100)
    parser_.add_argument("--orderly_sample", help="orderly sample", action="store_true")

    parser_.add_argument("-inference", "--inference", help="inferring", type=str)
    parser_.add_argument("--not_augment", help="don`t augment when generate", action="store_true")
    parser_.add_argument("--image", help="an inference image file", type=str)
    parser_.add_argument("--images", help="an inference folder contain images", type=str, default='')

    parser_.add_argument("--force", help="force to re-split test/train sets", action="store_true")
    parser_.add_argument("--single", help="could be done in parallel", action="store_true")

    args_ = parser_.parse_args()
    return args_


def main():
    args = parse_argument()
    print_("project:", args.project)
    if args.train:
        print_('training.....')
        from flow.train import train
        train(args)

    if args.inference:
        print_('inferring.....')
        from flow.inference import infer
        infer(args)

    print_('Done')


if __name__ == '__main__':
    main()
