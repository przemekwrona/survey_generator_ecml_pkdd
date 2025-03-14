import argparse

from survey_generator.data import warsaw_survey_generator


def generate_arff():
    parser = argparse.ArgumentParser(description='ARFF File Generator')
    parser.add_argument('--randomized', dest='randomized', type=bool, default=True, help='If True use Ones Matrix otherwise use Traffic Matrix')
    parser.add_argument('--shuffle', dest='shuffle', type=bool, default=False, help='If True shuffle attributes')
    parser.add_argument('--no-sample-train', dest='no_sample_train', type=int, default=10000, help='Limit number of real surveys')
    parser.add_argument('--no-sample-test', dest='no_sample_test', type=int, default=10000, help='Limit number of real surveys')

    args = parser.parse_args()

    warsaw_survey_generator.to_arff_all(randomized=args.randomized,
                                        shuffle=args.shuffle,
                                        no_sample_train=args.no_sample_train,
                                        no_sample_test=args.no_sample_test)


def hello():
    print("Hello")
