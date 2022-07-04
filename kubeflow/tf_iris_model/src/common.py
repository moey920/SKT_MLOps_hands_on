import argparse
import tensorflow as tf


def parse_args():
    parser = argparse.ArgumentParser(description="arguments needs to Tensorflow iris model training")
    parser.add_argument(
        "--batch_size",
        "-b",
        required=False,
        default=32,
        type=int,
        help="input hyper parameters - batch_size",
    )
    parser.add_argument(
        "--epochs",
        "-e",
        required=False,
        default=200,
        type=int,
        help="input hyper parameters - epochs",
    )
    parser.add_argument(
        "--optimizer",
        required=False,
        default="Adam",
        type=str,
        help="input hyper parameters - optimizer, like rmsprop, adam, sgd..",
    )
    parser.add_argument(
        "--save", "-s", required=False, type=str, help="model saved path at TFJob",
    )

    return parser.parse_args()


def get_hyperparams():
    args = parse_args()

    batch_size = args.batch_size
    epochs = args.epochs
    optimizer = args.optimizer
    save = args.save

    return (
        batch_size,
        epochs,
        optimizer,
        save
    )
