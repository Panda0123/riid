import argparse

import config
import model_dispatcher
from models.saint.train import run as saint


def run(model, bS, nEpochs, bufferSize, logInterval):
    print(model)
    if model == "saint":
        alwaysPath = config.SAINT_MODEL_ALWAYS_PATH
        bestPath = config.SAINT_MODEL_BEST_PATH
        saint(bS, nEpochs, bufferSize, logInterval, alwaysPath, bestPath)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str
    )
    # parser.add_argument(
    #     "--bs",
    #     type=int
    # )
    # parser.add_argument(
    #     "--nepochs",
    #     type=str
    # )
    # parser.add_argument(
    #     "--buffer",
    #     type=int
    # )
    # parser.add_argument(
    #     "--log",
    #     type=int
    # )

    args = parser.parse_args()
    model = args.model
    bS = args.bs
    nEpochs = args.nepochs
    bufferSize = args.buffer
    logInterval = args.log
    run(model, bS, nEpochs, bufferSize, logInterval)
