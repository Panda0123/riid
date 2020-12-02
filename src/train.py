import argparse
import config
import model_dispatcher 

def run(fold, model):
    print(fold)
    print(model)
    return 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fold",
        type=int
    )
    parser.add_argument(
        "--model",
        type=str
    )

    args = parser.parse_args()
    run(
        fold=args.fold,
        model=args.model)