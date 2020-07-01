from triku.tl import triku
from funcargparse import FuncArgParser


def main():
    parser = FuncArgParser()
    parser.setup_args(triku)
    parser.update_arg(
        "object_triku", help="Path to the annData object or the count matrix."
    )
    parser.create_arguments()
    parser.parse2func()


if __name__ == "__main__":
    main()
