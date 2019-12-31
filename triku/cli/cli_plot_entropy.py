from triku.pl import entropy
from funcargparse import FuncArgParser

def main():
    parser_entropy = FuncArgParser()
    parser_entropy.setup_args(entropy)
    parser_entropy.update_arg('object_triku', help="Path to the annData object or the count matrix.")
    parser_entropy.update_arg('show', default=False)
    parser_entropy.create_arguments()
    parser_entropy.parse2func()

if __name__ == '__main__':
    main()
