from triku.pl import entropy
from funcargparse import FuncArgParser

def main():
    parser = FuncArgParser()
    parser.setup_args(entropy)
    parser.update_arg('object_triku', help="Path to the annData object or the count matrix.")
    parser.update_arg('dict_triku', help="Path to the dictionary files '_entropy.txt' and '_selected_genes.txt' files."
                                         "Add the path and the name. For instance: my_data/example_entropy.txt -> "
                                         "mydata/example.")
    parser.create_arguments()
    parser.parse2func()

if __name__ == '__main__':
    main()
