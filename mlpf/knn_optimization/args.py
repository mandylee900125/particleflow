import argparse
from math import inf

def parse_args():
    parser = argparse.ArgumentParser()

    # from raw -> processed
    parser.add_argument("--dataset", type=str, default='../../../test_tmp_delphes/data/pythia8_ttbar', help="dataset path", required=False)
    parser.add_argument("--dataset_qcd", type=str, default='../../../test_tmp_delphes/data/pythia8_qcd', help="dataset path", required=False)
    parser.add_argument("--processed_dir", type=str, help="processed", required=False, default=None)
    parser.add_argument("--num-files-merge", type=int, default=10, help="number of files to merge")
    parser.add_argument("--num-proc", type=int, default=24, help="number of processes")

    # for training
    parser.add_argument("--n_train", type=int, default=1, help="number of data files to use for training.. each file contains 100 events")
    parser.add_argument("--n_valid", type=int, default=1, help="number of data files to use for validation.. each file contains 100 events")
    parser.add_argument("--n_test", type=int, default=1, help="number of data files to use for testing.. each file contains 100 events")
    parser.add_argument("--n_epochs", type=int, default=1, help="number of training epochs")
    parser.add_argument("--hidden_dim", type=int, default=256, help="hidden dimension")
    parser.add_argument("--hidden_dim_nn1", type=int, default=64, help="hidden dimension")
    parser.add_argument("--input_encoding", type=int, default=12, help="use an input encoding layer")
    parser.add_argument("--encoding_dim", type=int, default=64, help="encoded element dimension")
    parser.add_argument("--batch_size", type=int, default=1, help="Number of .pt files to load in parallel")
    parser.add_argument("--model", type=str, help="type of model to use", default="PFNet7")
    parser.add_argument("--target", type=str, choices=["cand", "gen"], help="Regress to PFCandidates or GenParticles", default="gen")
    parser.add_argument("--outpath", type=str, default = '../../../test_tmp_delphes/experiments/', help="Output folder")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--alpha", type=float, default=2e-4, help="Loss multiplier for pdg-id classification.. recall: loss = clf + alpha*reg")
    parser.add_argument("--space_dim", type=int, default=4, help="Spatial dimension for clustering in gravnet layer")
    parser.add_argument("--propagate_dimensions", type=int, default=22, help="The number of features to be propagated between the vertices")
    parser.add_argument("--nearest", type=int, default=16, help="k nearest neighbors in gravnet layer")
    parser.add_argument("--overwrite", action=BoolArg, default=True, help="Overwrites the model if True")
    parser.add_argument("--title", type=str, default='', help="Appends this title to the model's name")

    args = parser.parse_args()

    return args


class BoolArg(argparse.Action):
    """
    Take an argparse argument that is either a boolean or a string and return a boolean.
    """
    def __init__(self, default=None, nargs=None, *args, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")

        # Set default
        if default is None:
            raise ValueError("Default must be set!")

        default = _arg_to_bool(default)

        super().__init__(*args, default=default, nargs='?', **kwargs)

    def __call__(self, parser, namespace, argstring, option_string):

        if argstring is not None:
            # If called with an argument, convert to bool
            argval = _arg_to_bool(argstring)
        else:
            # BoolArg will invert default option
            argval = True

        setattr(namespace, self.dest, argval)

def _arg_to_bool(arg):
    # Convert argument to boolean

    if type(arg) is bool:
        # If argument is bool, just return it
        return arg

    elif type(arg) is str:
        # If string, convert to true/false
        arg = arg.lower()
        if arg in ['true', 't', '1']:
            return True
        elif arg in ['false', 'f', '0']:
            return False
        else:
            return ValueError('Could not parse a True/False boolean')
    else:
        raise ValueError('Input must be boolean or string! {}'.format(type(arg)))


# From https://stackoverflow.com/questions/12116685/how-can-i-require-my-python-scripts-argument-to-be-a-float-between-0-0-1-0-usin
class Range(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end
    def __eq__(self, other):
        return self.start <= other <= self.end
