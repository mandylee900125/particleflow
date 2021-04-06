import argparse
from math import inf

def parse_args():
    parser = argparse.ArgumentParser()

    # from raw -> processed
    parser.add_argument("--dataset", type=str, default='../../../test_tmp_delphes/data/pythia8_ttbar', help="dataset path", required=True)
    parser.add_argument("--dataset_qcd", type=str, default='../../../test_tmp_delphes/data/pythia8_qcd', help="dataset path", required=True)
    parser.add_argument("--processed_dir", type=str, help="processed", required=False, default=None)
    parser.add_argument("--num-files-merge", type=int, default=10, help="number of files to merge")
    parser.add_argument("--num-proc", type=int, default=24, help="number of processes")

    # for training
    parser.add_argument("--train", type=str, default=True, help="Trains the model")
    parser.add_argument("--n_train", type=int, default=3, help="number of data files to use for training.. each file contains 100 events")
    parser.add_argument("--n_valid", type=int, default=1, help="number of data files to use for validation.. each file contains 100 events")
    parser.add_argument("--n_test", type=int, default=2, help="number of data files to use for testing.. each file contains 100 events")
    parser.add_argument("--n_epochs", type=int, default=1, help="number of training epochs")
    parser.add_argument("--patience", type=int, default=100, help="patience before early stopping")
    parser.add_argument("--hidden_dim", type=int, default=32, help="hidden dimension")
    parser.add_argument("--encoding_dim", type=int, default=256, help="encoded element dimension")
    parser.add_argument("--batch_size", type=int, default=1, help="Number of .pt files to load in parallel")
    parser.add_argument("--model", type=str, help="type of model to use", default="PFNet7")
    parser.add_argument("--target", type=str, choices=["cand", "gen"], help="Regress to PFCandidates or GenParticles", default="cand")
    parser.add_argument("--outpath", type=str, default = '../../../test_tmp_delphes/experiments/', help="Output folder")
    parser.add_argument("--activation", type=str, default='leaky_relu', choices=["selu", "leaky_relu", "relu"], help="activation function")
    parser.add_argument("--optimizer", type=str, default='adam', choices=["adam", "adamw"], help="optimizer to use")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--l1", type=float, default=1.0, help="Loss multiplier for pdg-id classification")
    parser.add_argument("--l2", type=float, default=0.001, help="Loss multiplier for momentum regression")
    parser.add_argument("--l3", type=float, default=1.0, help="Loss multiplier for clustering")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate")
    parser.add_argument("--radius", type=float, default=0.1, help="Radius-graph radius")
    parser.add_argument("--convlayer", type=str, choices=["gravnet-knn", "gravnet-radius", "sgconv", "gatconv"], help="Convolutional layer", default="gravnet-knn")
    parser.add_argument("--convlayer2", type=str, choices=["sgconv", "graphunet", "gatconv", "none"], help="Convolutional layer", default="none")
    parser.add_argument("--space_dim", type=int, default=2, help="Spatial dimension for clustering in gravnet layer")
    parser.add_argument("--nearest", type=int, default=3, help="k nearest neighbors in gravnet layer")
    parser.add_argument("--overwrite", action='store_true', help="overwrite if model output exists")
    parser.add_argument("--input_encoding", type=int, help="use an input encoding layer", default=0)
    parser.add_argument("--load", type=str, help="Load the model (no training)", required=False, default=None)
    parser.add_argument("--load_model", type=str, help="Which model to load", default="PFNet7_cand_ntrain_2")
    parser.add_argument("--load_epoch", type=float, default=0, help="Which epoch of the model to load for evaluation")

    # for evaluation
    parser.add_argument("--evaluate", type=str, default=True, help="Evaluates the model on the test data")
    parser.add_argument("--evaluate_on_cpu", type=str, help="Check to evaluate on cpu", default=False)

    args = parser.parse_args()

    return args
