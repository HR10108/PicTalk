from pathlib import Path
from toolbox import Toolbox
from utils.argutils import print_args
from utils.modelutils import check_model_paths
import argparse
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Runs the toolbox",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("-e", "--enc_models_dir", type=Path, default="encoder/saved_models")
    parser.add_argument("-s", "--syn_models_dir", type=Path, default="synthesizer/saved_models")
    parser.add_argument("-v", "--voc_models_dir", type=Path, default="vocoder/saved_models")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    print_args(args, parser)
    args.cpu = False
    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    del args.cpu
    check_model_paths(encoder_path=args.enc_models_dir, synthesizer_path=args.syn_models_dir,
                      vocoder_path=args.voc_models_dir)

    Toolbox(**vars(args))
