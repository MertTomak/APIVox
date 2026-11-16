"""Define the CTC, attention encoder-decoder and RNN-T models trained from scratch."""
import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    return parser.parse_args()


def main() -> None:
    parse_args()


if __name__ == "__main__":
    main()
