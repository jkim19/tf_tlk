"""
Will Jinwoo Kim (Numberseed)
twitter: @numberseedsoft
github: jkim19
"""

import argparse
import datetime as dt
import logging
import os
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def main():
    st = time.time()
    logging.basicConfig(format='%(asctime)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.DEBUG)
    logging.info("Running main ...")

    # load dataset

    # configure model / name / load previous

    # train model / save model

    # eval model

    et = time.time()
    logging.info("main() finished. Duration {:.6f}"
                 " minutes.".format((et-st)/60.0))


if __name__ == "__main__":
    main()
