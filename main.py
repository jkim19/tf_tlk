"""
Will Jinwoo Kim (Numberseed)
twitter: @numberseedsoft
github: jkim19
"""

import argparse
import logging
import time


def main():
    st = time.time()
    logging.basicConfig(format='%(asctime)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.DEBUG)
    logging.info("Running main ...")

    et = time.time()
    logging.info("main() finished. Duration {:.6f}"
                 " minutes.".format((et-st)/60.0))


if __name__ == "__main__":
    main()
