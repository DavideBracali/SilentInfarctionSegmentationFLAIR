#!/usr/bin/python
# -*- coding: utf-8 -*-

from .__version__ import __version__
import argparse


__author__ = ['Davide Bracali']
__email__ = ['davide.bracali@studio.unibo.it']

def parse_args ():

    description = ('SilentInfarctionSegmentationFLAIR - '
    '°_° aggiungere descrizione'
    )

    parser = argparse.ArgumentParser(
    prog='SilentInfarctionSegmentationFLAIR',
    argument_default=None,
    add_help=True,
    prefix_chars='-',
    allow_abbrev=True,
    exit_on_error=True,
    description=description,
    epilog=f'SilentInfarctionSegmentationFLAIR Python package v{__version__}'
    )

    # version
    parser.add_argument(
    '--version', '-v',
    dest='version',
    required=False,
    action='store_true',
    default=False,
    help='Get the current version installed',
    )

    return parser.parse_args()

def main():
    print(__version__)

if __name__ == '__main__':
    main()