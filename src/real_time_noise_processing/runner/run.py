#!/usr/bin/env python3

""" Run a combination of the objects in the library.
"""

from __future__ import absolute_import

from .runner import Runner

import argparse
import yaml

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-f', '--filename', help='setting file with classes to create', required=True)
    return parser.parse_args()

def load_classes_dict(classes_file_name):
    with open(classes_file_name, "r") as f:
        classes = yaml.load(f, yaml.SafeLoader)
    return classes

def main():
    # Read the YAML of objects to create
    # Initialize the objects
    
    args = parse_arguments()

    classes = load_classes_dict(args.filename)

    runner = Runner(classes)
    runner.run()

if __name__ == '__main__':
    main()