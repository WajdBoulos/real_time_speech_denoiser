#!/usr/bin/env python3

"""Entry point of the library, to allow the user to use it in a CLI and use any of the built in objects in it.
Create the objects detailed in the setting file (that is passed as an argument to this script), and run them to let
them process audio.
"""

from __future__ import absolute_import

from .runner import Runner

import argparse
import yaml

def parse_arguments():
    """Get the arguments for this script from the user.
        
    Returns:
        Parameters to this scripts.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-f', '--filename', help='YAML setting file with classes to create', required=True)
    return parser.parse_args()

def load_classes_dict(classes_file_path):
    """Load the file in classes_file_path as a yaml file and return the dict that was in it.

    Args:
        classes_file_path (string): Path to the file that contains the classes to create.
        
    Returns:
        The dict of all the classes that are needed to be created.
    """
    with open(classes_file_path, "r") as f:
        classes = yaml.load(f, yaml.SafeLoader)
    return classes

def main():
    """Entry point for this script, to make it easy to use this library in a CLI.

    Create the objects detailed in the setting file (that is passed as an argument to this script), and run them to let
    them process audio.
    """
    # Get the arguments to the script
    args = parse_arguments()

    # Load the settings file as a dict
    classes = load_classes_dict(args.filename)

    # Create the classes and run them
    runner = Runner(classes)
    runner.run()

if __name__ == '__main__':
    main()