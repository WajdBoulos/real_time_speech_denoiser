#!/usr/bin/env python3

"""
Entry point of this library, to make the running of it easier. 
This script just calls the main function of the runner.run script.
"""
from __future__ import absolute_import

from .runner.run import main
import cProfile

if __name__ == '__main__':
    pr = cProfile.Profile()
    pr.enable()

    main()

    pr.disable()
    pr.print_stats(sort='cumtime')