#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 14:06:38 2023

@author: u0145079
"""
import os
import sys
import time
import threading

error2="""
 __ __  __  __  __  
|_ |__)|__)/  \|__) 
|__| \ | \ \__/| \  
                                     
"""


def exists_on_env_path(program):
    " Check whether program exists in PATH and is executable"
    for dir in os.environ["PATH"].split(os.pathsep):
        fpath = dir + "/" + program
        if os.path.exists(fpath) and os.access(fpath, os.X_OK):
            return True
    return False

def check_dependencies(programs):
    if isinstance(programs, str):  # if only a single program is given
        programs = [programs]  # convert to a list

    for program in programs:
        if not exists_on_env_path(program):
            error = f"\nRequired program '{program}' not found\n"
            error += "Make sure this program has been installed and added to your PATH\n"
            print(error2)
            sys.exit(error)

def spinning_cursor():
    while True:
        for cursor in '|/-\\':
            yield cursor

def spinning_cursor_task(task_done,program):
    spinner = spinning_cursor()
    print(f"Starting {program}in the background. This may take a while...")
    while not task_done.is_set():
        sys.stdout.write(next(spinner))  # write the next character
        sys.stdout.flush()                # flush stdout buffer (actual character display)
        sys.stdout.write('\b')            # erase the last written char
        time.sleep(0.1)