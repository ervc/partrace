import configparser
import numpy as np

import os


def read_input(f_in):
    if not os.path.exists(f_in):
        raise FileNotFoundError(f"Cannot find input file: '{f_in}'")

    strkeys = ['fargodir','outputdir','partfile','nout']
    fltkeys = ['t0','tf','partsize','partdens']
    intkeys = []
    boolkeys = ['diffusion']

    defaults = configparser.ConfigParser()
    defaults.read('partrace/defaults.ini')

    params = {}
    for key in defaults['params']:
        if key in strkeys:
            params[key] = defaults['params'][key]
        elif key in fltkeys:
            params[key] = float(defaults['params'][key])
        elif key in intkeys:
            params[key] = int(defaults['params'][key])
        elif key in boolkeys:
            params[key] = defaults['params'].getboolean(key)

    config = configparser.ConfigParser()
    config.read(f_in)

    for key in config['params']:
        if key in strkeys:
            params[key] = config['params'][key]
        elif key in fltkeys:
            params[key] = float(config['params'][key])
        elif key in intkeys:
            params[key] = int(config['params'][key])
        elif key in boolkeys:
            params[key] = config['params'].getboolean(key)
        else:
            raise KeyError(f"Unknown parameter {key} in input file")

    return params

def read_locations(fpart):
    locs = []
    with open(fpart,'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            n,x,y,z = line.split()
            locs.append((float(x),float(y),float(z)))
    return np.array(locs)

def write_paramsfile(params,fname):
    with open(fname,'w+') as f:
        f.write('[params]\n')
        for key in params:
            f.write(f'{key} = {params[key]}\n')
