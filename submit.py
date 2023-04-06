#!/usr/bin/env python

import subprocess
import argparse

def main(args):
    sb = '#!/bin/bash\n\n'
    if args.time:
        sb += f'#SBATCH --time={parse_time(args.time)}\n'
    else:
        print('no wall time given')
    sb += f'#SBATCH --ntasks={args.ntasks}\n'
    sb += f'#SBATCH --job-name={args.jobname}\n'
    sb += f'#SBATCH --output=j-%j.out\n'
    sb += f'#SBATCH --error=j-%j.err\n'
    if not args.nodelist is None:
        sb += f'#SBATCH --nodelist={args.nodelist}\n'
    sb += '\n'
    sb += f'./run_partrace.py -n {args.ntasks} {args.infile}'
    with open(f'sbatch_partrace.sbatch','w+') as f:
        f.write(sb)

    subprocess.run(['sbatch','sbatch_partrace.sbatch'])

def parse_time(time):
    days = int(time//24)
    hrs = int(time-days*24)
    mins = int((time-days*24-hrs)*60)
    secs = int((time-days*24-hrs-mins/60)*3600)
    if days>0:
        return f'{days}-{hrs:0>2}:{mins:0>2}:{secs:0>2}'
    else:
        return f'{hrs:0>2}:{mins:0>2}:{secs:0>2}'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='submit job to slurm scheduler')
    parser.add_argument('-n','--ntasks',type=int,default=1)
    parser.add_argument('-t','--time',type=float,default=None,help='walltime in hours')
    parser.add_argument('-j','--jobname',type=str,default='partrace')
    parser.add_argument('--nodelist',type=str,default=None)
    parser.add_argument('infile',type=str)
    args = parser.parse_args()
    main(args)
