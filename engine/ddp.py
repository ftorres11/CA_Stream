# -*- coding: utf-8 -*- 
# Author: Felipe Torres

# Torch Imports
import torch.distributed as dist
# In Package Imports

# Package Imports
import os
import hostlist
 

# ========================================================================
# Basic config

# Get SLURM variables
rank = int(os.environ['SLURM_PROCID'])
local_rank = int(os.environ['SLURM_LOCALID'])
size = int(os.environ['SLURM_NTASKS'])
cpus_per_task = int(os.environ['SLURM_CPUS_PER_TASK'])
  
# get node list from slurm
hostnames = hostlist.expand_hostlist(os.environ['SLURM_JOB_NODELIST'])
   
# get IDs of reserved GPU
gpu_ids = os.environ['SLURM_STEP_GPUS'].split(",")
    
# define MASTER_ADD & MASTER_PORT
os.environ['MASTER_ADDR'] = hostnames[0]
os.environ['MASTER_PORT'] = str(1694 + int(min(gpu_ids))) 
