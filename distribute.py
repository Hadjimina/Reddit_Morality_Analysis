import constants as CS
from helpers.helper_functions import *
import distributed_configuration as dist_config
import subprocess
import sys
import socket

import logging as lg
import coloredlogs
coloredlogs.install()

TMUX_SESSION_NAME = "rma"

def run_subprocess(cmd):
    """Run command on subprocess

    Args:
        cmd (string): command to run on subprocess
    """
    return  str(subprocess.check_output(cmd, shell=True)).replace('\\n', '\n').replace('\\t', '\t')

def run_cmd(cmd, verbose, verbose_msg):
    if verbose:
        lg.info(verbose_msg)
        #lg.info("Copying ssh key to {username}@{remote_host}".format(username=username, remote_host=remote_host))
        #print(cmd)
        
    p = run_subprocess(cmd)
    if verbose and len(p)>3:
        print(p)
    
def kill_tmux(username, remote_host, verbose):
    tmux_running = "[[ -n $(pgrep tmux) ]] && echo true || echo false"
    tmux_ls  = "tmux ls"
    tmux_kill = "tmux kill-session -t "+TMUX_SESSION_NAME
    
    # 1. check if server running
    cmd = tmux_running if remote_host == "main" else "ssh {username}@{remote_host} '{tmux_running}'".format(username=username, remote_host=remote_host,tmux_running=tmux_running)
    p = str(subprocess.check_output(cmd, shell=True)).replace('\\n', '\n').replace('\\t', '\t')
    if "true" in p:
        # 2. check session list
        cmd = tmux_ls if remote_host == "main" else "ssh {username}@{remote_host} '{tmux_ls}'".format(username=username, remote_host=remote_host,tmux_ls=tmux_ls)
        p = str(subprocess.check_output(cmd, shell=True)).replace('\\n', '\n').replace('\\t', '\t')
        
        # 3. only kill if session exists
        if "rma" in p:
            cmd = tmux_kill if remote_host == "main" else "ssh {username}@{remote_host} '{tmux_kill}'".format(username=username, remote_host=remote_host,tmux_kill=tmux_kill)
            verbose_msg = "Kill existing tmux on {username}@{remote_host}".format(username=username, remote_host=remote_host)
            run_cmd(cmd, verbose, verbose_msg)     
    
def main(args):
    """Distributed feature extraction to all hosts listed in distributed_configuration.py
    """
    repo_name = "Reddit_Morality_Analysis"
    
    # skip list
    only_skip = True
    to_skip = []
    for host in dist_config.feature_functions["hosts"].keys():
        if "skip" in dist_config.feature_functions["hosts"][host] and dist_config.feature_functions["hosts"][host]["skip"]:
            to_skip.append(host)
        else:
            only_skip = False
    
    if only_skip:
        lg.warning("No distribution! Skipping all instances")
    else:
        for sk in to_skip:
            lg.warning("Skipping host {sk}".format(sk=sk))
            
    
    # install cmds
    requirements_cmd = "pip3 install -r requirements.txt || pip install -r requirements.txt"
    spacy_install = "python3 -m spacy download en_core_web_trf || python -m spacy download en_core_web_trf"
    
    # tmux & create_features
    tmux_cmd = "tmux new-session -d  -s "+TMUX_SESSION_NAME
    create_features_cmd = "python3 create_features.py -d || python create_features.py -d"
    
    
    # 1. Check dist_config structure:
    # TODO
    verbose = dist_config.feature_functions["verbose"]
    # 2. Iterate over hosts 
    for host in dist_config.feature_functions["hosts"].keys():
        username = dist_config.feature_functions["hosts"][host]["username"]
        remote_host = dist_config.feature_functions["hosts"][host]["host_address"]
        path = dist_config.feature_functions["hosts"][host]["path"]
        skip = dist_config.feature_functions["hosts"][host]["skip"] if "skip" in dist_config.feature_functions["hosts"][host] else False
        should_upload = dist_config.feature_functions["hosts"][host]["upload"] if "upload" in dist_config.feature_functions["hosts"][host] else True
        
        if remote_host == "main"or skip: #we run the main host last and skip if necessary
            continue
        
        # sudo apt install libpython3.8-dev
        
        # Don't always upload the directory
        if should_upload:
            # 2.2 make sure that path exists over ssh & delete existing repo i.e. ssh username@remote_host mkdir -p path
            cmd = "ssh {username}@{remote_host} 'mkdir -p {path} && rm -rf {path}/{repo_name}'".format(path=path, repo_name=repo_name,username=username, remote_host=remote_host)
            verbose_msg = "Create directory on {username}@{remote_host}".format(username=username, remote_host=remote_host)
            run_cmd(cmd, verbose, verbose_msg)
            
            # 2.3 scp current code into remote-host
            cmd = "scp -r ../{repo} {username}@{remote_host}:{path} ".format(username=username, remote_host=remote_host, cmd=cmd, path=path, repo=repo_name)
            verbose_msg = "Copy directory to {username}@{remote_host}".format(username=username, remote_host=remote_host)
            run_cmd(cmd, verbose, verbose_msg)
        else:
            lg.info("Not uploading to {host}".format(host=host))
            
        # 2.4 ssh into remote host and check if all packages are installed
        cmd = "ssh {username}@{remote_host} 'cd {path}/{repo} && ({requirements}) && ({spacy_install})'".format(path=path ,repo=repo_name, requirements=requirements_cmd,spacy_install=spacy_install, username=username, remote_host=remote_host,)
        verbose_msg = "Check installed packages on {username}@{remote_host}".format(username=username, remote_host=remote_host)
        run_cmd(cmd, verbose, verbose_msg)            
        
        # 2.5 check if tmux is running, if yes kill if not create new instance
        kill_tmux(username, remote_host, verbose)  
        
         
        # 2.6 ssh into remote host & run create features over tmux
        msg = "Creating features on {0}".format(host)
        cmd = "ssh {username}@{remote_host} 'cd {path}; {tmux} \"({create_features})\"'".format(path=path+"/"+repo_name, tmux=tmux_cmd, create_features = create_features_cmd, username=username, remote_host=remote_host)
        run_cmd(cmd, True, msg)
        
        # cmd to connect to instance
        # sent_telegram_notification("* Create features on {host} sucessfull".format(host=host))
        cmd_to_connect = "ssh -t {username}@{remote_host} \"tmux a -t rma\"".format(username=username, remote_host=remote_host)
        sent_telegram_notification("Connect to {host} with: \n{cmd_to_connect}".format(host=host, cmd_to_connect=cmd_to_connect))

    # Start main session
    addresses = [dist_config.feature_functions["hosts"][host]["host_address"] for host in list(dist_config.feature_functions["hosts"].keys())]
    if "main" in addresses:
        if not("skip" in dist_config.feature_functions["hosts"]["phdesktop"] and dist_config.feature_functions["hosts"]["phdesktop"]["skip"]):
            kill_tmux(username, "main", verbose)
            msg = "Creating features on {0}".format(socket.gethostname())
            cmd = "{tmux} \"({create_features})\"".format(tmux=tmux_cmd, create_features = create_features_cmd)
            run_cmd(cmd, True, msg)
            
            #sent_telegram_notification("* Create features on {host} sucessfull".format(host=host))
            cmd_to_connect = "tmux a -t rma"
            sent_telegram_notification("Connect to {host} with: \n{cmd_to_connect}".format(host=socket.gethostname(), cmd_to_connect=cmd_to_connect))
        
    sent_telegram_notification("*** All distributions started ")                
    

if __name__ == "__main__":
    sent_telegram_notification("*** Started distribution on "+socket.gethostname())
    main(sys.argv)