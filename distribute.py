import constants as CS
from helpers.helper_functions import *
import distributed_configuration as dist_config
import subprocess
import sys
import socket

import logging as lg
import coloredlogs
coloredlogs.install()

def run_subprocess(cmd):
    """Run command on subprocess

    Args:
        cmd (string): command to run on subprocess
    """
    return subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def run_cmd(cmd, verbose, verbose_msg):
    if verbose:
        lg.info(verbose_msg)
        #lg.info("Copying ssh key to {username}@{remote_host}".format(username=username, remote_host=remote_host))
        #print(cmd)
        
    p = run_subprocess(cmd)
    p.wait()
    if verbose:
        out_str = repr(p.stdout.readline())#.decode("ascii")
        err_str = repr(p.stderr.readline())#.decode("ascii")
        if len(out_str) > 3:
            print("    Output:\n"+out_str)
        if len(err_str) > 3:
            print("    Errors:\n"+err_str)
    
def main(args):
    """Distributed feature extraction to all hosts listed in distributed_configuration.py
    """
    repo_name = "Reddit_Morality_Analysis"
    
    # install cmds
    requirements_cmd = "pip3 -r requirements.txt || pip -r requirements.txt"
    spacy_install = "python3 -m spacy download en_core_web_trf || python -m spacy download en_core_web_trf"
    
    # tmux & create_features
    tmux_kill_cmd  = "tmux kill-session -t rma"
    tmux_cmd = "tmux new-session -d  -s rma " 
    create_features_cmd = "python3 create_features.py -d || python create_features.py -d"
    
    
    # 1. Check dist_config structure:
    # TODO
    verbose = dist_config.feature_functions["verbose"]
    # 2. Iterate over hosts 
    for host in dist_config.feature_functions["hosts"].keys():
        username = dist_config.feature_functions["hosts"][host]["username"]
        remote_host = dist_config.feature_functions["hosts"][host]["host_address"]
        path = dist_config.feature_functions["hosts"][host]["path"]
        
        if remote_host == "main": #we run the main host last
            continue
        
        # 2.1 exchange ssh keys 
        #cmd = "ssh-copy-id {username}@{remote_host}".format(username=username, remote_host=remote_host)
        #verbose_msg = "Copying ssh key to {username}@{remote_host}".format(username=username, remote_host=remote_host)
        #run_cmd(cmd, verbose, verbose_msg)
        #
        # 2.2 make sure that path exists over ssh & delete existing repo i.e. ssh username@remote_host mkdir -p path
        cmd = "ssh {username}@{remote_host} 'mkdir -p {path} && rm -rf {path}/{repo_name}'".format(path=path, repo_name=repo_name,username=username, remote_host=remote_host)
        verbose_msg = "Create directory on {username}@{remote_host}".format(username=username, remote_host=remote_host)
        run_cmd(cmd, verbose, verbose_msg)
        
        # 2.3 scp current code into remote-host
        cmd = "scp -r ../{repo} {username}@{remote_host}:{path} ".format(username=username, remote_host=remote_host, cmd=cmd, path=path, repo=repo_name)
        verbose_msg = "Copy directory to {username}@{remote_host}".format(username=username, remote_host=remote_host)
        run_cmd(cmd, verbose, verbose_msg)
        
        # 2.4 ssh into remote host and check if all packages are installed and kill prev tmux sessions 
        cmd = "ssh {username}@{remote_host} 'cd {path}/{repo} && ({requirements}) && ({spacy_install}) && {tmux_kill}'".format(path=path ,repo=repo_name, requirements=requirements_cmd,spacy_install=spacy_install, tmux_kill=tmux_kill_cmd, username=username, remote_host=remote_host,)
        verbose_msg = "Check installed packages on {username}@{remote_host}".format(username=username, remote_host=remote_host)
        run_cmd(cmd, verbose, verbose_msg)            
        
        # 2.5 ssh into remote host & run create features over tmux
        msg = "Start create_features on {0}".format(remote_host)
        cmd = "ssh {username}@{remote_host} 'cd {path}; {tmux} \"({create_features})\"'".format(path=path+"/"+repo_name, tmux=tmux_cmd, create_features = create_features_cmd, username=username, remote_host=remote_host)
        run_cmd(cmd, True, msg)
        # cmd to connect to instance
        cmd_to_connect = "ssh -t {username}@{remote_host} \"tmux a -t rma\"".format(username=username, remote_host=remote_host)
        sent_telegram_notification("Connect to {host} with: \n{cmd_to_connect}".format(host=host, cmd_to_connect=cmd_to_connect))

    addresses = [dist_config.feature_functions["hosts"][host]["host_address"] for host in list(dist_config.feature_functions["hosts"].keys())]
    if "main" in addresses:
        lg.info("create_features on {0}".format(host))
        run_subprocess(tmux_kill_cmd)
        run_subprocess("{tmux} '({create_features})'".format(tmux=tmux_cmd, create_features = create_features_cmd))
    
    sent_telegram_notification("***All distributions started sucessfully ")                
    

if __name__ == "__main__":
    sent_telegram_notification("*** Started distribution on "+socket.gethostname())
    main(sys.argv)