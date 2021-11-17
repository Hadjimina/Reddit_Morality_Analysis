import constants as CS
from helpers.helper_functions import *
import distributed_configuration as dist_config
import subprocess
import sys
import os
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

def run_cmd(cmd, verbose, verbose_msg, ret=False):
    if verbose:
        lg.info(verbose_msg)
        #lg.info("Copying ssh key to {username}@{remote_host}".format(username=username, remote_host=remote_host))
        #print(cmd)
        
    p = run_subprocess(cmd)
    if ret:
        return str(p).replace('\\n', '\n').replace('\\t', '\t')
    elif verbose and len(p)>3:
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
  

def is_server_up(remote_host):
    """Given an ip address checks whether the ip address is reachable by ping

    Args:
        ip_addr (int): ip address of server we want to check

    Returns:
        bool: whether server is up
    """
    
    return os.system('ping -c 1 ' + remote_host + ' > /dev/null') == 0  

def is_server_creating(username, remote_host):
    """Check if all remote hosts are either shut down or creating features

    Args:
        username (string): ssh username
        remote_host (string): ip address of remote host

    Returns:
        bool: True if server is creating features or shutdown. False if not creating
    """
    if remote_host == "main" or is_server_up(remote_host):
        to_check = "ps -ef | grep create_features"
        cmd = "{location}{to_check}{quote_end}".format(location=f"ssh {username}@{remote_host} '"if remote_host != "main" else "", to_check = to_check, quote_end = "'" if remote_host != "main" else "" )
        p = run_cmd(cmd, False, "", ret=True).split("\n")
        p_filtered = list(filter(lambda x: "create_features.py -d" in x, p))
        is_running = len(p_filtered)> 0
        if is_running:
            lg.info(f"  {remote_host} is creating")
        else:
            lg.warning(f"  {remote_host} has crashed")
        return is_running
    else:
        lg.info(f"  {remote_host} is down")
        return True
    
def main(args):
    """Distributed feature extraction to all hosts listed in distributed_configuration.py
    """
    repo_name = "Reddit_Morality_Analysis"
    verbose = dist_config.feature_functions["verbose"]
    
    only_skip = True
    to_skip = []
    all_running = True
    
    if "-c" in args or "-check" in args:
        lg.info("Checking all hosts")
    #    sent_telegram_notification("*** Checking all hosts")
    
    
    # Create list of hosts to skip and run "check" if necessary
    for host in dist_config.feature_functions["hosts"].keys():
        if "-c" in args or "-check" in args:
            username = dist_config.feature_functions["hosts"][host]["username"]
            remote_host = dist_config.feature_functions["hosts"][host]["host_address"]
            
            all_running &= is_server_creating(username, remote_host)
            continue
        
        if "skip" in dist_config.feature_functions["hosts"][host] and dist_config.feature_functions["hosts"][host]["skip"]:
            to_skip.append(host)
        else:
            only_skip = False
    
    if "-c" in args or "-check" in args:
        if all_running:
            lg.info("All Okay")
        return
    
    sent_telegram_notification("*** Started distribution on "+socket.gethostname())
    if only_skip:
        lg.warning("No distribution! Skipping all instances")
    else:
        for sk in to_skip:
            lg.warning("Skipping host {sk}".format(sk=sk))
            
    
    # install cmds
    requirements_cmd = "pip3 install -r requirements.txt || pip install -r requirements.txt"
    click_install = "pip3 install click --upgrade || pip install click --upgrade"
    spacy_install = "python3 -m spacy download en_core_web_trf || python -m spacy download en_core_web_trf"
    
    # tmux & create_features
    tmux_cmd = "tmux new-session -d  -s "+TMUX_SESSION_NAME
    create_features_cmd = "python3 create_features.py -d || python create_features.py -d"
    
    
    # 2. Iterate over hosts 
    for host in dist_config.feature_functions["hosts"].keys():
        username = dist_config.feature_functions["hosts"][host]["username"]
        remote_host = dist_config.feature_functions["hosts"][host]["host_address"]
        path = dist_config.feature_functions["hosts"][host]["path"]
        skip = dist_config.feature_functions["hosts"][host]["skip"] if "skip" in dist_config.feature_functions["hosts"][host] else False
        should_upload = dist_config.feature_functions["hosts"][host]["upload"] if "upload" in dist_config.feature_functions["hosts"][host] else True
        
        if remote_host == "main"or skip: #we run the main host last and skip if necessary
            continue
        
        # Don't always upload the directory
        if should_upload:
            # 2.2 make sure that path exists over ssh & delete existing repo i.e. ssh username@remote_host mkdir -p path
            cmd = f"ssh {username}@{remote_host} 'mkdir -p {path} && rm -rf {path}/{repo_name}'"
            verbose_msg = f"Create directory on {username}@{remote_host}"
            run_cmd(cmd, verbose, verbose_msg)
            
            # 2.3 scp current code into remote-host
            cmd = f"scp -r ../{repo_name} {username}@{remote_host}:{path} "
            verbose_msg = f"Copy directory to {username}@{remote_host}"
            run_cmd(cmd, verbose, verbose_msg)
        else:
            lg.info(f"Not uploading to {host}")
            
        # 2.4 ssh into remote host and check if all packages are installed
        cmd = f"ssh {username}@{remote_host} 'cd {path}/{repo_name} && ({requirements_cmd}) && ({click_install}) &&({spacy_install})'"
        verbose_msg = f"Check installed packages on {username}@{remote_host}"
        run_cmd(cmd, verbose, verbose_msg)            
        
        # 2.5 check if tmux is running, if yes kill if not create new instance
        kill_tmux(username, remote_host, verbose)  
        
         
        # 2.6 ssh into remote host & run create features over tmux
        #msg = "Creating features on {0}".format(host)
        #cmd = "ssh {username}@{remote_host} 'cd {path}; {tmux_cmd} \"({create_features_cmd})\"'".format(path=path+"/"+repo_name)
        #run_cmd(cmd, True, msg)
        msg = f"Creating features on {0}".format(host)
        cmd = "ssh {username}@{remote_host} 'cd {path}; {tmux_cmd}; tmux send-keys -t \"rma:0\" \"python3 create_features -d\" Enter'".format(path=path+"/"+repo_name, username=username, tmux_cmd=tmux_cmd, remote_host=remote_host)
        run_cmd(cmd, True, msg)
        
        
        # cmd to connect to instance
        # sent_telegram_notification("* Create features on {host} sucessfull".format(host=host))
        cmd_to_connect = f"ssh -t {username}@{remote_host} \"tmux a -t rma\""
        sent_telegram_notification(f"Connect to {host} with: \n{cmd_to_connect}")

    # Start main session
    addresses = [dist_config.feature_functions["hosts"][host]["host_address"] for host in list(dist_config.feature_functions["hosts"].keys())]
    if "main" in addresses:
        if not("skip" in dist_config.feature_functions["hosts"]["phdesktop"] and dist_config.feature_functions["hosts"]["phdesktop"]["skip"]):
            kill_tmux(username, "main", verbose)
            msg = "Creating features on {0}".format(socket.gethostname())
            cmd = f"{tmux_cmd} \"({create_features_cmd})\""
            run_cmd(cmd, True, msg)
            
            #sent_telegram_notification("* Create features on {host} sucessfull".format(host=host))
            cmd_to_connect = "tmux a -t rma"
            sent_telegram_notification("Connect to {host} with: \n{cmd_to_connect}".format(host=socket.gethostname(), cmd_to_connect=cmd_to_connect))
        
    sent_telegram_notification("*** All distributions started ")                
    

if __name__ == "__main__":
    main(sys.argv)