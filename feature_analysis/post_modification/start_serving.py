from waitress import serve
import server
import subprocess, time
import threading
LIWC_PATH = "LIWC-22"

def start_server():
    serve(server.app, host='0.0.0.0', port=8080)
    
def start_keep_alive():
    while True:
        proc=subprocess.Popen(LIWC_PATH)
        time.sleep(1*60*60) # 3 hours
        proc.kill()
        time.sleep(10)  # 30 seconds
        
server_thread = threading.Thread(target=start_server,args=())
keep_alive_thread = threading.Thread(target=start_keep_alive,args=())

print("Starting keep alive")
keep_alive_thread.daemon = True
keep_alive_thread.start()
time.sleep(10)
print("Starting Server")
server_thread.daemon = True
server_thread.start()

while True:
    time.sleep(1)