import json, os, subprocess, time, sys
from pathlib import Path
cfg = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
gpu = cfg['gpu']
jobs = cfg['jobs']
log_path = Path(__file__).with_name(f"scheduler_gpu{gpu}_py.log")
logf = open(log_path, 'a', encoding='utf-8')
def log(msg):
    print(msg)
    logf.write(msg + "\n")
    logf.flush()
env = os.environ.copy()
env['CUDA_VISIBLE_DEVICES'] = str(gpu)
env['LD_LIBRARY_PATH'] = '/home/star/anaconda3/envs/zkx/lib:/home/star/anaconda3/lib'
log(f'START PY SCHED GPU{gpu} {time.ctime()}')
running = []
idx = 0
while idx < len(jobs) or running:
    while len(running) < 2 and idx < len(jobs):
        job = jobs[idx]
        log(f'START_JOB {job} {time.ctime()}')
        p = subprocess.Popen(['bash','-lc','./launch.sh'], cwd=job, env=env)
        running.append((job,p))
        idx += 1
    time.sleep(5)
    still = []
    for job,p in running:
        ret = p.poll()
        if ret is None:
            still.append((job,p))
        else:
            log(f'END_JOB {job} rc={ret} {time.ctime()}')
    running = still
log(f'DONE PY SCHED GPU{gpu} {time.ctime()}')
logf.close()
