//! Static trusted adapters, executed by `/usr/bin/python3 -I`. They are small
//! processes, not daemon forks. Payload argv/env are data, never interpolated
//! code; only the payload's intentional bash -c interprets the command text.
//! No descriptor is inherited implicitly: foreground handoff uses a held
//! CLOEXEC memfd via its explicit /proc/<parent>/fd path and closes after read.
pub const PYTHON: &str = "/usr/bin/python3";
pub const EXEC: &str = r#"
import os, sys, json
try:
    # Establishment precedes payload BASH_ENV and any payload effect.
    with open(sys.argv[2], 'wb') as marker: marker.write(b'1')
    with open(sys.argv[1], 'rb') as source: raw = source.read(524289)
    if len(raw) > 524288: raise ValueError()
    env = json.loads(raw)
    if not isinstance(env, dict) or any(not isinstance(k,str) or not isinstance(v,str) or '\0' in k+v or '=' in k for k,v in env.items()): raise ValueError()
    with open(sys.argv[2], 'ab') as marker: marker.write(b'2')
    os.execve(sys.argv[3], sys.argv[3:], env)
except Exception:
    os.write(2, b'workspace payload adapter setup/exec failed\n')
    sys.exit(125)
"#;

pub const JOB: &str = r#"
import os, sys, json, subprocess, signal, stat
signal.signal(signal.SIGTERM, lambda *_: None)
try:
    # Root ancestors are operator-trusted; every component below root is nofollow.
    directory = os.open(sys.argv[1], os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    for part in sys.argv[2:6]:
        if not part or part in ('.','..') or '/' in part: raise ValueError()
        child = os.open(part, os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW, dir_fd=directory)
        os.close(directory); directory = child
    directory_stat = os.fstat(directory)
    if str(directory_stat.st_dev) != sys.argv[6] or str(directory_stat.st_ino) != sys.argv[7]: raise ValueError()
    def read(name):
        fd = os.open(name, os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK, dir_fd=directory)
        with os.fdopen(fd, 'rb') as f: return f.read(524289)
    raw = read('env')
    if len(raw)>524288: raise ValueError()
    env = json.loads(raw)
    if not isinstance(env,dict) or any(not isinstance(k,str) or not isinstance(v,str) or '\0' in k+v or '=' in k for k,v in env.items()): raise ValueError()
    os.unlink('env', dir_fd=directory)
    head_cap, tail_cap = int(sys.argv[8]), int(sys.argv[9])
    output = os.open('output.log', os.O_WRONLY|os.O_APPEND|os.O_NOFOLLOW|os.O_NONBLOCK, dir_fd=directory)
    if not stat.S_ISREG(os.fstat(output).st_mode): raise ValueError()
    process = subprocess.Popen(sys.argv[10:], env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    kept, tail = 0, b''
    def write_all(fd, data):
        while data:
            n = os.write(fd,data); data = data[n:]
    while True:
        data = os.read(process.stdout.fileno(),65536)
        if not data: break
        head = min(len(data),head_cap-kept)
        write_all(output,data[:head]); kept += head
        tail = (tail+data[head:])[-tail_cap:]
    code = process.wait()
    if code < 0: code = 128-code
    os.fsync(output); os.close(output)
    def atomic(name, data):
        temp = '.'+name+'-'+str(os.getpid())
        fd = os.open(temp,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=directory)
        write_all(fd,data); os.fsync(fd); os.close(fd)
        # rename replaces a symlink itself, never its target; refuse static links too.
        try:
            target = os.stat(name,dir_fd=directory,follow_symlinks=False)
            if stat.S_ISLNK(target.st_mode): raise ValueError()
        except FileNotFoundError: pass
        os.rename(temp,name,src_dir_fd=directory,dst_dir_fd=directory)
        os.fsync(directory)
    if tail: atomic('output.tail',tail)
    atomic('exit',(str(code)+'\n').encode())
    os.close(directory)
    sys.exit(code)
except Exception:
    os.write(2,b'workspace job adapter setup/IO failed\n')
    sys.exit(125)
"#;
