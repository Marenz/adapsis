#!/usr/bin/env python3
"""Trusted prepare-only runner. Untrusted model Python executes inside bubblewrap.

stdin: {title, brief, material, references}; stdout: checked job report. No printer API,
credentials, upload/start action, or guest-selected host path is exposed.
"""
import base64
import binascii
import hashlib
import html
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import uuid
import zipfile

HOME=Path.home()
CAD_PROJECT=HOME/'Projects/piercing-gauge'
VENV=CAD_PROJECT/'.venv'
JOBS=HOME/'Projects/kronk-designs'
PUBLIC=Path('/srv/www/cat-bowl/kronk-designs')
SANDBOX_TIMEOUT=300
# systemd RuntimeMaxSec is a hard backstop just above the Python timeout. The
# Python timeout fires first so teardown is explicit and the failure message is
# actionable; RuntimeMaxSec guarantees the sandbox control group cannot outlive
# ~5.5 minutes even if the systemd-run client is killed or wedged.
SANDBOX_RUNTIME_LIMIT=SANDBOX_TIMEOUT+30
# The untrusted guest runs inside a transient user *service* whose cgroup is
# capped here. `TasksMax` bounds the whole guest process tree without touching
# the shared per-user RLIMIT_NPROC, so a fork loop cannot exhaust the budget the
# adapsis service itself forks from. Only the bwrap guest is inside this cgroup;
# font-cache warmup and trusted slicing run outside it.
SANDBOX_TASKS_MAX=64
SYSTEMD_RUN='/usr/bin/systemd-run'
SYSTEMCTL='/usr/bin/systemctl'
TRACEBACK_HEADER='Traceback (most recent call last):'
DETAIL_LIMIT=2000
GEOMETRY_ATTEMPTS=3
# Assembly placement is optional DATA from the guest. It is validated against the
# actually exported STL parts and a bounded author-declared envelope, then copied
# into the report/manifest as numbers only. It can never introduce host HTML/JS/SVG
# or untrusted paths: a rejected or absent layout degrades to print-parts-only.
ASSEMBLY_FILE='assembly.json'
ASSEMBLY_MAX_INSTANCES=64
ASSEMBLY_MAX_DIM_MM=1200.0
ASSEMBLY_MAX_BYTES=64_000
ASSEMBLY_TOLERANCE_MM=0.5
ASSEMBLY_KEYS={'units','envelope_mm','instances'}
MAX_REFERENCE_IMAGES=3
MAX_REFERENCE_BYTES=20*1024*1024
# The runner only ever executes absolute paths, so it needs no inheritable
# environment. This scrubbed env is what the *systemd-run client* gets; it no
# longer reaches the guest, because systemd gives the service the user manager's
# environment instead. The load-bearing /proc/1/environ scrub is the `env -i`
# layer inside the service (see sandbox_command).
RUNNER_ENV={'PATH':'/usr/bin:/bin'}
# The CAD venv bundles an old fontconfig (cadquery-ocp 7.9.3 -> libfontconfig 2.12).
# It cannot parse the generic-family constructs (xsi:nil / <const>) added in this
# node's newer fontconfig config files, so it floods stderr. The sandbox gets a
# conf.d copy without those files; every other rule (family aliases, rendering
# hints) is kept, so real text/engraving font matching is unchanged.
INCOMPATIBLE_FONTCONFIG={'48-guessfamily.conf','48-spacing.conf','49-sansserif.conf'}
# Benign sandbox-only noise that must never reach the surfaced error summary.
FONT_NOISE=re.compile(r"Fontconfig (?:error|warning):"
    r"|cannot open font '[^']*Font_Awesome_[^']*'"
    r"|skipping malformed name record")
# systemd-run writes its own status lines to the client's stderr, which we also
# capture. They are not guest output: left in, they would mask a real traceback
# or a signal death and turn 'terminated by signal 11' into systemd chatter. The
# client env has no LANG/LC_ALL, so these stay in the C locale (English) and are
# matchable. Do not let them reach failure_detail.
SYSTEMD_NOISE=re.compile(
    r"^\s*(?:Running as unit:"
    r"|Finished with result:"
    r"|Main processes terminated with:"
    r"|Service runtime:"
    r"|CPU time consumed:"
    r"|Failed to start transient service unit:"
    r"|Failed to connect to user scope bus)")

class SandboxUnavailable(Exception):
    """The untrusted sandbox cannot run with its required process cap.

    Distinct from RuntimeError/ValueError on purpose: prepare_geometry only
    retries those, and an unavailable user systemd manager must fail the job
    immediately rather than be mislabeled as a geometry failure.
    """

_SYSTEMD_REQUIRED_MESSAGE=(
    'CAD sandbox process cap requires a user systemd manager, but none is '
    'reachable ($XDG_RUNTIME_DIR unset or $XDG_RUNTIME_DIR/systemd/private '
    'missing, or the user bus probe failed). Refusing to run the untrusted sandbox without a process-count '
    'cap. Run adapsis as a user systemd service (systemctl --user) or make the '
    'user manager available on this host.')

def user_systemd_available():
    """Whether a transient unit can be created for the current user.

    The pids cap is a cgroup limit, so it requires a user systemd manager. This
    is a filesystem check only: systemd-run --user talks to the socket at
    $XDG_RUNTIME_DIR/systemd/private, and no subprocess is spawned.
    """
    if not (Path(SYSTEMD_RUN).exists() and Path(SYSTEMCTL).exists()):return False
    runtime=os.environ.get('XDG_RUNTIME_DIR')
    if not runtime or not Path(runtime,'systemd','private').exists():return False
    try:
        result=subprocess.run([SYSTEMCTL,'--user','show-environment'],
            env=_systemd_client_env(),stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,timeout=10)
    except (OSError,subprocess.TimeoutExpired):return False
    return result.returncode==0

def _require_user_systemd():
    if not user_systemd_available():raise SandboxUnavailable(_SYSTEMD_REQUIRED_MESSAGE)

def _systemd_client_env():
    """Environment for the systemd-run/systemctl clients.

    They need XDG_RUNTIME_DIR to locate the user manager; everything else stays
    PATH-only. This never leaks to the guest — the service gets the manager's
    environment, which sandbox_command scrubs with `env -i`.
    """
    env=dict(RUNNER_ENV)
    for name in ('XDG_RUNTIME_DIR','DBUS_SESSION_BUS_ADDRESS'):
        value=os.environ.get(name)
        if value:env[name]=value
    return env

def sandbox_unit_name():
    """A unique, namespaced transient unit name for one guest execution."""
    return 'adapsis-cad-sandbox-'+uuid.uuid4().hex

def _strip_runner_noise(text):
    return '\n'.join(line for line in text.splitlines()
        if not FONT_NOISE.search(line) and not SYSTEMD_NOISE.match(line)).strip()

def _header_only(detail):
    return not detail or detail.strip()==TRACEBACK_HEADER

def _bound_detail(detail,limit):
    if len(detail)<=limit:return detail
    head=(TRACEBACK_HEADER+'\n') if detail.startswith(TRACEBACK_HEADER) else ''
    return head+detail[-(limit-len(head)):]

def _signal_from_returncode(returncode):
    if returncode<0:return -returncode
    if returncode>128:return returncode-128
    return None

_SIGNAL_HINTS={6:'aborted',9:'killed (sandbox CPU/memory limit)',11:'native crash (segfault)',24:'sandbox CPU limit'}

def _signal_message(signal):
    hint=_SIGNAL_HINTS.get(signal)
    return f'guest process terminated by signal {signal}{" ("+hint+")" if hint else ""} without a Python exception'

def store_references(job,raw):
    """Decode the trusted stdin envelope into private files outside the sandbox.

    Only metadata is returned for persistence. Paths are kept in memory for the
    Astra transport and are never written to request/report/public manifests.
    """
    if raw is None:raw=[]
    if not isinstance(raw,list) or len(raw)>MAX_REFERENCE_IMAGES:
        raise ValueError(f'Reference images must be a list of at most {MAX_REFERENCE_IMAGES}')
    references=[];total=0
    directory=job/'references'
    for index,item in enumerate(raw):
        if not isinstance(item,dict) or not isinstance(item.get('mime'),str) or not isinstance(item.get('data_base64'),str):
            raise ValueError('Each reference image requires MIME type and base64 data')
        mime=item['mime'].split(';',1)[0].strip().lower()
        if not mime.startswith('image/'):
            raise ValueError('Reference MIME type must be an image')
        try:data=base64.b64decode(item['data_base64'],validate=True)
        except (binascii.Error,ValueError) as error:
            raise ValueError('Reference image contains invalid base64 data') from error
        total+=len(data)
        if not data or total>MAX_REFERENCE_BYTES:
            raise ValueError(f'Reference images must be non-empty and total at most {MAX_REFERENCE_BYTES} bytes')
        digest=hashlib.sha256(data).hexdigest()
        directory.mkdir(mode=0o700,exist_ok=True)
        path=directory/f'{index+1}-{digest}'
        path.write_bytes(data)
        path.chmod(0o600)
        references.append({'sha256':digest,'mime':mime,'size':len(data),'path':str(path)})
    return references

def reference_metadata(references):
    return [{key:reference[key] for key in ('sha256','mime','size')} for reference in references]

def delete_reference_bytes(job,references):
    """Remove only private files created under this job's reference directory."""
    directory=job/'references'
    try:resolved_directory=directory.resolve()
    except OSError:return
    for reference in references:
        path=Path(reference.get('path',''))
        try:
            if path.parent.resolve()!=resolved_directory:continue
            path.unlink(missing_ok=True)
        except OSError:pass
    try:directory.rmdir()
    except OSError:pass

def failure_detail(stderr,limit=DETAIL_LIMIT):
    """Final Python traceback from guest stderr, with benign font noise removed and
    the result bounded to `limit` chars while keeping the header and the final
    exception line."""
    text=stderr or ''
    index=text.rfind(TRACEBACK_HEADER)
    block=text[index:] if index!=-1 else text[-limit:]
    detail=_strip_runner_noise(block)
    if _header_only(detail):detail=''
    if not detail:
        detail=_strip_runner_noise(text)[-limit:]
        if _header_only(detail):detail=''
    return _bound_detail(detail,limit)

def _sandbox_fontconfig(sandbox,source_dir='/etc/fonts/conf.d'):
    source=Path(source_dir)
    if not source.is_dir():return None
    target=sandbox.parent/'fontconfig-conf.d'
    if target.exists():shutil.rmtree(target)
    target.mkdir(parents=True)
    for entry in sorted(source.iterdir()):
        if entry.suffix!='.conf' or entry.name in INCOMPATIBLE_FONTCONFIG:continue
        resolved=entry.resolve()
        if resolved.is_file():shutil.copy(resolved,target/entry.name)
    return target

def _warm_font_cache(sandbox):
    """Pre-seed ezdxf's font cache so its cold-start scan does not log dangling
    Font Awesome 6 symlinks and malformed name records. Best effort only: the
    summary filter also removes that noise if generation is unavailable."""
    cache=sandbox/'.cache'
    (cache/'ezdxf').mkdir(parents=True,exist_ok=True)
    try:
        subprocess.run([str(VENV/'bin/python'),'-c','from ezdxf.fonts import fonts; fonts.load()'],
            env={**os.environ,'XDG_CACHE_HOME':str(cache)},stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,timeout=120,check=True)
    except Exception:pass

def _sandbox_isolation_command(sandbox):
    # NOTE: `--as-pid-1` is deliberately absent. It would make the guest the
    # PID-namespace init, which (a) removes bwrap's reaper so a guest looping
    # fork()/_exit(0) accumulates zombies, and (b) is redundant — bwrap stays
    # PID 1 under the service and reaps, and the service's env is scrubbed by
    # `env -i` in sandbox_command. Do not "helpfully" re-add it.
    command=['/usr/bin/prlimit','--as=8589934592','--cpu=240','--fsize=314572800',
        '/usr/bin/bwrap','--unshare-all','--die-with-parent','--new-session','--clearenv',
        '--ro-bind','/usr','/usr','--symlink','usr/bin','/bin',
        '--symlink','usr/lib','/lib','--symlink','usr/lib64','/lib64',
        '--proc','/proc','--dev','/dev','--tmpfs','/tmp','--dir','/home',
        '--ro-bind',str(VENV),'/env','--bind',str(sandbox),'/job','--chdir','/job',
        '--setenv','HOME','/tmp','--setenv','PATH','/env/bin:/usr/bin',
        '--setenv','OPENBLAS_NUM_THREADS','1','--setenv','OMP_NUM_THREADS','1',
        '--setenv','XDG_CACHE_HOME','/job/.cache']
    for path in ('/etc/ld.so.cache','/etc/fonts','/var/cache/fontconfig'):
        if Path(path).exists():command.extend(['--ro-bind',path,path])
    conf_d=_sandbox_fontconfig(sandbox)
    if conf_d is not None:command.extend(['--ro-bind',str(conf_d),'/etc/fonts/conf.d'])
    return command+['/env/bin/python','/job/source.py']

def sandbox_command(sandbox,unit):
    """Wrap the bwrap guest in a transient user systemd *service* that caps its
    process tree.

    A service (not a scope) gets a private cgroup, so `TasksMax` bounds the
    guest's pids without touching the shared per-user RLIMIT_NPROC, and
    `KillMode=control-group` reaps the whole tree on stop. `--wait --collect
    --pipe` forwards stdio and the exit status and garbage-collects the unit.

    The `env -i PATH=...` layer is load-bearing: systemd starts the service with
    the *user manager's* environment (which holds API keys), so a PATH-only
    client environment is no longer enough to keep /proc/1/environ clean. Loading
    that scrub back into RUNNER_ENV, or dropping it, would reintroduce the leak.
    """
    return [SYSTEMD_RUN,'--user','--wait','--collect','--pipe',f'--unit={unit}',
        '-p',f'TasksMax={SANDBOX_TASKS_MAX}','-p','KillMode=control-group',
        '-p',f'RuntimeMaxSec={SANDBOX_RUNTIME_LIMIT}',
        '--','/usr/bin/env','-i','PATH=/usr/bin:/bin',
        *_sandbox_isolation_command(sandbox)]

def _teardown_unit(unit):
    """SIGKILL a transient sandbox unit's whole control group and let --collect
    reap it. Tolerates the unit never having started or already having exited,
    where systemctl reports 'not loaded'."""
    try:
        subprocess.run([SYSTEMCTL,'--user','kill','--kill-whom=all','--signal=SIGKILL',
            f'{unit}.service'],env=_systemd_client_env(),stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,timeout=30)
    except Exception:pass

def run_source(job,source):
    _require_user_systemd()
    sandbox=job/'sandbox';sandbox.mkdir()
    (sandbox/'source.py').write_text(source)
    (sandbox/'output').mkdir()
    _warm_font_cache(sandbox)
    unit=sandbox_unit_name()
    with (job/'model.log').open('wb') as log:
        try:
            result=subprocess.run(sandbox_command(sandbox,unit),stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,timeout=SANDBOX_TIMEOUT,env=_systemd_client_env())
        except subprocess.TimeoutExpired as error:
            log.write(b'== stdout ==\n');log.write(error.stdout or b'')
            log.write(b'\n== stderr ==\n');log.write(error.stderr or b'')
            _teardown_unit(unit)
            raise RuntimeError(f'Model script exceeded the {SANDBOX_TIMEOUT}s sandbox limit without finishing') from error
        except OSError as error:
            _teardown_unit(unit)
            raise SandboxUnavailable(
                f'CAD sandbox systemd service could not be started: {error}'
            ) from error
        except BaseException:
            # Any other teardown path (launch failure, interrupt, ...) must not
            # leave the transient unit's process tree running.
            _teardown_unit(unit)
            raise
        log.write(b'== stdout ==\n');log.write(result.stdout)
        log.write(b'\n== stderr ==\n');log.write(result.stderr)
    if result.returncode:
        raw_stderr=result.stderr.decode(errors='replace')
        if ('Failed to start transient service unit:' in raw_stderr
                or 'Failed to connect to user scope bus' in raw_stderr):
            raise SandboxUnavailable(
                'CAD sandbox systemd service could not be started: '
                +_bound_detail(raw_stderr.strip(),DETAIL_LIMIT))
        detail=failure_detail(raw_stderr)
        if not detail:
            detail=result.stdout.decode(errors='replace').strip()[-2000:]
        if not detail:
            signal=_signal_from_returncode(result.returncode)
            detail=_signal_message(signal) if signal else 'no output'
        raise RuntimeError(f'Model script failed: {detail}')

def validate_exports(job):
    import numpy as np
    import trimesh
    output=job/'sandbox/output'
    if output.is_symlink() or not output.is_dir():raise ValueError('Output directory must not be a symlink')
    candidates=sorted(output.glob('*.stl'))
    if not 1<=len(candidates)<=8:raise ValueError('Export 1–8 STL parts to /job/output/')
    scene=trimesh.Scene();parts=[]
    for path in candidates:
        if path.is_symlink() or not re.fullmatch(r'[a-zA-Z0-9_-]+\.stl',path.name):
            raise ValueError('Use simple regular STL filenames, no symlinks')
        if path.stat().st_size>100_000_000:raise ValueError('STL exceeds 100MB')
        mesh=trimesh.load_mesh(path)
        components=len(mesh.split(only_watertight=False))
        if not mesh.is_watertight or not mesh.is_winding_consistent or components!=1:
            raise ValueError(f'{path.name}: mesh must be one closed consistently oriented solid')
        if mesh.volume<=0 or not np.isfinite(mesh.vertices).all():raise ValueError('Invalid mesh')
        if abs(mesh.bounds[0,2])>.05:raise ValueError(f'{path.name}: place flat print base at Z=0')
        if any(mesh.extents>np.array([240,240,250])):raise ValueError(f'{path.name}: too large for A1 planning bounds')
        scene.add_geometry(mesh,node_name=path.stem,geom_name=path.stem)
        parts.append({'file':path.name,'size_mm':mesh.extents.tolist(),
                      'volume_mm3':mesh.volume,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    (job/'layout.3mf').write_bytes(scene.export(file_type='3mf'))
    return parts

class AssemblyError(ValueError):pass

def _finite_number(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)

def _validate_envelope(value):
    if not isinstance(value,list) or len(value)!=3:
        raise AssemblyError('envelope_mm must list [x, y, z] in millimetres')
    dimensions=[]
    for dimension in value:
        if not _finite_number(dimension) or dimension<=0 or dimension>ASSEMBLY_MAX_DIM_MM:
            raise AssemblyError(f'envelope_mm must be three positive finite millimetres no larger than {ASSEMBLY_MAX_DIM_MM:g}')
        dimensions.append(float(dimension))
    return dimensions

def _validate_transform(value):
    if not isinstance(value,list) or len(value)!=4 or any(not isinstance(row,list) or len(row)!=4 for row in value):
        raise AssemblyError('each transform must be a 4x4 numeric matrix')
    if any(not _finite_number(number) for row in value for number in row):
        raise AssemblyError('each transform must contain only finite numbers')
    matrix=[[float(number) for number in row] for row in value]
    if any(abs(matrix[3][column]-expected)>1e-6 for column,expected in enumerate((0.0,0.0,0.0,1.0))):
        raise AssemblyError('each transform must be affine (last row 0 0 0 1)')
    for first in range(3):
        for second in range(3):
            dot=sum(matrix[k][first]*matrix[k][second] for k in range(3))
            if abs(dot-(1.0 if first==second else 0.0))>1e-4:
                raise AssemblyError('each transform rotation must be orthonormal (no scale or shear)')
    determinant=(matrix[0][0]*(matrix[1][1]*matrix[2][2]-matrix[1][2]*matrix[2][1])
        -matrix[0][1]*(matrix[1][0]*matrix[2][2]-matrix[1][2]*matrix[2][0])
        +matrix[0][2]*(matrix[1][0]*matrix[2][1]-matrix[1][1]*matrix[2][0]))
    if abs(abs(determinant)-1.0)>1e-4:
        raise AssemblyError('each transform must be a rigid rotation or reflection')
    if any(abs(matrix[row][3])>ASSEMBLY_MAX_DIM_MM for row in range(3)):
        raise AssemblyError('transform translation is outside the allowed envelope')
    return matrix

def _validate_instance(value,part_names):
    if not isinstance(value,dict) or set(value)!={'part','transform'}:
        raise AssemblyError('each instance must be an object with only "part" and "transform"')
    part=value['part']
    if not isinstance(part,str) or part not in part_names:
        raise AssemblyError('instance part is not one of the exported STL parts')
    return {'part':part,'transform':_validate_transform(value['transform'])}

def _assembled_size(output,instances):
    import numpy as np
    import trimesh
    meshes={};low=high=None
    for instance in instances:
        name=instance['part']
        if name not in meshes:meshes[name]=trimesh.load_mesh(output/name)
        mesh=meshes[name].copy()
        mesh.apply_transform(np.array(instance['transform'],dtype=float))
        bounds=mesh.bounds
        if low is None:low,high=bounds[0].copy(),bounds[1].copy()
        else:low=np.minimum(low,bounds[0]);high=np.maximum(high,bounds[1])
    return (high-low).tolist()

def _validated_assembly(output,parts):
    path=output/ASSEMBLY_FILE
    if path.is_symlink() or not path.is_file():raise AssemblyError('assembly.json must be a regular file, not a symlink')
    if path.stat().st_size>ASSEMBLY_MAX_BYTES:raise AssemblyError('assembly.json is too large')
    try:raw=json.loads(path.read_text())
    except (OSError,UnicodeDecodeError,json.JSONDecodeError) as error:
        raise AssemblyError('assembly.json is not valid JSON') from error
    if not isinstance(raw,dict):raise AssemblyError('assembly.json must be a JSON object')
    unknown=set(raw)-ASSEMBLY_KEYS
    if unknown:raise AssemblyError('assembly.json has unexpected fields')
    if 'envelope_mm' not in raw or 'instances' not in raw:
        raise AssemblyError('assembly.json must declare envelope_mm and instances')
    if raw.get('units','mm')!='mm':raise AssemblyError('assembly.json units must be "mm"')
    envelope=_validate_envelope(raw['envelope_mm'])
    instances=raw['instances']
    if not isinstance(instances,list) or not 1<=len(instances)<=ASSEMBLY_MAX_INSTANCES:
        raise AssemblyError(f'instances must be a list of 1–{ASSEMBLY_MAX_INSTANCES} part placements')
    part_names={part['file'] for part in parts}
    checked=[_validate_instance(instance,part_names) for instance in instances]
    size=_assembled_size(output,checked)
    if any(not math.isfinite(dimension) or dimension<=0 for dimension in size):
        raise AssemblyError('assembled parts have an empty or non-finite size')
    if any(dimension>envelope[index]+ASSEMBLY_TOLERANCE_MM for index,dimension in enumerate(size)):
        raise AssemblyError(f'assembled size {[round(d,1) for d in size]} mm exceeds the declared envelope {envelope} mm')
    return {'available':True,'units':'mm','envelope_mm':envelope,'instances':checked,
            'instance_count':len(checked),'assembled_size_mm':[round(d,3) for d in size]}

def validate_assembly(job,parts):
    """Return validated, bounded placement data or a print-parts fallback.

    Never raises: an absent or rejected assembly.json must degrade to the
    print-parts view with a visible reason instead of failing an otherwise valid job.
    """
    output=job/'sandbox/output'
    path=output/ASSEMBLY_FILE
    if not path.exists() and not path.is_symlink():
        return {'available':False,'reason':'No assembly layout was supplied; showing print parts only.'}
    try:
        return _validated_assembly(output,parts)
    except AssemblyError as error:
        return {'available':False,'reason':f'Assembly layout rejected ({error}); showing print parts only.'}
    except Exception as error:
        return {'available':False,'reason':f'Assembly layout could not be checked ({type(error).__name__}); showing print parts only.'}

def _prepare_geometry(job,title,brief,material,references=(),previous_source=None,failure=None):
    """Feed real execution/mesh failures back to Astra, with isolated attempts.

    Author-service failures are not geometry failures and do not enter this loop.
    Failed files stay private; only a fully checked attempt is promoted for slicing.
    """
    from cad_astra import generate
    for index in range(1,GEOMETRY_ATTEMPTS+1):
        attempt=job/f'attempt-{index}';attempt.mkdir()
        astra_references=[{'mime':reference['mime'],'path':reference['path']} for reference in references]
        source,author=generate(title,brief,material,references=astra_references,
            previous_source=previous_source,failure=failure)
        (attempt/'cad-author.json').write_text(json.dumps(author,indent=2)+'\n')
        try:
            run_source(attempt,source)
            parts=validate_exports(attempt)
        except (RuntimeError,ValueError) as error:
            failure=str(error)
            (attempt/'failure.txt').write_text(failure+'\n')
            if index==GEOMETRY_ATTEMPTS:
                raise RuntimeError(f'CAD geometry failed after {index} checked attempts: {failure}') from error
            previous_source=source
            continue
        assembly=validate_assembly(attempt,parts)
        (attempt/'sandbox').rename(job/'sandbox')
        shutil.copyfile(attempt/'layout.3mf',job/'layout.3mf')
        author['geometry_attempts']=index
        (job/'cad-author.json').write_text(json.dumps(author,indent=2)+'\n')
        return parts,author,assembly

def prepare_geometry(job,title,brief,material,references=(),previous_source=None,failure=None):
    """Author and validate geometry, then erase private reference image bytes."""
    try:
        return _prepare_geometry(job,title,brief,material,references,previous_source,failure)
    finally:
        delete_reference_bytes(job,references)

def slice_job(job,material):
    profiles=Path(os.environ.get('ADAPSIS_CAD_PROFILES',str(HOME/'.local/share/cad-pipeline/squashfs-root/resources/profiles/BBL')))
    if not profiles.is_dir():raise RuntimeError('Official Bambu profiles missing; administrator must extract installed AppImage resources first')
    command=[str(VENV/'bin/python'),str(CAD_PROJECT/'prepare_slice.py'),
        '--profiles',str(profiles),'--studio',str(HOME/'.local/opt/BambuStudio-2.8.2.AppImage'),
        '--input',str(job/'layout.3mf'),'--output',str(job/'sliced'),
        '--filename','prepared-a1.3mf','--arrange','1','--filament',f'Bambu {material} Basic @BBL A1']
    with (job/'slicer.log').open('w') as log:
        result=subprocess.run(command,stdout=log,stderr=log,timeout=600)
    if result.returncode:raise RuntimeError('Slicing failed; administrator can inspect slicer.log')
    result=json.loads((job/'sliced/result.json').read_text())
    if result['return_code']!=0 or not result.get('sliced_plates'):raise ValueError('No successful sliced plates')
    warnings=[p.get('warning_message') for p in result['sliced_plates'] if p.get('warning_message')]
    if warnings:raise ValueError(f'Slicer reported warnings: {warnings}')
    with zipfile.ZipFile(job/'sliced/prepared-a1.3mf') as archive:
        if archive.testzip():raise ValueError('Corrupt sliced archive')
        settings=json.loads(archive.read('Metadata/project_settings.config'))
        if settings['printer_model']!='Bambu Lab A1' or settings['filament_type']!=[material]:
            raise ValueError('Slicer machine/material mismatch')
        codes=[n for n in archive.namelist() if re.fullmatch(r'Metadata/plate_\d+\.gcode',n)]
        if len(codes)!=len(result['sliced_plates']):raise ValueError('Missing plate G-code')
        for name in codes:
            if hashlib.md5(archive.read(name)).hexdigest()!=archive.read(name+'.md5').decode().strip().lower():
                raise ValueError('G-code checksum mismatch')
    return result

def publish(job,report):
    # Never copy source.py, logs, private metadata or arbitrary output directory.
    destination=PUBLIC/job.name;destination.mkdir(parents=True,exist_ok=False)
    for part in report['parts']:
        shutil.copyfile(job/'sandbox/output'/part['file'],destination/part['file'])
        step=(job/'sandbox/output'/part['file']).with_suffix('.step')
        if step.exists() and not step.is_symlink() and step.stat().st_size<100_000_000:
            shutil.copyfile(step,destination/step.name)
    shutil.copyfile(job/'sliced/prepared-a1.3mf',destination/'prepared-a1.3mf')
    public={k:v for k,v in report.items() if k not in ('job_dir',)}
    (destination/'manifest.json').write_text(json.dumps(public,indent=2))
    page=Path(__file__).with_name('cad_preview.html').read_text()
    (destination/'index.html').write_text(page.replace('__TITLE__',html.escape(report['title'])))

def main():
    request=json.load(sys.stdin)
    title=request['title'];brief=request.get('brief',request.get('source'));material=request.get('material','PLA')
    if not isinstance(title,str) or not 0<len(title)<=300:raise ValueError('Invalid title')
    if not isinstance(brief,str) or not 0<len(brief)<=64000:raise ValueError('Invalid design brief')
    if material not in ('PLA','PETG'):raise ValueError('Invalid material')
    # This must precede `cad_astra.generate`: authoring is paid/off-host, while
    # an unavailable mandatory sandbox makes every generated source unusable.
    _require_user_systemd()
    JOBS.mkdir(exist_ok=True);job=JOBS/('design-'+uuid.uuid4().hex[:16]);job.mkdir()
    references=store_references(job,request.get('references',[]))
    (job/'request.json').write_text(json.dumps({'title':title,'brief':brief,'material':material,
        'references':reference_metadata(references)},indent=2)+'\n')
    parts,author,assembly=prepare_geometry(job,title,brief,material,references=references)
    result=slice_job(job,material)
    plates=result['sliced_plates']
    report={'passed':True,'title':title,'job_dir':str(job),'parts':parts,'assembly':assembly,
        'material':material,'cad_model':author['requested_model'],
        'reference_images':len(references),
        'plates':len(plates),'minutes':round(sum(p['total_predication'] for p in plates)/60,1),
        'grams':round(sum(f['total_used_g'] for p in plates for f in p['filaments']),2),
        'url':f'https://home.mare.nz/cat-bowl/kronk-designs/{job.name}/',
        'print_approval':'Required from Marenz; not started',
        'limitations':'Mesh/slicer checks only; support strategy, physical fit and any anatomical claims require review'}
    (job/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    publish(job,report);print(json.dumps(report))

if __name__=='__main__':
    try:main()
    except Exception as error:
        print(str(error),file=sys.stderr);sys.exit(1)
