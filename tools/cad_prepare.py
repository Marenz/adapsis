#!/usr/bin/env python3
"""Trusted prepare-only runner. Untrusted model Python executes inside bubblewrap.

stdin: {title, brief, material}; stdout: checked job report. No printer API,
credentials, upload/start action, or guest-selected host path is exposed.
"""
import hashlib
import html
import json
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
TRACEBACK_HEADER='Traceback (most recent call last):'
DETAIL_LIMIT=2000
GEOMETRY_ATTEMPTS=3
# The runner only ever executes absolute paths, so it needs no inheritable
# environment. This scrubbed env is the load-bearing fix for the /proc/1/environ
# leak: bwrap is PID 1 in the sandbox and inherits the runner's environ, so a
# PATH-only env means guest code reading /proc/1/environ finds no secrets.
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

def _strip_font_noise(text):
    return '\n'.join(line for line in text.splitlines() if not FONT_NOISE.search(line)).strip()

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

def failure_detail(stderr,limit=DETAIL_LIMIT):
    """Final Python traceback from guest stderr, with benign font noise removed and
    the result bounded to `limit` chars while keeping the header and the final
    exception line."""
    text=stderr or ''
    index=text.rfind(TRACEBACK_HEADER)
    block=text[index:] if index!=-1 else text[-limit:]
    detail=_strip_font_noise(block)
    if _header_only(detail):detail=''
    if not detail:
        detail=_strip_font_noise(text)[-limit:]
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

def sandbox_command(sandbox):
    # NOTE: `--as-pid-1` is deliberately absent. It would make the guest the
    # PID-namespace init, which (a) removes bwrap's reaper so a guest looping
    # fork()/_exit(0) accumulates zombies against the shared user pids.max, and
    # (b) is redundant — RUNNER_ENV above already leaves bwrap's inherited
    # environ PATH-only, so /proc/1/environ exposes nothing regardless of which
    # process is PID 1. Do not "helpfully" re-add it.
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

def run_source(job,source):
    sandbox=job/'sandbox';sandbox.mkdir()
    (sandbox/'source.py').write_text(source)
    (sandbox/'output').mkdir()
    _warm_font_cache(sandbox)
    with (job/'model.log').open('wb') as log:
        try:
            result=subprocess.run(sandbox_command(sandbox),stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,timeout=SANDBOX_TIMEOUT,env=dict(RUNNER_ENV))
        except subprocess.TimeoutExpired as error:
            log.write(b'== stdout ==\n');log.write(error.stdout or b'')
            log.write(b'\n== stderr ==\n');log.write(error.stderr or b'')
            raise RuntimeError(f'Model script exceeded the {SANDBOX_TIMEOUT}s sandbox limit without finishing') from error
        log.write(b'== stdout ==\n');log.write(result.stdout)
        log.write(b'\n== stderr ==\n');log.write(result.stderr)
    if result.returncode:
        detail=failure_detail(result.stderr.decode(errors='replace'))
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

def prepare_geometry(job,title,brief,material,previous_source=None,failure=None):
    """Feed real execution/mesh failures back to Astra, with isolated attempts.

    Author-service failures are not geometry failures and do not enter this loop.
    Failed files stay private; only a fully checked attempt is promoted for slicing.
    """
    from cad_astra import generate
    for index in range(1,GEOMETRY_ATTEMPTS+1):
        attempt=job/f'attempt-{index}';attempt.mkdir()
        source,author=generate(title,brief,material,previous_source,failure)
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
        (attempt/'sandbox').rename(job/'sandbox')
        shutil.copyfile(attempt/'layout.3mf',job/'layout.3mf')
        author['geometry_attempts']=index
        (job/'cad-author.json').write_text(json.dumps(author,indent=2)+'\n')
        return parts,author

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
    JOBS.mkdir(exist_ok=True);job=JOBS/('design-'+uuid.uuid4().hex[:16]);job.mkdir()
    (job/'request.json').write_text(json.dumps({'title':title,'brief':brief,'material':material},indent=2)+'\n')
    parts,author=prepare_geometry(job,title,brief,material)
    result=slice_job(job,material)
    plates=result['sliced_plates']
    report={'passed':True,'title':title,'job_dir':str(job),'parts':parts,'material':material,'cad_model':author['requested_model'],
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
