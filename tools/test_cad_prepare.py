import base64
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import unittest
import uuid
from pathlib import Path
from unittest.mock import Mock, patch
import cad_prepare as cad

def _sandbox_prereqs_available():
    return (Path('/usr/bin/bwrap').exists() and Path('/usr/bin/prlimit').exists()
        and (Path.home()/'Projects/piercing-gauge/.venv/bin/python').exists()
        and cad.user_systemd_available())

def _identity():return [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]

def _write_box(path,extents,translate=(0,0,0)):
    import trimesh
    mesh=trimesh.creation.box(extents=extents)
    mesh.apply_translation(translate)
    path.write_bytes(mesh.export(file_type='stl'))

class CadPreparationTests(unittest.TestCase):
    def test_geometry_failure_is_revised_in_clean_sandbox(self):
        def run(attempt,source):
            (attempt/'sandbox/output').mkdir(parents=True)
            (attempt/'sandbox/output/model.stl').write_text(source)
            if source=='bad':raise RuntimeError('rack-left: 11 components')
            (attempt/'layout.3mf').write_text('checked')
        author={'requested_model':'chatgpt/gpt-6-astra'}
        references=[{'sha256':'abc','mime':'image/png','size':1,'path':'/private/reference'}]
        with tempfile.TemporaryDirectory() as d:
            job=Path(d)
            with patch('cad_astra.generate',side_effect=[('bad',dict(author)),('good',dict(author))]) as generate, \
                 patch('cad_prepare.run_source',side_effect=run), \
                 patch('cad_prepare.validate_exports',return_value=[{'file':'model.stl'}]):
                parts,result,assembly=cad.prepare_geometry(job,'rack','20 pegs','PLA',references=references)
            self.assertEqual(generate.call_args.args,('rack','20 pegs','PLA'))
            self.assertEqual(generate.call_args.kwargs['previous_source'],'bad')
            self.assertEqual(generate.call_args.kwargs['failure'],'rack-left: 11 components')
            for call in generate.call_args_list:
                self.assertEqual(call.kwargs['references'],[{'mime':'image/png','path':'/private/reference'}])
            self.assertEqual((job/'sandbox/output/model.stl').read_text(),'good')
            self.assertEqual((job/'attempt-1/sandbox/output/model.stl').read_text(),'bad')
            self.assertEqual(result['geometry_attempts'],2)
            self.assertFalse(assembly['available'])

    def test_failed_geometry_never_promoted_and_retry_count_bounded(self):
        with tempfile.TemporaryDirectory() as d:
            job=Path(d)
            with patch('cad_astra.generate',return_value=('bad',{})) as generate, \
                 patch('cad_prepare.run_source',side_effect=RuntimeError('invalid solid')):
                with self.assertRaisesRegex(RuntimeError,'after 3 checked attempts'):
                    cad.prepare_geometry(job,'rack','brief','PLA')
            self.assertEqual(generate.call_count,3)
            self.assertFalse((job/'sandbox').exists())

    def test_author_failure_is_not_retried_as_geometry(self):
        with tempfile.TemporaryDirectory() as d:
            with patch('cad_astra.generate',side_effect=RuntimeError('must be Astra')) as generate:
                with self.assertRaisesRegex(RuntimeError,'must be Astra'):
                    cad.prepare_geometry(Path(d),'rack','brief','PLA')
            self.assertEqual(generate.call_count,1)

    def test_host_mesh_validation_failure_is_sent_to_astra(self):
        with tempfile.TemporaryDirectory() as d:
            job=Path(d)
            def run(attempt,source):
                (attempt/'sandbox').mkdir()
                (attempt/'layout.3mf').write_text('layout')
            with patch('cad_astra.generate',return_value=('source',{})) as generate, \
                 patch('cad_prepare.run_source',side_effect=run), \
                 patch('cad_prepare.validate_exports',side_effect=[ValueError('mesh must be one closed solid'),[]]):
                cad.prepare_geometry(job,'rack','brief','PLA')
            self.assertEqual(generate.call_args.kwargs['previous_source'],'source')
            self.assertEqual(generate.call_args.kwargs['failure'],'mesh must be one closed solid')

    def test_reference_request_metadata_contains_no_private_bytes(self):
        image=b'\x89PNG\r\n\x1a\nprivate pixels'
        raw=[{'mime':'image/png','data_base64':base64.b64encode(image).decode('ascii')}]
        with tempfile.TemporaryDirectory() as directory:
            job=Path(directory)
            references=cad.store_references(job,raw)
            metadata=cad.reference_metadata(references)
            serialized=json.dumps(metadata)
            self.assertEqual(metadata,[{'sha256':hashlib.sha256(image).hexdigest(),
                'mime':'image/png','size':len(image)}])
            self.assertNotIn('base64',serialized)
            self.assertNotIn('private pixels',serialized)
            self.assertNotIn('path',serialized)
            self.assertEqual(Path(references[0]['path']).parent,job/'references')
            self.assertFalse((job/'sandbox').exists())

    def test_reference_bytes_are_deleted_after_successful_authoring(self):
        image=b'private pixels'
        raw=[{'mime':'image/png','data_base64':base64.b64encode(image).decode('ascii')}]
        with tempfile.TemporaryDirectory() as directory:
            job=Path(directory);references=cad.store_references(job,raw)
            path=Path(references[0]['path'])
            def run(attempt,source):
                (attempt/'sandbox/output').mkdir(parents=True)
                (attempt/'layout.3mf').write_text('checked')
            with patch('cad_astra.generate',return_value=('source',{})), \
                 patch('cad_prepare.run_source',side_effect=run), \
                 patch('cad_prepare.validate_exports',return_value=[]):
                cad.prepare_geometry(job,'rack','brief','PLA',references=references)
            self.assertFalse(path.exists())
            self.assertFalse((job/'references').exists())

    def test_reference_bytes_are_deleted_after_author_failure(self):
        image=b'private pixels'
        raw=[{'mime':'image/png','data_base64':base64.b64encode(image).decode('ascii')}]
        with tempfile.TemporaryDirectory() as directory:
            job=Path(directory);references=cad.store_references(job,raw)
            path=Path(references[0]['path'])
            with patch('cad_astra.generate',side_effect=RuntimeError('author unavailable')):
                with self.assertRaisesRegex(RuntimeError,'author unavailable'):
                    cad.prepare_geometry(job,'rack','brief','PLA',references=references)
            self.assertFalse(path.exists())
            self.assertFalse((job/'references').exists())

    def test_publish_allowlist_excludes_private_references(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);job=root/'design-test';public=root/'public'
            (job/'sandbox/output').mkdir(parents=True)
            (job/'sandbox/output/model.stl').write_bytes(b'stl')
            (job/'sliced').mkdir();(job/'sliced/prepared-a1.3mf').write_bytes(b'3mf')
            (job/'references').mkdir();(job/'references/private-image').write_bytes(b'private pixels')
            report={'passed':True,'title':'test','job_dir':str(job),
                'parts':[{'file':'model.stl'}],'reference_images':1}
            with patch('cad_prepare.PUBLIC',public):cad.publish(job,report)
            destination=public/job.name
            self.assertEqual({path.name for path in destination.iterdir()},
                {'model.stl','prepared-a1.3mf','manifest.json','index.html'})
            manifest=(destination/'manifest.json').read_text()
            self.assertNotIn('private pixels',manifest)
            self.assertNotIn('private-image',manifest)

    def test_generated_code_cannot_read_home_or_connect_to_host(self):
        source='''import os, socket
assert not os.path.exists('/home/marenz/.config/cat-bowl/buddy.json')
assert 'HOME' in os.environ and os.environ['HOME']=='/tmp'
s=socket.socket();s.settimeout(.2)
try:
    s.connect(('127.0.0.1',3002))
except OSError:
    pass
else:
    raise RuntimeError('Host API unexpectedly reachable')
import build123d as b
shape=b.Box(12,10,3,align=(b.Align.CENTER,b.Align.CENTER,b.Align.MIN))
b.export_stl(shape,'/job/output/model.stl')
'''
        with tempfile.TemporaryDirectory() as d:
            job=Path(d);cad.run_source(job,source)
            parts=cad.validate_exports(job)
            self.assertEqual(len(parts),1)
            self.assertAlmostEqual(parts[0]['volume_mm3'],360,places=2)

    def test_symlinks_are_not_exported(self):
        with tempfile.TemporaryDirectory() as d:
            job=Path(d);(job/'sandbox/output').mkdir(parents=True)
            (job/'sandbox/output/model.stl').symlink_to('/etc/passwd')
            with self.assertRaisesRegex(ValueError,'symlinks'):cad.validate_exports(job)

    def test_floating_model_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            job=Path(d)
            cad.run_source(job,"import build123d as b\nb.export_stl(b.Pos(0,0,20)*b.Box(4,4,4),'/job/output/model.stl')")
            with self.assertRaisesRegex(ValueError,'Z=0'):cad.validate_exports(job)

    def test_output_directory_cannot_redirect_trusted_host_processing(self):
        with tempfile.TemporaryDirectory() as d:
            job=Path(d);(job/'sandbox').mkdir()
            (job/'sandbox/output').symlink_to('/home/marenz')
            with self.assertRaisesRegex(ValueError,'symlink'):cad.validate_exports(job)

    def test_failure_detail_keeps_traceback_and_drops_font_noise(self):
        stderr='''Fontconfig error: "/etc/fonts/conf.d/48-guessfamily.conf", line 196: invalid constant used : ui-rounded
Fontconfig warning: "/etc/fonts/conf.d/49-sansserif.conf", line 42: invalid constant used : math
cannot open font '/usr/share/fonts/truetype/Font_Awesome_6_Brands-Regular-400.otf': [Errno 2] No such file or directory
skipping malformed name record #3
Traceback (most recent call last):
  File "/job/source.py", line 150, in <module>
    left_panel = left_panel.fuse(ring_peg(x, y))
AttributeError: 'ShapeList' object has no attribute 'fuse'
'''
        detail=cad.failure_detail(stderr)
        self.assertIn("AttributeError: 'ShapeList' object has no attribute 'fuse'",detail)
        self.assertIn('Traceback (most recent call last):',detail)
        self.assertIn('line 150',detail)
        for noise in ('Fontconfig','Font_Awesome','malformed name record'):
            self.assertNotIn(noise,detail)

    def test_guest_exception_surfaces_traceback_without_font_noise(self):
        source='''import build123d as b
t=b.Text("O",font_size=10,font="DejaVu Sans")
b.export_stl(b.extrude(t,amount=2),"/job/output/model.stl")
b.Box(1,1,1).no_such_method()
'''
        with tempfile.TemporaryDirectory() as d:
            job=Path(d)
            with self.assertRaises(RuntimeError) as caught:cad.run_source(job,source)
            message=str(caught.exception)
        self.assertIn('AttributeError',message)
        self.assertIn('Traceback (most recent call last):',message)
        for noise in ('Fontconfig','Font_Awesome','malformed name record'):
            self.assertNotIn(noise,message)

    @unittest.skipUnless(Path('/etc/fonts/conf.d').is_dir() and Path('/usr/share/fonts').is_dir(),'fontconfig/fonts unavailable')
    def test_sandbox_renders_text_without_font_noise(self):
        source='''import build123d as b
t=b.Text("O",font_size=10,font="DejaVu Sans",align=(b.Align.CENTER,b.Align.CENTER))
b.export_stl(b.extrude(t,amount=2),"/job/output/text.stl")
'''
        with tempfile.TemporaryDirectory() as d:
            job=Path(d);cad.run_source(job,source)
            log=(job/'model.log').read_text(errors='replace')
            import trimesh
            mesh=trimesh.load_mesh(job/'sandbox/output/text.stl')
        self.assertGreater(mesh.volume,0)
        for noise in ('Fontconfig','Font_Awesome','malformed name record'):
            self.assertNotIn(noise,log)

    def test_runner_environment_not_visible_via_proc_pid_1(self):
        source='''import os
data=open('/proc/1/environ','rb').read().decode(errors='replace')
assert 'ADAPSIS_CANARY_SECRET' not in data, 'LEAK: '+repr(data)
'''
        with tempfile.TemporaryDirectory() as d:
            job=Path(d)
            with patch.dict('os.environ',{'ADAPSIS_CANARY_SECRET':'super-secret-canary'}):
                cad.run_source(job,source)

    @unittest.skipUnless(_sandbox_prereqs_available(),'bwrap/venv/user systemd unavailable')
    def test_sandbox_pid1_environ_is_scrubbed_to_path_only(self):
        # systemd gives the service the *user manager's* environment, which holds
        # API keys on this host. A PATH-only client env no longer protects
        # /proc/1/environ; only the `env -i` layer inside the unit does. Demanding
        # exactly PATH proves the manager env cannot leak in.
        source='''import os
entries=open('/proc/1/environ').read().split('\\0')
keys={e.split('=',1)[0] for e in entries if '=' in e}
assert keys=={'PATH'}, 'manager environment leaked: '+repr(sorted(keys))
'''
        with tempfile.TemporaryDirectory() as d:
            cad.run_source(Path(d),source)

    @unittest.skipUnless(_sandbox_prereqs_available(),'bwrap/venv/user systemd unavailable')
    def test_fork_loop_is_capped_and_reaped(self):
        marker='capprobe'+uuid.uuid4().hex
        source=f'''import os,sys,time
marker={marker!r}
kids=[]
for _ in range(6000):
    try:pid=os.fork()
    except OSError:print('FORK_FAILED',len(kids),flush=True);break
    if pid==0:
        try:os.execv(sys.executable,[sys.executable,'-c','import time;time.sleep(120)','--marker',marker])
        except OSError:os._exit(127)
    kids.append(pid)
else:print('NO_LIMIT',len(kids),flush=True)
print('KIDS',len(kids),flush=True)
time.sleep(0.3)
os._exit(0)
'''
        with tempfile.TemporaryDirectory() as d:
            job=Path(d)
            try:
                cad.run_source(job,source)
                log=(job/'model.log').read_text(errors='replace')
                self.assertIn('FORK_FAILED',log,'TasksMax did not constrain the fork loop')
                self.assertNotIn('NO_LIMIT',log)
                count=int(re.search(r'FORK_FAILED (\d+)',log).group(1))
                self.assertLess(count,1000)
                self.assertTrue(self._wait_for_no_probe(marker),'guest children survived teardown')
                # The daemon's ability to fork is unaffected by the guest's cap.
                self.assertEqual(subprocess.run(['/bin/true'],timeout=10).returncode,0)
                pid=os.fork()
                if pid==0:os._exit(0)
                os.waitpid(pid,0)
            finally:
                subprocess.run(['pkill','-f',marker],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)

    @unittest.skipUnless(_sandbox_prereqs_available(),'bwrap/venv/user systemd unavailable')
    def test_real_timeout_tears_down_a_forking_unit(self):
        marker='timeoutprobe'+uuid.uuid4().hex
        source=f'''import os,sys,time
marker={marker!r}
for _ in range(6000):
    try:pid=os.fork()
    except OSError:break
    if pid==0:
        try:os.execv(sys.executable,[sys.executable,'-c','import time;time.sleep(120)','--marker',marker])
        except OSError:os._exit(127)
time.sleep(120)
'''
        with tempfile.TemporaryDirectory() as d:
            job=Path(d)
            try:
                with patch.object(cad,'SANDBOX_TIMEOUT',3):
                    with self.assertRaisesRegex(RuntimeError,'sandbox limit'):
                        cad.run_source(job,source)
                self.assertTrue(self._wait_for_no_probe(marker),'timed-out unit left children behind')
                self.assertEqual(subprocess.run(['/bin/true'],timeout=10).returncode,0)
            finally:
                subprocess.run(['pkill','-f',marker],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)

    def _wait_for_no_probe(self,marker,timeout=5.0):
        deadline=time.monotonic()+timeout
        while time.monotonic()<deadline:
            if subprocess.run(['pgrep','-f',marker],stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL).returncode!=0:return True
            time.sleep(0.1)
        return False

    def test_run_source_passes_path_only_env_to_the_systemd_client(self):
        unit='adapsis-cad-sandbox-env1'
        result=Mock(returncode=0,stdout=b'',stderr=b'')
        with tempfile.TemporaryDirectory() as d:
            with patch('cad_prepare.user_systemd_available',return_value=True), \
                 patch('cad_prepare.sandbox_unit_name',return_value=unit), \
                 patch('cad_prepare.subprocess.run',return_value=result) as run:
                cad.run_source(Path(d),'x = 1\n')
        sandbox_calls=[c for c in run.call_args_list if '/usr/bin/bwrap' in c.args[0]]
        self.assertEqual(len(sandbox_calls),1)
        command=sandbox_calls[0].args[0]
        self.assertEqual(command[0],cad.SYSTEMD_RUN)
        self.assertIn(f'--unit={unit}',command)
        env=sandbox_calls[0].kwargs.get('env')
        self.assertEqual(env,cad._systemd_client_env())
        self.assertNotIn('DEEPSEEK_API_KEY',env)
        self.assertEqual(set(env)-{'XDG_RUNTIME_DIR','DBUS_SESSION_BUS_ADDRESS'},{'PATH'})

    def test_sandbox_command_caps_process_tree_in_transient_service(self):
        with tempfile.TemporaryDirectory() as d:
            command=cad.sandbox_command(Path(d),'adapsis-cad-sandbox-abc123')
        self.assertEqual(command[0],cad.SYSTEMD_RUN)
        self.assertIn('--user',command)
        for flag in ('--wait','--collect','--pipe'):
            self.assertIn(flag,command)
        self.assertIn('--unit=adapsis-cad-sandbox-abc123',command)
        properties=[command[index+1] for index,value in enumerate(command) if value=='-p']
        self.assertIn(f'TasksMax={cad.SANDBOX_TASKS_MAX}',properties)
        self.assertIn('KillMode=control-group',properties)
        self.assertIn(f'RuntimeMaxSec={cad.SANDBOX_RUNTIME_LIMIT}',properties)
        # The systemd hard limit must not pre-empt the Python timeout, or the
        # actionable sandbox-limit message and explicit teardown are lost.
        self.assertGreater(cad.SANDBOX_RUNTIME_LIMIT,cad.SANDBOX_TIMEOUT)
        # The isolation entry point is only wrapped, not replaced.
        separator=command.index('--')
        self.assertEqual(command[separator:separator+4],
            ['--','/usr/bin/env','-i','PATH=/usr/bin:/bin'])
        self.assertEqual(command[separator+4],'/usr/bin/prlimit')
        self.assertLess(command.index('/usr/bin/prlimit'),command.index('/usr/bin/bwrap'))
        self.assertEqual(command[-1],'/job/source.py')

    def test_sandbox_unit_names_are_unique_and_namespaced(self):
        first=cad.sandbox_unit_name();second=cad.sandbox_unit_name()
        self.assertNotEqual(first,second)
        for name in (first,second):
            self.assertTrue(name.startswith('adapsis-cad-sandbox-'))
            self.assertRegex(name,r'^[A-Za-z0-9_.-]+$')
            self.assertLess(len(name)+len('.service'),256)

    def test_run_source_refuses_to_run_without_a_user_systemd(self):
        with tempfile.TemporaryDirectory() as d:
            with patch('cad_prepare.user_systemd_available',return_value=False), \
                 patch('cad_prepare.subprocess.run') as run:
                with self.assertRaises(cad.SandboxUnavailable) as caught:
                    cad.run_source(Path(d),'x = 1\n')
        run.assert_not_called()
        message=str(caught.exception)
        self.assertIn('systemd',message)
        self.assertIn('process-count cap',message)

    def test_unavailable_sandbox_is_not_retried_as_geometry(self):
        with tempfile.TemporaryDirectory() as d, \
             patch('cad_astra.generate',return_value=('source',{})) as generate, \
             patch('cad_prepare.run_source',side_effect=cad.SandboxUnavailable('no user systemd')):
            with self.assertRaises(cad.SandboxUnavailable):
                cad.prepare_geometry(Path(d),'rack','brief','PLA')
        self.assertEqual(generate.call_count,1)

    def test_timeout_kills_the_transient_unit_and_reports_the_limit(self):
        unit='adapsis-cad-sandbox-timeout1'
        def fake_run(command,**kwargs):
            if command and command[0]==cad.SYSTEMD_RUN:
                raise subprocess.TimeoutExpired(command,cad.SANDBOX_TIMEOUT,output=b'',stderr=b'')
            return Mock(returncode=0,stdout=b'',stderr=b'')
        with tempfile.TemporaryDirectory() as d:
            with patch('cad_prepare.user_systemd_available',return_value=True), \
                 patch('cad_prepare.sandbox_unit_name',return_value=unit), \
                 patch('cad_prepare.subprocess.run',side_effect=fake_run) as run:
                with self.assertRaisesRegex(RuntimeError,'sandbox limit'):
                    cad.run_source(Path(d),'x = 1\n')
        teardown=[c.args[0] for c in run.call_args_list if c.args and c.args[0][0]==cad.SYSTEMCTL]
        self.assertTrue(any(f'{unit}.service' in c for c in teardown))
        self.assertTrue(any('--signal=SIGKILL' in c for c in teardown))

    def test_launch_failure_still_tears_down_the_transient_unit(self):
        unit='adapsis-cad-sandbox-launch1'
        def fake_run(command,**kwargs):
            if command and command[0]==cad.SYSTEMD_RUN:
                raise OSError('systemd bus gone')
            return Mock(returncode=0,stdout=b'',stderr=b'')
        with tempfile.TemporaryDirectory() as d:
            with patch('cad_prepare.user_systemd_available',return_value=True), \
                 patch('cad_prepare.sandbox_unit_name',return_value=unit), \
                 patch('cad_prepare.subprocess.run',side_effect=fake_run) as run:
                with self.assertRaises(cad.SandboxUnavailable):
                    cad.run_source(Path(d),'x = 1\n')
        self.assertTrue(any(c.args[0][0]==cad.SYSTEMCTL and f'{unit}.service' in c.args[0]
            for c in run.call_args_list))

    def test_systemd_start_failure_is_sandbox_unavailable(self):
        result=Mock(returncode=1,stdout=b'',
            stderr=b'Failed to connect to user scope bus: Connection refused\n')
        with tempfile.TemporaryDirectory() as d:
            with patch('cad_prepare.user_systemd_available',return_value=True), \
                 patch('cad_prepare.subprocess.run',return_value=result):
                with self.assertRaises(cad.SandboxUnavailable) as caught:
                    cad.run_source(Path(d),'x = 1\n')
        self.assertIn('systemd service could not be started',str(caught.exception))

    def test_main_checks_sandbox_before_paid_authoring(self):
        request=json.dumps({'title':'part','brief':'make a cube','material':'PLA'})
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(cad,'JOBS',Path(directory)/'jobs'), \
             patch.object(sys,'stdin',io.StringIO(request)), \
             patch('cad_prepare.user_systemd_available',return_value=False), \
             patch('cad_prepare.prepare_geometry') as prepare:
            with self.assertRaises(cad.SandboxUnavailable):cad.main()
        prepare.assert_not_called()

    def test_signal_death_names_the_signal(self):
        source='import ctypes\nctypes.string_at(0)\n'
        with tempfile.TemporaryDirectory() as d:
            job=Path(d)
            with self.assertRaises(RuntimeError) as caught:cad.run_source(job,source)
        self.assertIn('signal 11',str(caught.exception))

    def test_signal_from_returncode_maps_128_plus_n(self):
        self.assertEqual(cad._signal_from_returncode(137),9)
        self.assertEqual(cad._signal_from_returncode(139),11)
        self.assertEqual(cad._signal_from_returncode(152),24)
        self.assertEqual(cad._signal_from_returncode(-9),9)
        self.assertIsNone(cad._signal_from_returncode(1))

    def test_failure_detail_bounded_keeps_final_exception(self):
        frames=''.join(f'  File "/job/source.py", line {i}, in f{i}\n    f{i+1}()\n' for i in range(1000))
        stderr='Traceback (most recent call last):\n'+frames+'RecursionError: maximum recursion depth exceeded\n'
        detail=cad.failure_detail(stderr)
        self.assertLessEqual(len(detail),cad.DETAIL_LIMIT)
        self.assertIn('Traceback (most recent call last):',detail)
        self.assertIn('RecursionError: maximum recursion depth exceeded',detail)

    def test_failure_detail_strips_noise_interleaved_and_after_traceback(self):
        stderr='''Fontconfig error: "/etc/fonts/conf.d/48-guessfamily.conf", line 196: invalid constant used : ui-rounded
Traceback (most recent call last):
  File "/job/source.py", line 3, in <module>
Fontconfig warning: "/etc/fonts/conf.d/49-sansserif.conf", line 10: invalid constant used : serif
    boom()
cannot open font '/usr/share/fonts/truetype/Font_Awesome_6_Free-Solid-900.otf': No such file or directory
AttributeError: boom
skipping malformed name record #5
'''
        detail=cad.failure_detail(stderr)
        self.assertIn('Traceback (most recent call last):',detail)
        self.assertIn('AttributeError: boom',detail)
        for noise in ('Fontconfig','Font_Awesome','malformed name record'):
            self.assertNotIn(noise,detail)

    def test_failure_detail_strips_noise_without_traceback_header(self):
        stderr='''Fontconfig error: "/etc/fonts/conf.d/48-guessfamily.conf", line 196: invalid constant used : ui-rounded
cannot open font '/usr/share/fonts/truetype/Font_Awesome_6_Brands-Regular-400.otf': No such file or directory
OSError: something real failed here
skipping malformed name record #3
'''
        detail=cad.failure_detail(stderr)
        self.assertIn('OSError: something real failed here',detail)
        for noise in ('Fontconfig','Font_Awesome','malformed name record'):
            self.assertNotIn(noise,detail)

    def test_failure_detail_bare_header_is_not_a_detail(self):
        self.assertEqual(cad.failure_detail('Traceback (most recent call last):\n'),'')

    def test_failure_detail_drops_systemd_run_chatter(self):
        stderr='''Running as unit: adapsis-cad-sandbox-x.service
          Finished with result: exit-code
Main processes terminated with: code=exited, status=139/n/a
               Service runtime: 187ms
               CPU time consumed: 34ms
'''
        self.assertEqual(cad.failure_detail(stderr),'')

    def test_failure_detail_keeps_traceback_and_drops_systemd_chatter(self):
        stderr='''Traceback (most recent call last):
  File "/job/source.py", line 3, in <module>
    boom()
ValueError: bad
Running as unit: adapsis-cad-sandbox-x.service
          Finished with result: exit-code
               Service runtime: 5ms
'''
        detail=cad.failure_detail(stderr)
        self.assertIn('ValueError: bad',detail)
        self.assertNotIn('Running as unit',detail)
        self.assertNotIn('Finished with result',detail)

    def test_fontconfig_copies_regular_files_not_symlinks(self):
        with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as d:
            (Path(src)/'00-custom.conf').write_text('<fontconfig>\n  <dir>/usr/share/fonts</dir>\n</fontconfig>\n')
            sandbox=Path(d)/'sandbox';sandbox.mkdir()
            target=cad._sandbox_fontconfig(sandbox,source_dir=src)
            copied=target/'00-custom.conf'
            self.assertFalse(copied.is_symlink())
            self.assertEqual(copied.read_text(),(Path(src)/'00-custom.conf').read_text())
            self.assertEqual(copied.stat().st_size,(Path(src)/'00-custom.conf').stat().st_size)

class CadAssemblyTests(unittest.TestCase):
    def _job(self,directory,layout,assembly,raw=None):
        job=Path(directory);output=job/'sandbox/output';output.mkdir(parents=True)
        parts=[]
        for name,extents,translate in layout:
            _write_box(output/name,extents,translate)
            parts.append({'file':name})
        if raw is not None:(output/cad.ASSEMBLY_FILE).write_text(raw)
        elif assembly is not None:(output/cad.ASSEMBLY_FILE).write_text(json.dumps(assembly))
        return job,parts

    def _rack(self):
        layout=[('base.stl',[250,100,20],(0,0,10)),('brace.stl',[20,20,180],(0,0,110))]
        assembly={'units':'mm','envelope_mm':[250,100,200],'instances':[
            {'part':'base.stl','transform':_identity()},
            {'part':'brace.stl','transform':_identity()}]}
        return layout,assembly

    def test_multipart_fixture_reports_authored_assembled_envelope(self):
        layout,assembly=self._rack()
        with tempfile.TemporaryDirectory() as d:
            job,parts=self._job(d,layout,assembly)
            result=cad.validate_assembly(job,parts)
        self.assertTrue(result['available'],result.get('reason'))
        self.assertEqual(result['envelope_mm'],[250,100,200])
        self.assertEqual(result['instance_count'],2)
        self.assertEqual(len(result['instances']),2)
        for measured,expected in zip(result['assembled_size_mm'],[250,100,200]):
            self.assertAlmostEqual(measured,expected,places=3)

    def test_assembly_translation_moves_part_without_rescaling(self):
        layout=[('base.stl',[40,40,10],(0,0,5)),('peg.stl',[10,10,10],(0,0,5))]
        assembly={'units':'mm','envelope_mm':[100,100,100],'instances':[
            {'part':'base.stl','transform':_identity()},
            {'part':'peg.stl','transform':[[1,0,0,20],[0,1,0,0],[0,0,1,0],[0,0,0,1]]}]}
        with tempfile.TemporaryDirectory() as d:
            job,parts=self._job(d,layout,assembly)
            result=cad.validate_assembly(job,parts)
        self.assertTrue(result['available'],result.get('reason'))
        dx,dy,dz=result['assembled_size_mm']
        self.assertAlmostEqual(dx,45,places=3)
        self.assertAlmostEqual(dy,40,places=3)
        self.assertAlmostEqual(dz,10,places=3)

    def test_unknown_part_is_rejected(self):
        layout,assembly=self._rack()
        assembly['instances'][1]['part']='not-exported.stl'
        with tempfile.TemporaryDirectory() as d:
            job,parts=self._job(d,layout,assembly)
            result=cad.validate_assembly(job,parts)
        self.assertFalse(result['available'])
        self.assertIn('not one of the exported STL parts',result['reason'])
        self.assertIn('print parts only',result['reason'])

    def test_nonfinite_transform_is_rejected(self):
        layout,_=self._rack()
        raw=json.dumps({'units':'mm','envelope_mm':[250,100,200],'instances':[
            {'part':'base.stl','transform':[[1,0,0,float('nan')],[0,1,0,0],[0,0,1,0],[0,0,0,1]]}]})
        with tempfile.TemporaryDirectory() as d:
            job,parts=self._job(d,layout,assembly=None,raw=raw)
            result=cad.validate_assembly(job,parts)
        self.assertFalse(result['available'])
        self.assertIn('finite numbers',result['reason'])

    def test_scaled_or_sheared_transform_is_rejected(self):
        layout,_=self._rack()
        assembly={'units':'mm','envelope_mm':[250,100,200],'instances':[
            {'part':'base.stl','transform':[[2,0,0,0],[0,2,0,0],[0,0,2,0],[0,0,0,1]]}]}
        with tempfile.TemporaryDirectory() as d:
            job,parts=self._job(d,layout,assembly)
            result=cad.validate_assembly(job,parts)
        self.assertFalse(result['available'])
        self.assertIn('orthonormal',result['reason'])

    def test_unbounded_envelope_is_rejected(self):
        layout,_=self._rack()
        assembly={'units':'mm','envelope_mm':[5000,100,200],'instances':[
            {'part':'base.stl','transform':_identity()}]}
        with tempfile.TemporaryDirectory() as d:
            job,parts=self._job(d,layout,assembly)
            result=cad.validate_assembly(job,parts)
        self.assertFalse(result['available'])
        self.assertIn('envelope_mm',result['reason'])

    def test_assembled_bounds_beyond_declared_envelope_are_rejected(self):
        layout,_=self._rack()
        assembly={'units':'mm','envelope_mm':[100,100,100],'instances':[
            {'part':'base.stl','transform':_identity()},
            {'part':'brace.stl','transform':_identity()}]}
        with tempfile.TemporaryDirectory() as d:
            job,parts=self._job(d,layout,assembly)
            result=cad.validate_assembly(job,parts)
        self.assertFalse(result['available'])
        self.assertIn('exceeds the declared envelope',result['reason'])

    def test_malicious_payload_fields_and_paths_are_rejected(self):
        layout,_=self._rack()
        cases={
            'extra-top-level':{'units':'mm','envelope_mm':[250,100,200],'html':'<script>evil()</script>',
                'instances':[{'part':'base.stl','transform':_identity()}]},
            'extra-instance-field':{'units':'mm','envelope_mm':[250,100,200],
                'instances':[{'part':'base.stl','transform':_identity(),'src':'http://evil/x.js'}]},
            'path-part':{'units':'mm','envelope_mm':[250,100,200],
                'instances':[{'part':'../../etc/passwd','transform':_identity()}]},
            'url-part':{'units':'mm','envelope_mm':[250,100,200],
                'instances':[{'part':'http://evil/model.stl','transform':_identity()}]},
            'bool-transform':{'units':'mm','envelope_mm':[250,100,200],
                'instances':[{'part':'base.stl','transform':[[True,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]}]},
        }
        for name,assembly in cases.items():
            with self.subTest(case=name),tempfile.TemporaryDirectory() as d:
                job,parts=self._job(d,layout,assembly)
                result=cad.validate_assembly(job,parts)
                self.assertFalse(result['available'],name)
                self.assertNotIn('<script>',json.dumps(result))
                self.assertNotIn('evil',json.dumps(result))

    def test_absent_assembly_falls_back_to_print_parts(self):
        layout,_=self._rack()
        with tempfile.TemporaryDirectory() as d:
            job,parts=self._job(d,layout,assembly=None)
            result=cad.validate_assembly(job,parts)
        self.assertFalse(result['available'])
        self.assertIn('print parts only',result['reason'])

    def test_valid_rotation_is_accepted_and_measured(self):
        layout=[('stick.stl',[40,10,10],(0,0,5))]
        assembly={'units':'mm','envelope_mm':[100,100,100],'instances':[
            {'part':'stick.stl','transform':[[0,-1,0,0],[1,0,0,0],[0,0,1,0],[0,0,0,1]]}]}
        with tempfile.TemporaryDirectory() as d:
            job,parts=self._job(d,layout,assembly)
            result=cad.validate_assembly(job,parts)
        self.assertTrue(result['available'],result.get('reason'))
        dx,dy,dz=result['assembled_size_mm']
        self.assertAlmostEqual(dx,10,places=3)
        self.assertAlmostEqual(dy,40,places=3)
        self.assertAlmostEqual(dz,10,places=3)

    def test_pathological_json_never_fails_the_job(self):
        layout,_=self._rack()
        raw='['*2000+']'*2000
        with tempfile.TemporaryDirectory() as d:
            job,parts=self._job(d,layout,assembly=None,raw=raw)
            result=cad.validate_assembly(job,parts)
        self.assertFalse(result['available'])
        self.assertIn('print parts only',result['reason'])

    def test_symlinked_assembly_is_rejected(self):
        layout,_=self._rack()
        with tempfile.TemporaryDirectory() as d:
            job,parts=self._job(d,layout,assembly=None)
            target=job/'sandbox/output'/'elsewhere.json';target.write_text('{}')
            (job/'sandbox/output'/cad.ASSEMBLY_FILE).symlink_to(target)
            result=cad.validate_assembly(job,parts)
        self.assertFalse(result['available'])
        self.assertIn('symlink',result['reason'])

    def test_invalid_assembly_does_not_trigger_a_revision(self):
        def run(attempt,source):
            (attempt/'sandbox/output').mkdir(parents=True)
            (attempt/'sandbox/output/model.stl').write_text('x')
            (attempt/'sandbox/output'/cad.ASSEMBLY_FILE).write_text(json.dumps(
                {'units':'mm','envelope_mm':[10,10,10],
                 'instances':[{'part':'other.stl','transform':_identity()}]}))
            (attempt/'layout.3mf').write_text('checked')
        author={'requested_model':'chatgpt/gpt-6-astra'}
        with tempfile.TemporaryDirectory() as d:
            job=Path(d)
            with patch('cad_astra.generate',side_effect=[('source',dict(author))]) as generate, \
                 patch('cad_prepare.run_source',side_effect=run), \
                 patch('cad_prepare.validate_exports',return_value=[{'file':'model.stl'}]):
                parts,result,assembly=cad.prepare_geometry(job,'rack','brief','PLA')
            self.assertEqual(generate.call_count,1)
            self.assertFalse(assembly['available'])
            self.assertIn('print parts only',assembly['reason'])
            self.assertTrue((job/'sandbox/output/model.stl').exists())

    def test_publish_only_copies_allowlisted_files(self):
        with tempfile.TemporaryDirectory() as d:
            job=Path(d)/'design-abc'
            output=job/'sandbox/output';output.mkdir(parents=True)
            (output/'model.stl').write_bytes(b'solid model')
            (output/'evil.svg').write_text('<svg onload="alert(1)"/>')
            (output/'evil.html').write_text('<script>alert(1)</script>')
            (output/'source.py').write_text('print(1)')
            (output/cad.ASSEMBLY_FILE).write_text('{"malicious":"<script>"}')
            (job/'sliced').mkdir();(job/'sliced/prepared-a1.3mf').write_bytes(b'3mf')
            report={'passed':True,'title':'Rack <x>','job_dir':str(job),
                'parts':[{'file':'model.stl','size_mm':[10,10,10],'volume_mm3':1,'sha256':'a'}],
                'assembly':{'available':False,'reason':'Assembly layout rejected (nope); showing print parts only.'},
                'material':'PLA','plates':1,'minutes':1,'grams':1,
                'url':'https://example/','print_approval':'x','limitations':'x'}
            with patch.object(cad,'PUBLIC',Path(d)/'public'):
                cad.publish(job,report)
            destination=Path(d)/'public'/'design-abc'
            names=sorted(p.name for p in destination.iterdir())
            self.assertEqual(names,['index.html','manifest.json','model.stl','prepared-a1.3mf'])
            manifest=json.loads((destination/'manifest.json').read_text())
            self.assertEqual(manifest['parts'][0]['file'],'model.stl')
            self.assertFalse(manifest['assembly']['available'])
            self.assertNotIn('job_dir',manifest)
            self.assertNotIn('<script>',(destination/'manifest.json').read_text())
            page=(destination/'index.html').read_text()
            self.assertIn('Rack &lt;x&gt;',page)
            self.assertNotIn('__TITLE__',page)

if __name__=='__main__':unittest.main()
