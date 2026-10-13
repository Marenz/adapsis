import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
import cad_prepare as cad

class CadPreparationTests(unittest.TestCase):
    def test_geometry_failure_is_revised_in_clean_sandbox(self):
        def run(attempt,source):
            (attempt/'sandbox/output').mkdir(parents=True)
            (attempt/'sandbox/output/model.stl').write_text(source)
            if source=='bad':raise RuntimeError('rack-left: 11 components')
            (attempt/'layout.3mf').write_text('checked')
        author={'requested_model':'chatgpt/gpt-6-astra'}
        with tempfile.TemporaryDirectory() as d:
            job=Path(d)
            with patch('cad_astra.generate',side_effect=[('bad',dict(author)),('good',dict(author))]) as generate, \
                 patch('cad_prepare.run_source',side_effect=run), \
                 patch('cad_prepare.validate_exports',return_value=[{'file':'model.stl'}]):
                parts,result=cad.prepare_geometry(job,'rack','20 pegs','PLA')
            self.assertEqual(generate.call_args.args,('rack','20 pegs','PLA','bad','rack-left: 11 components'))
            self.assertEqual((job/'sandbox/output/model.stl').read_text(),'good')
            self.assertEqual((job/'attempt-1/sandbox/output/model.stl').read_text(),'bad')
            self.assertEqual(result['geometry_attempts'],2)

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
            self.assertEqual(generate.call_args.args[-2:],('source','mesh must be one closed solid'))

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

    def test_run_source_passes_path_only_env(self):
        with tempfile.TemporaryDirectory() as d:
            job=Path(d)
            result=Mock(returncode=0,stdout=b'',stderr=b'')
            with patch('cad_prepare.subprocess.run',return_value=result) as run:
                cad.run_source(job,'x = 1\n')
        sandbox_calls=[c for c in run.call_args_list if '/usr/bin/bwrap' in c.args[0]]
        self.assertEqual(len(sandbox_calls),1)
        self.assertEqual(sandbox_calls[0].kwargs.get('env'),dict(cad.RUNNER_ENV))

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

    def test_fontconfig_copies_regular_files_not_symlinks(self):
        with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as d:
            (Path(src)/'00-custom.conf').write_text('<fontconfig>\n  <dir>/usr/share/fonts</dir>\n</fontconfig>\n')
            sandbox=Path(d)/'sandbox';sandbox.mkdir()
            target=cad._sandbox_fontconfig(sandbox,source_dir=src)
            copied=target/'00-custom.conf'
            self.assertFalse(copied.is_symlink())
            self.assertEqual(copied.read_text(),(Path(src)/'00-custom.conf').read_text())
            self.assertEqual(copied.stat().st_size,(Path(src)/'00-custom.conf').stat().st_size)

if __name__=='__main__':unittest.main()
