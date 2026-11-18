"""Diagnose coordinate sensitivity; never patch parameters or approve guidance."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.compare_opls_engines import compare, command
from scripts.diagnose_opls_benchmark import REPORT_SHA256


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def coordinate_cases(raw, relaxed):
    """Probe fixed prespecified perturbations; retain the original failures."""
    raw, relaxed = np.asarray(raw, float), np.asarray(relaxed, float)
    if raw.shape != relaxed.shape or raw.ndim != 2 or raw.shape[1] != 3:
        raise ValueError('Invalid paired coordinates')
    if not np.isfinite(raw).all() or not np.isfinite(relaxed).all():
        raise ValueError('Nonfinite coordinates')
    direction = np.random.default_rng(20261117).normal(size=relaxed.shape)
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    return [('raw',raw.copy()), ('relaxed',relaxed.copy()),
            ('relaxed_probe_0p001A', relaxed + direction*1e-4),
            ('relaxed_probe_0p01A', relaxed + direction*1e-3)]


def run(source, output, gmx):
    if digest(source/'report.json') != REPORT_SHA256:
        raise ValueError('Unexpected benchmark source')
    report = json.loads((source/'report.json').read_text())
    command([gmx, '--version'], output, 'gromacs-version.txt')
    results = {'status':'diagnostic_only', 'guidance_approved':False, 'physical_correctness_validated':False,
               'source_report_sha256':REPORT_SHA256, 'cases':[],
               'note':'Offsets are translations; perturbations change geometry and are diagnostic only. '
                      'A passing perturbed case never validates the original relaxed coordinates. '
                      'No force-field terms or tolerances are changed.'}
    for index in (1,7,13):
        graph = next(g for g in report['inputs'][index]['graphs'] if g['graph_index']==0)
        metadata = graph['parameters']; cache = Path(metadata['cache_directory'])
        for name, expected in metadata['file_sha256'].items():
            if digest(cache/name) != expected:
                raise ValueError('Parameter cache changed: '+name)
        saved = source/f'input_{index:03d}'/'graph_0'/'opls_guided_0.npz'
        with np.load(saved, allow_pickle=False) as arrays:
            mapping = np.asarray(metadata['model_to_engine'], int)
            cases=[]
            for stage in ('raw','relaxed'):
                points = arrays[stage+'_angstrom']
                if sorted(mapping.tolist()) != list(range(len(points))):
                    raise ValueError('Invalid atom mapping')
                engine = np.empty_like(points, dtype=float); engine[mapping]=points/10.
                cases.append(engine)
        for offset in (0.5,5.0):
            folder=output/f'input_{index}_offset_{offset}';folder.mkdir()
            row={'input_index':index,'offset_nm':offset,'source_npz_sha256':digest(saved)}
            try:
                row['comparison']=compare(cache/'export','MOL',folder,gmx,forces=True,
                    coordinate_cases=coordinate_cases(*cases),coordinate_offset_nm=offset)
                row['status']='evaluated'
            except Exception as exc:
                row.update(status='failed',error=f'{type(exc).__name__}: {exc}')
            results['cases'].append(row)
            (output/'report.json').write_text(json.dumps(results,indent=2,allow_nan=False))
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('outputs/opls_benchmark_1820024'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gmx',required=True)
    args=parser.parse_args();output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    result={'status':'failed','guidance_approved':False}
    try:
        gmx=shutil.which(args.gmx)
        if not gmx: raise ValueError('GROMACS executable unavailable')
        result=run(args.source.resolve(),output,gmx)
    except Exception as exc:
        result['error']=f'{type(exc).__name__}: {exc}'
    (output/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    with zipfile.ZipFile(output/'precision_results.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.rglob('*')):
            if path.is_file() and path.suffix in ('.json','.txt','.log','.mdp','.top','.itp','.gro','.xvg','.npy'):
                archive.write(path,path.relative_to(output))
    print('Download:',output/'precision_results.zip',flush=True)
    return 0 if result['status']=='diagnostic_only' and all(r['status']=='evaluated' for r in result['cases']) else 1


if __name__=='__main__': raise SystemExit(main())
