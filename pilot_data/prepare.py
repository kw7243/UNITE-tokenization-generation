"""Build fixed pilot IDs and byte identities. No image decoding or model imports."""
import argparse
import ast
import collections
import concurrent.futures
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import time

FIELDS = ['image_id', 'synset', 'class_index', 'filename', 'absolute_path',
          'resolved_path', 'split_sha256', 'class_rank', 'content_sha256', 'size_bytes']
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp'}


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + '\n')


def read_rows(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f, delimiter='\t'))


def write_rows(path, rows):
    with path.open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, delimiter='\t', lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def verify(out, classes):
    """Check serialized outputs and split ordering independently of builder slices."""
    rows = {s: read_rows(out / (s + '.tsv')) for s in ['CONTENT', 'FIT', 'ALLOC', 'DEV', 'CONFIRM']}
    mapping = json.loads((out / 'class_to_idx.json').read_text())
    assert mapping == {s: i for i, s in enumerate(classes)}
    ids = {}
    for split, rr in rows.items():
        ids[split] = {r['image_id'] for r in rr}
        assert len(ids[split]) == len(rr), ('duplicate image ID', split)
        assert {r['synset'] for r in rr} == set(classes), split
        assert all(int(r['class_index']) == mapping[r['synset']] for r in rr)
    assert ids['FIT'].isdisjoint(ids['DEV'])
    assert ids['FIT'].isdisjoint(ids['CONFIRM'])
    assert ids['DEV'].isdisjoint(ids['CONFIRM'])
    assert ids['ALLOC'] <= ids['FIT']
    assert ids['CONTENT'] == ids['FIT'] | ids['DEV'] | ids['CONFIRM']
    content = {r['image_id']: r for r in rows['CONTENT']}
    for split in ['FIT', 'ALLOC', 'DEV', 'CONFIRM']:
        assert all(r == content[r['image_id']] for r in rows[split])
    counts = {}
    for synset in classes:
        candidates = [r for r in rows['CONTENT'] if r['synset'] == synset]
        expected = sorted(candidates, key=lambda r: (hashlib.sha256(('split-v1|' + r['image_id']).encode('utf-8')).hexdigest(), r['image_id']))
        assert len(expected) >= 96
        for rank, row in enumerate(expected):
            assert int(row['class_rank']) == rank
            assert row['split_sha256'] == hashlib.sha256(('split-v1|' + row['image_id']).encode('utf-8')).hexdigest()
            assert row['image_id'] == synset + '/' + row['filename']
            assert len(row['content_sha256']) == 64
            assert int(row['size_bytes']) > 0
        expected_ids = [r['image_id'] for r in expected]
        limits = {'DEV': (0, 32), 'CONFIRM': (32, 64), 'FIT': (64, len(expected)), 'ALLOC': (64, 96)}
        for split, (start, end) in limits.items():
            actual = [r['image_id'] for r in rows[split] if r['synset'] == synset]
            assert actual == expected_ids[start:end], (synset, split)
        counts[synset] = {s: sum(r['synset'] == synset for r in rr) for s, rr in rows.items()}
    assert all(len(rows[s]) == 3200 for s in ['DEV', 'CONFIRM', 'ALLOC'])
    return {'passed': True, 'exact_class_count': len(classes), 'rows': {s: len(r) for s, r in rows.items()},
            'per_class': counts, 'unique_ids': True, 'fit_dev_confirm_disjoint': True,
            'alloc_subset_fit': True, 'coverage_complete': True, 'exact_hash_rank_membership': True,
            'serialized_rows_equal_content': True, 'content_bytes_hashed_once': True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--verify-only')
    args = parser.parse_args()
    cfgpath = Path(args.config)
    cfg = json.loads(cfgpath.read_text())  # JSON is a YAML 1.2 subset.
    source = Path.cwd()
    assert os.environ.get('SLURM_JOB_ID'), 'CPU Slurm allocation required'
    assert (source / 'STAGING_MANIFEST.json').is_file(), 'Verified full repository stage required'
    assert sha(source / 'pilot_data/inputs/plan.md') == cfg['plan_sha256']
    class_file = source / cfg['class_list_path']
    assert sha(class_file) == cfg['class_list_sha256']
    tree = ast.parse(class_file.read_text())
    values = [ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'synsets' for t in n.targets)]
    assert len(values) == 1 and len(values[0]) == len(set(values[0])) == 100
    classes = sorted(values[0])  # ImageFolder class discovery sorts synset names.
    if args.verify_only:
        print(json.dumps(verify(Path(args.verify_only), classes), sort_keys=True), flush=True)
        return
    config_hash = sha(cfgpath)
    run_id = 'PILOT_DATA__SPLIT_V1__snone__' + config_hash[:12]
    out = Path(cfg['output_parent']) / run_id
    out.mkdir(parents=True, exist_ok=False)
    (out / 'resolved.yaml').write_bytes(cfgpath.read_bytes())
    start = time.time()
    source_hashes = {str(p.relative_to(source)): sha(p) for p in sorted(source.rglob('*'))
                     if p.is_file() and '.git' not in p.relative_to(source).parts and p.name != 'STAGING_MANIFEST.json'}
    save(out / 'source_hashes.json', source_hashes)
    python_path = Path(sys.executable).resolve()
    environment = {'python': sys.version, 'python_executable': str(python_path), 'python_sha256': sha(python_path),
                   'platform': platform.platform(), 'stdlib_only': True, 'external_packages_used': [],
                   'host': socket.gethostname(), 'job_id': os.environ['SLURM_JOB_ID'],
                   'cpus': os.environ.get('SLURM_CPUS_PER_TASK'), 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                   'argv': sys.argv, 'stage': str(source), 'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()}
    save(out / 'environment.json', environment)
    frozen = {'config_sha256': config_hash, 'source_hashes_sha256': sha(out / 'source_hashes.json'),
              'environment_sha256': sha(out / 'environment.json'), 'staging_manifest_sha256': sha(source / 'STAGING_MANIFEST.json'),
              'script_sha256': sha(source / 'pilot_data/prepare.py'), 'class_list_sha256': sha(class_file),
              'preprocessing_pointers': {p: source_hashes[p] for p in cfg['preprocessing_pointers']}}
    save(out / 'provenance.json', frozen)  # Frozen before any scientific data reduction.
    root = Path(cfg['data_root'])
    resolved_root = root.resolve(strict=True)
    dataset = {'declared_root': str(root), 'resolved_root': str(resolved_root),
               'root_stat': {'device': root.stat().st_dev, 'inode': root.stat().st_ino, 'mtime_ns': root.stat().st_mtime_ns},
               'archive_sha256': None, 'archive_sha256_reason': 'Original train archive unavailable; extracted-file identity is provided.',
               'shared_imagenet100_reference': '/data/vision/torralba/datasets/imagenet100/train',
               'shared_imagenet100_used': False, 'classes': {}, 'ignored_entries': []}
    allrows = []
    splits = {s: [] for s in ['FIT', 'ALLOC', 'DEV', 'CONFIRM']}
    def identify(entry):
        synset, path = entry
        resolved = path.resolve(strict=True)
        assert resolved.is_relative_to(resolved_root), ('image outside approved data root', str(path), str(resolved))
        before = resolved.stat()
        assert resolved.is_file() and before.st_size > 0
        digest = sha(resolved)
        after = resolved.stat()
        assert (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns), str(path)
        image_id = synset + '/' + path.name
        return {'image_id': image_id, 'synset': synset, 'class_index': classes.index(synset),
                'filename': path.name, 'absolute_path': str(path), 'resolved_path': str(resolved),
                'split_sha256': hashlib.sha256(('split-v1|' + image_id).encode('utf-8')).hexdigest(),
                'class_rank': None, 'content_sha256': digest, 'size_bytes': before.st_size}
    with concurrent.futures.ThreadPoolExecutor(max_workers=cfg['hash_workers']) as pool:
        for synset in classes:
            directory = root / synset
            assert directory.is_dir()
            resolved = directory.resolve(strict=True)
            assert resolved.is_relative_to(resolved_root)
            names = sorted(directory.iterdir())
            assert not any(p.is_dir() for p in names), ('unexpected nested image directory', synset)
            files = [p for p in names if p.suffix.lower() in EXTENSIONS]
            assert all(p.is_file() for p in files)
            dataset['ignored_entries'].extend(str(p) for p in names if p not in files)
            rows = list(pool.map(identify, [(synset, p) for p in files]))
            rows.sort(key=lambda r: (r['split_sha256'], r['image_id']))
            assert len(rows) >= 96
            for rank, row in enumerate(rows):
                row['class_rank'] = rank
            dataset['classes'][synset] = {'absolute_path': str(directory), 'resolved_path': str(resolved),
                                         'is_symlink': directory.is_symlink(), 'readlink': os.readlink(directory) if directory.is_symlink() else None,
                                         'images': len(rows), 'mtime_ns': directory.stat().st_mtime_ns}
            allrows.extend(rows)
            splits['DEV'].extend(rows[:32])
            splits['CONFIRM'].extend(rows[32:64])
            splits['FIT'].extend(rows[64:])
            splits['ALLOC'].extend(rows[64:96])
            print(json.dumps({'class': synset, 'images_hashed': len(rows), 'completed_classes': len(dataset['classes'])}), flush=True)
    write_rows(out / 'CONTENT.tsv', allrows)
    for split, rows in splits.items():
        write_rows(out / (split + '.tsv'), rows)
    save(out / 'class_to_idx.json', {s: i for i, s in enumerate(classes)})
    save(out / 'classes.json', classes)
    save(out / 'dataset_identity.json', dataset)
    checks = verify(out, classes)
    save(out / 'checks.json', checks)
    manifest_files = ['CONTENT.tsv', 'FIT.tsv', 'ALLOC.tsv', 'DEV.tsv', 'CONFIRM.tsv', 'class_to_idx.json', 'classes.json']
    hashes = {name: sha(out / name) for name in manifest_files}
    save(out / 'manifest_hashes.json', hashes)
    interface = {'version': 'pilot-splits-v1', 'run_id': run_id, 'artifact_root': str(out),
                 'filenames': manifest_files, 'format': 'UTF-8 TSV with header, LF, no BOM', 'columns': FIELDS,
                 'integer_columns': ['class_index', 'class_rank', 'size_bytes'], 'image_id': 'synset/filename relative to data_root',
                 'data_root': str(root), 'class_map': 'class_to_idx.json', 'class_order': 'lexicographically sorted synsets (ImageFolder-compatible)',
                 'row_order': 'synset ascending, then (split_sha256, image_id) ascending', 'class_rank': 'zero-based within selected synset',
                 'split_hash': 'SHA256(UTF-8("split-v1|" + image_id)), no newline',
                 'tie_break': 'image_id ascending at exact digest tie', 'content_hash': 'SHA256 of raw file bytes; no image decode',
                 'memberships': {'DEV': '[0,32)', 'CONFIRM': '[32,64)', 'FIT': '[64,N)', 'ALLOC': '[64,96), subset of FIT'},
                 'rows': checks['rows'], 'checks': 'checks.json', 'hashes': hashes,
                 'confirm_policy': 'Integrity preparation only; sealed for scientific use until D1 method freeze',
                 'preprocessing': frozen['preprocessing_pointers'], 'training_sampler_implemented': False}
    save(out / 'interface.json', interface)
    results = {'node_id': 'PILOT_DATA', 'method_id': 'SPLIT_V1', 'seed': None, 'seed_reason': 'Deterministic identifier hashing; no RNG',
               'node_status': 'DATA_PREPARATION_VERIFIED', 'outcome_class': None,
               'outcome_class_reason': 'No model training or evaluation; not an A2 scientific pass',
               **frozen, 'checkpoint_sha256': None, 'checkpoint_sha256_reason': 'No model is used',
               'data_manifest_sha256': hashes, 'metric_protocol_sha256': None, 'metric_protocol_sha256_reason': 'No model metrics',
               'main_updates': 0, 'images_seen': 0, 'images_seen_reason': 'No model input; bytes hashed without decoding',
               'training_gpu_hours': 0, 'gpu_hours': 0, 'training': None, 'reconstruction': None, 'generation': None,
               'model_metrics_reason': 'Data preparation only', 'checks': checks, 'data_bytes': sum(r['size_bytes'] for r in allrows),
               'environment': environment, 'elapsed_preparation_seconds': time.time() - start,
               'attempts': [{'job_id': os.environ['SLURM_JOB_ID'], 'kind': 'CPU data preparation', 'status': 'script_complete'}],
               'allocated_cpu_core_hours': None, 'allocated_cpu_core_hours_reason': 'Fill from terminal Slurm receipt after job ends'}
    save(out / 'results.json', results)
    report = f'''## 1 Objective

Prepare only Section 5.4 ImageNet100 training splits under plan {cfg['plan_sha256']}.
Node PILOT_DATA; method SPLIT_V1; deterministic, no random seed.
The predeclared integrity gate is exact class coverage, split membership, byte identity, and ALLOC containment.

## 2 Setup

Source/stage: {source}.
Output: {out}.
Source commits: UNITE {cfg['unite_commit']}; KARL {cfg['karl_commit']}.
Config SHA256: {config_hash}.
CPU environment and exact argv: environment.json. Source/config/script locks: provenance.json and source_hashes.json.
Data root: {root}; resolved identity and symlinks: dataset_identity.json.
Class mapping sorts synsets as ImageFolder does. Full stage includes .git and untouched official sources under third_party.
Preprocessing is recorded only by pinned source pointers and hashes in interface.json; no implementation was added.

## 3 Baseline

No model or baseline comparison is run. Existing KARL content hashes cover validation only and cannot identify training files.
A1 validation and generation-label preparation are separate. No A1 workload is duplicated.

## 4 Results

Verified counts: {json.dumps(checks['rows'], sort_keys=True)}.
Exactly 100 official classes; DEV, CONFIRM and ALLOC each have 32 rows per class.
FIT preserves every remaining training image. Per-class actual counts: checks.json.
All integrity checks passed. Raw bytes hashed once: {results['data_bytes']} bytes.
Manifest SHA256 values: manifest_hashes.json. Row interface: interface.json.
Model metrics and outcome_class are null with reasons. Training/GPU time is zero.
One CPU attempt; terminal scheduler accounting will be retained separately after completion.

## 5 Interpretation

The pilot data artifacts satisfy the prescribed split and byte-identity rules.
This supports data preparation only. It does not establish A1, A2, training correctness, or any scientific hypothesis.

## 6 Limitations

The original archive hash is unavailable. Extracted file hashes identify the observed bytes; no archive hash is inferred.
CONFIRM remains sealed for scientific use until D1 method freeze. No images were decoded, visualized, scored, or ranked by difficulty.
The shared dataset can change later; consumers must require these frozen content identities.
No training sampler, selector, model, feature statistics, or full ImageNet1K split was built.

## 7 Next action

PASS for bounded data preparation only. Coordinator must accept interface.json, checks.json and exact manifest hashes.
The canonical campaign writer owns the artifact lock. A2 still requires A1 reference and adapter checks.
'''
    (out / 'report.md').write_text(report)
    (out / 'notebook-entry.md').write_text(f'''## 2026-10-24: fixed pilot data preparation

- Prepared Section 5.4 only; no A1/A2 or scientific pass claimed.
- Source branch: research/jat-pilot-data-20261024; stage: {source}.
- CPU job: {os.environ['SLURM_JOB_ID']}; host: {socket.gethostname()}; GPU-hours: zero.
- Retained output: {out}.
- Counts: {json.dumps(checks['rows'], sort_keys=True)}. All integrity checks passed; exact 100-class set and actual FIT counts retained.
- Manifest hashes: manifest_hashes.json; consumer contract: interface.json; report: report.md.
- CONFIRM remains sealed until D1 method freeze. Only IDs, hashes and integrity counts were prepared.
- Coordinator accepts these exact files; the sole canonical campaign writer updates its lock and main notebook.
''')
    print('PREPARATION_COMPLETE ' + str(out), flush=True)


if __name__ == '__main__':
    main()
