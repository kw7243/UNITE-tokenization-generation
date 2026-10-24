"""Immutable pilot interfaces, exact FIT queues, and independent paired streams."""
import copy
import csv
import hashlib
import io
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor
from utils.crop import CenterCropTransform


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def derive_seed(seed, name):
    return int.from_bytes(hashlib.sha256(f'a2-v1|{seed}|{name}'.encode()).digest()[:8], 'big') % (2**63-1)


def read_interface(cfg):
    """CONFIRM and CONTENT are never opened by this consumer."""
    root = Path(cfg['root'])
    classes = json.loads((root/'class_to_idx.json').read_text())
    assert sha(root/'class_to_idx.json') == cfg['hashes']['class_to_idx.json']
    assert len(classes) == 100 and sorted(classes.values()) == list(range(100))
    splits = {}
    for split in ('FIT', 'DEV', 'ALLOC'):
        path = root/f'{split}.tsv'
        assert sha(path) == cfg['hashes'][path.name], path
        with path.open(newline='') as stream:
            rows = list(csv.DictReader(stream, delimiter='\t'))
        for row in rows:
            row['class_index'] = int(row['class_index'])
            row['class_rank'] = int(row['class_rank'])
            assert row['class_index'] == classes[row['synset']]
        assert len(rows) == cfg['rows'][split]
        assert len({r['image_id'] for r in rows}) == len(rows)
        splits[split] = rows
    assert {r['image_id'] for r in splits['ALLOC']} <= {r['image_id'] for r in splits['FIT']}
    assert not ({r['image_id'] for r in splits['FIT']} & {r['image_id'] for r in splits['DEV']})
    return splits


class FitQueue:
    """One optimizer batch: 32 distinct shuffled classes, eight queued images each."""
    def __init__(self, rows, seed):
        self.rows = rows
        self.by_class = {}
        for i, row in enumerate(rows):
            self.by_class.setdefault(row['class_index'], []).append(i)
        assert len(self.by_class) >= 32 and min(map(len, self.by_class.values())) >= 8
        self.rng = random.Random(derive_seed(seed, 'data_order'))
        self.queues = {c: self.rng.sample(v, len(v)) for c, v in sorted(self.by_class.items())}
        self.cursor = {c: 0 for c in self.queues}
        self.updates = 0
        self.images_seen = 0
        self.exposure_chain = '0'*64

    def next(self):
        classes = self.rng.sample(sorted(self.by_class), 32)
        self.rng.shuffle(classes)
        indices = []
        for c in classes:
            for _ in range(8):
                if self.cursor[c] == len(self.queues[c]):
                    self.queues[c] = self.rng.sample(self.by_class[c], len(self.by_class[c]))
                    self.cursor[c] = 0
                indices.append(self.queues[c][self.cursor[c]])
                self.cursor[c] += 1
        self.updates += 1
        self.images_seen += len(indices)
        ids = [self.rows[i]['image_id'] for i in indices]
        self.exposure_chain = hashlib.sha256((self.exposure_chain+'\n'+'\n'.join(ids)).encode()).hexdigest()
        return indices

    def state_dict(self):
        return copy.deepcopy(dict(queues=self.queues, cursor=self.cursor, rng=self.rng.getstate(),
                                  updates=self.updates, images_seen=self.images_seen, exposure_chain=self.exposure_chain))

    def load_state_dict(self, state):
        state = copy.deepcopy(state)
        assert set(state['queues']) == set(self.by_class)
        for c, queue in state['queues'].items():
            assert sorted(queue) == sorted(self.by_class[c])
            assert 0 <= state['cursor'][c] <= len(queue)
        assert state['images_seen'] == state['updates']*256
        self.queues, self.cursor = state['queues'], state['cursor']
        self.rng.setstate(state['rng'])
        self.updates, self.images_seen = state['updates'], state['images_seen']
        self.exposure_chain = state['exposure_chain']


class Streams:
    """Draw all 64-position noise in original global image order, even for RECON."""
    names = ('flip', 'encoding_noise', 'encoding_time', 'recon_noise', 'recon_time',
             'recon_mask', 'diffusion_noise', 'class_dropout', 'assignment_shuffle', 'probe')

    def __init__(self, seed):
        self.generators = {name: torch.Generator().manual_seed(derive_seed(seed, name)) for name in self.names}
        self.times = np.random.RandomState(derive_seed(seed, 'time_draws') % 2**32)

    def draw(self, batch):
        def normal(name, shape):
            return torch.randn(shape, generator=self.generators[name], dtype=torch.float32)
        def uniform(name, shape):
            return torch.rand(shape, generator=self.generators[name], dtype=torch.float32)
        # B is the first axis before any grouping, rank partition, or native slice.
        v = self.times.normal(0., 1., (batch, 3)).astype(np.float32)
        u = torch.from_numpy(1/(1+np.exp(-v)))
        times = .5*u/(1-.5*u)
        return dict(flip=uniform('flip', (batch,)) < .5,
                    encoding_noise=normal('encoding_noise', (batch, 64, 32)),
                    encoding_time=.01*uniform('encoding_time', (batch,)),
                    recon_noise=normal('recon_noise', (batch, 64, 32)),
                    recon_time=.7+.3*uniform('recon_time', (batch,)),
                    recon_mask=uniform('recon_mask', (batch,)) < .5,
                    diffusion_noise=normal('diffusion_noise', (batch, 3, 64, 32)),
                    diffusion_time=times,
                    class_dropout=uniform('class_dropout', (batch, 3)) < .1)

    def state_dict(self):
        return dict(generators={k: v.get_state() for k, v in self.generators.items()}, times=self.times.get_state())

    def load_state_dict(self, state):
        assert set(state['generators']) == set(self.generators)
        for name, value in state['generators'].items():
            self.generators[name].set_state(value)
        self.times.set_state(state['times'])


def local_indices(rank, accumulation_index, world_size=4, micro=16):
    assert world_size == 4 and micro == 16
    start = accumulation_index*64 + rank*16
    return list(range(start, start+16))


def select_draws(draws, indices, device):
    return {name: value[indices].to(device) for name, value in draws.items()}


def load_images(rows, flips, device):
    transform = CenterCropTransform(256)
    images = []
    for row, flip in zip(rows, flips):
        payload = Path(row['absolute_path']).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == row['content_sha256'], row['image_id']
        with Image.open(io.BytesIO(payload)) as image:
            image = transform(image.convert('RGB'))
            if bool(flip):
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            images.append(to_tensor(image))
    return torch.stack(images).to(device)


def eval_rows(rows, count, seed):
    return sorted(rows, key=lambda r: hashlib.sha256(f'eval-v1|{seed}|{r["image_id"]}'.encode()).hexdigest())[:count]


def per_image_draws(rows, seed):
    """Per-image keys make evaluation noise independent of batch/rank ordering."""
    draws = [Streams(derive_seed(seed, row['image_id'])).draw(1) for row in rows]
    return {k: torch.cat([d[k] for d in draws]) for k in draws[0]}
