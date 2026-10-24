"""Independent exact NESTED40 permutations; source streams are reused unchanged."""
import hashlib
import torch
from campaign.a2.data import Streams, derive_seed


class ScreenStreams(Streams):
    def __init__(self, seed):
        super().__init__(seed)
        self.generators['b1_screen_assignment'] = torch.Generator().manual_seed(
            derive_seed(seed, 'b1_screen_assignment'))

    def counts(self, method):
        # All arms consume this independent stream. No image or label enters it.
        rng = self.generators['b1_screen_assignment']
        before = hashlib.sha256(rng.get_state().numpy().tobytes()).hexdigest()
        blocks = [torch.tensor([16]*32+[64]*32)[torch.randperm(64, generator=rng)] for _ in range(4)]
        counts = torch.cat(blocks)
        if method in ('PREFIX16', 'PREFIX64'):
            counts.fill_(16 if method == 'PREFIX16' else 64)
        elif method != 'NESTED_RANDOM':
            raise ValueError('Only the three SCREEN arms are implemented')
        return counts, dict(stream='b1_screen_assignment', state_before_sha256=before,
            state_after_sha256=hashlib.sha256(rng.get_state().numpy().tobytes()).hexdigest(),
            realized_counts=counts.tolist(), histogram=[int((counts == k).sum()) for k in (4,16,64)],
            distributed_microbatch_histograms=[[int((block == k).sum()) for k in (4,16,64)]
                                              for block in counts.reshape(4,64)],
            mean_tokens=counts.double().mean().item(), mean_tokens_sq=counts.double().square().mean().item())
