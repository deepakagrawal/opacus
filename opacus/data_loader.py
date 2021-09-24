import torch
from typing import Optional, Sequence
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import _worker_init_fn_t, _collate_fn_t
from opacus.utils.uniform_sampler import UniformWithReplacementSampler
from torch.utils.data._utils.collate import default_collate


def wrap_collate_with_empty(collate_fn: Optional[_collate_fn_t], sample_empty_shapes: Sequence):
    #TODO: does it work with padding and packed sequences?
    def collate(batch):
        if len(batch) > 0:
            return collate_fn(batch)
        else:
            return [torch.zeros(x) for x in sample_empty_shapes]

    return collate


def shape_safe(x):
    if hasattr(x, "shape"):
        return x.shape
    else:
        return ()


class DPDataLoader(DataLoader):

    def __init__(self, dataset: Dataset, sample_rate: float,
                 num_workers: int = 0, collate_fn: Optional[_collate_fn_t] = None,
                 pin_memory: bool = False, drop_last: bool = False,
                 timeout: float = 0, worker_init_fn: Optional[_worker_init_fn_t] = None,
                 multiprocessing_context=None, generator=None,
                 *, prefetch_factor: int = 2,
                 persistent_workers: bool = False):

        self.sample_rate = sample_rate
        batch_sampler = UniformWithReplacementSampler(
            num_samples=len(dataset), # type: ignore[assignment, arg-type]
            sample_rate=sample_rate,
            generator=generator,
        )
        sample_empty_shapes = [[0, *shape_safe(x)] for x in dataset[0]]
        if collate_fn is None:
            collate_fn = default_collate

        super().__init__(
            dataset=dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=wrap_collate_with_empty(collate_fn, sample_empty_shapes),
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
            multiprocessing_context=multiprocessing_context,
            generator=generator,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
        )

    @classmethod
    def from_data_loader(cls, data_loader: DataLoader):
        if isinstance(data_loader, cls):
            return data_loader

        return cls(
            dataset=data_loader.dataset,
            sample_rate=1/len(data_loader),
            num_workers=data_loader.num_workers,
            pin_memory=data_loader.pin_memory,
            drop_last=data_loader.drop_last,
            timeout=data_loader.timeout,
            worker_init_fn=data_loader.worker_init_fn,
            multiprocessing_context=data_loader.multiprocessing_context,
            generator=data_loader.generator,
            prefetch_factor=data_loader.prefetch_factor,
            persistent_workers=data_loader.persistent_workers
        )
