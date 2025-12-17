import torch
from torch.utils.data import DataLoader
from src.config import cfg
from src.data.multi_patch_dataset import MultiPatchDataset
import numpy as np

class MultiPatchLoaders():
    def __init__(self, data, base_batch_size = cfg.batch_size, base_patch_size = cfg.patch_size, shuffle=True, num_workers=0, pin_memory=False):
        self.loaders = []
        self.cum_lengths = []  # cumulative lengths to map global idx -> loader
        cum_len = 0
        for ds in data.datasets:
            height = ds.height
            width = ds.width
            avg_patch_size = np.sqrt(height * width)
            batch_size = max(1, int(base_batch_size * (base_patch_size / avg_patch_size)**2))
            print(f'[MultiPatchLoader] Width {width} batch size {batch_size}')
            loader = SinglePatchLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
            self.loaders.append(loader)
            cum_len += len(loader.dataset)
            self.cum_lengths.append(cum_len)
            self.loss_weights = np.concatenate([loader.loss_weights for loader in self.loaders])

    def __getitem__(self, index):
        return self.loaders[index]

    def __len__(self):
        return self.cum_lengths[-1]

    def get_samples(self, n_samples):
        # Generate n_samples random global indices
        indices = np.random.randint(0, len(self), size=n_samples)
        samples = []

        for idx in indices:
            # Find which loader this idx belongs to
            loader_idx = 0
            while idx >= self.cum_lengths[loader_idx]:
                loader_idx += 1
            # Compute local index within that loader
            local_idx = idx if loader_idx == 0 else idx - self.cum_lengths[loader_idx-1]
            batch = self.loaders[loader_idx].dataset[local_idx].unsqueeze(0)  # keep batch dim
            samples.append(batch)

        samples = torch.cat(samples, dim=0).to(cfg.device)
        return samples


class SinglePatchLoader(DataLoader):
    def __init__(self, dataset, batch_size=8, shuffle=True, num_workers=0, pin_memory=True):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
        self.height = dataset.height
        self.width = dataset.width

        # Compute loss weights based on dataset length and batch size
        full_batches = len(dataset) // batch_size
        last_batch_size = len(dataset) % batch_size
        self.loss_weights = np.array([batch_size] * full_batches + ([last_batch_size] if last_batch_size > 0 else []))

    def get_samples(self, n_samples):
        samples = []
        for batch in iter(self):
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            
            samples.append(batch)
            if len(samples) * cfg.batch_size >= n_samples:
                break
        samples = torch.cat(samples, dim=0)[:n_samples]
        samples = samples.to(cfg.device)
        return samples

def get_loaders(data : MultiPatchDataset):
    loaders = MultiPatchLoaders(data, shuffle=True, num_workers=0, pin_memory=False)
    return loaders


if __name__=='__main__':
    loaders = get_loaders()
