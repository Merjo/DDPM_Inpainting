import torch
from torch.utils.data import DataLoader
from src.data.read_data import read_data
from src.config import cfg

class PrecipitationPatchLoader(DataLoader):
    def __init__(self, dataset, batch_size=8, shuffle=True, num_workers=0, pin_memory=True):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
    def get_samples(self, n_samples):
        samples = []
        for batch in self:
            samples.append(batch)
            if len(samples) * cfg.batch_size >= n_samples:
                break
        samples = torch.cat(samples, dim=0)[:n_samples]
        samples = samples.to(cfg.device)
        return samples

def get_loader(ds=None):
    if ds is None:
        ds = cfg.data
    loader = PrecipitationPatchLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    return loader


if __name__=='__main__':
    loader = get_loader()
