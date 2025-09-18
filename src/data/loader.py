from torch.utils.data import DataLoader
from src.data.read_data import read_data
from src.config import cfg

def get_loader():
    ds = read_data(reload=False, scaler=cfg.scaler()) 
    loader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
    return loader


if __name__=='__main':
    loader = get_loader()
    # do a for loop
