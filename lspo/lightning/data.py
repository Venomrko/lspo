from __future__ import annotations
import hashlib
import json
from dataclasses import asdict
from lightning.pytorch import LightningDataModule
from torch.utils.data import Dataset, DataLoader
from ..data.corpus import build_training_items

class UpdateDataset(Dataset):
    def __init__(self, items, updates, batch_size, rank, world_size, offset=0):
        self.items,self.updates,self.batch_size = items,updates,batch_size
        self.rank,self.world_size,self.offset = rank,world_size,offset
    def __len__(self):
        return self.updates
    def __getitem__(self,index):
        start = (index+self.offset)*self.batch_size*self.world_size + self.rank*self.batch_size
        return [self.items[(start+i)%len(self.items)] for i in range(self.batch_size)]

def identity(value):
    return value

class LSPODataModule(LightningDataModule):
    def __init__(self,config,items=None,limit=None):
        super().__init__()
        self.config,self.items,self.limit = config,items,limit
    def setup(self,stage=None):
        if self.items is None:
            self.items = build_training_items(self.config,limit=self.limit)
        if not self.items:
            raise ValueError("The training dataset is empty")
    def fingerprint(self):
        digest = hashlib.sha256()
        for item in self.items:
            digest.update(json.dumps(asdict(item),sort_keys=True,ensure_ascii=False,default=str).encode())
            digest.update(b"\n")
        return digest.hexdigest()

    def train_dataloader(self):
        model = self.trainer.lightning_module
        world = self.trainer.world_size
        count = self.config.optim.prompts_per_update
        if count % world:
            raise ValueError("prompts_per_update must be divisible by world size")
        dataset = UpdateDataset(self.items,model.total_steps,
            count*self.config.optim.gradient_accumulation,0,1,0)
        return DataLoader(dataset,batch_size=None,num_workers=0,collate_fn=identity)
