import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

class StreamDataset(IterableDataset):
    def __init__(self, data, seq_len: int, attention_mask=None, stride: int = 1, split:float | None = None, is_val=False,  dtype=torch.long, device='cpu'):
        self.data = data
        self.seq_len = seq_len
        self.attention_mask = attention_mask
        self.stride = stride
        self.dtype = dtype
        self.device = device
        self.split = split
        self.is_val = is_val

    def _get_range(self):
        total_len = len(self.data) - self.seq_len - 1

        worker = get_worker_info()
        if worker is None:
            return 0, total_len

        per_worker = total_len // worker.num_workers
        start = worker.id * per_worker
        end = start + per_worker

        if worker.id == worker.num_workers - 1:
            end = total_len

        return start, end

    def __iter__(self):
        start, end = self._get_range()

        if self.split:
            split_point = int(end * self.split)

        if self.is_val:
            start, end = split_point, end
        else:
            start, end = start, split_point

        for idx in range(start, end, self.stride):
            x = self.data[idx : idx + self.seq_len]
            y = self.data[idx + 1 : idx + self.seq_len + 1]

            yield {
                "decoder_input": torch.from_numpy(x).to(self.dtype).to(self.device),
                "attention_mask": self.attention_mask,
            }, torch.from_numpy(y).to(self.dtype).to(self.device)

    

if __name__ == '__main__':
    from torch.utils.data import DataLoader

    npy_path = 'E:/Documents/VS/Python/AGAT/AGAT_System/Traning data/DataSet.npy'

    data = np.load(npy_path, mmap_mode='r')

    dataset = StreamDataset(data, 10)

    data_loader = DataLoader(dataset, 2, True, num_workers=2, pin_memory=True)
    
    for input, label in data_loader:
        print(input.size(), input.size())