import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import yaml


class HdfDataset(Dataset):
    def __init__(self, data_path, *names):
        super(HdfDataset, self).__init__()
        self.dataset = h5py.File(data_path, 'r')
        self.dtname = list(self.dataset.keys())
        self.names = names

    def __len__(self):
        return len(self.dtname)

    def __getitem__(self, idx):
        data_n = tuple()
        if self.names is not None:
            for name in self.names:
                datum = self.dataset.get(self.dtname[idx]).get(name)[:]
                datum = datum.reshape(-1, datum.shape[-2], datum.shape[-1])
                datum = torch.as_tensor(datum)
                datum = datum.type(torch.FloatTensor)
                data_n = data_n + (datum,)
            # data_n = data_n + (self.dtname[idx],)
        else:
            print('Wrong args!')

        return data_n


def HdfDataLoader(train, **kwargs):
    if train:
        kwargs_train = kwargs.get('training')
        names_train = kwargs_train['name']

        datasets = HdfDataset(kwargs_train['path'], *names_train)
        indices = torch.randperm(len(datasets)).tolist()
        occupy = int(kwargs_train['occupy'] * len(indices))
        data_train = torch.utils.data.Subset(datasets, indices[:occupy])
        data_val = torch.utils.data.Subset(datasets, indices[occupy:])

        dataloader_train = DataLoader(dataset=data_train,
                                      batch_size=kwargs_train['batch_size'],
                                      shuffle=kwargs_train['shuffle'],
                                      num_workers=kwargs_train['num_worker'])
        dataloader_val = DataLoader(dataset=data_val,
                                    batch_size=kwargs_train['batch_size'],
                                    shuffle=kwargs_train['shuffle'],
                                    num_workers=kwargs_train['num_worker'])

        return dataloader_train, dataloader_val
    else:
        kwargs_pred = kwargs.get('prediction')
        names_pred = kwargs_pred['name']

        data_pred = HdfDataset(kwargs_pred['path'], *names_pred)
        dataloader_pred = DataLoader(dataset=data_pred,
                                     batch_size=kwargs_pred['batch_size'],
                                     shuffle=kwargs_pred['shuffle'],
                                     num_workers=kwargs_pred['num_worker'])
        return dataloader_pred


if __name__ == '__main__':
    cfg = yaml.safe_load(open('../config.yml', 'r'))
    print(cfg)
    # dt, dv = HdfDataLoader(True, **cfg.get('data'))
    # for b, (x, y) in enumerate(dv):
    #     print(y[0])
    # print(len(dt))
