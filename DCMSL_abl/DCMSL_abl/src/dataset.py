import os.path as osp

from torch_geometric.datasets import Planetoid, CitationFull, WikiCS, Coauthor, Amazon
import torch_geometric.transforms as T

from ogb.nodeproppred import PygNodePropPredDataset
import numpy as np
import torch
from torch.utils.data import random_split
import matplotlib.pyplot as plt

from time import perf_counter as t
import os.path as osp
from torch_geometric.datasets import Planetoid, CitationFull, WikiCS, Coauthor, Amazon
import torch_geometric.transforms as T


import os
import os.path as osp
import ssl
import sys
import urllib
from typing import Optional

import os.path as osp
import ssl
import sys
import urllib
from typing import Optional

from torch_geometric.data.makedirs import makedirs


def download_url(url: str, folder: str, log: bool = True,
                 filename: Optional[str] = None):
    r"""Downloads the content of an URL to a specific folder.

    Args:
        url (str): The URL.
        folder (str): The folder.
        log (bool, optional): If :obj:`False`, will not print anything to the
            console. (default: :obj:`True`)
    """

    if filename is None:
        filename = url.rpartition('/')[2]
        filename = filename if filename[0] == '?' else filename.split('?')[0]

    path = osp.join(folder, filename)

    if osp.exists(path):  # pragma: no cover
        if log and 'pytest' not in sys.modules:
            print(f'Using existing file {filename}', file=sys.stderr)
        return path

    if log and 'pytest' not in sys.modules:
        print(f'Downloading {url}', file=sys.stderr)

    makedirs(folder)

    context = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=context)

    with open(path, 'wb') as f:
        # workaround for https://bugs.python.org/issue42853
        while True:
            chunk = data.read(10 * 1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

    return path

import os
import os.path as osp
from typing import Callable, List, Optional

import scipy.sparse as sp
import torch

from torch_geometric.data import (
    Data,
    InMemoryDataset,
    extract_zip,
)



import os
import os.path as osp
import shutil
from typing import Callable, List, Optional

import scipy.sparse as sp
import torch

from torch_geometric.data import (
    Data,
    InMemoryDataset,
    extract_zip,
)
from torch_geometric.typing import SparseTensor


class AttributedGraphDataset(InMemoryDataset):
    r"""A variety of attributed graph datasets from the
    `"Scaling Attributed Network Embedding to Massive Graphs"
    <https://arxiv.org/abs/2009.00826>`_ paper.

    Args:
        root (str): Root directory where the dataset should be saved.
        name (str): The name of the dataset (:obj:`"Wiki"`, :obj:`"Cora"`
            :obj:`"CiteSeer"`, :obj:`"PubMed"`, :obj:`"BlogCatalog"`,
            :obj:`"PPI"`, :obj:`"Flickr"`, :obj:`"Facebook"`, :obj:`"Twitter"`,
            :obj:`"TWeibo"`, :obj:`"MAG"`).
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)

    **STATS:**

    .. list-table::
        :widths: 10 10 10 10 10
        :header-rows: 1

        * - Name
          - #nodes
          - #edges
          - #features
          - #classes
        * - Wiki
          - 2,405
          - 17,981
          - 4,973
          - 17
        * - Cora
          - 2,708
          - 5,429
          - 1,433
          - 7
        * - CiteSeer
          - 3,312
          - 4,715
          - 3,703
          - 6
        * - PubMed
          - 19,717
          - 44,338
          - 500
          - 3
        * - BlogCatalog
          - 5,196
          - 343,486
          - 8,189
          - 6
        * - PPI
          - 56,944
          - 1,612,348
          - 50
          - 121
        * - Flickr
          - 7,575
          - 479,476
          - 12,047
          - 9
        * - Facebook
          - 4,039
          - 88,234
          - 1,283
          - 193
        * - TWeibo
          - 2,320,895
          - 9,840,066
          - 1,657
          - 8
        * - MAG
          - 59,249,719
          - 978,147,253
          - 2,000
          - 100
    """
    url = 'https://docs.google.com/uc?export=download&id={}&confirm=t'

    datasets = {
        'wiki': '1EPhlbziZTQv19OsTrKrAJwsElbVPEbiV',
        'cora': '1FyVnpdsTT-lhkVPotUW8OVeuCi1vi3Ey',
        'citeseer': '1d3uQIpHiemWJPgLgTafi70RFYye7hoCp',
        'pubmed': '1DOK3FfslyJoGXUSCSrK5lzdyLfIwOz6k',
        'blogcatalog': '178PqGqh67RUYMMP6-SoRHDoIBh8ku5FS',
        'ppi': '1dvwRpPT4gGtOcNP_Q-G1TKl9NezYhtez',
        'flickr': '1tZp3EB20fAC27SYWwa-x66_8uGsuU62X',
        'facebook': '12aJWAGCM4IvdGI2fiydDNyWzViEOLZH8',
        'twitter': '1fUYggzZlDrt9JsLsSdRUHiEzQRW1kSA4',
        'tweibo': '1-2xHDPFCsuBuFdQN_7GLleWa8R_t50qU',
        'mag': '1ggraUMrQgdUyA3DjSRzzqMv0jFkU65V5',
    }

    def __init__(self, root: str, name: str,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        self.name = name.lower()
        assert self.name in self.datasets.keys()
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> List[str]:
        return ['attrs.npz', 'edgelist.txt', 'labels.txt']

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        url = self.url.format(self.datasets[self.name])
        path = download_url(url, self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.unlink(path)
        path = osp.join(self.raw_dir, f'{self.name}.attr')
        if self.name == 'mag':
            path = osp.join(self.raw_dir, self.name)
        for name in self.raw_file_names:
            os.rename(osp.join(path, name), osp.join(self.raw_dir, name))
        shutil.rmtree(path)

    def process(self):
        import pandas as pd

        x = sp.load_npz(self.raw_paths[0])
        if x.shape[-1] > 10000 or self.name == 'mag':
            x = SparseTensor.from_scipy(x).to(torch.float)
        else:
            x = torch.from_numpy(x.todense()).to(torch.float)

        df = pd.read_csv(self.raw_paths[1], header=None, sep=None,
                         engine='python')
        edge_index = torch.from_numpy(df.values).t().contiguous()

        with open(self.raw_paths[2], 'r') as f:
            ys = f.read().split('\n')[:-1]
            ys = [[int(y) - 1 for y in row.split()[1:]] for row in ys]
            multilabel = max([len(y) for y in ys]) > 1

        if not multilabel:
            y = torch.tensor(ys).view(-1)
        else:
            num_classes = max([y for row in ys for y in row]) + 1
            y = torch.zeros((len(ys), num_classes), dtype=torch.float)
            for i, row in enumerate(ys):
                for j in row:
                    y[i, j] = 1.

        data = Data(x=x, edge_index=edge_index, y=y)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.name.capitalize()}()'

from ogb.nodeproppred import PygNodePropPredDataset
def get_dataset(path, name):
    #assert name in ['WikiCS', 'Coauthor-CS', 'Amazon-Computers', 'Amazon-Photo']
    #root_path = './datasets'

    if name == 'Coauthor-CS':
        return Coauthor(root=path, name='cs', transform=T.NormalizeFeatures())
    if name == 'WikiCS':
        return WikiCS(root=path)
    if name == 'Amazon-Computers':
        return Amazon(root=path, name='computers', transform=T.NormalizeFeatures())
    if name == 'Amazon-Photo':
        return Amazon(root="/home/dhc/POT-GCL/GCA/datasets/Amazon-Photo/", name='photo', transform=T.NormalizeFeatures())
    if name=='BlogCatalog':
        return AttributedGraphDataset(root=path, name='blogcatalog', transform=T.NormalizeFeatures())
    if name=='Flickr':
        return AttributedGraphDataset(root=path, name='flickr', transform=NormalizeFeaturesSparse())
    if name=='Ogbn':
        return PygNodePropPredDataset(root=path, name='ogbn-arxiv', transform=T.NormalizeFeatures())
        
from typing import List, Union
import copy
from typing import Any, Callable, Iterator, Optional, Sequence

import torch

from torch_geometric.data import Batch
def from_smiles(smiles: str, with_hydrogen: bool = False,
                kekulize: bool = False) -> 'torch_geometric.data.Data':
    r"""Converts a SMILES string to a :class:`torch_geometric.data.Data`
    instance.

    Args:
        smiles (str): The SMILES string.
        with_hydrogen (bool, optional): If set to :obj:`True`, will store
            hydrogens in the molecule graph. (default: :obj:`False`)
        kekulize (bool, optional): If set to :obj:`True`, converts aromatic
            bonds to single/double bonds. (default: :obj:`False`)
    """
    from rdkit import Chem, RDLogger

    from torch_geometric.data import Data

    RDLogger.DisableLog('rdApp.*')

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        mol = Chem.MolFromSmiles('')
    if with_hydrogen:
        mol = Chem.AddHs(mol)
    if kekulize:
        Chem.Kekulize(mol)

    xs: List[List[int]] = []
    for atom in mol.GetAtoms():
        row: List[int] = []
        row.append(x_map['atomic_num'].index(atom.GetAtomicNum()))
        row.append(x_map['chirality'].index(str(atom.GetChiralTag())))
        row.append(x_map['degree'].index(atom.GetTotalDegree()))
        row.append(x_map['formal_charge'].index(atom.GetFormalCharge()))
        row.append(x_map['num_hs'].index(atom.GetTotalNumHs()))
        row.append(x_map['num_radical_electrons'].index(
            atom.GetNumRadicalElectrons()))
        row.append(x_map['hybridization'].index(str(atom.GetHybridization())))
        row.append(x_map['is_aromatic'].index(atom.GetIsAromatic()))
        row.append(x_map['is_in_ring'].index(atom.IsInRing()))
        xs.append(row)

    x = torch.tensor(xs, dtype=torch.long).view(-1, 9)

    edge_indices, edge_attrs = [], []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        e = []
        e.append(e_map['bond_type'].index(str(bond.GetBondType())))
        e.append(e_map['stereo'].index(str(bond.GetStereo())))
        e.append(e_map['is_conjugated'].index(bond.GetIsConjugated()))

        edge_indices += [[i, j], [j, i]]
        edge_attrs += [e, e]

    edge_index = torch.tensor(edge_indices)
    edge_index = edge_index.t().to(torch.long).view(2, -1)
    edge_attr = torch.tensor(edge_attrs, dtype=torch.long).view(-1, 3)

    if edge_index.numel() > 0:  # Sort indices.
        perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
        edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smiles=smiles)

try:
    from torch.utils.data import IterDataPipe, functional_datapipe
    from torch.utils.data.datapipes.iter import Batcher as IterBatcher
except ImportError:
    IterDataPipe = IterBatcher = object  # type: ignore

    def functional_datapipe(name: str) -> Callable:  # type: ignore
        return lambda cls: cls


@functional_datapipe('batch_graphs')
class Batcher(IterBatcher):
    def __init__(
        self,
        dp: IterDataPipe,
        batch_size: int,
        drop_last: bool = False,
    ) -> None:
        super().__init__(
            dp,
            batch_size=batch_size,
            drop_last=drop_last,
            wrapper_class=Batch.from_data_list,
        )


@functional_datapipe('parse_smiles')
class SMILESParser(IterDataPipe):
    def __init__(
        self,
        dp: IterDataPipe,
        smiles_key: str = 'smiles',
        target_key: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.dp = dp
        self.smiles_key = smiles_key
        self.target_key = target_key

    def __iter__(self) -> Iterator:
        for d in self.dp:
            if isinstance(d, str):
                data = from_smiles(d)
            elif isinstance(d, dict):
                data = from_smiles(d[self.smiles_key])
                if self.target_key is not None:
                    y = d.get(self.target_key, None)
                    if y is not None:
                        y = float(y) if len(y) > 0 else float('NaN')
                        data.y = torch.tensor([y], dtype=torch.float)
            else:
                raise ValueError(
                    f"'{self.__class__.__name__}' expected either a string or "
                    f"a dict as input (got '{type(d)}')")

            yield data


class DatasetAdapter(IterDataPipe):
    def __init__(self, dataset: Sequence[Any]) -> None:
        super().__init__()
        self.dataset = dataset
        self.range = range(len(self))

    def is_shardable(self) -> bool:
        return True

    def apply_sharding(self, num_shards: int, shard_idx: int) -> None:
        self.range = range(shard_idx, len(self), num_shards)

    def __iter__(self) -> Iterator:
        for i in self.range:
            yield self.dataset[i]

    def __len__(self) -> int:
        return len(self.dataset)


def functional_transform(name: str) -> Callable:
    def wrapper(cls: Any) -> Any:
        @functional_datapipe(name)
        class DynamicMapper(IterDataPipe):
            def __init__(
                self,
                dp: IterDataPipe,
                *args: Any,
                **kwargs: Any,
            ) -> None:
                super().__init__()
                self.dp = dp
                self.fn = cls(*args, **kwargs)

            def __iter__(self) -> Iterator:
                for data in self.dp:
                    yield self.fn(copy.copy(data))

        return cls

    return wrapper
from torch_geometric.data import Data, HeteroData

from torch_geometric.transforms import BaseTransform


@functional_transform('normalize_features_sparse')
class NormalizeFeaturesSparse(BaseTransform):
    r"""Row-normalizes the attributes given in :obj:`attrs` to sum-up to one
    (functional name: :obj:`normalize_features`).

    Args:
        attrs (List[str]): The names of attributes to normalize.
            (default: :obj:`["x"]`)
    """
    def __init__(self, attrs: List[str] = ["x"]):
        self.attrs = attrs

    def __call__(
        self,
        data: Union[Data, HeteroData],
    ) -> Union[Data, HeteroData]:
        for store in data.stores:
            for key, value in store.items(*self.attrs):
                value = value.to_dense()
                value = value - value.min()
                value.div_(value.sum(dim=-1, keepdim=True).clamp_(min=1.))
                store[key] = value
        return data


from typing import List, Union

from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import BaseTransform


@functional_transform('normalize_features_sparse')
class NormalizeFeaturesSparse(BaseTransform):
    r"""Row-normalizes the attributes given in :obj:`attrs` to sum-up to one
    (functional name: :obj:`normalize_features`).

    Args:
        attrs (List[str]): The names of attributes to normalize.
            (default: :obj:`["x"]`)
    """
    def __init__(self, attrs: List[str] = ["x"]):
        self.attrs = attrs

    def __call__(
        self,
        data: Union[Data, HeteroData],
    ) -> Union[Data, HeteroData]:
        for store in data.stores:
            for key, value in store.items(*self.attrs):
                value = value.to_dense()
                value = value - value.min()
                value.div_(value.sum(dim=-1, keepdim=True).clamp_(min=1.))
                store[key] = value
        return data
def generate_split(num_samples: int, train_ratio: float, val_ratio: float):
    train_len = int(num_samples * train_ratio)
    val_len = int(num_samples * val_ratio)
    test_len = num_samples - train_len - val_len

    train_set, test_set, val_set = random_split(torch.arange(0, num_samples), (train_len, test_len, val_len))

    idx_train, idx_test, idx_val = train_set.indices, test_set.indices, val_set.indices
    train_mask = torch.zeros((num_samples,)).to(torch.bool)
    test_mask = torch.zeros((num_samples,)).to(torch.bool)
    val_mask = torch.zeros((num_samples,)).to(torch.bool)

    train_mask[idx_train] = True
    test_mask[idx_test] = True
    val_mask[idx_val] = True

    return train_mask, test_mask, val_mask

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_alpha_beta(l, u, alpha):
    alpha_L= torch.zeros(l.shape,device=l.device)
    alpha_U, beta_L, beta_U = torch.clone(alpha_L), torch.clone(alpha_L), torch.clone(alpha_L)
    pos_mask = l >= 0
    neg_mask = u <= 0
    alpha_L[pos_mask] = 1
    alpha_U[pos_mask] = 1
    alpha_L[neg_mask] = alpha
    alpha_U[neg_mask] = alpha
    not_mask = ~(pos_mask | neg_mask)
    alpha_not_upp = u[not_mask] - alpha * l[not_mask]
    alpha_not = alpha_not_upp / (u[not_mask] - l[not_mask])
    alpha_L[not_mask] = alpha_not
    alpha_U[not_mask] = alpha_not
    beta_U[not_mask] = (alpha - 1) * u[not_mask] * l[not_mask] / alpha_not_upp
    return alpha_L, alpha_U, beta_L, beta_U

def get_crown_weights(l1, u1, l2, u2, alpha, gcn_weights, Wcl):
    alpha_2_L, alpha_2_U, beta_2_L, beta_2_U = get_alpha_beta(l2, u2, alpha) # onehop
    alpha_1_L, alpha_1_U, beta_1_L, beta_1_U = get_alpha_beta(l1, u1, alpha) # twohop
    lambda_2 = torch.where(Wcl >= 0, alpha_2_L, alpha_2_U) # N * d
    Delta_2 = torch.where(Wcl >= 0, beta_2_L, beta_2_U) # N * d
    Lambda_2 = lambda_2 * Wcl # N * d
    W1_tensor, b1_tensor, W2_tensor, b2_tensor = gcn_weights
    W_tilde_2 = Lambda_2 @ W2_tensor.T
    b_tilde_2 = torch.diag(Lambda_2 @ (Delta_2 + b2_tensor).T)
    lambda_1 = torch.where(W_tilde_2 >= 0, alpha_1_L, alpha_1_U)
    Delta_1 = torch.where(W_tilde_2 >= 0, beta_1_L, beta_1_U)
    Lambda_1 = lambda_1 * W_tilde_2
    W_tilde_1 = Lambda_1 @ W1_tensor.T
    b_tilde_1 = torch.diag(Lambda_1 @ (Delta_1 + b1_tensor).T)
    return W_tilde_1, b_tilde_1, W_tilde_2, b_tilde_2
def get_batch(node_list, batch_size, epoch):
    num_nodes = len(node_list)
    num_batches = (num_nodes - 1) // batch_size + 1
    i = epoch % num_batches
    if (i + 1) * batch_size >= len(node_list):
        node_list_batch = node_list[i * batch_size:]
    else:
        node_list_batch = node_list[i * batch_size:(i + 1) * batch_size]
    return node_list_batch
def get_A_bounds(dataset, drop_rate):
    upper_lower_file = osp.join(osp.expanduser('~/datasets'),f"bounds/{dataset}_{drop_rate}_upper_lower.pkl")
    if osp.exists(upper_lower_file):
        A_upper, A_lower = torch.load(upper_lower_file)
    else:
        A_upper, A_lower = None, None
    return A_upper, A_lower
def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])