import os
import sys
import time
import csv
import json
import warnings
import numpy as np
import ase
import glob
from ase import io
from scipy.stats import rankdata
import pandas as pd

##torch imports
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader, Dataset, Data, InMemoryDataset
from torch_geometric.utils import dense_to_sparse, degree, add_self_loops
import torch_geometric.transforms as T
from torch_geometric.utils import degree
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.cif import CifParser


################################################################################
# Data splitting
################################################################################

##basic train, val, test split
def split_data_own(dataset,
                   data_path,
                   aug='False',
                   repeat=0,
                   find_disorder=False,
                   temperature=10,
                   ):
    if data_path.find('disorder') != -1:
        temperature = 17
    train_file = os.path.join(data_path, 'train_Tc.csv')
    val_file = os.path.join(data_path, 'val_Tc.csv')
    test_file = os.path.join(data_path, 'test_Tc.csv')
    whole_file = os.path.join(data_path, 'targets.csv')
    train_data = pd.read_csv(train_file)
    val_data = pd.read_csv(val_file)
    test_data = pd.read_csv(test_file)

    whole_data = pd.read_csv(whole_file, header=None)
    train_cf = train_data.iloc[:, 0].values
    val_cf = val_data.iloc[:, 0].values
    test_cf = test_data.iloc[:, 0].values
    try:
        whole_cf = whole_data.iloc[:, 0].values + '.cif'
    except:
        whole_cf = np.char.add(whole_data.iloc[:, 0].values.astype(str), '.cif')
    train_index = np.where(np.isin(whole_cf, train_cf) == True)[0]
    val_index = [list(whole_cf).index(i) for i in val_cf]
    test_index = [list(whole_cf).index(i) for i in test_cf]

    if aug == 'True':
        whole_10 = whole_data.iloc[train_index]
        train_index_10 = whole_10[whole_10.iloc[:, 3] > temperature].index.tolist()
        train_index = train_index.tolist() + train_index_10 * repeat

    train_dataset = dataset[train_index]
    val_dataset = dataset[val_index]
    test_dataset = dataset[test_index]
    if find_disorder:
        train_dataset = [data for data in train_dataset if data.if_order[0, 0].item() == 1]
        val_dataset = [data for data in val_dataset if data.if_order[0, 0].item() == 1]
        test_dataset = [data for data in test_dataset if data.if_order[0, 0].item() == 1]
        print('Find disorder,Train:', len(train_dataset), 'Val:', len(val_dataset), 'Test:', len(test_dataset))
    return train_dataset, val_dataset, test_dataset


################################################################################
# 数据集处理
################################################################################

## 获取预处理后数据集
def get_dataset(data_path, target_index, reprocess="False", processing_args=None):
    if processing_args == None:
        processed_path = "processed"
    else:
        processed_path = processing_args.get("processed_path", "processed")

    transforms = GetY(index=target_index)

    data_path = os.path.normpath(data_path.strip("'"))
    if os.path.exists(data_path) == False:
        print("Data not found in:", data_path)
        sys.exit()

    ## reprocess数据集
    if reprocess == "True":
        os.system("rm -rf " + os.path.join(data_path, processed_path))
        process_data(data_path, processed_path, processing_args)

    if os.path.exists(os.path.join(data_path, processed_path, "data.pt")) == True:
        dataset = StructureDataset(
            data_path,
            processed_path,
            transforms,
        )
    else: # 检查数据集是否存在，若不存在则进行reprocess
        process_data(data_path, processed_path, processing_args)
        dataset = StructureDataset(
            data_path,
            processed_path,
            transforms,
        )
    return dataset


## torch_geometric fornat 数据集
class StructureDataset(InMemoryDataset):
    def __init__(
            self, data_path, processed_path="processed", transform=None, pre_transform=None
    ):
        self.data_path = data_path
        self.processed_path = processed_path
        super(StructureDataset, self).__init__(data_path, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_dir(self):
        return os.path.join(self.data_path, self.processed_path)

    @property
    def processed_file_names(self):
        file_names = ["data.pt"]
        return file_names


################################################################################
# Pretrain原子特征构建
################################################################################
class AtomInitializer(object):
    def __init__(self, atom_types):
        self.atom_types = set(atom_types)
        self._embedding = {}

    def get_atom_fea(self, atom_type):
        assert atom_type in self.atom_types
        return self._embedding[atom_type]

    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())
        self._decodedict = {idx: atom_type for atom_type, idx in
                            self._embedding.items()}

    def state_dict(self):
        return self._embedding

    def decode(self, idx):
        if not hasattr(self, '_decodedict'):
            self._decodedict = {idx: atom_type for atom_type, idx in
                                self._embedding.items()}
        return self._decodedict[idx]

class AtomCustomJSONInitializer(AtomInitializer):
    def __init__(self, elem_embedding_file):
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)
        elem_embedding = {int(key): value for key, value
                          in elem_embedding.items()}
        atom_types = set(elem_embedding.keys())
        super(AtomCustomJSONInitializer, self).__init__(atom_types)
        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)

def get_pretrain_data(crystal):
    atom_init_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                  "atom_init.json", )
    ari = AtomCustomJSONInitializer(atom_init_file)

    ### disorder处理方式；
    atom_fea = []
    for site in crystal:
        fea = np.array([0.0] * 92)
        for element, occupancy in site.species.items():
            fea += np.array(ari.get_atom_fea(element.Z)) * occupancy
        atom_fea.append(fea)
    atom_fea = np.vstack(atom_fea).astype(float)
    atom_fea = torch.Tensor(atom_fea)

    return atom_fea


################################################################################

################################################################################
#  特征构建
################################################################################
def create_global_feat(atoms_index_arr):
    comp = np.zeros(108)
    temp = np.unique(atoms_index_arr, return_counts=True)
    for i in range(len(temp[0])):
        comp[temp[0][i]] = temp[1][i] / temp[1].sum()
    return comp.reshape(1, -1)


def process_data(data_path, processed_path, processing_args):
    ##Begin processing data
    print("Processing data to: " + os.path.join(data_path, processed_path))
    assert os.path.exists(data_path), "Data path not found in " + data_path # 检查数据路径

    # 导入原子特征字典
    dict_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             "dictionary_default.json",)
    assert os.path.exists(dict_path), "Atom Dict Not found! "

    print("Using default dictionary.")
    atom_dictionary = get_dictionary(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "dictionary_default.json",
        )
    )
        

    # 导入Tc target
    target_property_file = os.path.join(data_path, processing_args["target_path"])
    assert os.path.exists(target_property_file), (
            "targets not found in " + target_property_file
    )
    with open(target_property_file) as f:
        reader = csv.reader(f)
        target_data = [row for row in reader]

    # 晶胞特征构建，cif数据处理
    data_list = []
    for index in range(0, len(target_data)):

        structure_id = target_data[index][0]
        data = Data()
        file_name = os.path.join(data_path, structure_id + "." + processing_args["data_format"])
        # check if file exists
        if os.path.exists(file_name) == False:
            print("File not found: ", file_name)
            continue

        # 存在晶胞数据使用pymatgen报错，同时使用ase进行读取
        try:
            parser = CifParser(file_name)
            pym_atoms = parser.parse_structures(primitive=True)[0]
        except:
            pym_atoms = ase.io.read(file_name)
            pym_atoms = AseAtomsAdaptor.get_structure(pym_atoms)

        ## 超级晶胞构建：扩充单个原子的晶胞
        if len(pym_atoms) == 1:
            pym_atoms = pym_atoms * (2, 2, 2)
        data.pym_atoms = pym_atoms

        ### Pretrain ### 
        # 获取pretrain原子特征
        atom_fea = get_pretrain_data(pym_atoms)
        data.pretrain_data = atom_fea

        ##Obtain distance matrix with ase
        distance_matrix = pym_atoms.distance_matrix
        ##Create sparse graph from distance matrix
        distance_matrix_trimmed = threshold_sort(
            distance_matrix,
            processing_args["graph_max_radius"],
            processing_args["graph_max_neighbors"],
            adj=False,
        )

        distance_matrix_trimmed = torch.Tensor(distance_matrix_trimmed)
        out = dense_to_sparse(distance_matrix_trimmed)
        edge_index = out[0]
        edge_weight = out[1]

        # 原始自连接
        edge_index, edge_weight = add_self_loops(
            edge_index, edge_weight, fill_value=0)

        data.edge_index = edge_index
        data.edge_weight = edge_weight

        distance_matrix_mask = (
                distance_matrix_trimmed.fill_diagonal_(1) != 0
        ).int()

        data.edge_descriptor = {}
        data.edge_descriptor["distance"] = edge_weight
        data.edge_descriptor["mask"] = distance_matrix_mask

        # 基类特征构建
        target = target_data[index][3:]
        y = torch.Tensor(np.array([target], dtype=np.float32))
        family = target_data[index][1]
        family_index = None
        if family == "h_based":
            family_index = torch.tensor(0)
        elif family == "cuprate":
            family_index = torch.tensor(1)
        elif family == "iron-based":
            family_index = torch.tensor(2)
        elif family == "heavy-Fermion":
            family_index = torch.tensor(3)
        elif family == "others":
            family_index = torch.tensor(4)
        data.family = F.one_hot(family_index, num_classes=5).float().reshape(1, -1)
        if target_data[index][2] == "order":
            data.if_order = torch.tensor([0, 1]).float().reshape(1, -1)
        else:
            data.if_order = torch.tensor([1, 0]).float().reshape(1, -1)
        data.y = y

        # 全局特征构建
        disorder = False
        for site in pym_atoms:
            for element, occupancy in site.species.items():
                if occupancy < 1:
                    disorder = True
                    break
            break
        occupancy1 = []
        ### disorder根据occupy特殊处理
        if disorder:
            atom_index = []
            atom_index1 = []
            for site in pym_atoms:
                atom_index.append(site.species.elements[0].Z)
                try:
                    atom_index1.append(site.species.elements[1].Z)
                except:
                    atom_index1.append(site.species.elements[0].Z)
                for element, occupancy in site.species.items():
                    occupancy1.append(occupancy)
                    break
            glob_feat = create_global_feat(atom_index)
            glob_feat1 = create_global_feat(atom_index1)
            occupancy_mean = np.array(occupancy1).mean()
            glob_feat = glob_feat * occupancy_mean + glob_feat1 * (1 - occupancy_mean)
        else:
            atom_index = []
            for site in pym_atoms:
                atom_index.append(site.species.elements[0].Z)
            glob_feat = create_global_feat(atom_index)
            occupancy1 = [1] * len(atom_index)
            atom_index1 = atom_index
        data.glob_feat = torch.Tensor(glob_feat).float().repeat(len(atom_index), 1)

        # 编码disorder程度，[order占比，替代无序占比，位置无序占比]
        atom_num = len(atom_index)
        order_num = 0
        replace_num = 0
        position_num = 0
        for i in range(atom_num):
            if occupancy1[i] == 1:
                order_num += 1
            else:
                if atom_index[i] != atom_index1[i]:
                    replace_num += 1
                else:
                    position_num += 1
        data.disorder_feat = torch.Tensor(
            [order_num / atom_num, replace_num / atom_num, position_num / atom_num]).reshape(1, -1)

        # 记录lattice和angle等其他晶胞的信息
        info = np.array([pym_atoms.lattice.a, pym_atoms.lattice.b, pym_atoms.lattice.c,
                         pym_atoms.lattice.alpha, pym_atoms.lattice.beta, pym_atoms.lattice.gamma,
                         pym_atoms.volume, pym_atoms.density]).reshape(1, -1)
        info = torch.Tensor(info).float()

        if data_path.find('disorder') != -1:
            info = torch.cat([info, data.family, data.disorder_feat], dim=1)
        else:
            info = torch.cat([info, data.family], dim=1)

        info = info.repeat(len(atom_index), 1)
        data.info = torch.Tensor(info).float()
        data.glob_feat = torch.cat([data.glob_feat, data.info], dim=1)

        data.structure_id = [[structure_id] * len(data.y)]

        # 打印处理进度
        if processing_args["verbose"] == "True" and (
                (index + 1) % 500 == 0 or (index + 1) == len(target_data)
        ):
            print("Data processed: ", index + 1, "out of", len(target_data))

        # 加入nodes的特征（disorder版本）
        atom_fea = []
        for site in pym_atoms:
            fea = np.array([0.0] * 100)
            for element, occupancy in site.species.items():
                fea += np.array(atom_dictionary[str(element.Z)]) * occupancy
            atom_fea.append(fea)
        atom_fea = np.vstack(atom_fea).astype(float)
        data.x = torch.Tensor(atom_fea)

        data_list.append(data)

    ##Adds node degree to node features (appears to improve performance)
    for index in range(0, len(data_list)):
        data_list[index] = OneHotDegree(
            data_list[index], max_degree=processing_args["graph_max_neighbors"] + 1
        )

    ##Generate edge features
    
    # 边特征计算
    ## Distance descriptor using a Gaussian basis
    distance_gaussian = GaussianSmearing(
        0, 1, processing_args["graph_edge_length"], 0.2
    )
    NormalizeEdge(data_list, "distance")
    for index in range(0, len(data_list)):
        data_list[index].edge_attr = distance_gaussian(
            data_list[index].edge_descriptor["distance"]
        )
        if processing_args["verbose"] == "True" and (
                (index + 1) % 500 == 0 or (index + 1) == len(target_data)
        ):
            print("Edge processed: ", index + 1, "out of", len(target_data))

    Cleanup(data_list, ["ase", "edge_descriptor"])

    if os.path.isdir(os.path.join(data_path, processed_path)) == False:
        os.mkdir(os.path.join(data_path, processed_path))

    # 存储处理后数据集
    data, slices = InMemoryDataset.collate(data_list)
    torch.save((data, slices), os.path.join(data_path, processed_path, "data.pt"))


################################################################################
#  预处理函数
################################################################################

##Selects edges with distance threshold and limited number of neighbors
def threshold_sort(matrix, threshold, neighbors, reverse=False, adj=False):
    mask = matrix > threshold
    distance_matrix_trimmed = np.ma.array(matrix, mask=mask)
    if reverse == False:
        distance_matrix_trimmed = rankdata(
            distance_matrix_trimmed, method="ordinal", axis=1
        )
    elif reverse == True:
        distance_matrix_trimmed = rankdata(
            distance_matrix_trimmed * -1, method="ordinal", axis=1
        )
    distance_matrix_trimmed = np.nan_to_num(
        np.where(mask, np.nan, distance_matrix_trimmed)
    )
    distance_matrix_trimmed[distance_matrix_trimmed > neighbors + 1] = 0

    if adj == False:
        distance_matrix_trimmed = np.where(
            distance_matrix_trimmed == 0, distance_matrix_trimmed, matrix
        )
        return distance_matrix_trimmed
    elif adj == True:
        adj_list = np.zeros((matrix.shape[0], neighbors + 1))
        adj_attr = np.zeros((matrix.shape[0], neighbors + 1))
        for i in range(0, matrix.shape[0]):
            temp = np.where(distance_matrix_trimmed[i] != 0)[0]
            adj_list[i, :] = np.pad(
                temp,
                pad_width=(0, neighbors + 1 - len(temp)),
                mode="constant",
                constant_values=0,
            )
            adj_attr[i, :] = matrix[i, adj_list[i, :].astype(int)]
        distance_matrix_trimmed = np.where(
            distance_matrix_trimmed == 0, distance_matrix_trimmed, matrix
        )
        return distance_matrix_trimmed, adj_list, adj_attr


##Slightly edited version from pytorch geometric to create edge from gaussian basis
class GaussianSmearing(torch.nn.Module):
    def __init__(self, start=0.0, stop=5.0, resolution=50, width=0.05, **kwargs):
        super(GaussianSmearing, self).__init__()
        offset = torch.linspace(start, stop, resolution)
        # self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.coeff = -0.5 / ((stop - start) * width) ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist):
        dist = dist.unsqueeze(-1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


##Obtain node degree in one-hot representation
def OneHotDegree(data, max_degree, in_degree=False, cat=True):
    idx, x = data.edge_index[1 if in_degree else 0], data.x
    deg = degree(idx, data.num_nodes, dtype=torch.long)
    deg = F.one_hot(deg, num_classes=max_degree + 6).to(torch.float)

    if x is not None and cat:
        x = x.view(-1, 1) if x.dim() == 1 else x
        data.x = torch.cat([x, deg.to(x.dtype)], dim=-1)
    else:
        data.x = deg

    return data


##Obtain dictionary file for elemental features
def get_dictionary(dictionary_file):
    with open(dictionary_file) as f:
        atom_dictionary = json.load(f)
    return atom_dictionary


##Deletes unnecessary data due to slow dataloader
def Cleanup(data_list, entries):
    for data in data_list:
        for entry in entries:
            try:
                delattr(data, entry)
            except Exception:
                pass


##Get min/max ranges for normalized edges
def GetRanges(dataset, descriptor_label):
    mean = 0.0
    std = 0.0
    feature_max = 0.0
    feature_min = 0.0
    for index in range(0, len(dataset)):
        if len(dataset[index].edge_descriptor[descriptor_label]) > 0:
            if index == 0:
                feature_max = dataset[index].edge_descriptor[descriptor_label].max()
                feature_min = dataset[index].edge_descriptor[descriptor_label].min()
            mean += dataset[index].edge_descriptor[descriptor_label].mean()
            std += dataset[index].edge_descriptor[descriptor_label].std()
            if dataset[index].edge_descriptor[descriptor_label].max() > feature_max:
                feature_max = dataset[index].edge_descriptor[descriptor_label].max()
            if dataset[index].edge_descriptor[descriptor_label].min() < feature_min:
                feature_min = dataset[index].edge_descriptor[descriptor_label].min()

    mean = mean / len(dataset)
    std = std / len(dataset)
    return mean, std, feature_min, feature_max


##Normalizes edges
def NormalizeEdge(dataset, descriptor_label):
    mean, std, feature_min, feature_max = GetRanges(dataset, descriptor_label)

    for data in dataset:
        data.edge_descriptor[descriptor_label] = (
                                                         data.edge_descriptor[descriptor_label] - feature_min
                                                 ) / (feature_max - feature_min)


################################################################################
#  Transforms
################################################################################

##Get specified y index from data.y
class GetY(object):
    def __init__(self, index=0):
        self.index = index

    def __call__(self, data):
        # Specify target.
        if self.index != -1:
            try:
                data.y = data.y[0][self.index]
            except:
                pass
        return data
