import torch
from pymatgen.io.cif import CifParser
from collections import defaultdict
import itertools
import numpy as np
from ase.io import read
from pymatgen.io.ase import AseAtomsAdaptor


def periodic_edge_feature(
        cif_file_path,
        max_neighbors,
        cutoff=8,
):
    """Radius based edge feature & periodic boundary condition"""
    try:
        parser = CifParser(cif_file_path)
        atoms = parser.parse_structures(primitive=True)[0]
    except:
        atoms = read(cif_file_path)
        atoms = AseAtomsAdaptor.get_structure(atoms)
    lat = atoms.lattice
    all_neighbors = atoms.get_all_neighbors(r=cutoff)

    edges = defaultdict(list)
    for site_idx, neighborlist in enumerate(all_neighbors):

        # sort on distance
        neighborlist = sorted(neighborlist, key=lambda x: x[1])
        distances = np.array([nbr[1] for nbr in neighborlist])
        ids = np.array([nbr[2] for nbr in neighborlist])

        # find the distance to the k-th nearest neighbor
        try:
            max_dist = distances[max_neighbors - 1]
        except IndexError:
            max_dist = distances[-1]
        ids = ids[distances <= max_dist]
        distances = distances[distances <= max_dist]
        for id, distance in zip(ids, distances):
            edges[(site_idx, id)].append(distance)

        abc = lat.abc
        combinations = list(itertools.combinations(abc, 2)) + list(itertools.combinations(abc, 1))
        combinations = [np.array(combination) for combination in combinations]
        lattice_distances = [np.sum(combination ** 2) ** 0.5 for combination in combinations]
        for lattice_distance in lattice_distances:
            if lattice_distance <= cutoff:
                edges[(site_idx, site_idx)].append(lattice_distance)
    edge_index = []
    edge_features = []
    for edge, edge_values in edges.items():
        # 将元组和列表分开
        edge_tuple = edge  # 元组部分
        for edge_value in edge_values:
            edge_index.append(edge_tuple)
            edge_features.append(edge_value)

    edge_index = torch.tensor(edge_index).t()
    edge_features = torch.tensor(edge_features, dtype=torch.float32)
    return edge_index, edge_features


if __name__ == "__main__":
    cif_file_path = r"C:\Users\79050\Desktop\AI4S\code\deeperGATGNN-main\data\order_data\Y1H6.cif"
    edge_index, edge_features = periodic_edge_feature(cif_file_path=cif_file_path, max_neighbors=12)
