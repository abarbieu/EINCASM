from typing import Callable, TypeVar, Type
from dataclasses import dataclass
import os
import taichi as ti
import torch
import pickle
from typing import List
import uuid
from matplotlib import pyplot as plt
import neat
import numpy as np
import networkx as nx
from numpy.typing import NDArray

from pytorch_neat.linear_net import LinearNet
from pytorch_neat.activations import identity_activation


from coralai.substrate.substrate import Substrate
from coralai.coralai_config import CoralaiConfig

from .reportable import Reportable
from .reportable_neat_config import ReportableNeatConfig
T = TypeVar('T', bound='Population')

def create_torch_net(config: CoralaiConfig, genome: neat.DefaultGenome):
    input_coords = []
    # TODO: adjust for direcitonal kernel?
    for ch in range(config.n_senses):
        input_coords.append([0, 0, config.sense_chinds[ch]])
        for offset_i in range(config.dir_order.shape[0]):
            offset_x = config.dir_kernel[config.dir_order[offset_i], 0]
            offset_y = config.dir_kernel[config.dir_order[offset_i], 1]
            input_coords.append([offset_x, offset_y, config.sense_chinds[ch]])

    output_coords = []
    for ch in range(config.n_acts):
        output_coords.append([0, 0, config.act_chinds[ch]])

    net = LinearNet.create(
        genome,
        config.neat_config,
        input_coords=input_coords,
        output_coords=output_coords,
        weight_threshold=0.0,
        weight_max=3.0,
        activation=identity_activation,
        cppn_activation=identity_activation,
        device=config.torch_device,
    )
    return net

@ti.kernel
def apply_weights_and_biases(mem: ti.types.ndarray(), out_mem: ti.types.ndarray(),
                             sense_chinds: ti.types.ndarray(),
                             combined_weights: ti.types.ndarray(), combined_biases: ti.types.ndarray(),
                             dir_kernel: ti.types.ndarray(), dir_order: ti.types.ndarray(),
                             ti_inds: ti.template()):
    inds = ti_inds[None]
    for i, j, act_k in ti.ndrange(mem.shape[2], mem.shape[3], out_mem.shape[0]):
        val = 0.0
        rot = mem[0, inds.rot, i, j]
        genome_key = int(mem[0, inds.genome, i, j])
        for sense_ch_n in ti.ndrange(sense_chinds.shape[0]):
            # base case [0,0]
            start_weight_ind = sense_ch_n * (dir_kernel.shape[0]+1)
            val += (mem[0, sense_chinds[sense_ch_n], i, j] *
                    combined_weights[genome_key, 0, act_k, start_weight_ind])
            for offset_m in ti.ndrange(dir_kernel.shape[0]):
                ind = int((rot + dir_order[offset_m]) % dir_kernel.shape[0])
                neigh_x = (i + dir_kernel[ind, 0]) % mem.shape[2]
                neigh_y = (j + dir_kernel[ind, 1]) % mem.shape[3]
                weight_ind = start_weight_ind + offset_m
                val += mem[0, sense_chinds[sense_ch_n], neigh_x, neigh_y] * combined_weights[genome_key, 0, act_k, weight_ind]
        out_mem[act_k, i, j] = val + combined_biases[genome_key, 0, act_k, 0]
    

def apply_population_nets(population: T, substrate: Substrate, config: CoralaiConfig):
    pass

class Population(Reportable):
    neat_config: ReportableNeatConfig
    genomes: List[neat.DefaultGenome]
    ages: NDArray[np.int64]
    alive_keys: List[int]
    dead_keys: List[int]
    create_torch_net: Callable[[CoralaiConfig, neat.DefaultGenome], LinearNet]
    apply_population_nets: Callable[[T, Substrate, CoralaiConfig]]
    nets: List[LinearNet]
    weights: torch.Tensor # shape: (n_genomes, 1, n_actuators, n_sensors)
    biases: torch.Tensor  # shape: (n_genomes, 1, n_actuators, 1)

    SAD: NDArray[np.int64]
    substrate_coverage: NDArray[np.float64]
    distance_matrix: NDArray[np.float64]
    knn_net: nx.Graph

    report_prefix: str = "population"

    def __init__(self, neat_config: ReportableNeatConfig, genomes: List[neat.DefaultGenome], ages: List[int],
                 create_torch_net: Callable[[CoralaiConfig, neat.DefaultGenome], LinearNet]):
        self.neat_config = neat_config
        self.genomes = genomes
        self.ages = ages
        self.create_torch_net = create_torch_net

    @classmethod
    def gen_random_pop(cls: Type[T], neat_config: ReportableNeatConfig, pop_size: int,
                       create_torch_net: Callable[[CoralaiConfig, neat.DefaultGenome], LinearNet]) -> T:
        genomes = [neat.DefaultGenome(uuid.uuid4()) for _ in range(pop_size)]
        for genome in genomes:
            genome.configure_new(neat_config.genome_config)

        ages = [0 for _ in range(pop_size)]
        return cls(neat_config, genomes, ages, create_torch_net)
    
    def gen_torch_nets(self):
        self.nets = [self.create_torch_net(self.neat_config, genome) for genome in self.genomes]
        self.weights = torch.stack([net.weight for net in self.nets])
        self.biases = torch.stack([net.bias for net in self.nets])
        return self.nets, self.weights, self.biases
    
    def save_snapshot(self, snapshot_dir: str, report_suffix: str = None) -> str:
        """Implements Reportable.save_snapshot, dumps genomes and ages to pkl files, reports config if it hasn't been"""
        
        snap_name = self.get_snapshot_name(report_suffix)
        snapshot_base_path = os.path.join(snapshot_dir, snap_name)
        os.makedirs(snapshot_base_path, exist_ok=True)  # Ensure the snapshot directory exists
        
        self.neat_config.save_snapshot(snapshot_base_path, report_suffix)
        
        genome_snapshot_path = os.path.join(snapshot_base_path, f"{snap_name}_genomes.pkl")
        ages_snapshot_path = os.path.join(snapshot_base_path, f"{snap_name}_ages.pkl")

        with open(genome_snapshot_path, "wb") as genome_file:
            pickle.dump(self.genomes, genome_file)
        with open(ages_snapshot_path, "wb") as ages_file:
            pickle.dump(self.ages, ages_file)

        return snapshot_base_path

    @classmethod
    def load_snapshot(cls: Type[T], snapshot_dir: str, neat_config: ReportableNeatConfig = None, report_suffix: str = None) -> T:
        """Implements Reportable.load_snapshot"""
        snap_name = cls.get_snapshot_name(report_suffix)

        genome_snapshot_path = os.path.join(snapshot_dir, snap_name, f"{snap_name}_genomes.pkl")
        ages_snapshot_path = os.path.join(snapshot_dir, snap_name, f"{snap_name}_ages.pkl")

        with open(genome_snapshot_path, "rb") as f:
            genomes = pickle.load(f)
        with open(ages_snapshot_path, "rb") as f:
            ages = pickle.load(f)
        
        if neat_config is None:
            neat_config = ReportableNeatConfig.load_snapshot(os.path.join(snapshot_dir, snap_name), report_suffix)

        return cls(neat_config, genomes, ages)
    
    @classmethod
    def load_snapshot_old(cls: Type[T], step_dir: str, neat_config: ReportableNeatConfig) -> T:
        genome_path = os.path.join(step_dir, "genomes")
        ages_path = os.path.join(step_dir, "ages")

        with open(genome_path, "rb") as f:
            genomes = pickle.load(f)
        with open(ages_path, "rb") as f:
            ages = pickle.load(f)
        
        return cls(neat_config, genomes, ages)
