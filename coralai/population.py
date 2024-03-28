from typing import TypeVar, Type
from dataclasses import dataclass
import os
import taichi as ti
import pickle
from typing import List
import uuid
from matplotlib import pyplot as plt
import neat
import numpy as np
import networkx as nx
from numpy.typing import NDArray

from coralai.substrate.substrate import Substrate

from .reportable import Reportable
from .reportable_neat_config import ReportableNeatConfig

T = TypeVar('T', bound='Population')

def calc_abundance(substrate, genome_i):
    return substrate['genome'].eq(genome_i).sum().item()

def calc_SAD(substrate, genomes):
    SAD = [] # species abundance distribution
    for i in range(len(genomes)):
        SAD.append(calc_abundance(substrate, i))
    return np.array(SAD)

def get_genome_coverage(substrate, SAD):
    return SAD / (substrate.mem.shape[2] * substrate.mem.shape[3])

def gen_genome_distance_matrix(genomes, neat_config):
    # Creates a similarity matrix between all genomes in a list using NEAT's genome distance
    num_genomes = len(genomes)
    # Initialize the matrix with zeros
    distance_matrix = [[0 for _ in range(num_genomes)] for _ in range(num_genomes)]
    
    # Fill the matrix with distances
    for i in range(num_genomes):
        for j in range(i+1, num_genomes):  # Start from i+1 to avoid redundant calculations
            distance = genomes[i].distance(genomes[j], neat_config)
            distance_matrix[i][j] = distance
            distance_matrix[j][i] = distance  # The matrix is symmetric
    
    return distance_matrix

def plot_genome_distance_matrix(distance_matrix, title):
    # Plots a heatmap of the genome distance matrix with labels for run name and step number
    fig, ax = plt.subplots()
    cax = ax.imshow(distance_matrix, cmap="viridis", vmin=0)
    fig.colorbar(cax, label="Distance")
    ax.set_title(title)
    plt.show()


def create_knn_net(distance_matrix: NDArray[np.float64], k: int, sub_cov_distr: NDArray[np.float64], ages: NDArray[np.int64]):
    distance_matrix_np = np.array(distance_matrix)
    
    num_nodes = distance_matrix_np.shape[0]
    
    G = nx.Graph()
    
    for i in range(num_nodes):
        if sub_cov_distr[i] > 0:
            G.add_node(i, genome_i=i, substrate_coverage=sub_cov_distr[i], age=ages[i])
    
    for i in G.nodes:
        genome_i = G.nodes[i]['genome_i']
        distances = distance_matrix_np[genome_i, :]
        nearest_indices = np.argsort(distances)[0:k+1]
        for j in nearest_indices:
            G.add_edge(i, j, weight=distances[j])
    
    return G

def plot_knn_net(G: nx.Graph, title: str):
    plt.figure(figsize=(10, 8))  # Adjust figure size
    pos = nx.spring_layout(G)  # Compute layout
    
    # Extract node attributes for plotting
    coverages_scaled = np.array([G.nodes[node]['substrate_coverage'] for node in G.nodes]) * 100000  # Scale pop size for visibility
    ages = np.array([G.nodes[node]['age'] for node in G.nodes])

    max_age = max(ages) if max(ages) > 0 else 1  # Ensure max_age is not zero to avoid division by zero\

    age_colors = [plt.cm.summer(age / max_age) for age in ages]  # Normalize ages and map to colormap
    nx.draw(G, pos, with_labels=True, node_size=coverages_scaled, node_color=age_colors, font_size=8, edge_color="gray")
    plt.title(title, fontsize=14)  # Set title with run name and step number
    plt.colorbar(plt.cm.ScalarMappable(cmap='summer'), label='Genome Age')
    plt.show()


class Population(Reportable):
    neat_config: ReportableNeatConfig
    genomes: List[neat.DefaultGenome]
    alive_keys: List[int]
    dead_keys: List[int]
    SAD: NDArray[np.int64]
    substrate_coverage: NDArray[np.float64]
    distance_matrix: NDArray[np.float64]
    ages: NDArray[np.int64]
    knn_net: nx.Graph

    report_prefix: str = "population"

    def __init__(self, neat_config: ReportableNeatConfig, genomes: List[neat.DefaultGenome], ages: List[int]):
        self.neat_config = neat_config
        self.genomes = genomes
        self.ages = ages
    
    def gen_distance_matrix(self):
        self.distance_matrix = gen_genome_distance_matrix(self.genomes, self.neat_config)
        return self.distance_matrix
    
    def plot_genome_distance_matrix(self, title: str, distance_matrix: NDArray[np.float64] = None):
        if distance_matrix is None:
            if self.distance_matrix is None:
                self.gen_distance_matrix()
            distance_matrix = self.distance_matrix
        plot_genome_distance_matrix(distance_matrix, title)
    
    def plot_knn_net(self, k: int, title: str, substrate: Substrate = None, substrate_coverage: NDArray[np.float64] = None):
        if not substrate_coverage:
            if not substrate:
                raise ValueError("Substrate must be provided if substrate_coverage is not provided")
            substrate_coverage = calc_SAD(substrate, self.genomes)
            self.SAD = substrate_coverage
            self.substrate_coverage = get_genome_coverage(substrate, substrate_coverage)
        self.distance_matrix = gen_genome_distance_matrix(self.genomes, self.neat_config, substrate_coverage)
        self.knn_net = create_knn_net(self.distance_matrix, k, self.substrate_coverage, self.ages)
        plot_knn_net(self.knn_net, title)

    @classmethod
    def gen_random_pop(cls: Type[T], neat_config: ReportableNeatConfig, pop_size: int) -> T:
        genomes = [neat.DefaultGenome(uuid.uuid4()) for _ in range(pop_size)]
        for genome in genomes:
            genome.configure_new(neat_config.genome_config)

        ages = [0 for _ in range(pop_size)]
        return cls(neat_config, genomes, ages)
    
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
