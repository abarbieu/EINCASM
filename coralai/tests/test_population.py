import os
import unittest

import numpy as np
from coralai.population import Population
from coralai.reportable_neat_config import ReportableNeatConfig

class TestPopulation(unittest.TestCase):
    def setUp(self):
        # Setup code here. This will run before each test method.
        self.current_dir = os.path.dirname(__file__)
        self.data_dir = os.path.join(self.current_dir, "data_dump")
        self.neat_config = ReportableNeatConfig()
        self.pop_size = 10
        self.population = Population.gen_random_pop(self.neat_config, self.pop_size)
        self.snap_dir = None

    def test_population_creation(self):
        self.assertEqual(len(self.population.genomes), self.pop_size, "Population size does not match expected.")

    def test_population_snapshot_saving_and_loading(self):
        self.snap_dir = self.population.save_snapshot(self.data_dir)  # Save the snapshot name to the class variable
        loaded_population = Population.load_snapshot(self.data_dir)
        # Assert that ages are as expected (all zeros for a new population)
        expected_ages = [0 for _ in range(self.pop_size)]
        self.assertEqual(list(loaded_population.ages), expected_ages, "Loaded population ages do not match expected.")

    def test_knn_graph(self):
        dist_matrix = self.population.gen_distance_matrix()
        self.population.plot_knn_net(5, "test", np.random.rand(self.pop_size))

if __name__ == "__main__":
    unittest.main()

