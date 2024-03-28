import unittest
from unittest.mock import patch, MagicMock
from coralai.coralai import Coralai, CoralaiConfig
from coralai.reportable_neat_config import ReportableNeatConfig
from coralai.substrate.substrate import Substrate
from coralai.population import Population

class TestCoralaiConfig(unittest.TestCase):
    def setUp(self):
        # Mocking the components to avoid actual file system operations
        self.mock_config = MagicMock(spec=CoralaiConfig)
        self.mock_neat_config = MagicMock(spec=ReportableNeatConfig)
        self.mock_substrate = MagicMock(spec=Substrate)
        self.mock_population = MagicMock(spec=Population)
        
        # Mocking the load_snapshot class method to return mock instances
        self.mock_config.load_snapshot = MagicMock(return_value=self.mock_config)
        self.mock_neat_config.load_snapshot = MagicMock(return_value=self.mock_neat_config)
        self.mock_substrate.load_snapshot = MagicMock(return_value=self.mock_substrate)
        self.mock_population.load_snapshot = MagicMock(return_value=self.mock_population)

    @patch('os.path.join', return_value='dummy_path')
    def test_save_snapshot(self, mock_join):
        coralai_instance = Coralai(self.mock_config, self.mock_neat_config, self.mock_substrate, self.mock_population)
        snapshot_dir = 'test_dir'
        report_suffix = 'suffix'
        
        expected_snap_path = 'dummy_path'
        snap_path = coralai_instance.save_snapshot(snapshot_dir, report_suffix)
        
        self.assertEqual(snap_path, expected_snap_path)
        self.mock_config.save_snapshot.assert_called_once_with(snapshot_dir, report_suffix)
        self.mock_neat_config.save_snapshot.assert_called_once_with(snapshot_dir, report_suffix)
        self.mock_substrate.save_snapshot.assert_called_once_with(snapshot_dir, report_suffix)
        self.mock_population.save_snapshot.assert_called_once_with(snapshot_dir, report_suffix)

    @patch('coralai.coralai.CoralaiConfig.load_snapshot')
    @patch('coralai.coralai.ReportableNeatConfig.load_snapshot')
    @patch('coralai.coralai.Substrate.load_snapshot')
    @patch('coralai.coralai.Population.load_snapshot')
    def test_load_snapshot(self, mock_pop_load, mock_sub_load, mock_neat_load, mock_config_load):
        snapshot_path = 'dummy_path'
        loaded_instance = Coralai.load_snapshot(snapshot_path)
        
        self.assertIsInstance(loaded_instance, Coralai)
        mock_config_load.assert_called_once_with(snapshot_path)
        mock_neat_load.assert_called_once_with(snapshot_path)
        mock_sub_load.assert_called_once_with(snapshot_path)
        mock_pop_load.assert_called_once_with(snapshot_path)

if __name__ == '__main__':
    unittest.main()