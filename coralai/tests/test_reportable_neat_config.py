import os
import unittest
from coralai.reportable_neat_config import ReportableNeatConfig

class TestConfigSnapshot(unittest.TestCase):
    def setUp(self):
        # Setup code here. This will run before each test method.
        self.current_dir = os.path.dirname(__file__)
        self.data_dir = os.path.join(self.current_dir, "data_dump")
        self.config = ReportableNeatConfig()
        self.snap_name = None

    def test_config_snapshot_saving(self):
        self.snap_name = self.config.save_snapshot(self.data_dir)  # Save the snapshot name to the class variable
        config_snap_dir = os.path.join(self.data_dir, self.snap_name)
        expected_config_file = os.path.join(config_snap_dir, f"{self.snap_name}_config.pkl")
        expected_og_config_file = os.path.join(config_snap_dir, f"{self.snap_name}_og_config_default_neat.config")
        self.assertTrue(os.path.isfile(expected_config_file), f"Expected config file {expected_config_file} does not exist.")
        self.assertTrue(os.path.isfile(expected_og_config_file), f"Expected original config file {expected_og_config_file} does not exist.")
    
    def test_config_snapshot_loading(self):
        loaded_config = ReportableNeatConfig.load_snapshot(self.data_dir)
        # just assert it is not None
        self.assertIsNotNone(loaded_config, "Loaded config should not be None.")
        # go through all the attributes and make sure they match
        # for attr in dir(loaded_config):
        #     if not attr.startswith("__"):
        #         self.assertEqual(getattr(loaded_config, attr), getattr(self.config, attr), f"Attribute {attr} does not match.")


if __name__ == "__main__":
    unittest.main()
