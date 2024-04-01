import os
import pickle
from typing import TypeVar, Type

import torch

T = TypeVar('T', bound='Reportable')

class Reportable():
    # TODO: Save initial configuration separately from snapshot. E.g. substrate layout vs mem @ timestep
    """
    ## A reportable object can be stored and loaded from a snapshot.
    - `report_prefix` is used to construct the file name of the snapshot
    - `report_prefix` is a class attribute that should be defined outside of `__init__`
    - A snapshot can be a folder containing different parts of the object

    ### Example:
    - A substrate snapshot might store its channel layout separately from its data.
    - `report_prefix = "substrate_snap"`
    - Resulting snapshot: `substrate_snap/channel_layout.json` and `substrate_snap/data.pt`
    """
    report_prefix: str

    @classmethod
    def get_snapshot_name(cls, report_suffix: str = None) -> str:
        return cls.report_prefix + (f"_{report_suffix}" if report_suffix is not None else "")

    def save_snapshot(self, snapshot_dir: str, report_suffix: str = None) -> str:
        """Saves to snapshot_dir/report_prefix[_report_suffix][.extension] 

        Suffix is optional, useful if multiple snapshots are stored in the same path.
        It is recommended to create a unique directory for each snapshot of a simulation.

        Returns the path of the saved snapshot.
        """
        snap_name = self.get_snapshot_name(report_suffix)
        snap_path = os.path.join(snapshot_dir, snap_name)
        with open(snap_path, "wb") as f:
            pickle.dump(self, f)
        return snap_path

    @classmethod
    def load_snapshot(cls: Type[T], snapshot_dir: str, torch_device: torch.DeviceObjType = None, report_suffix: str = None) -> T:
        """Loads from snapshot_dir/report_prefix[_report_suffix][.extension]

        Suffix is optional, useful if multiple snapshots are stored in the same path.
        It is recommended to create a unique directory for each snapshot of a simulation

        Returns an instance of the reportable object.
        """
        snap_name = cls.get_snapshot_name(report_suffix)
        snap_dir = os.path.join(snapshot_dir, snap_name)
        with open(snap_dir, "rb") as f:
            return pickle.load(f)

