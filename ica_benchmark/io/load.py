from mne.io import read_raw_gdf, read_raw_edf
from mne import Epochs, events_from_annotations, concatenate_raws, concatenate_epochs
from abc import ABC, abstractmethod
from pathlib import Path
import pandas as pd
import re

PRELOAD = False


class class_or_instancemethod(classmethod):
    def __get__(self, instance, type_):
        descr_get = super().__get__ if instance is None else self.__func__.__get__
        return descr_get(instance, type_)


class Dataset(ABC):

    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)

    @property
    @abstractmethod
    def EVENT_MAP_DICT(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def EOG_CHANNELS(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def EVENT_MAP_DICT(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def SUBJECT_INFO_KEYS(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def FILE_LOADER_FN(self):
        raise NotImplementedError

    @classmethod
    def load_as_raw(cls, filepath, load_eog=False, preload=False):
        
        raw_obj = cls.FILE_LOADER_FN(
            filepath,
            preload=preload,
            eog=cls.EOG_CHANNELS,
            exclude=list() if load_eog else cls.EOG_CHANNELS,
        )
    
        return raw_obj
    
    @classmethod
    def load_as_epochs(cls, filepath, tmin=-.3, tmax=.7, reject=None, load_eog=False, has_labels=True):
        # Default value of MNE is to not reject but default from
        # this class is using the REJECT_MAGNITUDE dict
        
        if reject is None:
            reject = dict(eeg=cls.REJECT_MAGNITUDE)
        elif reject is False:
            reject = None

        raw_obj = cls.load_as_raw(filepath, preload=True, load_eog=load_eog)
        events, _ = events_from_annotations(raw_obj, event_id=cls.EVENT_MAP_DICT if has_labels else cls.UNKNOWN_EVENT_MAP_DICT)
        epochs = Epochs(
            raw_obj,
            events,
            event_repeated="drop",
            reject_by_annotation=True,
            tmin=tmin,
            tmax=tmax,
            reject=reject,
            baseline=None,
            # proj=False
        )
        return epochs

    @classmethod
    def _parse_info(cls, info_dict):

        parsed_info = {k: v for k, v in info_dict.items() if k in cls.SUBJECT_INFO_KEYS}
        return parsed_info

    @classmethod
    def read_metadata_from_raw(cls, raw):
        return cls._parse_info(
            raw._raw_extras[0]["subject_info"]
        )

    @classmethod
    def read_metadata(cls, filepath):
        raw_obj = cls.load_as_raw(filepath, preload=False)
        return cls.read_metadata_from_raw(raw_obj)

    @classmethod
    def load_from_filepath(cls, filepath, as_epochs=True, **load_kwargs):

        load_fn = cls.load_as_epochs if as_epochs else cls.load_as_raw

        obj = load_fn(filepath, **load_kwargs)
        metadata = cls.read_metadata(filepath)

        return obj, metadata

    @classmethod
    def load_from_filepaths(cls, filepaths, as_epochs=True, concatenate=False, return_metadata=False, **load_kwargs):

        concatenate_fn = concatenate_epochs if as_epochs else concatenate_raws

        objs = list()
        metadata = list()
        for filepath in filepaths:
            obj, metadata = cls.load_from_filepath(filepath, as_epochs=as_epochs, **load_kwargs)
            objs.append(obj)
            metadata.append(metadata)
            
        if concatenate:
            objs = concatenate_fn(objs)
        
        if return_metadata:
            return objs, metadata
            
        return objs
    
    @abstractmethod
    def list_subject_filepaths(self) -> pd.DataFrame:
        raise NotImplementedError

    def list_uids(self):
        uids = self.list_subject_filepaths().uid.to_numpy()
        return uids


class BCI_IV_Comp_Dataset(Dataset):

    EVENT_MAP_DICT = {
        "769": 0,
        "770": 1,
        "771": 2,
        "772": 3,
#        "768": 4,
    }

    UNKNOWN_EVENT_MAP_DICT = {
        "783": 10,
    }

    EOG_CHANNELS = ["EOG-left", "EOG-central", "EOG-right"]

    REJECT_MAGNITUDE = 1e-4

    SUBJECT_INFO_KEYS = [
        "id",
        "smoking",
        "alcohol_abuse",
        "drug_abuse",
        "medication",
        "weight",
        "height",
        "sex",
        "handedness",
        "age",
    ]

    FILE_LOADER_FN = read_raw_gdf

    def __init__(self, dataset_path, test_folder=None):
        super(BCI_IV_Comp_Dataset, self).__init__(dataset_path)
        self.test_folder = test_folder or self.dataset_path
        self.test_folder = Path(self.test_folder)

    def get_uid_filename(self, uid, train=False):
        return "A{}{}.gdf".format(
            str(uid).rjust(2, "0"),
            "T" if train else "E"
        )

    def uid_from_filepath(self, filepath):
        return re.search("A0([0-9])[ET].gdf", filepath.name).group(1)

    def list_subject_filepaths(self) -> pd.DataFrame:
        filepaths = self.dataset_path.glob("A*.gdf")
        filepaths_list = [
            (
                str(filepath),
                "T" in filepath.name,
                self.uid_from_filepath(filepath)
            )
            for filepath
            in filepaths
        ]
        filepath_df = pd.DataFrame(filepaths_list, columns=["path", "train", "uid"])
        return filepath_df

    def load_subject(self, uid, train=False, **kwargs):
        filepaths_df = self.list_subject_filepaths()
        filepath = filepaths_df.query("uid == @uid").query("train == @train").path.item()
        epochs, _ = self.load_from_filepath(filepath, as_epochs=True, has_labels=train, **kwargs)
        if train:
            events = epochs.events[:, 2].flatten()
        else:
            test_label_filepath = self.test_folder / "A0{}E.csv".format(uid)
            events = pd.read_csv(test_label_filepath, header=None).to_numpy().flatten() - 1
            epochs.events[:, 2] = events
        return epochs, events


class OpenBMI_Dataset(Dataset):

    EVENT_MAP_DICT = {
        "0": 0,
        "1": 1,
    }
    UNKNOWN_EVENT_MAP_DICT = {}

    REJECT_MAGNITUDE = 1e-3

    SUBJECT_INFO_KEYS = [
        'id',
        'sex',
        'birthday',
        'name'
    ]

    FILE_LOADER_FN = read_raw_edf

    EOG_CHANNELS = list()

    def uid_from_filepath(self, filepath):
        return re.search("([0-9]+)_[\w]+.edf", filepath.name).group(1)

    def list_subject_filepaths(self):
        filepaths = Path("/home/paulo/Documents/datasets/OpenBMI/edf").glob("**/*.edf")

        filepaths_list = [
            (
                str(filepath),
                "train" in filepath.name,
                int(filepath.parts[-2][-1]),
                self.uid_from_filepath(filepath)
            )
            for filepath
            in filepaths
        ]
        filepath_df = pd.DataFrame(filepaths_list, columns=["path", "train", "session", "uid"])
        return filepath_df

    def load_subject(self, uid, session, train=False, **kwargs):
        filepaths_df = self.list_subject_filepaths()
        filepath = (
            filepaths_df
            .query("uid == @uid")
            .query("train == @train")
            .query("session == @session")
            .path.item()
        )

        epochs, _ = self.load_from_filepath(filepath, as_epochs=True, **kwargs)
        events = epochs.events[:, 2].flatten()
        return epochs, events
