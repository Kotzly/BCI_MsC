from mne.io import read_raw_gdf, read_raw_edf
from mne import Epochs, events_from_annotations, concatenate_raws, concatenate_epochs
from abc import ABC, abstractmethod
from pathlib import Path
import pandas as pd
import re
from warnings import warn
import numpy as np

PRELOAD = False


class DefaultSessionWarning(Warning):
    pass


class Dataset(ABC):
    """In the load_subject function we use the session and run notations.
    Session refers to one time that the subject has put on the EEG equipment until it removes it.
    The run refers to one repetition of the whole protocol.

    In the BCI IV Competition dataset, the run refers to the Train and Test sets.
    In the OpenBMI dataset, the run also refers to the Train and Test sets.

    """

    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)

    def uid_decorator(fn):
        def decorated(*args, **kwargs):
            # load_subject function returns the epochs and the events
            epochs, events = fn(*args, **kwargs)
            # uid is the second argument
            uid = args[1]
            epochs.info.update(
                dict(
                    subject_info=dict(
                        uid=uid
                    )
                )
            )
            return epochs, events
        return decorated

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
    def SUBJECT_INFO_KEYS(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def FILE_LOADER_FN(self):
        raise NotImplementedError

    @classmethod
    def load_as_raw(cls, filepath, load_eog=False, preload=False, verbose=None):

        raw_obj = cls.FILE_LOADER_FN(
            filepath,
            preload=preload,
            eog=cls.EOG_CHANNELS,
            exclude=list() if load_eog else cls.EOG_CHANNELS,
            verbose=verbose,
        )

        return raw_obj

    @classmethod
    def load_as_epochs(
        cls,
        filepath,
        tmin=-0.3,
        tmax=0.7,
        reject=None,
        load_eog=False,
        verbose=None
    ):
        # Default value of MNE is to not reject but default from
        # this class is using the REJECT_MAGNITUDE dict

        if reject is None:
            reject = dict(eeg=cls.REJECT_MAGNITUDE)
        elif reject is False:
            reject = None

        raw_obj = cls.load_as_raw(
            filepath, preload=False, load_eog=load_eog, verbose=verbose
        )
        events, _ = events_from_annotations(
            raw_obj,
            event_id=cls.EVENT_MAP_DICT,
            verbose=verbose,
        )

        epochs = Epochs(
            raw_obj,
            events,
            event_repeated="drop",
            reject_by_annotation=True,
            tmin=tmin,
            tmax=tmax,
            reject=reject,
            baseline=None,
            verbose=verbose
            # proj=False
        )
        return epochs

    @classmethod
    def _parse_info(cls, info_dict):

        parsed_info = {k: v for k, v in info_dict.items() if k in cls.SUBJECT_INFO_KEYS}
        return parsed_info

    @classmethod
    def read_metadata_from_raw(cls, raw):
        return cls._parse_info(raw._raw_extras[0]["subject_info"])

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
    def load_from_filepaths(
        cls,
        filepaths,
        as_epochs=True,
        concatenate=False,
        return_metadata=False,
        **load_kwargs
    ):

        concatenate_fn = concatenate_epochs if as_epochs else concatenate_raws

        objs = list()
        metadata = list()
        for filepath in filepaths:
            obj, metadata = cls.load_from_filepath(
                filepath, as_epochs=as_epochs, **load_kwargs
            )
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
        uids = self.list_subject_filepaths().uid.unique()
        return uids

    def _validate_session(self, session):
        if session is None:
            warn(
                f"Using session 1, as you did not pass the session argument. Using the first session, but you can choose the sessions: {self.SESSIONS}.",
                DefaultSessionWarning,
            )
            session = 1
        assert session in self.SESSIONS, "You asked for session {}, but the available sessions are {}".format(session, self.SESSIONS)
        return session

    def _validate_run(self, run):
        if run is None:
            warn(
                f"Using run 1, as you did not pass the session argument. Using the first session, but you can choose the runs: {self.RUNS}.",
                DefaultSessionWarning,
            )
            run = 1

        assert run in self.RUNS, "You asked for run {}, but the available runs are {}".format(run, self.RUNS)

        return run


class BCI_IV_Comp_Dataset(Dataset):

    EVENT_MAP_DICT = {
        "769": 0,
        "770": 1,
        "771": 2,
        "772": 3,
        # "768": 4,
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

    SESSIONS = [1, 2]
    RUNS = [1]

    HAS_LABELS = [1]

    def __init__(self, dataset_path, test_folder=None):
        super(BCI_IV_Comp_Dataset, self).__init__(dataset_path)
        self.test_folder = test_folder or self.dataset_path
        self.test_folder = Path(self.test_folder)

    @classmethod
    def uid_from_filepath(self, filepath):
        return re.search("A0([0-9])[ET].gdf", filepath.name).group(1)

    def list_subject_filepaths(self) -> pd.DataFrame:
        filepaths = self.dataset_path.glob("A*.gdf")
        filepaths_list = [
            (
                str(filepath),
                1 if "T" in filepath.name else 2,
                1,
                self.uid_from_filepath(filepath)
            )
            for filepath in filepaths
        ]
        filepath_df = pd.DataFrame(filepaths_list, columns=["path", "session", "run", "uid"])
        return filepath_df

    @Dataset.uid_decorator
    def load_subject(self, uid, session=None, run=None, **kwargs):

        session = self._validate_session(session)
        run = self._validate_run(run)

        filepaths_df = self.list_subject_filepaths()
        filepath = (
            filepaths_df
            .query("uid == @uid")
            .query("session == @session")
            .query("run == @run")
            .path.item()
        )
        epochs, _ = self.load_from_filepath(
            filepath, as_epochs=True, **kwargs
        )
        if session in self.HAS_LABELS:
            events = epochs.events[:, 2].flatten()
        else:
            test_label_filepath = self.test_folder / "A0{}E.csv".format(uid)
            events = (
                pd.read_csv(test_label_filepath, header=None).to_numpy().flatten() - 1
            )
            epochs.events[:, 2] = events
            epochs.event_id = {
                str(v): v
                for v in
                sorted(np.unique(events))
            }
        return epochs, events


class OpenBMI_Dataset(Dataset):

    EVENT_MAP_DICT = {
        "1": 0,
        "2": 1,
    }

    REJECT_MAGNITUDE = 1e-3

    SUBJECT_INFO_KEYS = ["id", "sex", "birthday", "name"]

    FILE_LOADER_FN = read_raw_edf

    EOG_CHANNELS = list()

    SESSIONS = [1, 2]
    RUNS = [1, 2]

    @classmethod
    def uid_from_filepath(self, filepath):
        return re.search("([0-9]+)_[\w]+.edf", filepath.name).group(1)

    def list_subject_filepaths(self):
        filepaths = Path("/home/paulo/Documents/datasets/OpenBMI/edf").glob("**/*.edf")

        filepaths_list = [
            (
                str(filepath),
                int(filepath.parts[-2][-1]),
                1 if ("train" in filepath.name) else 2,
                self.uid_from_filepath(filepath),
            )
            for filepath in filepaths
        ]
        filepath_df = pd.DataFrame(
            filepaths_list, columns=["path", "session", "run", "uid"]
        )
        return filepath_df

    @Dataset.uid_decorator
    def load_subject(self, uid, session=None, run=None, **kwargs):

        session = self._validate_session(session)
        run = self._validate_run(run)

        filepaths_df = self.list_subject_filepaths()
        filepath = (
            filepaths_df.query("uid == @uid")
            .query("run == @run")
            .query("session == @session")
            .path.item()
        )

        epochs, _ = self.load_from_filepath(filepath, as_epochs=True, **kwargs)
        events = epochs.events[:, 2].flatten()
        return epochs, events


class Physionet_2009_Dataset(Dataset):

    TRIALS = list(range(1, 14 + 1))
    TASKS = list(map(str, [1, 2, 3, 4, 10, 11]))
    TASK_DICT = {
        1: "each_fist_execution",
        2: "each_fist_imagery",
        3: "both_fist_execution",
        4: "both_fist_imagery",
    }

    EVENT_MAP_DICT = {
        str(i): i
        for i in range(12)
        if i
        not in [
            9,
        ]  # There is no "9" annotation, as it can be seen in ANNOTATION_REMAPPING
    }

    # Map the task number, defined in TRIAL_INFO_DF, to the EVENT_ID mappings
    ANNOTATION_REMAPPING = {
        # Each dict maps the original annotation string to a new annotation string,
        # which is unique amongst all trials, as in the original dataset the T1 and T2
        # annotations were used for multiple tasks
        #
        # Keys -> Values: Tasks -> Annotation to Label remapping
        "10": {"T0": "10"},  # Rest in Baseline, eyes open
        "11": {"T0": "11"},  # Rest in Baseline, eyes closed
        "1": {
            "T0": "0",  # Rest
            "T1": "1",  # Left, execution
            "T2": "2",  # Right, execution
        },
        "2": {
            "T0": "0",  # Rest
            "T1": "3",  # Left, imagination
            "T2": "4",  # Right, imagination
        },
        "3": {
            "T0": "0",  # Rest
            "T1": "5",  # Both fists, execution
            "T2": "6",  # Both feet, execution
        },
        "4": {
            "T0": "0",  # Rest
            "T1": "7",  # Both fists, imagery
            "T2": "8",  # Both feet, imagery
        },
    }

    # The trial number refers to the 'run', as defined in the dataset documentation page
    # (https://physionet.org/content/eegmmidb/1.0.0/). The task is one of the 4 tasks plus the 2
    # baseline protocols (eyes open and closed).
    #
    # Tasks description:
    # - Task 10: Baseline, eyes open
    # - Task 11: Baseline, eyes closed
    # - Task 1: (open and close left or right fist)
    # - Task 2: (imagine opening and closing left or right fist)
    # - Task 3: (open and close both fists or both feet)
    # - Task 4: (imagine opening and closing both fists or both feet)
    # - Task 0: Rests that happened during Trials 3 to 14 (while performing tasks 1 to 4)
    #
    # In summary, the experimental runs were:
    # Trial 1: Baseline, eyes open (Task 10, but is similar to Rest - task 0)
    # Trial 2: Baseline, eyes closed (Task 11, similar to Rest - task 0)
    # Trial 3: Task 1 (open and close left or right fist)
    # Trial 4: Task 2 (imagine opening and closing left or right fist)
    # Trial 5: Task 3 (open and close both fists or both feet)
    # Trial 6: Task 4 (imagine opening and closing both fists or both feet)
    # Trial 7: Task 1
    # Trial 8: Task 2
    # Trial 9: Task 3
    # Trial 10: Task 4
    # Trial 11: Task 1
    # Trial 12: Task 2
    # Trial 13: Task 3
    # Trial 14: Task 4

    # [TODO] Make as Taskes 3 to 6 are run 1, 7 to 10 are run 2, and 11 to 14 are run 3

    EOG_CHANNELS = []

    REJECT_MAGNITUDE = 1e-4

    SUBJECT_INFO_KEYS = ["id", "sex", "birthday", "name"]

    SESSIONS = [1]
    RUNS = [1]

    # [TODO] This @classmethod followd by @property is not a usual thing to do,
    # so maybe it would be better to change it? The issue is that it would not 
    # be trivial where to put trials 0 and 1. Maybe create 14 runs and keep
    # the trial and task logic alongside it?
    @classmethod
    @property
    def TRIAL_INFO_DF(cls):
        return pd.DataFrame(
            [
                [1, "baseline_open", "10"],
                [2, "baseline_closed", "11"],
            ] + [
                [i, cls.TASK_DICT[(i - 3) % 4 + 1], str((i - 3) % 4 + 1)]
                for i in range(3, 14 + 1)
            ],
            columns=["trial", "short_description", "task"],
        )

    @classmethod
    def FILE_LOADER_FN(cls, filepath, **kwargs):
        filepath_info = cls.parse_filepath_info(filepath)
        trial = filepath_info["trial"]
        task = cls.task_from_trial(trial)
        raw = read_raw_edf(filepath, **kwargs)
        raw.annotations.rename(cls.ANNOTATION_REMAPPING[task])
        return raw

    def __init__(self, dataset_path, test_folder=None):
        super(Physionet_2009_Dataset, self).__init__(dataset_path)
        self.test_folder = test_folder or self.dataset_path
        self.test_folder = Path(self.test_folder)

    @classmethod
    def task_from_trial(cls, trial):
        return cls.TRIAL_INFO_DF.query("trial == @trial").task.item()

    @classmethod
    def parse_filepath_info(cls, filepath):
        pattern = re.compile("S(?P<uid>[0-9]{3})R(?P<trial>[0-9]{2})")
        info = pattern.match(filepath.name).groupdict()
        info = {k: str(int(v)) for k, v in info.items()}
        info["trial"] = int(info["trial"])
        info["task"] = cls.task_from_trial(info["trial"])
        return info

    @classmethod
    def uid_from_filepath(self, filepath):
        return self.parse_filepath_info(filepath)["uid"]

    def list_subject_filepaths(self) -> pd.DataFrame:
        filepaths = self.dataset_path.glob("**/*.edf")
        filepaths_list = [
            (
                str(filepath),
                *self.parse_filepath_info(filepath).values()
            )
            for filepath in filepaths
        ]
        filepath_df = pd.DataFrame(
            filepaths_list, columns=["path", "uid", "trial", "task"]
        )
        filepath_df["run"] = 1  # Dummy run
        filepath_df["session"] = 1  # Dummy session

        return filepath_df

    @classmethod
    def _check_tasks(self, tasks):
        assert isinstance(tasks, (list, tuple)), "Tasks must be a list of tasks"
        assert all(
            [isinstance(task, str) for task in tasks]
        ), "The tasks must be strings!"
        assert all(
            [task in self.TASKS for task in tasks]
        ), "You passed a invalid task, the possible tasks are: {}".format(
            ", ".join(self.TASKS)
        )

    @Dataset.uid_decorator
    def load_subject(self, uid, tasks=None, trials=None, session=None, run=None, **kwargs):

        session = self._validate_session(session)
        run = self._validate_run(run)

        tasks = tasks or self.TASKS
        trials = trials or self.TRIALS
        self._check_tasks(tasks)

        filepaths_df = self.list_subject_filepaths()
        filepaths = (
            filepaths_df.query("uid == @uid")
            .query("task in @tasks")
            .query("trial in @trials")
            .query("session == @session")
            .query("run == @run")
            .path.apply(Path)
        )

        epochs_list = list()
        for filepath in filepaths:
            epochs, _ = self.load_from_filepath(filepath, as_epochs=True, **kwargs)
            epochs_list.append(epochs)

        epochs = concatenate_epochs(epochs_list)
        events = epochs.events[:, 2]

        return epochs, events
