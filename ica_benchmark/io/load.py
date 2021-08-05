from mne.io import read_raw_gdf
from mne import Epochs, events_from_annotations, concatenate_raws, concatenate_epochs

PRELOAD = False

class class_or_instancemethod(classmethod):
    def __get__(self, instance, type_):
        descr_get = super().__get__ if instance is None else self.__func__.__get__
        return descr_get(instance, type_)

class BCI_IV_Comp_Dataset():

    EVENT_MAP_DICT = {
        "769": 0,
        "770": 1,
        "771": 2,
        "772": 3
    }

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

    def __init__(self):
        pass

    @classmethod
    def load_as_raw(cls, filepath, preload=False):
        eog_channels = ["EOG-left", "EOG-central", "EOG-right"]
        gdf_obj = read_raw_gdf(
            filepath,
            preload=preload,
            eog=eog_channels,
            exclude=eog_channels,
        )
    
        return gdf_obj
    

    @classmethod
    def load_as_epochs(cls, filepath, tmin=-.3, tmax=.7, reject=None):
        if reject is None:
            reject = cls.REJECT_MAGNITUDE

        gdf_obj = cls.load_as_raw(filepath, preload=True)
        events, _ = events_from_annotations(gdf_obj, event_id=cls.EVENT_MAP_DICT)
        epochs = Epochs(
            gdf_obj,
            events,
            event_repeated="drop",
            reject_by_annotation=True,
            tmin=tmin,
            tmax=tmax,
            reject=dict(eeg=reject)
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
        gdf_obj = cls.load_as_raw(filepath, preload=False)
        return cls.read_metadata_from_raw(gdf_obj)

    @classmethod
    def load_dataset(cls, filepaths, as_epochs=False, concatenate=False, drop_bad=False, **load_kwargs):
        objs = list()
        load_fn = cls.load_as_epochs if as_epochs else cls.load_as_raw
        concatenate_fn = concatenate_epochs if as_epochs else concatenate_raws

        objs = list()
        for filepath in filepaths:
            obj = load_fn(filepath, **load_kwargs)
            objs.append(obj)
        
        if concatenate:
            objs = concatenate_fn(objs)
        
        return objs


def join_gdfs_to_numpy(gdfs):

    all_labels = []
    for gdf in gdfs:
        all_labels.append(get_annotations_from_gdf(gdf))

    labels = np.concatenate(all_labels, axis=0)
    gdf_base = gdfs[0].copy()
    for gdf in gdfs[1:]:
        gdf_base._data = np.concatenate([gdf_base._data, gdf._data], axis=1)

    data = gdf_base._data.T

    return data, labels


def load_gdf_file(filepath):
    gdf_data = read_raw_gdf(filepath, preload=PRELOAD)

    chs = gdf_data.ch_names

    gdf_data = read_raw_gdf(
        filepath,
        preload=True,
        eog=["EOG-left", "EOG-central", "EOG-right"],
        exclude=[x for x in chs if "EOG" in x],
    )
    ch_names = gdf_data.ch_names
    info = parse_info(gdf_data._raw_extras[0]["subject_info"])

    labels = get_annotations_from_gdf(gdf_data)

    return gdf_data, labels, ch_names, info


def load_subject_data(root, subject, filepaths=None, return_as_gdf=True):

    if filepaths is None:
        filepaths = root.glob("{subject}..gdf")

    gdf_all_data = []
    labels_all_data = []

    ch_names_t, info_t = None, None

    for filepath in filepaths:
        gdf_data, labels, ch_names, info = load_gdf_file(filepath)
        gdf_all_data.append(gdf_data.get_data())
        labels_all_data.append(labels)

        assert np.all(ch_names_t == ch_names) or ch_names_t is None
        assert np.all(info_t == info) or info_t is None
        ch_names_t = ch_names
        info_t = info

    gdf_data._data = np.concatenate(gdf_all_data, axis=1)

    labels = np.concatenate(labels_all_data, axis=0)

    if not return_as_gdf:
        gdf_data = gdf_data._data.T
        assert len(gdf_data) == len(labels)
    else:
        assert len(gdf_data._data.T) == len(labels)

    return gdf_data, labels, ch_names, info


def id_from_filepath(filepath):
    return filepath.name[:3]


def load_subjects_data(root, subjects=None, mode=None, return_as_gdf=True):
    root = Path(root)
    data_dict = {}
    if subjects is None:
        subjects = dict()
        if mode is None:
            filepaths = root.glob("*.gdf")
        elif mode.lower() in ("t", "train"):
            filepaths = root.glob("*T.gdf")
        elif mode.lower() in ("v", "test", "validation", "val", "e", "evaluation"):
            filepaths = root.glob("*E.gdf")

        for filepath in filepaths:
            subject_id = id_from_filepath(filepath)
            if subject_id in subjects:
                subjects[subject_id].append(filepath.name)
            else:
                subjects[subject_id] = [filepath.name]

        subjects[subject_id] = sorted(subjects[subject_id])

    chs_ = None
    for subject_id, filenames in subjects.items():
        #        data_dict[subject_id] =  load_subject_data(root, subject_id, filepaths)
        filepaths = [root / filename for filename in filenames]
        gdf, labels, chs, info = load_subject_data(
            root, subject_id, filepaths=filepaths, return_as_gdf=return_as_gdf
        )
        if chs_ is None:
            chs_ = chs
        else:
            assert chs_ == chs
        data_dict[subject_id] = {"gdf": gdf, "chs": chs, "info": info, "labels": labels}

    return data_dict


def load_dataset(root, dataset, mode=None):
    dataset_dict = dict()
    for dataset_name in dataset:
        dataset_dict[dataset_name] = load_subjects_data(
            root, dataset[dataset_name], mode=None
        )
    return dataset_dict
