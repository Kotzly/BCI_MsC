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

    def __init__(self):
        pass

    @classmethod
    def load_as_raw(cls, filepath, load_eog=False, preload=False):
        
        gdf_obj = read_raw_gdf(
            filepath,
            preload=preload,
            eog=cls.EOG_CHANNELS,
            exclude=list() if load_eog else cls.EOG_CHANNELS,
        )
    
        return gdf_obj

    @classmethod
    def load_as_epochs(cls, filepath, tmin=-.3, tmax=.7, reject=None, load_eog=False, has_labels=True):
        if reject is None:
            reject = dict(eeg=cls.REJECT_MAGNITUDE)
        elif reject is False:
            reject = None

        gdf_obj = cls.load_as_raw(filepath, preload=True, load_eog=load_eog)
        events, _ = events_from_annotations(gdf_obj, event_id=cls.EVENT_MAP_DICT if has_labels else cls.UNKNOWN_EVENT_MAP_DICT)
        epochs = Epochs(
            gdf_obj,
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
        gdf_obj = cls.load_as_raw(filepath, preload=False)
        return cls.read_metadata_from_raw(gdf_obj)

    @classmethod
    def load_dataset(cls, filepaths, as_epochs=False, concatenate=False, drop_bad=False, return_metadata=False, **load_kwargs):
        
        objs = list()
        load_fn = cls.load_as_epochs if as_epochs else cls.load_as_raw
        concatenate_fn = concatenate_epochs if as_epochs else concatenate_raws

        objs = list()
        metadata = list()
        for filepath in filepaths:
            obj = load_fn(filepath, **load_kwargs)
            objs.append(obj)

            if return_metadata:
                metadata.append(
                    cls.read_metadata(filepath)
                )
        if concatenate:
            objs = concatenate_fn(objs)
        
        if return_metadata:
            return objs, metadata
            
        return objs
