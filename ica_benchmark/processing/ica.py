import mne
from ica_benchmark.processing.jade import jade, JadeICA
from coroica import UwedgeICA, CoroICA
from abc import ABC, abstractmethod
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning) 
mne.set_log_level("ERROR")

def _get_kwargs(m, is_extended=False):
    if is_extended:
        return dict(method=m, fit_params=dict(extended=True))
    return dict(method=m)

def create_gdf_obj(arr):
    if isinstance(arr, mne.io.Raw):
        return arr

    n_channels = arr.shape[1]
    info = mne.create_info(
        ch_names=["C" + str(x+1) for x in range(n_channels)],
        ch_types=['eeg'] * n_channels,
        sfreq=SAMPLING_FREQ
    )
    return mne.io.RawArray(arr.T, info)

class ICABase(ABC):

    def __init__(self, n_components=None, method=None):
        self.n_components = n_components
        self.method = method
        self.transformer = None
        self.setup()

    def fit(self, x, n_components=None):
        self.transformer.fit(x, n_components=self.n_components)
    
    def transform(self, x):
        return self.transformer.transform(x)
    
    @abstractmethod
    def setup(self):
        raise NotImplementedError

class MNETransformerWrapper():

    def __init__(self, transformer):
        self.transformer = transformer
    
    def fit(self, x, n_components=None):
        gdf_data = create_gdf_obj(x)
        self.transformer.fit(gdf_data, picks="all")
    
    def transform(self, x):
        gdf_data = create_gdf_obj(x)
        sources = self.transformer.get_sources(gdf_data).get_data().T
        return sources

class MNEICA(ICABase):

    def setup(self):
        #self.transformer = mne.preprocessing.ICA(n_components=self.n_components, verbose=0, **_ica_kwargs_dict[self.method])
        self.transformer = mne.preprocessing.ICA(
            n_components=self.n_components,
            #n_components=None,
            #n_pca_components=self.n_components,
            verbose=0,
            **_ica_kwargs_dict[self.method]
        )
        self.transformer = MNETransformerWrapper(self.transformer)

class CoroPackICA(ICABase):
    def setup(self):
        kwargs = _coro_kwargs_dict[self.method]
        ica_constructor = UwedgeICA if self.method != "coro" else CoroICA
        self.transformer = ica_constructor(n_components=self.n_components, **kwargs)

    def fit(self, x, n_components=None):
        self.transformer.fit(x)

SAMPLING_FREQ = 250.

_ica_kwargs_dict = {
    "fastica": _get_kwargs("fastica"),
    "infomax": _get_kwargs("infomax"),
    "picard": _get_kwargs("picard"),
    "ext_infomax": _get_kwargs("infomax", is_extended=True),
    "ext_picard": _get_kwargs("picard", is_extended=True)
}

_coro_kwargs_dict = {
    "sobi": dict(partitionsize=int(10**6), timelags=list(range(1, 101))),
    "choi_var": dict(),
    "choi_vartd": dict(timelags=[1, 2, 3, 4, 5]),
    "choi_td": dict(instantcov=False, timelags=[1, 2, 3, 4, 5]),
    "coro": dict()
}

def get_ica_transformers(method="sobi", n_components=None):

    if method in _ica_kwargs_dict:
        return MNEICA(n_components=n_components, method=method)
    elif method in _coro_kwargs_dict:
        return CoroPackICA(n_components=n_components, method=method)
    else:
        raise Exception("Method {} not found.".format(method))
        