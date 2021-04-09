import mne
from jade import jade, JadeICA
from coroica import UwedgeICA, CoroICA

def _get_kwargs(m, is_extended=False):
    if is_extended:
        return dict(method=m, fit_params=dict(extended=True))
    return dict(method=m)

SAMPLING_FREQ = 250.

_ica_kwargs_dict = {
    "fastica": _get_kwargs("fastica"),
    "infomax": _get_kwargs("infomax"),
    "picard": _get_kwargs("picard"),
    "ext_infomax": _get_kwargs("infomax", is_extended=True),
    "ext_picard": _get_kwargs("picard", is_extended=True)
}

def create_gdf_obj(arr):
    n_channels = arr.shape[1]
    info = mne.create_info(
        ch_names=["C" + str(x+1) for x in range(n_channels)],
        ch_types=['eeg'] * n_channels,
        sfreq=SAMPLING_FREQ
    )
    return mne.io.RawArray(arr.T, info)

class ICABase():

    def __init__(self, n_components=None, method=None)
        self.n_components = n_components
        self.method = method
        self.setup()

    def fit(self, x):
        self.transformer.fit(x, n_components=self.n_components)
    
    def transform(self, x):
        self.transform.transform(x)
    
    def setup(self):
        pass

class MNETransformerWrapper():

    def __init__(self, transform):
        self.transform = transform
    
    def fit(self, x, n_components=None):
        gdf_data = create_gdf_obj(x)
        self.transform.fit(gdf_data, picks="all")
    
    def transform(self, x):
        gdf_data = create_gdf_obj(x)
        sources = self.transform.get_sources(gdf_data).get_data().T
        return sources

class MNEICA(ICABase):

    def setup(self):
        self.transform = mne.preprocessing.ICA(n_components=self.n_components, verbose=None, **_ica_kwargs_dict[method])
        self.transform = MNETransformerWrapper(self.transform)

MNE_ICA_LIST = {
    k: lambda n_components: MNEICA(n_components=n, method=k) for k in _ica_kwargs_dict
}


_coro_kwargs = {
    "sobi": dict(partitionsize=int(10**6), timelags=list(range(1, 101))),
    "choi_var": dict(),
    "choi_vartd": dict(timelags=[1, 2, 3, 4, 5]),
    "choi_td": dict(instantcov=False, timelags=[1, 2, 3, 4, 5]),
    
}

class CoroPackICA(ICABase):
    def setup(self):
        kwargs = _coro_kwargs[self.method]
        self.transform = UwedgeICA(n_components=self.n_components, **kwargs)

def coro_ica(x, n_components=None, method="sobi"):
    kwargs = _coro_kwargs[method]
    ica_transform = UwedgeICA(n_components=n_components, **kwargs)
    ica_transform.fit(x)
    sources = ica_transform.transform(x)
    return sources

def ica_choi_var(x, n_components=None):
    return coro_ica(x, n_components=n_components, method="choi_var")

def ica_choi_vartd(x, n_components=None):
    return coro_ica(x, n_components=n_components, method="choi_vartd")

def ica_choi_td(x, n_components=None):
    return coro_ica(x, n_components=n_components, method="choi_td")

def ica_sobi(x, n_components=None):
    return coro_ica(x, n_components=n_components, method="sobi")

def ica_coroica(x, n_components=None):
    ica_transform = CoroICA()
    ica_transform.fit(x)
    #ica_transform.fit(Xtrain, group_index=groups, partition_index=partition)
    sources = ica_transform.transform(x)
    return sources

ICA_METHODS = [
    ica_jade,
    ica_picard,
    ica_ext_picard,
    ica_infomax,
    ica_ext_infomax,
    ica_fastica,
    ica_choi_var,
    ica_choi_vartd,
    ica_choi_td,
    ica_sobi,
    ica_coroica
]