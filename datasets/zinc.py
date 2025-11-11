import time
from typing import Any, Callable

from torch_geometric.data.separate import separate
from torch_geometric.datasets import ZINC as PyG_ZINC
from tqdm import tqdm


class ZINC(PyG_ZINC):
    def __init__(
        self, root: str, subset: bool = False, split: str = 'train',
        full_pre_transform: Callable[..., Any] | None = None,
        inmemory_transform: Callable[..., Any] | None = None,
        onthefly_transform: Callable[..., Any] | None = None,
        force_reload: bool = False
    ) -> None:
        super().__init__(root, subset, split, onthefly_transform, force_reload=force_reload)
        self._split = split
        if full_pre_transform is not None:
            self._full_pre_process(full_pre_transform)
        if inmemory_transform is not None:
            self._inmemory_process(inmemory_transform)

        # Use just the first dimension
        self._seperate_data()
        for data in self._data_list:
            data.x = data.x[:, 0]

    def _seperate_data(self):
        if not hasattr(self, '_data_list') or self._data_list is None:
            self._data_list = [
                separate(self._data.__class__, self._data, idx, self.slices, decrement=False)
                for idx in range(self.len())
            ]
        return self._data_list

    def _full_pre_process(self, full_pre_transform):
        data_list = self._seperate_data()
        processed_dir = self.processed_dir + '/' + self._split
        self._data_list = full_pre_transform(data_list, processed_dir)

    def _inmemory_process(self, inmemory_transform):
        data_list = self._seperate_data()
        print(f"Computing {inmemory_transform}")
        time_start = time.perf_counter()
        data_list = [inmemory_transform(data) for data in tqdm(data_list)]
        time_elapsed = time.perf_counter() - time_start
        print(f"Took {time_elapsed:.2f}s.")
        self._data_list.clear()
        del self._data_list
        self._data_list = data_list
