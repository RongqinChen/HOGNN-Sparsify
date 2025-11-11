from itertools import permutations, product


class KCVSet:
    "K-Connected Vertex Set"

    def __init__(self, k, verts, child_list=None):
        self.k: int = k
        # suppose $J=K-1, L=K+1, N=K+2$

        self.verts: set = verts  # the K-connected vertices

        # L-connected vertex sets within this set
        # it should be an empty list if no children (higher-order connected) vertex sets exist
        assert isinstance(child_list, list)
        self.child_list: list["KCVSet"] = child_list

        # K-connected L-verts
        # e.g., 0-con. 1-vertex, 1-con. 2-vertices, etc.
        self._kc_lverts_list = None

        # exactly K-connected N-verts
        # e.g., 0-con. 2-vertices, 1-con. 3-vertices, etc.
        self._ke_nverts_list = None

    @property
    def kc_lverts_list(self):
        if self._kc_lverts_list is not None:
            return self._kc_lverts_list

        if self.k == 0:
            self._kc_lverts_list = list(self.verts)
        else:
            self._kc_lverts_list = list(permutations(self.verts, self.k + 1))
        return self._kc_lverts_list

    @property
    def ke_nverts_list(self):
        if self._ke_nverts_list is not None:
            return self._ke_nverts_list

        if self.child_list is None:
            raise ValueError("This k-set does not have any child l-sets.")

        self._ke_nverts_list = []
        for cld in self.child_list:
            cld_vs = cld.verts
            others = self.verts - cld_vs

            lverts_list = list(permutations(cld_vs, self.k + 1))
            # N verts that are K-connected but not L-connected
            nverts_list = [
                lvert + (oth,) for lvert, oth in product(lverts_list, others)
            ]
            self._ke_nverts_list.extend(nverts_list)

        return self._ke_nverts_list

    def __repr__(self):
        num_child = len(self.child_list) if self.child_list else 0
        return f"KCVSet({self.k}, {len(self.verts)} verts, {num_child} child-sets)"

    def __len__(self):
        return len(self.verts)
