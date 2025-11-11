from .compute_conn_and_poly import ConnAndPoly
from .compute_rwp_of_nodeedge import RWPOfNodeEdge
from .compute_connrwp_of_nodeedge import ConnRWPOfNodeEdge
from .compute_distances import RD, SPD
from .compute_polynomial import Polynomials
from .compute_poly_conn_and_kblocks import PolyConnAndKblock
from .fully_compute_conn_and_poly import FullConnectivityPolynomial
from .fully_compute_poly_conn_and_kblocks import FullyComputePolyPairConnAndKBlocks
from .qm9_input_transform import QM9InputTransform
# from .kset.kcv_transform import V1KCVTransform as V3KCVTransform

from .compute_2fwl import K2FWLTransform
from .compute_2fwl_connsp import K2FWLConnSpTransform
# from .compute_2fwl_connsplit import K2FWLConnSplitDataTransform
from .compute_2fwl_conndistsp import K2FWLConnDistSpTransform
# from .v1_kcv_transform import V1KCVTransform
# from .v2_kcv_transform import V2KCVTransform


transform_dict = {
    "qm9_input_transform": QM9InputTransform,
    "poly": Polynomials,

    "2fwl": K2FWLTransform,  # ((a, b), (a, c), (c, b)) for all 3-tuples (a, b, c)
    "2fwl_connsp": K2FWLConnSpTransform,  # connectivity-guilded sparsifying
    "2fwl_conndistsp": K2FWLConnDistSpTransform,  # connectivity and distance co-guilded sparsifying
    # "2fwl_connsplit": K2FWLConnSplitDataTransform,  # splitting exactly 1-connected triples

    # "kset_v3": V3KCVTransform,
    "conn_poly": ConnAndPoly,
    "connrwp_of_nodeedge": ConnRWPOfNodeEdge,
    "rwp_of_nodeedge": RWPOfNodeEdge,

    "poly_conn_kblock": PolyConnAndKblock,
    "full_conn_poly": FullConnectivityPolynomial,
    "full_poly_conn_kblock": FullyComputePolyPairConnAndKBlocks,
    # "k-set": V1KCVTransform,
    # "kset_v2": V2KCVTransform,
    "RD": RD,
    "SPD": SPD,
}
