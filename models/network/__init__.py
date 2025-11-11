from .grit import GRIT_Net
from .ppgn import PPGN
from .ppgn_brec import PPGN_BREC
from .sppgn1 import SPPGN1
from .sppgn2 import SPPGN2
from .sp2fwlgin import Sp2FwlGIN
from .classic_gnns import ClassicGNNs


network_dict = {
    "grit": GRIT_Net,
    "ppgn": PPGN,
    "sppgn1": SPPGN1,
    "sp2fwlgin": Sp2FwlGIN,
    "sppgn2": SPPGN2,
    "ppgn_brec": PPGN_BREC,
    "classic": ClassicGNNs,
}
