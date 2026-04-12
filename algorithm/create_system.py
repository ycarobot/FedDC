from .Base import BaseServer
from .Fedavg_dy import FedAvgServer
from .FedAwi import FedAwiServer

# ======== Below are algorithms for submission ========

from .Test import TestServer

from .ATP import ATPServer
from .ATPTest import ATPTestServer
from .BatchNorm import BatchNormServer
from .Tent import TentServer
from .pasle import PasleServer
from .feddc_o import Feddc_oServer
from .feddc import feddcServer
from .tsd import TSDServer

from .MEMO import MEMOServer
from .program import PROGRAMServer

from .T3A import T3AServer
from .SHOT import SHOTServer

from .EM import EMServer
from .BBSE import BBSEServer

from .Surgical import SurgicalServer

def create_system(train_datasets, test_datasets, args):
    algorithm = args.algorithm

    if algorithm == 'central':
        server = BaseServer(train_datasets, test_datasets, args)

    elif algorithm == 'fedavg':
        server = FedAvgServer(train_datasets, test_datasets, args)

    elif algorithm == 'fedawi':
        server = FedAwi(train_datasets, test_datasets, args)

    # ======== Below are algorithms for submission ========

    elif algorithm == 'test':
        server = TestServer(train_datasets, test_datasets, args)

    elif algorithm == 'program':
        server = PROGRAMServer(train_datasets, test_datasets, args)

    elif algorithm == 'tsd':
        server = TSDServer(train_datasets, test_datasets, args)

    elif algorithm == 'pasle':
        server = PasleServer(train_datasets, test_datasets, args)

    elif algorithm == 'feddc_o':
        server = Feddc_oServer(train_datasets, test_datasets, args)

    elif algorithm == 'feddc':
        server = FeddcServer(train_datasets, test_datasets, args)

    elif algorithm == 'atp':
        server = ATPServer(train_datasets, test_datasets, args)

    elif algorithm == 'atptest':
        server = ATPTestServer(train_datasets, test_datasets, args)

    elif algorithm == 'bn':
        server = BatchNormServer(train_datasets, test_datasets, args)

    elif algorithm == 'tent':
        server = TentServer(train_datasets, test_datasets, args)

    elif algorithm == 'memo':
        server = MEMOServer(train_datasets, test_datasets, args)

    elif algorithm == 't3a':
        server = T3AServer(train_datasets, test_datasets, args)

    elif algorithm == 'shot':
        server = SHOTServer(train_datasets, test_datasets, args)

    elif algorithm == 'em':
        server = EMServer(train_datasets, test_datasets, args)

    elif algorithm == 'bbse':
        server = BBSEServer(train_datasets, test_datasets, args)

    elif algorithm == 'surgical':
        server = SurgicalServer(train_datasets, test_datasets, args)

    # ======== Above are algorithms for submission ========

    else:
        raise NotImplementedError

    return server
