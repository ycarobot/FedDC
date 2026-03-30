from .Base import BaseServer
from .FedAvg import FedAvgServer
from .FedAU_init4 import FedAU_Init4Server

# ======== Below are algorithms for submission ========

from .Test import TestServer

from .ATP import ATPServer
from .ATPTest import ATPTestServer
from .BatchNorm import BatchNormServer
from .Tent import TentServer
from .pasle6_1 import ATPTestPasle6_1Server
from .pasle_only import ATPTestPasleOServer
from .pasle_p import ATPPALSEPTestServer
from .tsd5 import ATPTestTSD5Server
from .Fedavg_dy3 import FedAvgServer3
from .MEMO import MEMOServer
from .program4 import ATPTestPROGRAM4Server

from .T3A import T3AServer
from .SHOT import SHOTServer

from .EM import EMServer
from .BBSE import BBSEServer

from .Surgical import SurgicalServer



# ======== Above are algorithms for submission ========


def create_system(train_datasets, test_datasets, args):
    algorithm = args.algorithm

    if algorithm == 'central':
        server = BaseServer(train_datasets, test_datasets, args)
    # elif algorithm == 'fedavg':
    #     server = FedAvgServer(train_datasets, test_datasets, args)

    elif algorithm == 'fedavg':
        server = FedAvgServer3(train_datasets, test_datasets, args)

    elif algorithm == 'fedawi':
        server = FedAU_Init4Server(train_datasets, test_datasets, args)

    # ======== Below are algorithms for submission ========

    elif algorithm == 'test':
        server = TestServer(train_datasets, test_datasets, args)

    elif algorithm == 'program':
        server = ATPTestPROGRAM4Server(train_datasets, test_datasets, args)

    elif algorithm == 'tsd':
        server = ATPTestTSD5Server(train_datasets, test_datasets, args)

    elif algorithm == 'pasle':
        server = ATPTestPasle6_1Server(train_datasets, test_datasets, args)

    elif algorithm == 'feddc_o':
        server = ATPTestPasleOServer(train_datasets, test_datasets, args)

    elif algorithm == 'feddc':
        server = ATPPALSEPTestServer(train_datasets, test_datasets, args)

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
