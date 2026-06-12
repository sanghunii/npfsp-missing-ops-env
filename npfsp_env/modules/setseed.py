import random
import numpy as np
import torch

SEED = 42       # GLOBAL SEED

MC1_SEED = 1
MC2_SEED = 3
MC3_SEED = 5
MC4_SEED = 7
MC5_SEED = 9
MC6_SEED = 11
MC7_SEED = 13
MC8_SEED = 15
MC9_SEED = 17
MC10_SEED = 19


INSP_SEED = 7

def set_seed(val):
    torch.manual_seed(val)
    torch.cuda.manual_seed(val)
    torch.cuda.manual_seed_all(val)  # 멀티 GPU를 사용하는 경우
    np.random.seed(val)
    random.seed(val)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False