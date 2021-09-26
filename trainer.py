"""
"""

import os
import sys
import time
import logging
import argparse
import textwrap
from copy import deepcopy
from collections import deque, OrderedDict
from typing import Any, Union, Optional, Tuple, Sequence, NoReturn
from numbers import Real, Number

import numpy as np
np.set_printoptions(precision=5, suppress=True)
# try:
#     from tqdm.auto import tqdm
# except ModuleNotFoundError:
#     from tqdm import tqdm
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch import Tensor
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP, DataParallel as DP
from tensorboardX import SummaryWriter
from easydict import EasyDict as ED

# from torch_ecg.torch_ecg.models._nets import BCEWithLogitsWithClassWeightLoss
from torch_ecg.torch_ecg.utils.utils_nn import default_collate_fn as collate_fn
from torch_ecg.torch_ecg.utils.misc import (
    init_logger, get_date_str, dict_to_str, str2bool,
)
from model import (
    ECG_SEQ_LAB_NET_CPSC2021,
    ECG_UNET_CPSC2021, ECG_SUBTRACT_UNET_CPSC2021,
    RR_LSTM_CPSC2021,
)
from utils.scoring_metrics import compute_challenge_metric
from cfg import BaseCfg, TrainCfg, ModelCfg
from dataset import CPSC2021

if BaseCfg.torch_dtype.lower() == "double":
    torch.set_default_tensor_type(torch.DoubleTensor)
    _DTYPE = torch.float64
else:
    _DTYPE = torch.float32


__all__ = [
    "train",
]


def train(model:nn.Module,
          model_config:dict,
          device:torch.device,
          config:dict,
          logger:Optional[logging.Logger]=None,
          debug:bool=False) -> OrderedDict:
    """ NOT finished, NOT checked,

    Parameters
    ----------
    model: Module,
        the model to train
    model_config: dict,
        config of the model, to store into the checkpoints
    device: torch.device,
        device on which the model trains
    config: dict,
        configurations of training, ref. `ModelCfg`, `TrainCfg`, etc.
    logger: Logger, optional,
        logger
    debug: bool, default False,
        if True, the training set itself would be evaluated 
        to check if the model really learns from the training set

    Returns
    -------
    best_state_dict: OrderedDict,
        state dict of the best model
    """
    raise NotImplementedError


@torch.no_grad()
def evaluate(model:nn.Module,
             data_loader:DataLoader,
             config:dict,
             device:torch.device,
             debug:bool=True,
             logger:Optional[logging.Logger]=None) -> Tuple[float,...]:
    """ NOT finished, NOT checked,

    Parameters
    ----------
    model: Module,
        the model to evaluate
    data_loader: DataLoader,
        the data loader for loading data for evaluation
    config: dict,
        evaluation configurations
    device: torch.device,
        device for evaluation
    debug: bool, default True,
        more detailed evaluation output
    logger: Logger, optional,
        logger to record detailed evaluation output,
        if is None, detailed evaluation output will be printed

    Returns
    -------
    eval_res: tuple of float,
        evaluation results, including
        auroc, auprc, accuracy, f_measure, f_beta_measure, g_beta_measure, challenge_metric
    """
    model.eval()
    raise NotImplementedError


def get_args(**kwargs:Any):
    """
    """
    cfg = deepcopy(kwargs)
    parser = argparse.ArgumentParser(
        description="Train the Model on CPSC2021",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-b", "--batch-size",
        type=int, default=128,
        help="the batch size for training",
        dest="batch_size")
    # parser.add_argument(
    #     "-c", "--cnn-name",
    #     type=str, default="multi_scopic_leadwise",
    #     help="choice of cnn feature extractor",
    #     dest="cnn_name")
    # parser.add_argument(
    #     "-r", "--rnn-name",
    #     type=str, default="none",
    #     help="choice of rnn structures",
    #     dest="rnn_name")
    # parser.add_argument(
    #     "-a", "--attn-name",
    #     type=str, default="se",
    #     help="choice of attention structures",
    #     dest="attn_name")
    parser.add_argument(
        "--keep-checkpoint-max", type=int, default=20,
        help="maximum number of checkpoints to keep. If set 0, all checkpoints will be kept",
        dest="keep_checkpoint_max")
    # parser.add_argument(
    #     "--optimizer", type=str, default="adam",
    #     help="training optimizer",
    #     dest="train_optimizer")
    parser.add_argument(
        "--debug", type=str2bool, default=False,
        help="train with more debugging information",
        dest="debug")
    
    args = vars(parser.parse_args())

    cfg.update(args)
    
    return ED(cfg)


_MODEL_MAP = {
    "seq_lab": ECG_SEQ_LAB_NET_CPSC2021,
    "unet": ECG_UNET_CPSC2021,
    "lstm_crf": RR_LSTM_CPSC2021,
    "lstm": RR_LSTM_CPSC2021,
}


if __name__ == "__main__":
    config = get_args(**TrainCfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = init_logger(log_dir=config.log_dir, verbose=2)
    logger.info(f"\n{'*'*20}   Start Training   {'*'*20}\n")
    logger.info(f"Using device {device}")
    logger.info(f"Using torch of version {torch.__version__}")
    logger.info(f"with configuration\n{dict_to_str(config)}")

    # TODO: adjust for CPSC2021
    for task in config.tasks:
        model_cls = _MODEL_MAP[config[task].model_name]
        model_config = deepcopy(ModelCfg[task])
        model = model_cls(
            classes=config.classes,
            n_leads=config.n_leads,
            config=model_config,
        )
        model.__DEBUG__ = False
        if torch.cuda.device_count() > 1:
            model = DP(model)
            # model = DDP(model)
        model.to(device=device)

        try:
            train(
                model=model,
                model_config=model_config,
                config=config,
                device=device,
                logger=logger,
                debug=config.debug,
            )
        except KeyboardInterrupt:
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_config": model_config,
                "train_config": config,
            }, os.path.join(config.checkpoints, "INTERRUPTED.pth.tar"))
            logger.info("Saved interrupt")
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)
