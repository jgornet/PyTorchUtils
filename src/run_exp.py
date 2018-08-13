#!/usr/bin/env python
__doc__ = """

Training Script

Put all the ugly things that change with every experiment here

Nicholas Turner, 2017-8
"""

import os, imp
import collections

import torch
import tensorboardX

import utils
import train
import loss
import models
import samplers


def main(**args):

    #args should be the info you need to specify the params
    # for a given experiment, but only params should be used below
    params = fill_params(**args)

    utils.set_gpus(params["gpus"])

    utils.make_required_dirs(**params)

    tstamp = utils.timestamp()
    utils.log_params(params, tstamp=tstamp)
    utils.log_tagged_modules(params["modules_used"],
                             params["log_dir"], "train",
                             chkpt_num=params["chkpt_num"],
                             tstamp=tstamp)

    start_training(**params)


def fill_params(expt_name, chkpt_num, batch_sz, gpus,
                sampler_name, model_name, **args):

    params = {}

    #Model params
    params["in_dim"]       = 1
    params["output_spec"]  = collections.OrderedDict(soma_label=1)
    params["depth"]        = 4
    params["batch_norm"]   = True

    #Training procedure params
    params["max_iter"]    = 1000000
    params["lr"]          = 0.0001
    params["test_intv"]   = 1000
    params["test_iter"]   = 100
    params["avgs_intv"]   = 50
    params["chkpt_intv"]  = 500
    params["warm_up"]     = 50
    params["chkpt_num"]   = chkpt_num
    params["batch_size"]  = batch_sz

    #Sampling params
    # params["data_dir"]     = os.path.expanduser("~/seungmount/research/Nick/datasets/CSHL_GAD/")
    params["data_dir"] = "/home/gornet/Data/OLST"
    assert os.path.isdir(params["data_dir"]),"nonexistent data directory"
    params["train_sets"]   = ["train"]
    params["val_sets"]     = ["val"]

    #GPUS
    params["gpus"] = gpus

    #IO/Record params
    params["expt_name"]  = expt_name
    params["expt_dir"]   = "experiments/{}".format(expt_name)
    params["model_dir"]  = os.path.join(params["expt_dir"], "models")
    params["log_dir"]    = os.path.join(params["expt_dir"], "logs")
    params["fwd_dir"]    = os.path.join(params["expt_dir"], "forward")
    params["tb_train"]   = os.path.join(params["expt_dir"], "tb/train")
    params["tb_val"]     = os.path.join(params["expt_dir"], "tb/val")

    #Use-specific Module imports
    sampler_module = getattr(samplers,sampler_name)
    params["sampler_class"] = sampler_module.Sampler
    model_module = getattr(models,model_name)
    params["model_class"]   = model_module.Model

    #"Schema" for turning the parameters above into arguments
    # for the model class
    params["model_args"]     = [ params["in_dim"], params["output_spec"],
                                 params["depth"] ]
    params["model_kwargs"]   = { "bn" : params["batch_norm"] }

    #modules used for record-keeping
    params["modules_used"] = [__file__, model_module.__file__,
                              sampler_module.__file__, "loss.py"]

    return params


def start_training(model_class, model_args, model_kwargs, sampler_class,
                   chkpt_num, lr, train_sets, val_sets, data_dir,
                   model_dir, log_dir, tb_train, tb_val,
                   **params):

    #PyTorch Model
    net = utils.create_network(model_class, model_args, model_kwargs)
    train_writer = tensorboardX.SummaryWriter(tb_train)
    val_writer = tensorboardX.SummaryWriter(tb_val)
    monitor = utils.LearningMonitor()

    #Loading model checkpoint (if applicable)
    if chkpt_num != 0:
        utils.load_chkpt(net, monitor, chkpt_num, model_dir, log_dir)

    #DataProvider Sampler
    train_sampler = utils.AsyncSampler(sampler_class(data_dir, dsets=train_sets,
                                                     mode="train"))

    val_sampler   = utils.AsyncSampler(sampler_class(data_dir, dsets=val_sets,
                                                     mode="val"))

    loss_fn = loss.TopologyWarpingBCE()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    train.train(net, loss_fn, optimizer, train_sampler, val_sampler,
                train_writer=train_writer, val_writer=val_writer,
                last_iter=chkpt_num, model_dir=model_dir, log_dir=log_dir,
                monitor=monitor,
                **params)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description= __doc__)

    parser.add_argument("expt_name",
                        help="Experiment Name")
    parser.add_argument("sampler_name",
                        help="DataProvider Sampler name")
    parser.add_argument("model_name",
                        help="Model Template name")
    parser.add_argument("--batch_sz",  type=int, default=1,
                        help="Batch size for each sample")
    parser.add_argument("--chkpt_num", type=int, default=0,
                        help="Checkpoint Number")
    parser.add_argument("--gpus", default=["0"], nargs="+")

    args = parser.parse_args()


    main(**vars(args))
