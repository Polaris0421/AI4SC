import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import argparse
import time
import sys
import json
import numpy as np
import pprint
import yaml
import warnings
warnings.filterwarnings("ignore")

import torch
from matdeeplearn import process, training

################################################################################
#
################################################################################
#  MatDeepLearn code
################################################################################
#
################################################################################
def main():
    start_time = time.time()
    print("Starting...")
    print(
        "GPU is available:",
        torch.cuda.is_available(),
        ", Quantity: ",
        torch.cuda.device_count(),
    )

    parser = argparse.ArgumentParser(description="MatDeepLearn inputs")
    ###Job arguments
    parser.add_argument(
        "--config_path",
        default="config.yml",
        type=str,
        help="Location of config file (default: config.json)",
    )
    parser.add_argument(
        "--run_mode",
        default=None,
        type=str,
        help="run modes: Training, Predict, Repeat, CV, Hyperparameter, Ensemble, Analysis",
    )
    parser.add_argument(
        "--job_name",
        default=None,
        type=str,
        help="name of your job and output files/folders",
    )
    parser.add_argument(
        "--model",
        default=None,
        type=str,
        help="BiSPGCN, DEEP_GATGNN_demo",
    )
    parser.add_argument(
        "--seed",
        default=None,
        type=int,
        help="seed for data split, 0=random",
    )
    parser.add_argument(
        "--model_path",
        default=None,
        type=str,
        help="path of the model .pth file",
    )
    parser.add_argument(
        "--save_model",
        default=None,
        type=str,
        help="Save model",
    )
    parser.add_argument(
        "--load_model",
        default=None,
        type=str,
        help="Load model",
    )
    parser.add_argument(
        "--write_output",
        default=None,
        type=str,
        help="Write outputs to csv",
    )
    parser.add_argument(
        "--parallel",
        default=None,
        type=str,
        help="Use parallel mode (ddp) if available",
    )
    parser.add_argument(
        "--reprocess",
        default=None,
        type=str,
        help="Reprocess data since last run",
    )
    ###Processing arguments
    parser.add_argument(
        "--data_path",
        default=None,
        type=str,
        help="Location of data containing structures (json or any other valid format) and accompanying files",
    )
    parser.add_argument("--format", default=None, type=str, help="format of input data")
    ###Training arguments
    parser.add_argument("--train_ratio", default=None, type=float, help="train ratio")
    parser.add_argument(
        "--val_ratio", default=None, type=float, help="validation ratio"
    )
    parser.add_argument("--test_ratio", default=None, type=float, help="test ratio")
    parser.add_argument(
        "--verbosity", default=None, type=int, help="prints errors every x epochs"
    )

    parser.add_argument(
        "--target_index",
        default=None,
        type=int,
        help="which column to use as target property in the target file",
    )
    ###Model arguments
    parser.add_argument(
        "--epochs",
        default=None,
        type=int,
        help="number of total epochs to run",
    )
    parser.add_argument("--batch_size", default=None, type=int, help="batch size")
    parser.add_argument("--lr", default=None, type=float, help="learning rate")
    parser.add_argument(
        "--gc_count",
        default=None,
        type=int,
        help="number of gc layers",
    )
    parser.add_argument(
        "--info_fc_count",
        default=None,
        type=int,
        help="number of info processing layers",
    )
    parser.add_argument(
        "--dropout_rate",
        default=None,
        type=float,
        help="dropout rate",
    )

    ### Others
    # pretrain
    parser.add_argument(
        "--pt",
        default='False',
        type=str,
        help="Using Pretrain Model Embedding",
    )

    # 是否单独打印disorder的内容
    parser.add_argument(
        "--find_disorder",
        default=False,
        type=bool,
    )

    # augumentation
    parser.add_argument(
        "--aug",
        default='False',
        type=str, 
        help="If Using Augumentation",
    ) 
    
    parser.add_argument(
        "--aug_times",
        default=5,
        type=int,
        help="Augumentation times",
    ) 
    parser.add_argument(
        "--aug_stage",
        default=0.,
        type=float,
        help="Augumentation stage; i.e. 0.6 is 60%",
    ) 

    ##Get arguments from command line
    args = parser.parse_args(sys.argv[1:])

    ##Open provided config file
    assert os.path.exists(args.config_path), (
        "Config file not found in " + args.config_path
    )
    with open(args.config_path, "r") as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)

    ##Update config values from command line
    if args.run_mode != None:
        config["Job"]["run_mode"] = args.run_mode
    run_mode = config["Job"].get("run_mode")
    config["Job"] = config["Job"].get(run_mode)
    if config["Job"] == None:
        print("Invalid run mode")
        sys.exit()

    if args.job_name != None:
        config["Job"]["job_name"] = args.job_name
    if args.model != None:
        config["Job"]["model"] = args.model
    if args.seed != None:
        config["Job"]["seed"] = args.seed
    if args.model_path != None:
        config["Job"]["model_path"] = args.model_path
    if args.load_model != None:
        config["Job"]["load_model"] = args.load_model
    if args.save_model != None:
        config["Job"]["save_model"] = args.save_model
    if args.write_output != None:
        config["Job"]["write_output"] = args.write_output
    if args.parallel != None:
        config["Job"]["parallel"] = args.parallel
    if args.reprocess != None:
        config["Job"]["reprocess"] = args.reprocess

    if args.data_path != None:
        config["Processing"]["data_path"] = args.data_path
    if args.format != None:
        config["Processing"]["data_format"] = args.format

    if args.train_ratio != None:
        config["Training"]["train_ratio"] = args.train_ratio
    if args.val_ratio != None:
        config["Training"]["val_ratio"] = args.val_ratio
    if args.test_ratio != None:
        config["Training"]["test_ratio"] = args.test_ratio
    if args.verbosity != None:
        config["Training"]["verbosity"] = args.verbosity
    if args.target_index != None:
        config["Training"]["target_index"] = args.target_index

    if args.aug != None:
        config["Training"]["aug"] = args.aug
    if args.aug_times != None:
        config["Training"]["aug_times"] = args.aug_times
    if args.aug_stage != None:
        config["Training"]["aug_stage"] = args.aug_stage

    for key in config["Models"]:
        if args.epochs != None:
            config["Models"][key]["epochs"] = args.epochs
        if args.batch_size != None:
            config["Models"][key]["batch_size"] = args.batch_size
        if args.lr != None:
            config["Models"][key]["lr"] = args.lr
        if args.gc_count != None:
            config["Models"][key]["gc_count"] = args.gc_count
        if args.info_fc_count != None:
            config["Models"][key]["info_fc_count"] = args.info_fc_count
        if args.find_disorder != None:
            config["Models"][key]["find_disorder"] = args.find_disorder
        if args.dropout_rate != None:
            config["Models"][key]["dropout_rate"] = args.dropout_rate
        if args.pt != None: ### Pretrain ###
            config["Models"][key]["pt"] = args.pt

    if run_mode == "Predict":
        config["Models"] = {}
    else:
        config["Models"] = config["Models"].get(config["Job"]["model"].strip("'"))

    if config["Job"]["seed"] == 0:
        config["Job"]["seed"] = np.random.randint(1, 1e6)

    ## 参数集存储
    print("Settings: ")
    pprint.pprint(config)
    with open(str(config["Job"]["job_name"]) + "_settings.txt", "w") as log_file:
        pprint.pprint(config, log_file)

    ################################################################################
    #  导入数据集
    ################################################################################
    process_start_time = time.time()

    dataset = process.get_dataset(
        config["Processing"]["data_path"],
        config["Training"]["target_index"],
        config["Job"]["reprocess"],
        config["Processing"],
    )

    print("Dataset used:", dataset)
    print(dataset[0])

    print("--- %s seconds for processing ---" % (time.time() - process_start_time))

    ################################################################################
    #  模型训练
    ################################################################################
    ## 预测模式
    if run_mode == "Predict":
        print("Starting prediction from trained model")
        train_error = training.predict(
            dataset, config["Training"]["loss"], config["Job"]
        )
        print("Test Error: {:.5f}".format(train_error))

    ## 重复训练模式
    elif run_mode == "Repeat":
        print("Repeat training for " + str(config["Job"]["repeat_trials"]) + " trials")
        training.train_repeat(
            config["Processing"]["data_path"],
            config["Job"],
            config["Training"],
            config["Models"],
        )
    else:
        print("No valid mode selected, try again")

    print("--- %s total seconds elapsed ---" % (time.time() - start_time))
    print("--- %s total hours elapsed ---" % ((time.time() - start_time)/3600))

if __name__ == "__main__":
    main()
