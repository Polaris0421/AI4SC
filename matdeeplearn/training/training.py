##General imports
import csv
import os
import time
from datetime import datetime
import shutil
import copy
import numpy as np
from functools import partial
import platform
import random

##Torch imports
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch_geometric.data import DataLoader, Dataset
from torch_geometric.nn import DataParallel
import torch_geometric.transforms as T
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import torch.multiprocessing as mp

##Matdeeplearn imports
from matdeeplearn import models
import matdeeplearn.process as process
import matdeeplearn.training as training
from matdeeplearn.models.utils import model_summary

# 固定随机种子
def seed_everything(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

################################################################################
#  Training functions
################################################################################

## 训练函数
def train(model, optimizer, loader, loss_method, rank, pt_model=None):
    model.train()
    loss_all = 0
    count = 0
    for data in loader:
        data = data.to(rank)
        optimizer.zero_grad()
        output = model(data)

        ### 预训练嵌入损失 ###
        if pt_model != None:
            output, atom_emb_cg = model(data, pt=True)

            input = data.pretrain_data
            atom_emb_pt = pt_model(input)
            loss = getattr(F, loss_method)(output, data.y) + 1e-4*F.mse_loss(atom_emb_cg, atom_emb_pt)
        else:
            output = model(data)
            loss = getattr(F, loss_method)(output, data.y)

        loss.backward()
        loss_all += loss.detach() * output.size(0)

        optimizer.step()
        count = count + output.size(0)

    loss_all = loss_all / count
    return loss_all


## 指标计算函数
def evaluate(loader, model, loss_method, rank, out=False):
    model.eval()
    loss_all = 0
    count = 0
    for data in loader:
        data = data.to(rank)
        with torch.no_grad():
            output = model(data)
            loss = getattr(F, loss_method)(output, data.y)
            loss_all += loss * output.size(0)
            if out == True:
                if count == 0:
                    ids = [item for sublist in data.structure_id for item in sublist]
                    ids = [item for sublist in ids for item in sublist]
                    predict = output.data.cpu().numpy()
                    target = data.y.cpu().numpy()
                else:
                    ids_temp = [
                        item for sublist in data.structure_id for item in sublist
                    ]
                    ids_temp = [item for sublist in ids_temp for item in sublist]
                    ids = ids + ids_temp
                    predict = np.concatenate(
                        (predict, output.data.cpu().numpy()), axis=0
                    )
                    target = np.concatenate((target, data.y.cpu().numpy()), axis=0)
            count = count + output.size(0)

    loss_all = loss_all / count

    if out == True:
        test_out = np.column_stack((ids, target, predict))
        residual = target - predict
        ssr = np.sum(residual ** 2)
        r2 = 1 - ssr / np.sum((target - np.mean(target)) ** 2)
        mae = np.mean(np.abs(residual))
        print('MAE&R2: ', [mae, r2])
        return loss_all, test_out, [mae, r2]
    elif out == False:
        return loss_all


##Model trainer
def trainer(
        rank,
        world_size,
        model,
        optimizer,
        scheduler,
        loss,
        train_loader,
        val_loader,
        train_sampler,
        epochs,
        verbosity,
        filename="my_model_temp.pth",
        pt_model=None,
        aug_params=None
):
    train_error = val_error = test_error = epoch_time = float("NaN")
    train_start = time.time()
    best_val_error = 1e10
    model_best = model

    ### Two Stage Training ###
    not_aug_yet = True
    if aug_params != None:
        aug, aug_times, aug_stage, dataset_params = aug_params
    else:
        aug = 'False'

    ##Start training over epochs loop
    for epoch in range(1, epochs + 1):

        lr = scheduler.optimizer.param_groups[0]["lr"]

        ### Two Stage Training ###
        if aug=='True' and not_aug_yet and epoch > int(epochs * aug_stage):
            dataset, data_path, batch_size = dataset_params
            train_dataset, _, _ = process.split_data_own(dataset, data_path, aug=aug, repeat=aug_times)

            train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                      shuffle=True, num_workers=0, pin_memory=True, )
            not_aug_yet = False
            print('Having Augument Tc > 10, the training length now is:', len(train_dataset))

        ##Train model
        train_error = train(model, optimizer, train_loader, loss, rank=rank, pt_model=pt_model)
        val_error = evaluate(val_loader, model, loss, rank=rank, out=False)

        ##Train loop timings
        epoch_time = time.time() - train_start
        train_start = time.time()

        ##remember the best val error and save model and checkpoint
        if val_error == float("NaN") or val_error < best_val_error:
            model_best = copy.deepcopy(model)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "full_model": model,
                },
                filename,
            )
        best_val_error = min(val_error, best_val_error)
    
        ##scheduler on train error
        scheduler.step(train_error)

        ##Print performance
        if epoch % verbosity == 0:
            if rank in (0, "cpu", "cuda"):
                print(
                    "Epoch: {:04d}, Learning Rate: {:.6f}, Training Error: {:.5f}, Val Error: {:.5f}, Time per epoch (s): {:.5f}".format(
                        epoch, lr, train_error, val_error, epoch_time
                    )
                )

    return model_best


##Write results to csv file
def write_results(output, filename):
    shape = output.shape
    with open(filename, "w") as f:
        csvwriter = csv.writer(f)
        for i in range(0, len(output)):
            if i == 0:
                csvwriter.writerow(
                    ["ids"]
                    + ["target"] * int((shape[1] - 1) / 2)
                    + ["prediction"] * int((shape[1] - 1) / 2)
                )
            elif i > 0:
                csvwriter.writerow(output[i - 1, :])


## 设定模型
def model_setup(
        rank,
        model_name,
        model_params,
        dataset,
        load_model=False,
        model_path=None,
        print_model=True,
):
    model = getattr(models, model_name)(
        data=dataset, **(model_params if model_params is not None else {})
    ).to(rank)
    if load_model == "True":
        assert os.path.exists(model_path), "Saved model not found"
        if str(rank) in ("cpu"):
            saved = torch.load(model_path, map_location=torch.device("cpu"))
        else:
            saved = torch.load(model_path)
        model.load_state_dict(saved["model_state_dict"])
        # optimizer.load_state_dict(saved['optimizer_state_dict'])

    if print_model == True and rank in (0, "cpu", "cuda"):
        model_summary(model)
    return model


##Pytorch loader setup
def loader_setup(
        data_path,
        train_ratio,
        val_ratio,
        test_ratio,
        batch_size,
        dataset,
        rank,
        find_disorder=False,
        world_size=0,
        num_workers=0,
):
    # 获取数据集
    train_dataset, val_dataset, test_dataset = process.split_data_own(dataset, data_path, find_disorder=find_disorder)
    train_sampler = None

    ##Load data
    train_loader = val_loader = test_loader = None
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # (train_sampler is None),
        num_workers=num_workers,
        pin_memory=True,
        # sampler=train_sampler,
    )
    # may scale down batch size if memory is an issue
    # 导入验证集与测试集
    if len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
    if len(test_dataset) > 0:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

    return (
        train_loader,
        val_loader,
        test_loader,
        train_sampler,
        train_dataset,
        val_dataset,
        test_dataset,
    )


################################################################################
#  Trainers
################################################################################

###Regular training with train, val, test split
def train_regular(
        rank,
        world_size,
        data_path,
        job_parameters=None,
        training_parameters=None,
        model_parameters=None,
        pt_model=None,
        seed=None,
):
    ##DDP
    # ddp_setup(rank, world_size)
    ##some issues with DDP learning rate
    if rank not in ("cpu", "cuda"):
        model_parameters["lr"] = model_parameters["lr"] * world_size

    ##Get dataset
    dataset = process.get_dataset(data_path, training_parameters["target_index"], False)

    ### Two Stage Training ###
    if training_parameters['aug'] == 'True':
        aug_params = [training_parameters['aug'], training_parameters['aug_times'], training_parameters['aug_stage'],
                      [dataset, data_path, model_parameters["batch_size"]]]
    else:
        aug_params = None

    if rank not in ("cpu", "cuda"):
        dist.barrier()

    ##Set up loader
    (
        train_loader,
        val_loader,
        test_loader,
        train_sampler,
        train_dataset,
        _,
        _,
    ) = loader_setup(
        data_path,
        training_parameters["train_ratio"],
        training_parameters["val_ratio"],
        training_parameters["test_ratio"],
        model_parameters["batch_size"],
        dataset,
        rank,
    )

    ##Set up model
    model = model_setup(
        rank,
        model_parameters["model"],
        model_parameters,
        dataset,
        job_parameters["load_model"],
        job_parameters["model_path"],
        model_parameters.get("print_model", True),
    )

    ##Set-up optimizer & scheduler
    optimizer = getattr(torch.optim, model_parameters["optimizer"])(
        model.parameters(),
        lr=model_parameters["lr"],
        **model_parameters["optimizer_args"]
    )
    scheduler = getattr(torch.optim.lr_scheduler, model_parameters["scheduler"])(
        optimizer, **model_parameters["scheduler_args"]
    )

    ##Start training
    model = trainer(
        rank,
        world_size,
        model,
        optimizer,
        scheduler,
        training_parameters["loss"],
        train_loader,
        val_loader,
        train_sampler,
        model_parameters["epochs"],
        training_parameters["verbosity"],
        "my_model_temp.pth",
        pt_model=pt_model,
        aug_params=aug_params
    )

    if rank in (0, "cpu", "cuda"):

        train_error = val_error = test_error = float("NaN")

        ##workaround to get training output in DDP mode
        ##outputs are slightly different, could be due to dropout or batchnorm?
        train_loader = DataLoader(
            train_dataset,
            batch_size=model_parameters["batch_size"],
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )

        ##Get train error in eval mode
        train_error, train_out, train_metrics = evaluate(
            train_loader, model, training_parameters["loss"], rank, out=True
        )

        ##Get val error
        if val_loader != None:
            val_error, val_out, val_metrics = evaluate(
                val_loader, model, training_parameters["loss"], rank, out=True
            )

        ##Get test error
        if test_loader != None:
            test_error, test_out, test_metrics = evaluate(
                test_loader, model, training_parameters["loss"], rank, out=True
            )

        ##Save model
        if job_parameters["save_model"] == "True":
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "full_model": model,
                },
                job_parameters["model_path"],
            )

        ##Write outputs
        if job_parameters["write_output"] == "True":

            write_results(
                train_out, str(job_parameters["seed"]) + "_train_outputs.csv"
            )
            if val_loader != None:
                write_results(
                    val_out, str(job_parameters["seed"]) + "_val_outputs.csv"
                )
            if test_loader != None:
                write_results(
                    test_out, str(job_parameters["seed"]) + "_test_outputs.csv"
                )

        if rank not in ("cpu", "cuda"):
            dist.destroy_process_group()

        ##Write out model performance to file
        error_values = np.array(
            [train_error.cpu(), val_error.cpu(), test_error.cpu()] + train_metrics + val_metrics + test_metrics)
        if job_parameters.get("write_error") == "True":
            np.savetxt(
                job_parameters["job_name"] + "_errorvalues.csv",
                error_values[np.newaxis, ...],
                delimiter=",",
            )

        return error_values


###Predict using a saved movel
def predict(dataset, loss, job_parameters=None):
    rank = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ##Loads predict dataset in one go, care needed for large datasets)
    loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    ##Load saved model
    assert os.path.exists(job_parameters["model_path"]), "Saved model not found"
    if str(rank) == "cpu":
        saved = torch.load(
            job_parameters["model_path"], map_location=torch.device("cpu")
        )
    else:
        saved = torch.load(
            job_parameters["model_path"], map_location=torch.device("cuda")
        )
    model = saved["full_model"]
    model = model.to(rank)
    model_summary(model)

    ##Get predictions
    time_start = time.time()
    test_error, test_out, test_metrics = evaluate(loader, model, loss, rank, out=True)
    elapsed_time = time.time() - time_start

    print("Evaluation time (s): {:.5f}".format(elapsed_time))

    ##Write output
    if job_parameters["write_output"] == "True":
        write_results(
            test_out, str(job_parameters["job_name"]) + "_predicted_outputs.csv"
        )

    return test_error


### Repeat training for n times
def train_repeat(
        data_path,
        job_parameters=None,
        training_parameters=None,
        model_parameters=None,

):
    world_size = torch.cuda.device_count()
    assert world_size <= 1, "This Version doesn't support MultiGPU Training. "

    job_name = job_parameters["job_name"]
    model_path = job_parameters["model_path"]
    job_parameters["write_error"] = "True"

    ### 设定预训练模型 ###
    if model_parameters['pt'] == 'True':
        print("using pretrain model")
        checkpoint_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                            "../models/pretrain_model.pth.tar")
        checkpoint = torch.load(checkpoint_file_path)

        pt_model = nn.Linear(92, 64, bias=False)
        pt_model.load_state_dict({'weight': checkpoint['state_dict']['embedding.weight']})
        pt_model = pt_model.cuda()
    else:
        pt_model = None

    ### 重复试验
    for i in range(0, job_parameters["repeat_trials"]):

        ##new seed each time for different data split
        job_parameters["seed"] = i+1
        seed_everything(job_parameters["seed"])

        if i == 0:
            model_parameters["print_model"] = True
        else:
            model_parameters["print_model"] = False

        job_parameters["job_name"] = job_name + str(i)
        job_parameters["model_path"] = str(i+1) + "_" + model_path

        if world_size == 0:
            print("Running on CPU - this will be slow")
            training.train_regular(
                "cpu",
                world_size,
                data_path,
                job_parameters,
                training_parameters,
                model_parameters,
                pt_model,
                seed=job_parameters["seed"]
            )
        else: ## 现版本仅支持单卡训练
            print("Running on one GPU")
            training.train_regular(
                "cuda",
                world_size,
                data_path,
                job_parameters,
                training_parameters,
                model_parameters,
                pt_model,
                seed=job_parameters["seed"]
            )
            
    ##Compile error metrics from individual trials
    print("Individual training finished.")
    print("Compiling metrics from individual trials...")
    error_values = np.zeros((job_parameters["repeat_trials"], 9))
    for i in range(0, job_parameters["repeat_trials"]):
        filename = job_name + str(i) + "_errorvalues.csv"
        error_values[i] = np.genfromtxt(filename, delimiter=",")
    mean_values = np.mean(error_values, axis=0)
    std_values = np.std(error_values, axis=0)

    ##Print error
    print(
        "Training Error Avg: {:.3f}, Training Standard Dev: {:.3f}".format(
            mean_values[0], std_values[0]
        )
    )
    print(
        "Val Error Avg: {:.3f}, Val Standard Dev: {:.3f}".format(
            mean_values[1], std_values[1]
        )
    )
    print(
        "Test Error Avg: {:.3f}, Test Standard Dev: {:.3f}".format(
            mean_values[2], std_values[2]
        )
    )
    print('\n r2:', mean_values[3:])
    print(std_values[3:])

    ##Write error metrics
    if job_parameters["write_output"] == "True":
        with open(job_name + "_all_errorvalues.csv", "w") as f:
            csvwriter = csv.writer(f)
            csvwriter.writerow(
                [
                    "",
                    "Training",
                    "Validation",
                    "Test",
                ]
            )
            for i in range(0, len(error_values)):
                csvwriter.writerow(
                    ["Trial " + str(i)] + error_values[i].tolist()
                )
            csvwriter.writerow(["Mean"] + mean_values.tolist())
            csvwriter.writerow(["Std"] + std_values.tolist())
    elif job_parameters["write_output"] == "False":
        for i in range(0, job_parameters["repeat_trials"]):
            filename = job_name + str(i) + "_errorvalues.csv"
            os.remove(filename)
