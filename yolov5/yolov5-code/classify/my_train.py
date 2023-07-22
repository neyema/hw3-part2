# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Train a YOLOv5 classifier model on a classification dataset

Usage - Single-GPU training:
    $ python classify/train.py --model yolov5s-cls.pt --data imagenette160 --epochs 5 --img 224

Usage - Multi-GPU DDP training:
    $ python -m torch.distributed.run --nproc_per_node 4 --master_port 2022 classify/train.py
    --model yolov5s-cls.pt --data imagenet --epochs 5 --img 224 --device 0,1,2,3

Datasets:           --data mnist, fashion-mnist, cifar10, cifar100, imagenette, imagewoof,
imagenet, or 'path/to/data'
YOLOv5-cls models:  --model yolov5n-cls.pt, yolov5s-cls.pt, yolov5m-cls.pt, yolov5l-cls.pt,
yolov5x-cls.pt
Torchvision models: --model resnet50, efficientnet_b0, etc. See
https://pytorch.org/vision/stable/models.html
"""

import argparse
import os
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
import torch.hub as hub
import torch.optim.lr_scheduler as lr_scheduler
import torchvision
from torch.cuda import amp
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from classify import val as validate
from models.experimental import attempt_load
from models.yolo import ClassificationModel, DetectionModel
from utils.dataloaders import create_classification_dataloader
from utils.general import (DATASETS_DIR, LOGGER, TQDM_BAR_FORMAT, WorkingDirectory, check_git_info,
                           check_git_status,
                           check_requirements, colorstr, download, increment_path, init_seeds,
                           print_args, yaml_save)
from utils.loggers import GenericLogger
from utils.plots import imshow_cls
from utils.torch_utils import (ModelEMA, de_parallel, model_info, reshape_classifier_output,
                               select_device, smart_DDP,
                               smart_optimizer, smartCrossEntropyLoss, torch_distributed_zero_first)

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))
GIT_INFO = check_git_info()


def train(opt, device):
    init_seeds(opt.seed + 1 + RANK, deterministic=True)
    save_dir, data, bs, epochs, nw, imgsz, pretrained = \
        opt.save_dir, Path(opt.data), opt.batch_size, opt.epochs, min(os.cpu_count() - 1,
                                                                      opt.workers), \
            opt.imgsz, str(opt.pretrained).lower() == 'true'
    cuda = device.type != 'cpu'

    # Directories
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)  # make dir
    last, best = wdir / 'last.pt', wdir / 'best.pt'

    # Save run settings
    yaml_save(save_dir / 'opt.yaml', vars(opt))

    # Logger
    logger = GenericLogger(opt=opt, console_logger=LOGGER) if RANK in {-1, 0} else None

    # Download Dataset
    with torch_distributed_zero_first(LOCAL_RANK), WorkingDirectory(ROOT):
        data_dir = data if data.is_dir() else (DATASETS_DIR / data)
        if not data_dir.is_dir():
            LOGGER.info(f'\nDataset not found ⚠️, missing path {data_dir}, attempting download...')
            t = time.time()
            if str(data) == 'imagenet':
                subprocess.run(['bash', str(ROOT / 'data/scripts/get_imagenet.sh')], shell=True,
                               check=True)
            else:
                url = f'https://github.com/ultralytics/yolov5/releases/download/v1.0/{data}.zip'
                download(url, dir=data_dir.parent)
            s = f"Dataset download success ✅ ({time.time() - t:.1f}s), saved to " \
                f"{colorstr('bold', data_dir)}\n"
            LOGGER.info(s)

    # Dataloaders
    nc = len([x for x in (data_dir / 'train').glob('*') if x.is_dir()])  # number of classes
    train_loader = create_classification_dataloader(path=data_dir / 'train', imgsz=imgsz,
                                                    batch_size=bs // WORLD_SIZE, augment=True,
                                                    cache=opt.cache, rank=LOCAL_RANK, workers=nw)

    if RANK in {-1, 0}:
        val_loader = create_classification_dataloader(path=data_dir / 'val', imgsz=imgsz,
                                                      augment=False, cache=opt.cache, workers=nw,
                                                      batch_size=bs // WORLD_SIZE * 2, rank=-1)
        test_loader = create_classification_dataloader(path=data_dir / 'test', imgsz=imgsz,
                                                       augment=False, cache=opt.cache, workers=nw,
                                                       batch_size=bs // WORLD_SIZE * 2, rank=-1)

    # Model
    with torch_distributed_zero_first(LOCAL_RANK), WorkingDirectory(ROOT):
        if Path(opt.model).is_file() or opt.model.endswith('.pt'):
            model = attempt_load(opt.model, device='cpu', fuse=False)
        elif opt.model in torchvision.models.__dict__:  # TorchVision models i.e. resnet50,
            # efficientnet_b0
            model = torchvision.models.__dict__[opt.model](
                weights='IMAGENET1K_V1' if pretrained else None)
        else:
            m = hub.list('ultralytics/yolov5')  # + hub.list('pytorch/vision')  # models
            raise ModuleNotFoundError(
                f'--model {opt.model} not found. Available models are: \n' + '\n'.join(m))
        if isinstance(model, DetectionModel):
            LOGGER.warning(
                "WARNING ⚠️ pass YOLOv5 classifier model with '-cls' suffix, i.e. '--model "
                "yolov5s-cls.pt'")
            model = ClassificationModel(model=model, nc=nc,
                                        cutoff=opt.cutoff or 10)  # convert to classification model
        reshape_classifier_output(model, nc)  # update class count
    for m in model.modules():
        if not pretrained and hasattr(m, 'reset_parameters'):
            m.reset_parameters()
        if isinstance(m, torch.nn.Dropout) and opt.dropout is not None:
            m.p = opt.dropout  # set dropout

    for k, v in model.named_parameters():
        v.requires_grad = False  # freeze for training
    model.model[9].linear = torch.nn.Linear(1280, 102, bias=True)
    model = model.to(device)

    # Info
    if RANK in {-1, 0}:
        model.names = train_loader.dataset.classes  # attach class names
        model.transforms = test_loader.dataset.torch_transforms  # attach inference transforms
        model_info(model)
        if opt.verbose:
            LOGGER.info(model)
        images, labels = next(iter(train_loader))
        file = imshow_cls(images[:25], labels[:25], names=model.names,
                          f=save_dir / 'train_images.jpg')
        logger.log_images(file, name='Train Examples')
        logger.log_graph(model, imgsz)  # log model

    # Optimizer
    optimizer = smart_optimizer(model, opt.optimizer, opt.lr0, momentum=0.9, decay=opt.decay)

    # Scheduler
    lrf = 0.01  # final lr (fraction of lr0)
    # lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - lrf) + lrf  # cosine
    lf = lambda x: (1 - x / epochs) * (1 - lrf) + lrf  # linear
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    # scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=lr0, total_steps=epochs, pct_start=0.1,
    #                                    final_div_factor=1 / 25 / lrf)

    # EMA
    ema = ModelEMA(model) if RANK in {-1, 0} else None

    # DDP mode
    if cuda and RANK != -1:
        model = smart_DDP(model)

    # Train
    t0 = time.time()
    criterion = smartCrossEntropyLoss(label_smoothing=opt.label_smoothing)  # loss function
    best_fitness = 0.0
    scaler = amp.GradScaler(enabled=cuda)
    LOGGER.info(f'Image sizes {imgsz} train, {imgsz} test\n'
                f'Using {nw * WORLD_SIZE} dataloader workers\n'
                f"Logging results to {colorstr('bold', save_dir)}\n"
                f'Starting {opt.model} training on {data} dataset with {nc} classes for {epochs} '
                f'epochs...\n\n'
                f"{'Epoch':>10}{'GPU_mem':>10}{'train_loss':>12}{f'val_loss':>12}"
                f"{f'train_loss':>12}{'val_loss':>12}{f'test_loss':>12}"
                f"{f'train_acc':>12}{'val_acc':>12}{f'test_acc':>12}")

    for epoch in range(epochs):  # loop over the dataset multiple times
        tloss, val_loss, fitness = 0.0, 0.0, 0.0  # train loss, val loss, fitness

        model.train()
        if RANK != -1:
            train_loader.sampler.set_epoch(epoch)
        pbar = enumerate(train_loader)
        if RANK in {-1, 0}:
            pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                        bar_format=TQDM_BAR_FORMAT)
        for i, (images, labels) in pbar:  # progress bar
            images, labels = images.to(device, non_blocking=True), labels.to(device)

            # Forward
            with amp.autocast(enabled=cuda):  # stability issues when enabled
                loss = criterion(model(images), labels)

            # Backward
            scaler.scale(loss).backward()

            # Optimize
            scaler.unscale_(optimizer)  # unscale gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # clip gradients
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if ema:
                ema.update(model)

            if RANK in {-1, 0}:
                # Print
                tloss = (tloss * i + loss.item()) / (i + 1)  # update mean losses
                mem = '%.3gG' % (
                    torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                pbar.desc = f"{f'{epoch + 1}/{epochs}':>10}{mem:>10}{tloss:>12.3g}" + ' ' * 36

                if i == len(pbar) - 1:  # last batch
                    # eval on train
                    train_top1, _, train_loss = validate.run(model=ema.ema,
                                                             dataloader=train_loader,
                                                             criterion=criterion,
                                                             pbar=pbar)
                    # eval on validation
                    val_top1, val_top5, val_loss = validate.run(model=ema.ema,
                                                                dataloader=val_loader,
                                                                criterion=criterion,
                                                                pbar=pbar)
                    fitness = val_top1  # define fitness as top1 accuracy
                    # eval on test
                    test_top1, _, test_loss = validate.run(model=ema.ema,
                                                           dataloader=test_loader,
                                                           criterion=criterion,
                                                           pbar=pbar)
        # Scheduler
        scheduler.step()

        # Log metrics
        if RANK in {-1, 0}:
            # Best fitness
            if fitness > best_fitness:
                best_fitness = fitness

            # Log
            metrics = {
                'train/loss': tloss,
                f'val/loss': val_loss,
                'metrics/train_loss': train_loss,
                'metrics/val_loss': val_loss,
                'metrics/test_loss': test_loss,
                'metrics/train_accuracy': train_top1,
                'metrics/val_accuracy': val_top1,
                'metrics/test_accuracy': test_top1,
                'lr/0': optimizer.param_groups[0]['lr']}  # learning rate
            logger.log_metrics(metrics, epoch)

            # Save model
            final_epoch = epoch + 1 == epochs
            if (not opt.nosave) or final_epoch:
                ckpt = {
                    'epoch': epoch,
                    'best_fitness': best_fitness,
                    'model': deepcopy(ema.ema).half(),  # deepcopy(de_parallel(model)).half(),
                    'ema': None,  # deepcopy(ema.ema).half(),
                    'updates': ema.updates,
                    'optimizer': None,  # optimizer.state_dict(),
                    'opt': vars(opt),
                    'git': GIT_INFO,  # {remote, branch, commit} if a git repo
                    'date': datetime.now().isoformat()}

                # Save last, best and delete
                torch.save(ckpt, last)
                if best_fitness == fitness:
                    torch.save(ckpt, best)
                del ckpt

    # Train complete
    if RANK in {-1, 0} and final_epoch:
        LOGGER.info(f'\nTraining complete ({(time.time() - t0) / 3600:.3f} hours)'
                    f"\nResults saved to {colorstr('bold', save_dir)}"
                    f'\nPredict:         python classify/predict.py --weights {best} --source '
                    f'im.jpg'
                    f'\nValidate:        python classify/val.py --weights {best} --data {data_dir}'
                    f'\nExport:          python export.py --weights {best} --include onnx'
                    f"\nPyTorch Hub:     model = torch.hub.load('ultralytics/yolov5', 'custom', "
                    f"'{best}')"
                    f'\nVisualize:       https://netron.app\n')

        # Plot examples
        images, labels = (x[:25] for x in next(iter(test_loader)))  # first 25 images and labels
        pred = torch.max(ema.ema(images.to(device)), 1)[1]
        file = imshow_cls(images, labels, pred, de_parallel(model).names, verbose=False,
                          f=save_dir / 'test_images.jpg')

        # Log results
        meta = {'epochs': epochs, 'top1_acc': best_fitness, 'date': datetime.now().isoformat()}
        logger.log_images(file, name='Test Examples (true-predicted)', epoch=epoch)
        logger.log_model(best, epochs, metadata=meta)


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='yolov5s-cls.pt', help='initial weights path')
    parser.add_argument('--data', type=str, default='imagenette160',
                        help='cifar10, cifar100, mnist, imagenet, ...')
    parser.add_argument('--epochs', type=int, default=10, help='total training epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='total batch size for all GPUs')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=224,
                        help='train, val image size (pixels)')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--cache', type=str, nargs='?', const='ram',
                        help='--cache images in "ram" (default) or "disk"')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8,
                        help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--project', default=ROOT / 'runs/train-cls', help='save to project/name')
    parser.add_argument('--name', default='exp-1', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true',
                        help='existing project/name ok, do not increment')
    parser.add_argument('--pretrained', nargs='?', const=True, default=True,
                        help='start from i.e. --pretrained False')
    parser.add_argument('--optimizer', choices=['SGD', 'Adam', 'AdamW', 'RMSProp'], default='Adam',
                        help='optimizer')
    parser.add_argument('--lr0', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--decay', type=float, default=5e-5, help='weight decay')
    parser.add_argument('--label-smoothing', type=float, default=0.1,
                        help='Label smoothing epsilon')
    parser.add_argument('--cutoff', type=int, default=None,
                        help='Model layer cutoff index for Classify() head')
    parser.add_argument('--dropout', type=float, default=None, help='Dropout (fraction)')
    parser.add_argument('--verbose', action='store_true', help='Verbose mode')
    parser.add_argument('--seed', type=int, default=0, help='Global training seed')
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='Automatic DDP Multi-GPU argument, do not modify')
    return parser.parse_known_args()[0] if known else parser.parse_args()


def main(opt):
    # Checks
    if RANK in {-1, 0}:
        print_args(vars(opt))
        check_git_status()
        check_requirements(ROOT / 'requirements.txt')

    # DDP mode
    device = select_device(opt.device, batch_size=opt.batch_size)
    if LOCAL_RANK != -1:
        assert opt.batch_size != -1, 'AutoBatch is coming soon for classification, please pass a ' \
                                     'valid --batch-size'
        assert opt.batch_size % WORLD_SIZE == 0, f'--batch-size {opt.batch_size} must be multiple' \
                                                 f' of WORLD_SIZE'
        assert torch.cuda.device_count() > LOCAL_RANK, 'insufficient CUDA devices for DDP command'
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device('cuda', LOCAL_RANK)
        dist.init_process_group(backend='nccl' if dist.is_nccl_available() else 'gloo')

    # Parameters
    opt.save_dir = increment_path(Path(opt.project) / opt.name,
                                  exist_ok=opt.exist_ok)  # increment run

    # Train
    train(opt, device)


def run(**kwargs):
    # Usage: from yolov5 import classify; classify.train.run(data=mnist, imgsz=320, model='yolov5m')
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)
    return opt


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
