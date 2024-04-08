from utils import parser
from utils.logger import *
from utils.config import *
import time
from models import build_model_from_cfg
import os
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from COSC_Dataset import COSC_Dataset
from COSC_loss import NBVLoss
from utils.metrics import Metrics
import torch.optim as optim
from tqdm import tqdm
from utils.utils import wblue
from tensorboardX import SummaryWriter
import numpy as np

def save_ckpt(base_model, optimizer, epoch, logger, ckpt_pth):
    torch.save({
                'base_model': base_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                }, 
                ckpt_pth)
    print_log(f"Save checkpoint at {ckpt_pth}", logger = logger)


def random_check_loader(config, logger):
    train_objs = []
    with open(config['dataset']['train_split'], 'r') as f:
        for line in f:
            train_objs.append(line.strip())

    # random select 4 obj from tran_objs
    import random
    random.shuffle(train_objs)
    check_objs = train_objs[:4]
    print_log(f'check_objs: {check_objs}', logger=logger)

    check_dataset = COSC_Dataset(root=config['dataset']['root_dir'], obj_list=check_objs, num_points=config.model.num_points, do_sample=config.do_sample, logger=logger)
    check_loader = torch.utils.data.DataLoader(check_dataset,
                                                batch_size = config['total_bs'],
                                                shuffle=True,
                                                num_workers=8,
                                                drop_last=True,
                                            )

    return check_loader

def build_dataset(config, logger):
    train_objs, test_objs = [], []
    with open(config['dataset']['train_split'], 'r') as f:
        for line in f:
            train_objs.append(line.strip())

    with open(config['dataset']['test_split'], 'r') as f:
        for line in f:
            test_objs.append(line.strip())

    train_dataset = COSC_Dataset(root=config['dataset']['root_dir'], obj_list=train_objs, num_points=config.model.num_points, do_sample=config.do_sample, logger=logger)

    dataset_size = len(train_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(0.8 * dataset_size))
    np.random.seed(42)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[:split], indices[split:]

    print_log(f'train_dataset_size: {dataset_size}, train set size: {len(train_indices)}, test set size: {len(val_indices)}', logger=logger)    
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
   
    train_loader = DataLoader(train_dataset, batch_size=config['total_bs'],shuffle=True)
    test_loader = DataLoader(train_dataset, batch_size=config['total_bs'], sampler=valid_sampler)


    # test_dataset = COSC_Dataset(root=config['dataset']['root_dir'], obj_list=test_objs, num_points=config.model.num_points)
    # train_loader = torch.utils.data.DataLoader(train_dataset, 
    #                                             batch_size = config['total_bs'],
    #                                             shuffle=True,
    #                                             num_workers=8,
    #                                             drop_last=True,
                                                
    #                                         )
    # test_loader = torch.utils.data.DataLoader(test_dataset, 
    #                                             batch_size = config['total_bs'],
    #                                             num_workers=8,
    #                                         )
    return train_loader, test_loader


def check_accuracy(model, test_loader, threshold_gamma=0.5, device=torch.device('cuda')): 
    print('EVALUATING')
    recall, precision = 0.0, 0.0
    label_thresh = threshold_gamma
    test_case_num = len(test_loader.dataset)
    with torch.no_grad():
        t = tqdm(
            test_loader, 
            leave=True
            )
        for sample in t:
            grid = sample[0].to(device)
            vs = sample[1].to(device)
            label = sample[2].to(device)

            output = model(grid, vs)

            output = torch.where(output >= label_thresh, torch.tensor(1).to(device), torch.tensor(0).to(device))
            correct1 = torch.sum(torch.logical_and(output == 1, label == 1), dim=1)
            wrong1 = torch.sum(torch.logical_and(output == 1, label == 0), dim=1)
            cnt1 = torch.sum(label == 1, dim=1)
            recall += torch.sum(correct1 / cnt1).item()
            precision += torch.sum(correct1 / (correct1 + wrong1 + 1e-6)).item()
        
        print(f'recall: {recall}, precision: {precision}, case num: {test_case_num}')

        recall /= test_case_num
        precision /= test_case_num
        print(f'test recall:{recall}, precision:{precision}')
        model.train()
        return recall, precision


def run_net_on_cosc(args, config, train_writer=None, val_writer=None):

    base_model = build_model_from_cfg(config.model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_model = base_model.to(device)
    logger = get_logger(args.log_name)
    # train_loader, test_loader = build_dataset(config, logger=logger)

    # exit()
    start_epoch = 0

    # print model info
    print_log(args.description, logger=logger)
    print_log('Trainable_parameters:', logger = logger)
    print_log('=' * 25, logger = logger)
    for name, param in base_model.named_parameters():
        if param.requires_grad:
            print_log(name, logger=logger)
    print_log('=' * 25, logger = logger)
    
    print_log('Untrainable_parameters:', logger = logger)
    print_log('=' * 25, logger = logger)
    for name, param in base_model.named_parameters():
        if not param.requires_grad:
            print_log(name, logger=logger)
    print_log('=' * 25, logger = logger)

    # Criterion
    criterion = NBVLoss(lambda_for_1=config.lambda_for_1, device=device)

    # optimizer & scheduler
    optimizer = optim.AdamW(base_model.parameters(), lr=0.0001, weight_decay=0.005)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    scaler = torch.cuda.amp.GradScaler()

    if args.resume:
        ckpt = torch.load(os.path.join(args.experiment_path, './best_ckpt.pth'),  map_location=device)
        base_model.load_state_dict(ckpt)
        print_log('load ckpt from ./best_ckpt.pth', logger=logger)

    # training
    for epoch in range(start_epoch, config.max_epoch + 1):
        train_loader, test_loader = build_dataset(config, logger=logger)
        epoch_start_time = time.time()
        losses = []
        best_f1_score = 0.0
        
        t = tqdm(
            train_loader, 
            leave=True
            )

        base_model.train()  # set model to training mode
        for data in t:
            partial, vs, gt = data
            partial = partial.to(device)
            vs = vs.to(device)
            gt = gt.to(device)
        
            pred = base_model(partial, vs)
            loss = criterion(pred, gt)

            losses.append(loss)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            description = f"Epoch {epoch}: loss {loss.item()}"
            t.set_description_str(wblue(description))

        scheduler.step()
        epoch_end_time = time.time()

        train_writer.add_scalar('Loss/Epoch/NBVLoss', sum(losses) / len(losses), epoch)
        print_log(f'[Training] EPOCH:{epoch}, EPOCHTIME:{epoch_end_time-epoch_start_time}, LOSS:{sum(losses) / len(losses)} ', logger = logger)


        check_loader = random_check_loader(config, logger)
        recall_on_train, precision_on_train = check_accuracy(base_model, check_loader, threshold_gamma=config.threshold_gamma, device=device)
        print_log(f'Training epoch: {epoch} with recall:{recall_on_train}, precision:{precision_on_train}', logger=logger)
        f1_score_on_train = 2 * (precision_on_train * recall_on_train) / (precision_on_train + recall_on_train + 1e-6)
        if best_f1_score < f1_score_on_train:
            best_f1_score = f1_score_on_train
            print_log(f"saving better model on epoch: {epoch} with recall={recall_on_train} precision={precision_on_train} and f1={best_f1_score}")
            save_ckpt(base_model, optimizer, epoch, logger, os.path.join(args.experiment_path, 'best_ckpt.pth'))

        if train_writer is not None:
            train_writer.add_scalar('recall_on_train', recall_on_train, epoch)
            train_writer.add_scalar('precision_on_train', precision_on_train, epoch)
            train_writer.add_scalar('f1_score_on_train', f1_score_on_train, epoch)

        if epoch % 2 == 0:
            # Validate the current model
            recall, precision = check_accuracy(base_model, test_loader, threshold_gamma=config.threshold_gamma, device=device)
            f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)
            print_log(f"test model on validation, epoch: {epoch} with recall={recall} precision={precision} and f1={f1_score}", logger=logger)
            val_writer.add_scalar('f1_score', f1_score, epoch)          
            val_writer.add_scalar('recall', recall, epoch)
            val_writer.add_scalar('precision', precision, epoch)

        if (config.max_epoch - epoch) < 2:
            save_ckpt(base_model, optimizer, epoch, logger, os.path.join(args.experiment_path, f'ckpt-epoch-{epoch:03d}.pth'))
    # done training

    print_log(f'best result: {best_f1_score}', logger=logger)

    if train_writer is not None and val_writer is not None:
        train_writer.close()
        val_writer.close()


def main():
    args = parser.get_args()

    args.use_gpu = torch.cuda.is_available()
    if args.use_gpu:
        torch.backends.cudnn.benchmark = True

    # logger
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(args.experiment_path, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, name=args.log_name)

    # config
    config = get_config(args, logger = logger)
    train_writer = SummaryWriter(os.path.join(args.tfboard_path, 'train'))
    val_writer = SummaryWriter(os.path.join(args.tfboard_path, 'test'))

    run_net_on_cosc(args, config, train_writer, val_writer)

if __name__ == "__main__":
    main()
