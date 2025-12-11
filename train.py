import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import torch, numpy as np, glob, torch.utils.data, scipy.ndimage, multiprocessing as mp
import torch.nn.functional as F
import time
import math
import datetime
import logging
import importlib
from tqdm import tqdm
from pathlib import Path
from tools.euler_tools import quat2mat
from dataset.kitti_dataset_voxel import OdometryDataset
import cmd_args
import random

ranseed = 6
np.random.seed(ranseed)
random.seed(ranseed)
torch.manual_seed(ranseed)
torch.cuda.manual_seed(ranseed)
torch.cuda.manual_seed_all(ranseed)

if 'NUMBA_DISABLE_JIT' in os.environ:
    del os.environ['NUMBA_DISABLE_JIT']

global args
args = cmd_args.parse_args_from_yaml(sys.argv[1])
if args.multi_gpu is None:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

'''CREATE DIR'''
experiment_dir = Path('./experiment/')  # output_path
experiment_dir.mkdir(exist_ok=True)
file_dir = Path(str(experiment_dir) + '/%s-' % (args.dataset) + str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')))
file_dir.mkdir(exist_ok=True)
checkpoints_dir = file_dir.joinpath('checkpoints/')
checkpoints_dir.mkdir(exist_ok=True)
eval_dir = file_dir.joinpath('eval/')
eval_dir.mkdir(exist_ok=True)
log_dir = file_dir.joinpath('logs/')
log_dir.mkdir(exist_ok=True)

os.system('cp %s %s' % (args.model_name + '.py', log_dir))
os.system('cp %s %s' % ('model_util.py', log_dir))
os.system('cp %s %s' % ('kitti_dataset_voxel.py', log_dir))
os.system('cp %s %s' % ('config.yaml', log_dir))
os.system('cp %s %s' % ('train.py', log_dir))


def main():
    '''LOG'''
    logger = logging.getLogger(args.model_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(str(log_dir) + '/log_%s.txt' % args.model_name)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info('---------------------------TRAINING----------------------------------')
    logger.info('PARAMETER ...')
    logger.info(args)

    module = importlib.import_module(args.model_name)
    model = getattr(module, 'RCP_LO')()

    train_dir_list = [0, 1, 2, 3, 4, 5, 6]
    test_dir_list = [7, 8, 9, 10]

    if torch.cuda.is_available():
        num_devices = torch.cuda.device_count()
        print("Obtain the number of available CUDA devices:", num_devices)
        if num_devices > 1:
            torch.backends.cudnn.benchmark = True
            model = torch.nn.DataParallel(model)
            model.cuda()
        else:
            model.cuda()
    else:
        raise EnvironmentError("CUDA is not available.")

    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08,
                                     weight_decay=args.weight_decay)
    elif args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08,
                                      weight_decay=args.weight_decay)

    optimizer.param_groups[0]['initial_lr'] = args.learning_rate
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=9, gamma=0.8, last_epoch=0)
    LEARNING_RATE_CLIP = 1e-5

    if args.pretrain is not None:

        checkpoint = torch.load(args.pretrain)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['opt_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        init_epoch = checkpoint['epoch']
        print('load model %s' % args.pretrain)
        logger.info('load model %s' % args.pretrain)

    else:
        init_epoch = 0
        print('Training from scratch')
        logger.info('Training from scratch')

    if args.eval_before == 1:
        eval_pose(model, test_dir_list, init_epoch)

    train_dataset = OdometryDataset(is_training=1, num_points=args.num_points, data_root=args.data_root,
        data_dir_list=train_dir_list, )
    logger.info('train_dataset: ' + str(train_dataset))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True,
        worker_init_fn=lambda x: np.random.seed((torch.initial_seed()) % (2 ** 32)))

    for epoch in range(init_epoch, args.epochs):

        lr = max(optimizer.param_groups[0]['lr'], LEARNING_RATE_CLIP)
        print('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        total_loss = 0.0
        total_seen = 0
        optimizer.zero_grad()

        for i, data in tqdm(enumerate(train_loader, 0), total=len(train_loader), smoothing=0.9, ncols=100,
                            mininterval=60.0):
            pos1, norm1, pos2, norm2, T_gt, qq_gt, t_gt, Tr = data
            pos1 = pos1.float().cuda()
            pos2 = pos2.float().cuda()
            norm1 = pos1
            norm2 = pos2
            qq_gt = qq_gt.float().cuda()
            t_gt = t_gt.float().cuda()

            model = model.train()
            _, loss = model(pos1, pos2, norm1, norm2, qq_gt, t_gt)

            loss = loss.mean()  # 多卡训练

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.detach().item() * args.batch_size
            total_seen += args.batch_size

        scheduler.step()

        train_loss = total_loss / total_seen
        str_out = 'EPOCH {} train mean loss: {:04f}'.format(epoch, float(train_loss))
        print(str_out)
        logger.info(str_out)

        model_name = '%s_%.3d%.4f.pth' % (args.model_name, epoch, train_loss)
        save_path = os.path.join(checkpoints_dir, model_name)
        torch.save(
            {'model_state_dict': model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict(),
                'opt_state_dict': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'epoch': epoch},
            save_path)

        print('Save model ...')
        logger.info('Save model ...')

        if epoch % 4 == 0:
            eval_pose(model, test_dir_list, epoch)


def eval_pose(model, test_list, epoch):
    for item in test_list:

        test_dataset = OdometryDataset(is_training=0, num_points=args.num_points, data_root=args.val_path,
            data_dir_list=[item], )
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.val_batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True,
            worker_init_fn=lambda x: np.random.seed((torch.initial_seed()) % (2 ** 32)))

        line = 0
        total_time = 0

        for batch_id, data in tqdm(enumerate(test_loader), total=len(test_loader), smoothing=0.9):

            pos1, norm1, pos2, norm2, T_gt, qq_gt, t_gt, Tr = data

            pos1 = pos1.float().cuda()
            pos2 = pos2.float().cuda()
            norm1 = pos1
            norm2 = pos2
            qq_gt = qq_gt.float().cuda()
            t_gt = t_gt.float().cuda()
            model = model.eval()

            with torch.no_grad():

                start_time = time.time()
                pred_T = model(pos1, pos2, norm1, norm2, qq_gt, t_gt)
                total_time += (time.time() - start_time)
                pred_T = pred_T.cpu().numpy()
                for n0 in range(pred_T.shape[0]):

                    cur_Tr = Tr[n0, :, :]
                    TT = pred_T[n0:n0 + 1, :].reshape(4, 4)
                    TT = np.matmul(cur_Tr, TT)
                    TT = np.matmul(TT, np.linalg.inv(cur_Tr))

                    if line == 0:
                        T_final = TT
                        T = T_final[:3, :]
                        T = T.reshape(1, 1, 12)
                        line += 1
                    else:
                        T_final = np.matmul(T_final, TT)
                        T_current = T_final[:3, :]
                        T_current = T_current.reshape(1, 1, 12)
                        T = np.append(T, T_current, axis=0)

        T = T.reshape(-1, 12)
        print(T.shape)

        fname_txt = os.path.join(eval_dir, str(item).zfill(2) + '_pred.npy')
        fname_txt2 = os.path.join(eval_dir, str(item).zfill(2) + '_pred.txt')

        data_dir = os.path.join(eval_dir, 'DLO_' + str(item).zfill(2))
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        np.save(fname_txt, T)
        np.savetxt(fname_txt2, T)
        os.system('cp %s %s' % (fname_txt, data_dir))
        os.system('python eval.py --result_dir ' + data_dir + ' --eva_seqs ' + str(item).zfill(
            2) + '_pred' + ' --epoch ' + str(epoch))
        print("This seq is over!\n")
    return 0


if __name__ == '__main__':
    main()
