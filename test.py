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
from dataset.kitti_dataset_voxel  import OdometryDataset 
import cmd_args 
import random


ranseed =6 
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
experiment_dir = Path('./experiment/')   #output_path
experiment_dir.mkdir(exist_ok=True)
file_dir = Path(str(experiment_dir) + '/%s-'%(args.dataset) + str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')))
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
os.system('cp %s %s' % ('config_train.yaml', log_dir))
os.system('cp %s %s' % ('train.py', log_dir))


def main():

    '''LOG'''
    logger = logging.getLogger(args.model_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(str(log_dir) + '/log_%s.txt'%args.model_name)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info('---------------------------TRAINING----------------------------------')
    logger.info('PARAMETER ...')
    logger.info(args)

    module = importlib.import_module(args.model_name)
    model = getattr(module, 'RCP_LO')()

    def load_and_eval(model, ckpt_path, test_dir_list, logger):
        print(f'\n Loading checkpoint: {ckpt_path}')
        logger.info(f'Loading checkpoint: {ckpt_path}')

        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.cuda()
        epoch = checkpoint['epoch']

        print(f"Loaded model epoch: {epoch}")
        logger.info(f"Loaded model epoch: {epoch}")
        eval_pose1(model, test_dir_list, epoch)

    if args.pretrain is not None:

        if os.path.isfile(args.pretrain):
            load_and_eval(model, args.pretrain, test_dir_list=[7,8,9,10], logger=logger)
            init_epoch = ['epoch']

        elif os.path.isdir(args.pretrain):
            print(f"Pretrain is a directory. Testing all checkpoints in: {args.pretrain}")
            logger.info(f"Pretrain is a directory. Testing all checkpoints in: {args.pretrain}")
            ckpt_files = sorted(
                [os.path.join(args.pretrain, f)
                for f in os.listdir(args.pretrain)
                if f.endswith(('.pth', '.pt'))],
                reverse=True
            )

            if len(ckpt_files) == 0:
                raise ValueError(f"No checkpoint files found in directory {args.pretrain}")
            print(f"Found {len(ckpt_files)} checkpoints:\n")

            for ckpt_path in ckpt_files:
                load_and_eval(model, ckpt_path, test_dir_list=[7,8,9,10], logger=logger)

        else:
            raise ValueError(f"Invalid args.pretrain: {args.pretrain}")

    else:
        init_epoch = 0
        print("Training from scratch")
        logger.info("Training from scratch")


def eval_pose(model, test_list, epoch ):

    for item in test_list:

        test_dataset = OdometryDataset(
            is_training = 0,
            num_points = args.num_points,
            data_root = args.data_root,
            data_dir_list = [item],
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size= args.val_batch_size,
            shuffle=False,
            num_workers= args.workers, 
            pin_memory=True,
            worker_init_fn=lambda x: np.random.seed((torch.initial_seed()) % (2 ** 32))
        )

        line = 0
        total_time = 0

        for batch_id, data in tqdm(enumerate(test_loader), total=len(test_loader), smoothing=0.9):

            pos1,norm1, pos2,norm2, T_gt , qq_gt ,t_gt ,Tr   = data  
            pos1 = pos1.float().cuda()
            pos2 = pos2.float().cuda() 
            norm1 = pos1
            norm2 = pos2
            qq_gt =qq_gt.float().cuda()
            t_gt  =t_gt.float().cuda()
            model = model.eval()

            with torch.no_grad():

                start_time = time.time()
                pred_T = model(pos1, pos2, norm1, norm2, qq_gt,t_gt)                
                total_time += (time.time() - start_time)
                pred_T = pred_T.cpu().numpy()
                for n0 in range(pred_T.shape[0]):

                    cur_Tr = Tr[n0, :, :]
                    TT =pred_T[n0:n0 + 1, :].reshape(4,4)
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
        os.system('python eval.py --result_dir ' + data_dir + ' --eva_seqs ' + str(item).zfill(2) + '_pred' + ' --epoch ' + str(epoch))
        print("This seq is over!\n")
    return 0


def eval_pose1(model, test_list, epoch):  
    MAX_SUBMAP = 3

    for item in test_list:

        test_dataset = OdometryDataset(
            is_training=0,
            num_points=args.num_points,
            data_root=args.data_root,
            data_dir_list=[item],
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            worker_init_fn=lambda x: np.random.seed((torch.initial_seed()) % (2 ** 32))
        )

        submap_points = []    
        # set batch_size = 1
        for batch_id, data in tqdm(enumerate(test_loader), total=len(test_loader), smoothing=0.9):

            pos1, _, pos2, _, T_gt, qq_gt, t_gt, Tr = data

            pos1 = pos1.float().cuda()    
            qq_gt = qq_gt.float().cuda()
            t_gt = t_gt.float().cuda()
            model = model.eval()

            if len(submap_points) == 0:
                current_pc = pos1[0].cpu().numpy()      
                submap_points.append(current_pc)
                T_final = np.eye(4)
                T = T_final[:3, :].reshape(1, 1, 12)
                continue
            map_pc = np.concatenate(submap_points, axis=0)
            map_pc_down= crop_down(map_pc,args.num_points)
            map_pc_tensor = torch.from_numpy(map_pc_down).float().cuda().unsqueeze(0)

            with torch.no_grad():
                pred_T = model(pos1, map_pc_tensor, pos1, map_pc_tensor, qq_gt, t_gt)
            pred_TT = pred_T[0].cpu().numpy().reshape(4, 4) 

            pred_TT_inv = np.linalg.inv(pred_TT)
            R = pred_TT_inv[:3, :3]
            t = pred_TT_inv[:3, 3]

            for i in range(len(submap_points)):
                pc = submap_points[i]
                pc = (R @ pc.T).T + t
                submap_points[i] = pc
            new_pc = pos1[0].cpu().numpy()
            submap_points.append(new_pc)
            
            if len(submap_points) > MAX_SUBMAP:
                submap_points.pop(0)

            cur_Tr = Tr[0].cpu().numpy()
            pred_T = np.matmul(cur_Tr, pred_TT)  
            pred_T = np.matmul(pred_T, np.linalg.inv(cur_Tr))     
            T_final = T_final @ pred_T
            T_current = T_final[:3, :].reshape(1, 1, 12)
            T = np.append(T, T_current, axis=0)

        T = T.reshape(-1, 12)
        print(T.shape)

        fname_npy = os.path.join(eval_dir, f"{str(item).zfill(2)}_pred.npy")
        fname_txt = os.path.join(eval_dir, f"{str(item).zfill(2)}_pred.txt")

        data_dir = os.path.join(eval_dir, 'DLO_' + str(item).zfill(2))
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        np.save(fname_npy, T)
        np.savetxt(fname_txt, T)
        os.system(f'cp {fname_npy} {data_dir}')
        os.system(
            f'python eval.py --result_dir {data_dir} '
            f'--eva_seqs {str(item).zfill(2)}_pred --epoch {epoch}'
        )
        print("This seq is over!\n")

    return 0


def crop_down(pc, num_points, voxel_size=0.2):  #体素采样固定点数量

    xmin, xmax = -60, 60
    ymin, ymax = -40, 40
    zmin, zmax = -2, 4

    mask = (
        (pc[:, 0] >= xmin) & (pc[:, 0] <= xmax) &
        (pc[:, 1] >= ymin) & (pc[:, 1] <= ymax) &
        (pc[:, 2] >= zmin) & (pc[:, 2] <= zmax)
    )
    pc = pc[mask]

    if pc.shape[0] == 0:
        return np.zeros((num_points, 3))

    voxel_idx = np.floor(pc / voxel_size).astype(np.int32)

    _, inv_idx, counts = np.unique(
        voxel_idx, axis=0, return_inverse=True, return_counts=True)

    rand_offsets = (np.random.rand(len(counts)) * counts).astype(np.int32)
    sort_idx = np.argsort(inv_idx)

    start_idx = np.cumsum(np.concatenate(([0], counts[:-1])))
    global_idx = start_idx + rand_offsets
    pcs_idx = sort_idx[global_idx]
    down_pc = pc[pcs_idx]

    N = down_pc.shape[0]
    if N >= num_points:
        idx = np.random.choice(N, num_points, replace=False)
        down_pc = down_pc[idx]
    else:
        idx = np.random.choice(N, num_points - N, replace=True)
        down_pc = np.concatenate([down_pc, down_pc[idx]], axis=0)

    return down_pc


if __name__ == '__main__':
    main()




