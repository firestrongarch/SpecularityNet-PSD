from os.path import join, basename
from options.specularitynet.train_options import TrainOptions
from engine import Engine
from data.image_folder import read_fns
from data.transforms import __scale_width
import torch.backends.cudnn as cudnn
import data.reflect_dataset as datasets
import util.util as util

from data import spec
from torch.utils.data import DataLoader


opt = TrainOptions().parse()

opt.isTrain = False
cudnn.benchmark = True
opt.no_log =True
opt.display_id=0
opt.verbose = False

dataset_wild = spec.TestDataset('/home/fu/CodeSpace/SLAM/Experiments/Imgs/inputs/ori',imgsize='origin')
dataloader_wild = DataLoader(dataset_wild,1,num_workers=opt.nThreads,shuffle=not opt.serial_batches,drop_last=False)

# kitti = "08"

# dataset_wild = spec.TestDataset('/home/fu/CodeSpace/SLAM/Experiments/Imgs/inputs',imgsize='origin')
# dataloader_wild = DataLoader(dataset_wild,1,num_workers=opt.nThreads,shuffle=not opt.serial_batches,drop_last=False)

# dataset_kitti_image_0 = spec.TestDataset(f'/home/fu/CodeSpace/Datasets/Kitti/{kitti}/image_0',imgsize='origin')
# dataloader_kitti_image0 = DataLoader(dataset_kitti_image_0,1,num_workers=opt.nThreads,shuffle=not opt.serial_batches,drop_last=False)

# dataset_kitti_image_1 = spec.TestDataset(f'/home/fu/CodeSpace/Datasets/Kitti/{kitti}/image_1',imgsize='origin')
# dataloader_kitti_image1 = DataLoader(dataset_kitti_image_1,1,num_workers=opt.nThreads,shuffle=not opt.serial_batches,drop_last=False)

engine = Engine(opt)

engine.test(dataloader_wild, savedir=join('./results','speccgan'))

# engine.test(dataloader_kitti_image0, savedir=join('/home/fu/CodeSpace/Datasets/Kitti_Net',f'{kitti}/image_0'))
# engine.test(dataloader_kitti_image1, savedir=join('/home/fu/CodeSpace/Datasets/Kitti_Net',f'{kitti}/image_1'))



