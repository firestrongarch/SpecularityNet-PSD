import torch
import util.util as util
import models
import time
import os
import sys
from os.path import join
from tqdm import tqdm  # 导入 tqdm
from util.visualizer import Visualizer


class Engine(object):
    def __init__(self, opt):
        self.opt = opt
        self.writer = None
        self.visualizer = None
        self.model = None
        self.best_val_loss = 1e6

        self.__setup()

    def __setup(self):
        self.basedir = join('checkpoints', self.opt.name)
        if not os.path.exists(self.basedir):
            os.mkdir(self.basedir)
        
        opt = self.opt
        
        """Model"""
        self.model = models.__dict__[self.opt.model]()
        self.model.initialize(opt)
        if opt.resume:
            self.model.epoch += 1
        if not opt.no_log:
            self.writer = util.get_summary_writer(os.path.join(self.basedir, 'logs'))
            self.visualizer = Visualizer(opt)

    def train(self, train_loader, **kwargs):
        avg_meters = util.AverageMeters()
        opt = self.opt
        model = self.model
        epoch = self.epoch

        epoch_start_time = time.time()
        
        # 使用 tqdm 包装数据加载器，并设置进度条格式
        train_loader_tqdm = tqdm(
            train_loader, 
            desc=f'Epoch {epoch}', 
            dynamic_ncols=True, 
            bar_format='{l_bar}{bar:20}{r_bar}'
        )
        
        for i, data in enumerate(train_loader_tqdm):
            iter_start_time = time.time()
            iterations = self.iterations
            
            model.set_input(data, mode='train')
            model.optimize_parameters(**kwargs)
            
            errors = model.get_current_errors()
            avg_meters.update(errors)
            
            # 动态更新进度条的统计信息（保留4位小数）
            current_metrics = {k: round(avg_meters[k], 4) for k in avg_meters.keys()}
            train_loader_tqdm.set_postfix(current_metrics)
            
            if not opt.no_log:
                util.write_loss(self.writer, 'train', avg_meters, iterations)
            
                if iterations % opt.display_freq == 0 and opt.display_id != 0:
                    save_result = iterations % opt.update_html_freq == 0
                    self.visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

                if iterations % opt.print_freq == 0 and opt.display_id != 0:
                    t = (time.time() - iter_start_time)          

            self.iterations += 1
    
        if not self.opt.no_log:
            if (self.epoch+1) % opt.save_epoch_freq == 0:
                print('saving the model at epoch %d, iters %d' %
                    (self.epoch+1, self.iterations))
                model.save()
            
            print('saving the latest model at the end of epoch %d, iters %d' % 
                (self.epoch+1, self.iterations))
            model.save(label='latest')

            print('Time Taken: %d sec' %
                (time.time() - epoch_start_time))

    def eval(self, val_loader, dataset_name, savedir=None, loss_key=None, **kwargs):
        avg_meters = util.AverageMeters()
        model = self.model
        opt = self.opt
        
        # 使用 tqdm 包装验证数据加载器
        val_loader_tqdm = tqdm(
            val_loader, 
            desc=f'Eval {dataset_name}', 
            dynamic_ncols=True,
            bar_format='{l_bar}{bar:20}{r_bar}'
        )
        
        with torch.no_grad():
            for i, data in enumerate(val_loader_tqdm):                
                index = model.eval(data, savedir=savedir, **kwargs)
                avg_meters.update(index)
                
                # 动态更新指标（保留4位小数）
                current_metrics = {k: round(avg_meters[k], 4) for k in avg_meters.keys()}
                val_loader_tqdm.set_postfix(current_metrics)
                
        if not opt.no_log:
            util.write_loss(self.writer, join('eval', dataset_name), avg_meters, self.epoch)
        
        if loss_key is not None:
            val_loss = avg_meters[loss_key]
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                print('saving the best model at the end of epoch %d, iters %d' % 
                    (self.epoch, self.iterations))
                model.save(label='best_{}_{}'.format(loss_key, dataset_name))

        return avg_meters

    def test(self, test_loader, savedir=None, **kwargs):
        model = self.model
        opt = self.opt
        
        # 使用 tqdm 包装测试数据加载器
        test_loader_tqdm = tqdm(
            test_loader, 
            desc='Testing', 
            dynamic_ncols=True,
            bar_format='{l_bar}{bar:20}{r_bar}'
        )
        
        with torch.no_grad():
            for i, data in enumerate(test_loader_tqdm):
                model.test(data, savedir=savedir, **kwargs)
                # 可以添加附加信息（例如：当前处理文件名）
                # test_loader_tqdm.set_postfix(file_name=data['file_name'][0])

    @property
    def iterations(self):
        return self.model.iterations

    @iterations.setter
    def iterations(self, i):
        self.model.iterations = i

    @property
    def epoch(self):
        return self.model.epoch

    @epoch.setter
    def epoch(self, e):
        self.model.epoch = e