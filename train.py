#!/usr/bin/env python3

import os
import sys
import re
import json

import numpy as np
from PIL import Image

import chainer
import chainer.cuda
from chainer import Variable
from chainer import training
from chainer.datasets import TransformDataset, ImageDataset
from chainer.training import extension
from chainer.training import extensions
import cupy

try:
    import chainermn
    from mpi4py import MPI

    mpi_is_master = False
    mpi_available = True
except:  # pylint:disable=bare-except
    mpi_is_master = True
    mpi_available = False

import yaml
from updater import Updater
from net import Discriminator, StyleGenerator, MappingNetwork

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.dirname(__file__)) + os.path.sep + os.path.pardir)

from common.utils.record import record_setting
from common.utils.save_images import convert_batch_images

import utils.yaml_utils as yaml_utils


def sample_generate_light(gen, mapping, dst, rows=8, cols=8, z=None, seed=0, subdir='preview'):
    @chainer.training.make_extension()
    def make_image(trainer):
        nonlocal rows, cols, z
        if trainer.updater.stage > 15:
            rows = min(rows, 2)
            cols = min(cols, 2)
        elif trainer.updater.stage > 13:
            rows = min(rows, 3)
            cols = min(cols, 3)
        elif trainer.updater.stage > 11:
            rows = min(rows, 4)
            cols = min(cols, 4)

        np.random.seed(seed)
        n_images = rows * cols
        xp = gen.xp
        if z is None:
            z = Variable(xp.asarray(mapping.make_hidden(n_images)))
        else:
            z = z[:n_images]
        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            x = gen(mapping(z), stage=trainer.updater.stage)
        x = chainer.cuda.to_cpu(x.data)
        np.random.seed()

        x = convert_batch_images(x, rows, cols)

        preview_dir = '{}/{}'.format(dst, subdir)
        if not os.path.exists(preview_dir):
            os.makedirs(preview_dir)

        preview_path = preview_dir + '/image_latest.png'
        Image.fromarray(x).save(preview_path)
        preview_path = preview_dir + '/image{:0>8}.png'.format(trainer.updater.iteration)
        Image.fromarray(x).save(preview_path)

    return make_image


def make_iterator_func(dataset, batch_size):
    return chainer.iterators.MultithreadIterator(dataset, batch_size=batch_size, repeat=True, shuffle=None,
                                                 n_threads=config.dataset_worker_num)


def batch_generate_func(gen, mapping, trainer):
    def generate(n_images):
        xp = gen.xp
        z = Variable(xp.asarray(mapping.make_hidden(n_images)))
        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            x = gen(mapping(z), stage=trainer.updater.stage)
        x = chainer.cuda.to_cpu(x.data)
        return x

    return generate


class RunningHelper(object):

    def __init__(self, use_mpi, config):
        self.use_mpi = use_mpi
        self.config = config

        # Setup
        if self.use_mpi:
            if not mpi_available:
                raise RuntimeError('ChainerMN required for MPI but cannot be imported. Abort.')
            comm = chainermn.create_communicator(config.comm_name)
            if comm.mpi_comm.rank == 0:
                print('==========================================')
                print('Num process (COMM_WORLD): {}'.format(MPI.COMM_WORLD.Get_size()))
                print('Communcator name: {}'.format(config.comm_name))
                print('==========================================')
            fleet_size = MPI.COMM_WORLD.Get_size()
            device = comm.intra_rank
        else:
            fleet_size = 1
            comm = None
            device = config.gpu

        self.fleet_size, self.comm, self.device = fleet_size, comm, device

        self.is_master = is_master = not self.use_mpi or (self.use_mpi and comm.rank == 0)

        # Early works
        if is_master:
            record_setting(config.out)

        # Show effective hps
        effective_hps = {
            'is_master': self.is_master,
            'stage_interval': self.stage_interval,
            'dynamic_batch_size': self.dynamic_batch_size
        }
        self.print_log('Effective hps: {}'.format(effective_hps))

    @property
    def keep_smoothed_gen(self):
        return self.config.keep_smoothed_gen and self.is_master

    @property
    def use_cleargrads(self):
        # 1. Chainer 2/3 does not support clear_grads when running with MPI, so use zero_grads instead
        # 2. zero_grads on chainer >= 5.0.0 has a critical bug when running with MPI 
        return True

    @property
    def stage_interval(self):
        return self.config.stage_interval // self.fleet_size

    @property
    def dynamic_batch_size(self):
        fleet_size = self.fleet_size
        return [int(_) for _ in self.config.dynamic_batch_size.split(',')]

    def print_log(self, msg):
        print('[Device {}] {}'.format(self.device, msg))

    def check_hps_consistency(self):
        assert self.config.max_stage % 2 == 1
        # assert 4 * (2 ** ((config.max_stage - 1) // 2)) == config.image_size
        # assert 2 ** int(np.floor(np.log2(config.image_size))) == config.image_size
        assert len(self.dynamic_batch_size) >= self.config.max_stage

    def make_optimizer(self, model, alpha, beta1, beta2):
        self.print_log('Use Adam Optimizer with alpah = {}, beta1 = {}, beta2 = {}'.format(alpha, beta1, beta2))
        optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1, beta2=beta2)
        if self.use_mpi:
            self.print_log('Use Optimizer with MPI')
            optimizer = chainermn.create_multi_node_optimizer(optimizer, self.comm)
        optimizer.setup(model)
        return optimizer

    def make_dataset(self, stage_int):
        if self.is_master:
            size = 4 * (2 ** ((stage_int + 1) // 2))
            _dataset = BaseDataset(
                json.load(open(self.config.dataset_config, 'r')),
                '%dx%d' % (size, size),
                [["resize", {"probability": 1, "width": size, "height": size, "resample_filter": "ANTIALIAS"}]]
            )
            self.print_log('Add (master) dataset for size {}'.format(size))
        else:
            _dataset = None
            self.print_log('Add (slave) dataset')

        if self.use_mpi:
            _dataset = chainermn.scatter_dataset(_dataset, self.comm)

        return _dataset


config = yaml_utils.Config(yaml.load(open('configs/default.yml')))


def main():
    # FLAGS(sys.argv)

    running_helper = RunningHelper(config.use_mpi, config)
    global mpi_is_master
    mpi_is_master = running_helper.is_master
    # Check stage / image size / dynamic batch size / data consistency.
    running_helper.check_hps_consistency()

    # Setup Models
    mapping = MappingNetwork(config.ch)
    generator = StyleGenerator(config.ch, enable_blur=config.enable_blur)
    discriminator = Discriminator(ch=config.ch, enable_blur=config.enable_blur)

    if running_helper.keep_smoothed_gen:
        smoothed_generator = StyleGenerator(config.ch, enable_blur=config.enable_blur)
        smoothed_mapping = MappingNetwork(config.ch)

    models = [mapping, generator, discriminator]
    model_names = ['Mapping', 'Generator', 'Discriminator']
    if running_helper.keep_smoothed_gen:
        models.append(smoothed_generator)
        models.append(smoothed_mapping)
        model_names.append('SmoothedGenerator')
        model_names.append('SmoothedMapping')

    if running_helper.device > -1:
        chainer.cuda.get_device_from_id(running_helper.device).use()
        for model in models:
            model.to_gpu()

    train = TransformDataset(ImageDataset(config.train_img_list), lambda x: x / 127.5 - 1)
    train_iter = chainer.iterators.MultiprocessIterator(train, config.batchsize, n_processes=4)
    # stage_manager = StageManager(
    #     stage_interval=running_helper.stage_interval,
    #     dynamic_batch_size=running_helper.dynamic_batch_size,
    #     make_dataset_func=running_helper.make_dataset,
    #     make_iterator_func=make_iterator_func,
    #     debug_start_instance=config.debug_start_instance)

    # if running_helper.is_master:
    #    chainer.global_config.debug = True

    updater_args = {
        "models": models,
        "optimizer": {
            "map": running_helper.make_optimizer(mapping, config.adam_alpha_g / 100, config.adam_beta1,
                                                 config.adam_beta2),
            "gen": running_helper.make_optimizer(generator, config.adam_alpha_g, config.adam_beta1, config.adam_beta2),
            "dis": running_helper.make_optimizer(discriminator, config.adam_alpha_d, config.adam_beta1,
                                                 config.adam_beta2)
        },
        "iterator": train_iter,
        'config': config,
        'lambda_gp': config.lambda_gp,
        'smoothing': config.smoothing,
        'style_mixing_rate': config.style_mixing_rate,
        'total_gpu': running_helper.fleet_size
    }
    updater = Updater(**updater_args)
    trainer = training.Trainer(
        updater, (config.iteration, 'iteration'), out=config.out)

    # Set up extensions
    if running_helper.is_master:
        for model, model_name in zip(models, model_names):
            trainer.extend(
                extensions.snapshot_object(model, model_name + '_{.updater.iteration}.npz'),
                trigger=(config.snapshot_interval, 'iteration'))

        trainer.extend(
            extensions.snapshot(filename='snapshot_iter_{.updater.iteration}.npz'),
            trigger=(config.snapshot_interval, 'iteration'))

        trainer.extend(
            extensions.ProgressBar(update_interval=10))

        trainer.extend(
            sample_generate_light(generator, mapping, config.out, rows=8, cols=8),
            trigger=(config.evaluation_sample_interval, 'iteration'),
            priority=extension.PRIORITY_WRITER)

        if running_helper.keep_smoothed_gen:
            trainer.extend(
                sample_generate_light(smoothed_generator, smoothed_mapping, config.out, rows=8, cols=8,
                                      subdir='preview_smoothed'),
                trigger=(config.evaluation_sample_interval, 'iteration'),
                priority=extension.PRIORITY_WRITER)
        report_keys = [
            'iteration', 'elapsed_time', 'stage', 'batch_size', 'image_size', 'gen/loss_adv', 'dis/loss_adv',
            'dis/loss_gp'
        ]
        if config.fid_interval > 0:
            assert False, "FID is not supported for debug."
            # report_keys += 'FID'
            # fidapi = FIDAPI(config.fid_clfs_type,
            #                 config.fid_clfs_path,
            #                 gpu=running_helper.device,
            #                 load_real_stat=config.fid_real_stat)
            # trainer.extend(
            #     fid_extension(fidapi,
            #                   batch_generate_func(generator, mapping, trainer),
            #                   seed=config.seed,
            #                   report_key='FID'
            #                   ),
            #     trigger=(config.fid_interval, 'iteration')
            # )
            # if running_helper.keep_smoothed_gen:
            #     report_keys += 'S_FID'
            #     trainer.extend(
            #         fid_extension(fidapi,
            #                       batch_generate_func(smoothed_generator, smoothed_mapping, trainer),
            #                       seed=config.seed,
            #                       report_key='S_FID'
            #                       ),
            #         trigger=(config.fid_interval, 'iteration')
            #     )

        trainer.extend(extensions.LogReport(keys=report_keys, trigger=(config.display_interval, 'iteration')))
        trainer.extend(extensions.PrintReport(report_keys), trigger=(config.display_interval, 'iteration'))

    # Recover if possible
    if config.get_model_from_interation != '':
        resume_iteration_str = config.get_model_from_interation
        print('Resume from {}'.format(resume_iteration_str))
        for model, model_name in zip(models, model_names):
            chainer.serializers.load_npz(
                config.out + '/' + model_name + '_%s.npz' % resume_iteration_str,
                model, )
        chainer.serializers.load_npz(
            config.out + '/' + 'snapshot_iter_%s.npz' % resume_iteration_str,
            trainer, )

    elif config.auto_resume:
        print("Auto Resume")
        candidates = []
        auto_resume_dir = config.auto_resume_dir if config.auto_resume_dir != '' else config.out
        for fname in [f for f in os.listdir(auto_resume_dir) if f.startswith('Generator_') and f.endswith('.npz')]:
            fname = re.sub(r'^Generator_', '', fname)
            fname = re.sub('\.npz$', '', fname)
            fname_int = None
            try:
                fname_int = int(fname)
            except ValueError:
                pass
            if fname_int is not None:
                all_model_exist = True
                for m in model_names:
                    if not os.path.exists(auto_resume_dir + '/' + m + '_' + fname + '.npz'):
                        all_model_exist = False

                if not os.path.exists(auto_resume_dir + '/' + ('snapshot_iter_%s.npz' % fname)):
                    all_model_exist = False

                if all_model_exist:
                    candidates.append(fname)

        # print(candidates)
        candidates.sort(key=lambda _: int(_), reverse=True)
        if len(candidates) > 0:
            resume_iteration_str = candidates[0]
        else:
            resume_iteration_str = None
        if resume_iteration_str is not None:
            print('Automatic resuming: use iteration %s' % resume_iteration_str)
            for model, model_name in zip(models, model_names):
                chainer.serializers.load_npz(
                    auto_resume_dir + '/' + model_name + '_%s.npz' % resume_iteration_str,
                    model, )
            chainer.serializers.load_npz(
                auto_resume_dir + '/' + 'snapshot_iter_%s.npz' % resume_iteration_str,
                trainer, )

    # Run the training
    if config.enable_cuda_profiling:
        with cupy.cuda.profile():
            trainer.run()
    else:
        # with chainer.using_config('debug', True):
        trainer.run()

    for model, model_name in zip(models, model_names):
        chainer.serializers.save_npz(config.out + '/' + model_name + '_latest.npz', model)


import pdb, traceback, sys, code

if __name__ == '__main__':
    try:
        main()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
