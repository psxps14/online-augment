import os
import time
import shutil
from collections import defaultdict
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.nn as nn

from utils import AverageMeter
import utils
# import cifar_models as cifar_models
# from torch.utils.data.sampler import SubsetRandomSampler
# import json
# from torchvision.utils import make_grid, save_image
# import math
# from warmup_scheduler import GradualWarmupScheduler
from data import get_dataloaders
from models import get_model, num_class
from augment_stn import Augment

def train_and_validate(config):

    # data loaders
    trainloader, testloader = get_dataloaders(config)

    # model
    # if config.bn_num == 1:
    #     target_net = get_model(config, num_class(config.dataset))
    if config.bn_num == 2:
        target_net = get_model(config, num_class(config.dataset), bn_types=['base', 'stn'])
    else:
        raise Exception('invalid bn_num: {}'.format(config.bn_num))

    # aug_stn
    print('aug_stn: {}'.format(config.aug_stn))
    print('noise_dim: {}'.format(config.noise_dim))
    if config.aug_stn == 'stn_2cycle_diverse':
        from models.aug_stn import STN
        aug_net = STN(config.noise_dim, linear_size=config.linear_size)
        aug_net = nn.DataParallel(aug_net).cuda()
    else:
        raise Exception('invalid aug_stn: {}'.format(config.aug_stn))

    model = Augment(target_net=target_net, aug_net=aug_net, config=config)


    start_epoch = 0
    best_test_acc = 0.0
    test_acc = 0.0
    best_epoch = 0
    accuracies = []
    if config.resume:
        best_test_acc, test_acc, start_epoch = \
            utils.load_checkpoint(config, model.target_net, model.target_net_optim)

    print('trainloader length: {}'.format(len(trainloader)))
    print('testloader length: {}'.format(len(testloader)))

    exp_dir = utils.get_log_dir_path(config.exp_dir, config.exp_id)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    print('exp_dir: {}'.format(exp_dir))
    log_file = os.path.join(exp_dir, 'log.txt')
    names = ['epoch', 'lr', 'Train Acc', 'Test Acc', 'Best Test Acc']
    with open(log_file, 'a') as f:
        f.write('batch size: {}\n'.format(config.batch_size))
        f.write('lr: {}\n'.format(config.lr))
        f.write('momentum: {}\n'.format(config.momentum))
        f.write('weight_decay: {}\n'.format(config.weight_decay))
        for per_name in names:
            f.write(per_name + '\t')
        f.write('\n')
    # print('=> Training the base model')
    # print('start_epoch {}'.format(start_epoch))
    # print(type(start_epoch))
    # exit()
    print('target net grad clip: {}'.format(config.grad_clip))
    for epoch in range(start_epoch, config.epochs):
        # lr = adjust_learning_rate(optimizer, epoch, model.module, config)
        lr = model.target_net_optim.param_groups[0]['lr']
        print('lr: {}'.format(lr))
        # inner_lr = get_lr_cosine_decay(config, epoch)
        # print('inner_lr: {}'.format(inner_lr))
        # train for one epoch
        train_acc = train_epoch_two_bns(trainloader, model, epoch, config)
        # evaluate on test set
        # print('testing epoch ...')
        test_acc = validate_epoch(testloader, model, config)
        # remember best acc, evaluate on test set and save checkpoint
        is_best = test_acc > best_test_acc
        if is_best:
            best_test_acc = test_acc
            best_epoch = epoch

        accuracies.append(test_acc)

        utils.save_checkpoint(model,{
            'epoch': epoch + 1,
            'state_dict': model.target_net.state_dict(),
            'test_acc': test_acc,
            'optimizer': model.target_net_optim.state_dict(),
        }, is_best, exp_dir)

        values = [train_acc, test_acc, best_test_acc]
        with open(log_file, 'a') as f:
            f.write('{:d}\t'.format(epoch))
            f.write('{:g}\t'.format(lr))
            for per_value in values:
                f.write('{:2.2f}\t'.format(per_value))
            f.write('\n')
        print('exp_dir: {}'.format(exp_dir))

    print("max validation accuracy: " + str(best_test_acc))
    print("max validation epoch: " + str(best_epoch))

    abs_changes = [abs(accuracies[i] - accuracies[i-1]) for i in range(1, len(accuracies))]
    mean_abs_change = sum(abs_changes) / len(abs_changes)
    print("MAC: " + str(mean_abs_change))

    if (len(abs_changes) >= 99):
            mean_abs_change100 = sum(abs_changes[0:99]) / 99
            print("MAC of first 100 epochs: " + str(mean_abs_change100))


# not using implicit gradients from validation data
def train_epoch_two_bns(trainloader, model, epoch, config):
    print('using function train_epoch_two_bns...')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_adv = AverageMeter()
    losses_div = AverageMeter()
    losses_diversity = AverageMeter()
    # losses3 = AverageMeter()
    # losses4 = AverageMeter()
    top1 = AverageMeter()

    model.target_net.train()
    model.aug_net.train()
    loader_len = len(trainloader)
    end = time.time()
    for i, (input_list, target) in enumerate(trainloader):
        # measure data loading time
        # print('iter: {}'.format(i))
        data_time.update(time.time() - end)
        assert isinstance(input_list, list)
        assert len(input_list) == 2
        input, input_preaug = input_list[0], input_list[1]
        # print('input size: {}'.format(input.size()))
        # print('input_autoaug size: {}'.format(input_autoaug.size()))
        # print('target size: {}'.format(target.size()))
        input, input_preaug, target = input.cuda(), input_preaug.cuda(), target.cuda()

        for j in range(config.inner_num):
            model.aug_net_optim.zero_grad()
            model.target_net_optim.zero_grad()
            noise = torch.randn(input.size(0), config.noise_dim).cuda()
            input_aug, target_aug, div_loss, diversity_loss = \
                model.aug_net(noise, input, target, require_loss=True)
            input_aug.register_hook(lambda grad: grad * (-model.args.adv_weight_stn))
            output_aug = model.target_net(input_aug, 'stn')
            loss_aug = model.criterion(output_aug, target_aug)

            # loss_adv = -loss_aug *
            loss_div = div_loss * model.args.div_weight_stn
            loss_diversity = -diversity_loss * model.args.diversity_weight_stn
            loss_aug_net = loss_aug + loss_div + loss_diversity
            loss_aug_net.backward()
            model.aug_net_optim.step()

            if config.grad_clip and config.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.target_net.parameters(), config.grad_clip)
            model.target_net_optim.step()

        model.target_net_optim.zero_grad()
        output_preaug = model.target_net(input_preaug, 'base')
        loss_preaug = model.criterion(output_preaug, target)
        loss_preaug.backward()

        if config.grad_clip and config.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.target_net.parameters(), config.grad_clip)

        model.target_net_optim.step()

        # update lr
        model.lr_scheduler.step(epoch + float(i + 1) / loader_len)

        losses_adv.update((loss_aug * (-model.args.adv_weight_stn)).item(), input.size(0))
        losses_div.update(loss_div.item(), input.size(0))
        losses_diversity.update(loss_diversity.item(), input.size(0))
        losses.update((loss_aug+loss_preaug).item(), input.size(0))
        acc = utils.accuracy(output_preaug, target)[0]
        top1.update(acc.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # print('momentum_buffer: {}'.format(momentum_buffer[0][0, 0, 0:10]))

        if i % config.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Acc {top1.val:.3f}% ({top1.avg:.3f}%)\t'
                  'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                  'loss_adv {losses_adv.val:.4f} ({losses_adv.avg:.4f})\t'
                  'loss_div {losses_div.val:.4f} ({losses_div.avg:.4f})\t'.format(
                   epoch, i, len(trainloader), top1=top1, losses=losses,
                   losses_adv=losses_adv, losses_div=losses_div))
            # exit()

    print(' * Acc {top1.avg:.3f}% '.format(top1=top1))
    # exit()
    return top1.avg


def validate_epoch(val_loader, model, config):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.target_net.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            input, target = input.cuda(), target.cuda()

            # compute output
            output = model.target_net(input)
            loss = model.criterion(output, target)

            # measure accuracy and record loss
            acc = utils.accuracy(output, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(acc.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % config.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc {top1.val:.3f}% ({top1.avg:.3f}%)'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1))

    print(' * Validation Acc {top1.avg:.3f}% '.format(top1=top1))

    return top1.avg
