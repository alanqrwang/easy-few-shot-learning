from pathlib import Path
import random
from statistics import mean
import os

import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
import argparse
import wandb

from easyfsl.datasets import CUB
from easyfsl.samplers import TaskSampler
from easyfsl.methods import PrototypicalNetworks, FewShotClassifier, NW, Finetune, MatchingNetworks
from easyfsl.modules import resnet12
from easyfsl.utils import evaluate

from util.utils import ParseKwargs, initialize_wandb

class Parser(argparse.ArgumentParser):
    def __init__(self):
        super(Parser, self).__init__(description='NW Head Training')
        self.add_argument('--method', type=str,
            required=True, help='Method')
        self.add_argument('--dataset', type=str,
            default='cub', help='Dataset')
        self.add_argument('--num_steps_per_epoch', type=int,
            default=1000000000, help='Seed')
        self.add_argument('--seed', type=int,
            default=0, help='Seed')
        self.add_argument('--fine_tuning_steps', type=int,
            default=0, help='Number of ft steps')
        self.add_argument('--class_dropout', type=float,
            default=0, help='Class Dropout')
        self.add_argument('--batch_size', type=int,
            default=128, help='Batch size')
        self.add_argument('--n_workers', type=int,
            default=1, help='N workers')
        self.add_argument('--lr', type=float,
            default=0, help='Learning rate')
        self.add_bool_arg('debug_mode', False)
        
        # FSL
        self.add_bool_arg('episodic_training', False)
        self.add_argument('--n_tasks_per_epoch', type=int,
            default=500, help='Num tasks per training epoch')
        self.add_argument('--n_validation_tasks', type=int,
            default=100, help='Num validation tasks')
        self.add_argument('--n_test_tasks', type=int,
            default=1000, help='Num test tasks')

        # Weights & Biases
        self.add_bool_arg('use_wandb', False)
        self.add_argument('--wandb_api_key_path', type=str,
                            help="Path to Weights & Biases API Key. If use_wandb is set to True and this argument is not specified, user will be prompted to authenticate.")
        self.add_argument('--wandb_kwargs', nargs='*', action=ParseKwargs, default={},
                            help='keyword arguments for wandb.init() passed as key1=value1 key2=value2')

    def add_bool_arg(self, name, default=True):
        """Add boolean argument to argparse parser"""
        group = self.add_mutually_exclusive_group(required=False)
        group.add_argument('--' + name, dest=name, action='store_true')
        group.add_argument('--no_' + name, dest=name, action='store_false')
        self.set_defaults(**{name: default})

    def parse(self):
        args = self.parse_args()
        # args.run_dir = os.path.join(args.models_dir,
        #               'method{method}_dataset{dataset}_arch{arch}_lr{lr}_bs{batch_size}_projdim{proj_dim}_numsupp{numsupp}_subsample{subsample}_wd{wd}_seed{seed}'.format(
        #                 method=args.train_method,
        #                 dataset=args.dataset,
        #                 arch=args.arch,
        #                 lr=args.lr,
        #                 batch_size=args.batch_size,
        #                 proj_dim=args.proj_dim,
        #                 numsupp=args.supp_num_per_class,
        #                 subsample=args.subsample_classes,
        #                 wd=args.weight_decay,
        #                 seed=args.seed
        #               ))
        # args.ckpt_dir = os.path.join(args.run_dir, 'checkpoints')
        # if not os.path.exists(args.run_dir):
        #     os.makedirs(args.run_dir)
        # if not os.path.exists(args.ckpt_dir):
        #     os.makedirs(args.ckpt_dir)

        # Print args and save to file
        # print('Arguments:')
        # pprint(vars(args))
        # with open(args.run_dir + "/args.txt", 'w') as args_file:
        #     json.dump(vars(args), args_file, indent=4)
        
        if args.debug_mode:
            args.num_steps_per_epoch = 5
            args.num_val_steps_per_epoch = 5
        return args

def training_epoch(
    model: FewShotClassifier, data_loader: DataLoader, optimizer: Optimizer,
    criterion: nn.Module, args
):
    all_loss = []
    model.train()
    with tqdm(
        enumerate(data_loader), total=len(data_loader), desc="Training"
    ) as tqdm_train:
        for episode_index, batch in tqdm_train:
            optimizer.zero_grad()
            if args.episodic_training:
                support_images, support_labels, \
                query_images, query_labels, ids, = batch
                model.process_support_set(
                    support_images.to(args.device), support_labels.to(args.device)
                )
            else:
                query_images, query_labels = batch

            if args.method == 'nw':
                classification_scores = model(query_images.to(args.device), query_labels.to(args.device))
            else:
                classification_scores = model(query_images.to(args.device))

            loss = criterion(classification_scores, query_labels.to(args.device))
            loss.backward()
            optimizer.step()

            all_loss.append(loss.item())

            tqdm_train.set_postfix(loss=mean(all_loss))

            if episode_index == args.num_steps_per_epoch:
                break

    return mean(all_loss)

def main():
    # Parse arguments
    args = Parser().parse()

    # Set device
    if torch.cuda.is_available():
        args.device = torch.device('cuda:0')
    else:
        args.device = torch.device('cpu')
        print('No GPU detected... Training will be slow!')
    os.environ["CUDA_VISIBLE_DEVICES"] = '0' 

    # Random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if args.dataset == 'cub':
        train_set = CUB(split="train", training=True)
        val_set = CUB(split="val", training=False)
        num_classes = 200
    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))

    n_way = 5
    n_shot = 5
    n_query = 10

    if args.episodic_training:
        train_sampler = TaskSampler(
            train_set, n_way=n_way, n_shot=n_shot, n_query=n_query, n_tasks=args.n_tasks_per_epoch
        )
        train_loader = DataLoader(
            train_set,
            batch_sampler=train_sampler,
            num_workers=args.n_workers,
            pin_memory=True,
            collate_fn=train_sampler.episodic_collate_fn,
        )
    else:
        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            num_workers=args.n_workers,
            pin_memory=True,
            shuffle=True,
        )

    # Validation is always episodic
    val_sampler = TaskSampler(
        val_set, n_way=n_way, n_shot=n_shot, n_query=n_query, n_tasks=args.n_validation_tasks
    )
    val_loader = DataLoader(
        val_set,
        batch_sampler=val_sampler,
        num_workers=args.n_workers,
        pin_memory=True,
        collate_fn=val_sampler.episodic_collate_fn,
    )

    convolutional_network = resnet12()
    feat_dim = 640

    if args.method == 'protonets':
        few_shot_classifier = PrototypicalNetworks(convolutional_network).to(args.device)
    elif args.method == 'nw':
        train_set.targets = train_set.labels
        few_shot_classifier = NW(convolutional_network, 
                                 num_classes=num_classes,
                                 support_set=train_set,
                                 class_dropout=args.class_dropout,
                                 fine_tuning_steps=args.fine_tuning_steps, 
                                 debug_mode=args.debug_mode).to(args.device)
    elif args.method == 'matchingnets':
        few_shot_classifier = MatchingNetworks(
                                  convolutional_network,
                                  feature_dimension=feat_dim).to(args.device)
    else:
        raise ValueError('Unknown method: {}'.format(args.method))

    if args.method in ['nw', 'matchingnets']:
        criterion = nn.NLLLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    n_epochs = 200
    if args.method == 'nw':
        scheduler_milestones = [200, 250]
    else:
        scheduler_milestones = [120, 160]
    scheduler_gamma = 0.1

    train_optimizer = SGD(
        few_shot_classifier.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4
    )
    train_scheduler = MultiStepLR(
        train_optimizer,
        milestones=scheduler_milestones,
        gamma=scheduler_gamma,
    )

    if args.use_wandb and not args.debug_mode:
        initialize_wandb(args)

    best_state = few_shot_classifier.state_dict()
    best_validation_accuracy = 0.0
    for epoch in range(n_epochs):
        print(f"Epoch {epoch}")
        average_loss = training_epoch(few_shot_classifier, train_loader, train_optimizer, criterion, args)
        validation_accuracy = evaluate(
            few_shot_classifier, val_loader, args, device=args.device, tqdm_prefix="Validation"
        )

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            best_state = few_shot_classifier.state_dict()
            print("Ding ding ding! We found a new best model!")

        if args.use_wandb and not args.debug_mode:
            wandb.log({'Train/loss': average_loss, 'Val/acc': validation_accuracy})

        # Warn the scheduler that we did an epoch
        # so it knows when to decrease the learning rate
        train_scheduler.step()

    few_shot_classifier.load_state_dict(best_state) 

    test_set = CUB(split="test", training=False)
    test_sampler = TaskSampler(
        test_set, n_way=n_way, n_shot=n_shot, n_query=n_query, n_tasks=args.n_test_tasks
    )
    test_loader = DataLoader(
        test_set,
        batch_sampler=test_sampler,
        num_workers=args.n_workers,
        pin_memory=True,
        collate_fn=test_sampler.episodic_collate_fn,
    )

    accuracy = evaluate(few_shot_classifier, test_loader, args, device=args.device)
    print(f"Average accuracy : {(100 * accuracy):.2f} %")

if __name__ == '__main__':
    main()