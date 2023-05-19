import os
import time
from time import strftime
import pandas as pd
import warnings
import datetime
import torch
import torch.utils.data
import torchvision
import transforms
import argparse
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data.dataloader import default_collate
import presets
import models
import utils
from torch import nn
import tqdm
import random
import torch.backends.cudnn as cudnn
import numpy as np
from sklearn.metrics import confusion_matrix
from datetime import date

from torchvision.models import swin_v2_s
from torchvision.models import Swin_V2_S_Weights
from torchvision.models import efficientnet_v2_s
from torchvision.models.resnet import ResNet18_Weights
from torchvision.models import resnet18
from torchvision.models.efficientnet import EfficientNet_V2_S_Weights
from torchvision.models import regnet_y_16gf
from torchvision.models.regnet import RegNet_Y_16GF_Weights
from torchvision.models import swin_v2_t
from torchvision.models import Swin_V2_T_Weights

from pytorch_metric_learning import losses, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
import torchmetrics

torch.manual_seed(66)
torch.cuda.manual_seed(66)
torch.cuda.manual_seed_all(66)
np.random.seed(66)
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(66)

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def get_args_parser(add_help=True):

    parser = argparse.ArgumentParser(description="PyTorch Classification Training", add_help=add_help)

    parser.add_argument("--data-path", default="./hyfinal", type=str,
                        help="dataset path")

    parser.add_argument("--model", default="effnetv2_s", type=str, help="model name, resnet18, regnet_16gf, effnetv2_s, swinv2_m")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument("--gpu", default=0,  type=list)
    parser.add_argument("--distributed", default=False, type=bool )
    parser.add_argument(
        "-b", "--batch-size", default=8, type=int,
        help="images per gpu, the total batch size is $NGPU x batch_size"
    )
    parser.add_argument("--epochs", default=21, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument(
        "-j", "--workers", default=0, type=int, metavar="N", help="number of data loading workers (default: 16)"
    )
    parser.add_argument("--opt", default="sgd", type=str, help="optimizer")
    parser.add_argument("--lr", default=0.001, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument(
        "--norm-weight-decay",
        default=None,
        type=float,
        help="weight decay for Normalization layers (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--bias-weight-decay",
        default=0.01,
        type=float,
        help="weight decay for bias parameters of all layers (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--label-smoothing", default=0, type=float, help="label smoothing (default: 0.0)", dest="label_smoothing"
    )
    parser.add_argument("--mixup-alpha", default=0, type=float, help="mixup alpha (default: 0.0)")
    parser.add_argument("--cutmix-alpha", default=0, type=float, help="cutmix alpha (default: 0.0)")
    parser.add_argument("--lr-scheduler", default="cosineannealinglr", type=str, help="the lr scheduler (default: steplr)")
    parser.add_argument("--lr-warmup-epochs", default=3, type=int,
                        help="the number of epochs to warmup (default: 0)")
    parser.add_argument(
        "--lr-warmup-method", default="linear", type=str, help="the warmup method (default: constant)"
    )
    parser.add_argument("--lr-warmup-decay", default=0.01, type=float, help="the decay for lr")
    parser.add_argument("--lr-step-size", default=20, type=int, help="decrease lr every step-size epochs")
    parser.add_argument("--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma")
    parser.add_argument("--lr-min", default=0.0, type=float, help="minimum lr of lr schedule (default: 0.0)")
    parser.add_argument("--print-freq", default=10, type=int, help="print frequency")
    parser.add_argument("--output-dir", default="./model_log", type=str, help="path to save outputs")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument(
        "--valid-only",
        dest="valid_only",
        help="Only valid the model",
        action="store_true",
    )
    parser.add_argument(
        "--val-only",
        dest="val_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument("--test", default=False, type=bool, help='including test_')
    # distributed training parameters
    parser.add_argument(
        "--interpolation", default="bilinear", type=str, help="the interpolation method (default: bilinear)"
    )
    parser.add_argument(
        "--val-resize-size", default=300, type=int, help="the resize size used for validation (default: 256)"
    )
    parser.add_argument(
        "--val-crop-size", default=280, type=int, help="the central crop size used for validation (default: 224)"
    )
    parser.add_argument(
        "--train-crop-size", default=280, type=int, help="the random crop size used for training (default: 224)"
    )
    parser.add_argument(
        "--n-mean", default=[0.139, 0, 0], type=list, help="normalize mean --n-mean 0.485,0.456,0.406 --n-std 0.229,0.224,0.225"
    )
    parser.add_argument(
        "--n-std", default=[0.073, 1, 1], type=list, help="normalize std"
    )
    parser.add_argument(
        "--transfer-learning", default=False, type=bool, help="transfer learning backpropogate last layer "
    )        
    parser.add_argument(
        "--channel1", default=False, type=bool, help='use only 1channel'
    )
    parser.add_argument("--clip-grad-norm", default=None, type=float,
                        help="the maximum gradient norm (default None)")
    parser.add_argument("--ra-sampler", action="store_true",
                        help="whether to use Repeated Augmentation in training")
    parser.add_argument(
        "--ra-reps", default=3, type=int, help="number of repetitions for Repeated Augmentation (default: 3)"
    )
    parser.add_argument("--weights", default=None, type=str, help="the weights enum name to load")
    parser.add_argument("--use_deterministic_algorithms", default=False, type=bool)
    parser.add_argument("--confusion-matrix", default = False, type=bool)
    return parser

def add_txt(output_dir, msg):
    with open(os.path.join(output_dir, "train_log.txt"), 'a') as f:
        print(os.path.join(output_dir, "train_log.txt"))
        f.write(msg)

def train_one_epoch(model, loss_func, classifier_loss, optimizer, data_loader, device, epoch, loss_optimizer, num_classes):
    print('\n[ Train epoch: %d ]' % epoch)
    model.train()
    acc_metric = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
    f1_metric = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average='macro')
    train_loss = 0
    total = 0 

    for batch_idx, (inputs, targets) in enumerate(data_loader):
        
        inputs, targets = inputs.to(device), targets
        optimizer.zero_grad()
        #loss_optimizer.zero_grad()

        embeddings = model(inputs)[0]
        preds = model(inputs)[1].cpu()
        

        loss = 0.8 * loss_func(embeddings, targets) + 0.2 * classifier_loss(preds, targets)
        loss.backward()
        optimizer.step()
        #loss_optimizer.step()
        
        train_loss += loss.item()
        total += targets.size(0)
        
        
        
        acc = acc_metric(preds, targets)
        f1 = f1_metric(preds, targets)
    
    acc = acc_metric.compute()
    f1 = f1_metric.compute()
    print(f'Epoch {epoch:<4} ,train_Loss = {train_loss / total :<10}, train_acc = {acc:<10}, train_f1 = {f1:<10}')
    
    acc_metric.reset()
    f1_metric.reset()

        
def get_all_embeddings(dataset, model):
    tester = testers.BaseTester()
    return tester.get_all_embeddings(dataset, model)


def evaluate(model, loss_func, classifier_loss, valid_loader, device, epoch, args, num_classes, test_dataset, accuracy_calculator):
    acc_metric = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
    f1_metric = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average='macro')
    confmat = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=num_classes)
    
    print('\n[ Test epoch: %d ]' % epoch)
    model.eval()
    valid_loss = 0
    total = 0
    with torch.inference_mode():
        for batch_idx, (inputs, targets) in enumerate(valid_loader):
            inputs, targets = inputs.to(device), targets
            embeddings = model(inputs)[0]
            preds = model(inputs)[1].cpu()
            
        
            
            loss = 0.8 * loss_func(embeddings, targets) + 0.2 * classifier_loss(preds, targets)
            valid_loss += loss.item()
            total += targets.size(0)
            

            acc = acc_metric(preds, targets)
            f1 = f1_metric(preds, targets)
            c_matrix = confmat(preds, targets)
            


    # (train_embeddings, _), train_labels = get_all_embeddings(train_dataset, model)
    # (test_embeddings, _) , test_labels = get_all_embeddings(test_dataset, model)
    # train_labels = train_labels.squeeze(1)
    # test_labels = test_labels.squeeze(1)
    print("Computing accuracy")
    # metric_acc = accuracy_calculator.get_accuracy(
    #     test_embeddings, test_labels, train_embeddings, train_labels, False
    # )
                
    acc = acc_metric.compute()
    f1 = f1_metric.compute()    
    print(f'Epoch {epoch:<4} ,valid_Loss = {valid_loss / total :<10}, valid_acc = {acc:<10}, valid_f1 = {f1:<10}')
    #print(f'Epoch {epoch:<4} ,valid_Loss = {valid_loss / total :<10}, valid_acc = {acc:<10}, valid_f1 = {f1:<10}, metric_acc = {metric_acc:<10}')
    
    acc_metric.reset()
    f1_metric.reset()
    confmat.reset()


    return acc, f1, c_matrix


def load_data(traindir, valdir, args):
    val_resize_size, val_crop_size, train_crop_size = (
        args.val_resize_size,
        args.val_crop_size,
        args.train_crop_size,
    )
    interpolation = InterpolationMode(args.interpolation)
    

    print("Loading training data")
    st = time.time()

    # We need a default value for the variables below because args may come
    # from train_quantization.py which doesn't define them.
    channel1 = args.channel1
    auto_augment_policy = getattr(args, "auto_augment", None)
    random_erase_prob = getattr(args, "random_erase", 0.0)
    ra_magnitude = getattr(args, "ra_magnitude", None)
    augmix_severity = getattr(args, "augmix_severity", None)
    train_dataset = torchvision.datasets.ImageFolder(
        traindir,
        presets.ClassificationPresetTrain(
            mean=args.n_mean,
            std=args.n_std,
            channel1 = channel1,
            crop_size=train_crop_size,
            interpolation=interpolation,
            auto_augment_policy=auto_augment_policy,
            random_erase_prob=random_erase_prob,
            ra_magnitude=ra_magnitude,
            augmix_severity=augmix_severity,
        ),
    )
    print("Took", time.time() - st)

    print("Loading validation data")
    if args.weights and args.val_only:
        weights = torchvision.models.get_weight(args.weights)
        preprocessing = weights.transforms()
    else:
        preprocessing = presets.ClassificationPresetEval(
            mean=args.n_mean,
            std=args.n_std,
            crop_size=val_crop_size,
            resize_size=val_resize_size, 
            interpolation=interpolation, 
            channel1=channel1
        )

    dataset_test = torchvision.datasets.ImageFolder(
            valdir,
            preprocessing,
        )


    train_sampler = torch.utils.data.RandomSampler(train_dataset)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return train_dataset, dataset_test, train_sampler, test_sampler


def main(args):
    today = time.time()
    today = time.strftime('%Y_%m_%d_%I%M%S')
    data_dir = str(args.data_path).split('/')[-1]
    output_dir = os.path.join(args.output_dir, f'{data_dir}resize_{args.val_resize_size}_model_{args.model}_{today}')
    
    os.makedirs(output_dir, exist_ok = True)
    
    device = torch.device("cuda")
    print('model is', args.model)

    interpolation = InterpolationMode(args.interpolation)

    val_resize_size, val_crop_size, train_crop_size = (
        args.val_resize_size,
        args.val_crop_size,
        args.train_crop_size,
    )

    if args.use_deterministic_algorithms:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True

    train_dir = os.path.join(args.data_path, "train")
    val_dir = os.path.join(args.data_path, "val")

    dataset, dataset_test, train_sampler, test_sampler = load_data(train_dir, val_dir, args)
    


    collate_fn = None
    num_classes = len(dataset.classes)
    print('num_classes', num_classes)
    mixup_transforms = []
    if args.mixup_alpha > 0.0:
        mixup_transforms.append(transforms.RandomMixup(num_classes, p=1.0, alpha=args.mixup_alpha))
    if args.cutmix_alpha > 0.0:
        mixup_transforms.append(transforms.RandomCutmix(num_classes, p=1.0, alpha=args.cutmix_alpha))
    if mixup_transforms:
        mixupcutmix = torchvision.transforms.RandomChoice(mixup_transforms)

        def collate_fn(batch):
            return mixupcutmix(*default_collate(batch))

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size, sampler=test_sampler, num_workers=args.workers, pin_memory=True
    )

    print("Creating model")


    if args.model == 'resnet18':
        model = models.EModel(resnet18(weights = ResNet18_Weights.IMAGENET1K_V1), len(dataset.classes))
    elif args.model == 'regnet_16gf':
        model = models.EModel(regnet_y_16gf(weights=RegNet_Y_16GF_Weights.IMAGENET1K_V2), len(dataset.classes))
    elif args.model == 'effnetv2_s':
        model = models.EModel(efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1), len(dataset.classes))
    elif args.model == 'mlp':
        model = models.MLP(len(dataset.classes))
    elif args.model == 'swinv2_t':
        model = models.EModel(swin_v2_t(weights=Swin_V2_T_Weights.IMAGENET1K_V1), len(dataset.classes))
    elif args.model == 'swinv2_s':
        model = models.EModel(swin_v2_s(weights=Swin_V2_S_Weights.IMAGENET1K_V1), len(dataset.classes))
        
    if args.transfer_learning == True:
        for param in model.parameters(): # False로 설정함으로써 마지막 classifier를 제외한 모든 부분을 고정하여 backward()중에 경사도 계산이 되지 않도록 합니다.
            param.requires_grad = False
        for name, param in model.named_parameters():
            if name == 'FC1.weight' or name == 'FC1.bias':
                param.requires_grad = True
            elif name == 'FC2.weight' or name == 'FC2.bias':
                param.requires_grad = True
        
        print('train layer is')
            
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                print(name)
        

    #model = torchvision.models.get_model(args.model, weights=args.weights, num_classes=num_classes)
    model.to(device)

    if args.distributed:
        model = torch.nn.DataParallel(model, device_ids=args.gpu)

    classifier_loss = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    accuracy_calculator = AccuracyCalculator(include=("precision_at_1",), k=1)
    loss_func = losses.SubCenterArcFaceLoss(num_classes=num_classes, embedding_size=128).to(device)

    opt_name = args.opt.lower()
    if opt_name.startswith("sgd"):
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr*0.1,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov="nesterov" in opt_name,
        ) 
        loss_optimizer = torch.optim.SGD(loss_func.parameters(), lr=args.lr)
    elif opt_name == "rmsprop":
        optimizer = torch.optim.RMSprop(
            model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, eps=0.0316, alpha=0.9
        )
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise RuntimeError(f"Invalid optimizer {args.opt}. Only SGD, RMSprop and AdamW are supported.")

    args.lr_scheduler = args.lr_scheduler.lower()
    if args.lr_scheduler == "steplr":
        main_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    elif args.lr_scheduler == "cosineannealinglr":
        main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs - args.lr_warmup_epochs, eta_min=args.lr_min
        )
    elif args.lr_scheduler == "exponentiallr":
        main_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{args.lr_scheduler}'. Only StepLR, CosineAnnealingLR and ExponentialLR "
            "are supported."
        )

    if args.lr_warmup_epochs > 0:
        if args.lr_warmup_method == "linear":
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
            )
        elif args.lr_warmup_method == "constant":
            warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
            )
        else:
            raise RuntimeError(
                f"Invalid warmup lr method '{args.lr_warmup_method}'. Only linear and constant are supported."
            )

        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[args.lr_warmup_epochs]
        )
    else:
        lr_scheduler = main_lr_scheduler

    if args.resume:
        model.load_state_dict(torch.load(args.resume))
        # if not args.test_only:
        #     optimizer.load_state_dict(checkpoint["optimizer"])
        #     lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        # args.start_epoch = checkpoint["epoch"] + 1

    if args.val_only:
        # We disable the cudnn benchmarking because it can noticeably affect the accuracy
        torch.backends.cudnn.benchmark = False
        evaluate(model, loss_func, classifier_loss=classifier_loss, valid_loader= data_loader_test, device=device, args=args, num_classes = num_classes, 
                test_dataset = load_data(train_dir, val_dir, args)[1], 
                 accuracy_calculator=accuracy_calculator, epoch=epoch)
        return

    print("Start training")
    start_time = time.time()
    max_eval_acc = 0
    max_eval_f1 = 0
    max_metric_acc = 0
    max_matrix = []
    
    
    for epoch in range(args.start_epoch, args.epochs):
        train_one_epoch(model, loss_func, classifier_loss, optimizer, data_loader, device, epoch, loss_optimizer, num_classes)
        lr_scheduler.step()
        eval_acc, eval_f1, c_matrix = evaluate(model, loss_func, classifier_loss=classifier_loss, valid_loader= data_loader_test, device=device, args=args, num_classes = num_classes, 
                 test_dataset = load_data(train_dir, val_dir, args)[1], 
                 accuracy_calculator=accuracy_calculator, epoch=epoch)


        if max_eval_acc < eval_acc:
            max_eval_acc = eval_acc
            torch.save(model.state_dict(), os.path.join(output_dir, str(args.data_path).split('/')[-1]+'_'+args.model+"_acc_best.pth"))
        if max_eval_f1 < eval_f1:
            max_eval_f1 = eval_f1
            max_matrix = c_matrix
            torch.save(model.state_dict(), os.path.join(output_dir, str(args.data_path).split('/')[-1]+'_'+args.model+"_f1_best.pth"))
        # if max_metric_acc < metric_acc:
        #     max_metric_acc = eval_acc, eval_f1, metric_acc, c_matrix
        #     torch.save(model.state_dict(), os.path.join(output_dir, str(args.data_path).split('/')[-1]+'_'+args.model+"_,metric_best.pth"))
        
        

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    add_txt(output_dir, f"Training time {total_time_str} "+ '\n')
    add_txt(output_dir, 'dataset class는' + str(dataset.classes) + '\n')
    add_txt(output_dir, args.model + ' max_acc is ' + str(max_eval_acc)+ '\n')
    add_txt(output_dir, args.model + ' max_acc is ' + str(max_eval_f1)+ '\n')
    add_txt(output_dir, args.model + ' max_acc is ' + str(max_metric_acc)+ '\n')
    if args.confusion_matrix:
        add_txt(output_dir, str(max_matrix)+'\n')
    
    

    if args.test:
        import cv2
        import glob
        from torchvision import transforms
        from torch.utils.data import Dataset, DataLoader
        class Testset(Dataset):
            def __init__(self, image_folder, transforms):        
                self.image_folder = image_folder   
                self.transforms = transforms

            def __len__(self):
                return len(self.image_folder)
            
            def __getitem__(self, index):        
                image_fn = self.image_folder[index]                                       
                image = cv2.imread(image_fn, cv2.IMREAD_COLOR)        
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                if self.transforms:            
                    image = self.transforms(image)

                return image
        
        test_list = glob.glob(args.data_path+'/test/*.png')
        test_list=sorted(test_list, key=lambda x: int(x.split('/')[-1].split('.')[0]))
        
        test_transform = transforms.Compose([
            transforms.ToTensor(), #이미지 데이터를 tensor 데이터 포멧으로 바꾸어줍니다.
            transforms.Resize([args.val_crop_size, args.val_crop_size]), #이미지의 크기가 다를 수 있으니 크기를 통일해 줍니다.
            #transforms.Normalize(mean=args.n_mean, std=args.n_std) #픽셀 단위 데이터를 정규화 시켜줍니다.
        ])
        test_data = Testset(test_list, test_transform)
        test_loader = DataLoader(test_data, batch_size=args.batch_size)
        
        model.load_state_dict(torch.load(os.path.join(output_dir, str(args.data_path).split('/')[-1]+'_'+args.model+"_f1_best.pth")))
        
        model.eval()
        test_idx = []
        test_prob = []
        pd_dict = {}
        for x in test_loader:
            x = x.to(device)
            outputs = model(x)[1]
            prob = nn.functional.softmax(outputs, dim=1)
            top_p, top_class = prob.topk(1, dim = 1)
            top_class = top_class.detach().cpu().numpy().tolist()
            prob = prob.detach().cpu().numpy().tolist()
            test_idx.extend(top_class[0])
            test_prob.extend(prob)
        
        test_list = list(map(lambda x: x.split('/')[-1], test_list))
        pd_dict['test_list'] = test_list
        pd_dict['test_idx'] = test_idx
        #pd_dict['test_prob'] = test_prob
        
        
        test_df = pd.DataFrame(pd_dict)
        print(os.path.join(output_dir, str(args.data_path).split('/')[-1] +'_'+args.model+".csv"))
        test_df.to_csv(os.path.join(output_dir, str(args.data_path).split('/')[-1] +'_'+args.model+".csv"))
        



if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)