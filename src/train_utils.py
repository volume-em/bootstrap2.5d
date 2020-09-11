import os, sys
import torch
from torch.optim.lr_scheduler import OneCycleLR
from metrics import EMAMeter, AverageMeter, calculate_iou
from tqdm import tqdm
    
class DataFetcher:
    """
    Loads batches of images and masks from a dataloader onto the gpu.
    """
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.loader_iter = iter(dataloader)

    def __len__(self):
        return len(self.dataloader)
    
    def reset_loader(self):
        self.loader_iter = iter(self.dataloader)

    def load(self):
        try:
            batch = next(self.loader_iter)
        except StopIteration:
            self.reset_loader()
            batch = next(self.loader_iter)
            
        #get the images and masks as cuda float tensors
        images = batch['image'].float().cuda(non_blocking=True)
        masks = batch['mask'].float().cuda(non_blocking=True)
        return images, masks
        
class Trainer:
    """
    Handles model training and evaluation.
    
    Arguments:
    ----------
    model: A pytorch segmentation model (e.g. DeepLabV3)
    
    optimizer. A pytorch optimizer, usually AdamW or SGD
    
    criterion: Loss function for evaluating model performance.
    
    trn_data: A pytorch dataloader object that will return pairs of images and
    segmentation masks from a training dataset
    
    val_data: A pytorch dataloader object that will return pairs of images and
    segmentation masks from a validation dataset.
    
    """
    def __init__(self, model, optimizer, criterion, trn_data, val_data=None):
        self.model = model
        self.optimizer = optimizer
        
        self.criterion = criterion
        self.train = DataFetcher(trn_data)
        if val_data is not None:
            self.valid = DataFetcher(val_data)
        else:
            self.valid = None
        
        self.trn_losses = []
        self.val_losses = []
        self.trn_ious = []
        self.val_ious = []
        
    def resume(self, checkpoint_fpath):
        """
        Sets model parameters, scheduler and optimizer states to the
        last recorded values in the given checkpoint file.
        """
        checkpoint = self.load_state(checkpoint_fpath)
        self.model.load_state_dict(checkpoint['model'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        print(f'Loaded state from {checkpoint_fpath}')
        
    def train_one_cycle(self, total_iters=10000, max_lr=1e-3, eval_iters=None, save_path=None, resume=None):
        """
        Trains model using the OneCycleLR policy.
        
        Arguments:
        ----------
        total_iters: Int. Total number of training iterations.
        
        max_lr: Float. Maximum learning rate reached during the OneCycle policy.
        
        eval_iters: Int. Frequency to print training metrics and run evaluation on the
        
        validation dataset (if there is one). If None, then evaluation is run after every
        complete pass through the training data (i.e. after 1 epoch).
        
        save_path: Str. Location to save training state of the model. If None, the model
        will not be saved.
        
        resume: Std. Path to a previously saved state file. Resumes training from the last
        iteration recorded by the scheduler. If total_iters and max_lr will be overwritten
        if they are different from the ones recorded in the state file.
        
        """
        #wrap the optimizer in the OneCycleLR policy
        self.scheduler = OneCycleLR(self.optimizer, max_lr=max_lr, total_steps=total_iters)

        if resume is not None:
            self.resume(resume)
            
        #move the model to cuda
        self.model = self.model.cuda()
        print('Moved model to cuda device')
        
        #if no eval iters are given, use 1 epoch as evaluation period
        if eval_iters is None:
            eval_iters = len(self.train)
        
        #wrap the optimizer in the OneCycleLR policy
        last_iter = self.scheduler.last_epoch
        last_iter = 1 if last_iter == 0 else last_iter

        loss_meter = EMAMeter()
        iou_meter = EMAMeter()
        for ix in tqdm(range(last_iter, total_iters + 1), file=sys.stdout):
            images, masks = self.train.load()
            
            #run a training step
            self.model.train()
            self.optimizer.zero_grad()
            
            #forward pass
            output = self.model(images)
            loss = self.criterion(output, masks)
            
            #backward pass
            loss.backward()
            self.optimizer.step()
            
            #update the optimizer schedule
            self.scheduler.step()
            
            #record loss and iou
            loss_meter.update(loss.item())
            iou_meter.update(calculate_iou(output.detach(), masks.detach()))

            if (ix % eval_iters == 0):
                #print the training loss and ious and reset the meters
                print(f'train_loss: {loss_meter.avg}')
                print(f'train_mean_iou: {iou_meter.avg}')
                loss_meter.reset()
                iou_meter.reset()
                
                #run evaluation step if there is validation data
                if self.valid is not None:
                    rl = self.evaluate()

                #save the current training state
                if save_path is not None:
                    self.save_state(save_path)
                    print(f'State saved to {save_path}')

    def evaluate(self):
        """
        Runs model inference on validation data and prints the average loss
        and iou for the dataset.
        """
        loss_meter = AverageMeter()
        iou_meter = AverageMeter()
        
        self.model.eval()
        for _ in range(len(self.valid)):
            with torch.no_grad(): #necessary to prevent CUDA memory errors
                #load the next batch of validation data
                images, masks = self.valid.load()
                
                #forward pass in eval mode
                output = self.model(images)
                loss = self.criterion(output, masks)
                
                #record loss and iou
                loss_meter.update(loss.item())
                iou_meter.update(calculate_iou(output.detach(), masks.detach()))
                
        #print the loss and iou, no need to reset the meters
        print(f'valid_loss: {loss_meter.avg}')
        print(f'valid_mean_iou: {iou_meter.avg}')
    
    def save_state(self, save_path):
        """
        Saves the current training state, including model, scheduler, and optimizer.
        
        Arguments:
        ------------
        
        save_path: Path to save pytorch file

        """
        
        state = {
            'model': self.model.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        
        torch.save(state, save_path)
        
    def load_state(self, state_path):
        """
        Loads the saved training state to cpu
        
        Arguments:
        ------------
        
        state_path: Path of pytorch file to load

        """
        return torch.load(state_path, map_location='cpu')