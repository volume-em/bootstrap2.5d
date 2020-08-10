import sys, os
import numpy as np
import torch
import torch.nn as nn
import pickle
import functools, traceback
import torchvision.transforms.functional as TF
from torch.optim.lr_scheduler import LambdaLR
import copy
from skimage.measure import label, regionprops, marching_cubes_lewiner, mesh_surface_area
from skimage import morphology as morph
from PIL import Image as pil
from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import OneCycleLR
import mlflow

#we want to use the right version of tqdm based on the 
#environment, if imported in a jupyter notebook we use
#tqdm_notebook
program_name = os.path.basename(os.getenv('_', ''))

in_notebook = (
    'jupyter-notebook' in program_name or
    'ipython'          in program_name or
    'JPY_PARENT_PID'   in os.environ
)

#if in_notebook:
#    from tqdm.notebook import tqdm
#else:
from tqdm import tqdm

def gpu_mem_restore(func):
    """
    Reclaim GPU RAM if CUDA out of memory happened, or execution was interrupted
    Stolen from fastai.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except:
            type, val, tb = sys.exc_info()
            traceback.clear_frames(tb)
            raise type(val).with_traceback(tb) from None
    return wrapper
        
class Trainer:
    """
    Trainer class handles the training loop of a model along
    with a few other tools like the lr_finder and methods for saving and loading models.
    
    Arguments:
    -----------
    
    model: A pytorch model (model device will use all GPUs or cpu! Must change this to use only 1
    GPU, if you have 2!)
    optimizer: A pytorch optimizer compiled with model parameters
    loss: A pytorch loss function compiled previously
    trn_data: A pytorch dataloader. Currently each data batch dict must have 'im' and 'msk'
    keys. This can be modified for the given training scenario.
    val_data: A pytorch dataloader, default None. Currently each data batch dict must have 
    'im' and 'msk' keys. This can be modified for the given training scenario.
    metrics: A composed metrics class (see Compose in metrics.py), default None
    
    Public attributes:
    -------------
    
    device
    model
    trn_metrics, val_metrics
    trn_losses, val_losses
    
    Public methods:
    -------------
    
    train
    evaluate
    lr_finder
    save_model
    load_model
    
    Example:
    -------------
    
    model = UResnet()
    optimizer = torch.optim.Adam(lr=1e-3)
    loss = nn.BCELoss()
    trn_data = DataLoader(trn_dataset, batch_size=64, shuffle=True) #from torch.utils.data
    val_data = DataLoader(val_dataset, batch_size=64, shuffle=False) #from torch.utils.data
    metrics = Compose({'IoU': IoU()}) #see metrics.py
    
    trainer = Trainer(model, optimizer, loss, trn_data, val_data, metrics)
    trainer.train(epochs=10, train_iters_before_val=len(trn_data))
    """
    
    def __init__(self, model, optimizer, loss, trn_data, val_data=None, metrics=None, logging=None,
                 val_masking=False, freeze_bn=False):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        
        self.model = model.to(self.device)
        print("Model loaded onto " + self.device.type)
        self.optimizer = optimizer
        
        self.loss = loss.cuda()
        self.trn_data = Prefetcher(trn_data)
        self.val_data = val_data
        self.freeze_bn = freeze_bn
        
        self.trn_metrics = metrics
        self.val_metrics = copy.deepcopy(self.trn_metrics)
        if val_masking:
            self.val_metrics.metrics_dict['IoU'].masking = True

        self.trn_losses = []
        self.val_losses = []
        
        #instantiate our logging parameters if there are any
        self.logging = logging
        if self.logging is not None:
            #everytime that Trainer is instantiated we want to
            #end the current active run and let a new one begin
            mlflow.end_run()
            #first we need to extract the experiment name so that
            #we know where to save our files, if experiment name
            #already exists, we'll use it, otherwise we create a
            #new experiment
            mlflow.set_experiment(self.logging['experiment_name'])

            #to make things easy, we'll log all the parameters in the
            #logging_params dict (including experiment_name)
            #by default we resume the latest active run, or create
            #a new run if there are no active runs
            mlflow.log_params(self.logging)
    
    #@gpu_mem_restore
    def train(self, epochs, train_iters_per_epoch, save_path=None):
        """
        Defines a pytorch style training loop for the model withtqdm progress bar
        for each epoch and handles printing loss/metrics at the end of each epoch.
        
        epochs: Number of epochs to train model
        train_iters_per_epoch: Number of training iterations is each epoch. Reducing this 
        number will give more frequent updates but result in slower training time.
        
        Results:
        ----------
        
        After train_iters_per_epoch iterations are completed, it will evaluate the model
        on val_data if there is any, then prints loss and metrics for train and validation
        datasets.
        """
        
        for e in range(epochs):
            print(f'Epoch {e + 1}')
            self._one_epoch(train_iters_per_epoch)
            
            #if self.val_data is not None:
                #if self.val_losses[-1] <= min(self.val_losses):
                    
            if save_path:
                self.save_model(save_path)
                print("model saved to {}".format(save_path))
                    
                        
    #@gpu_mem_restore
    def train_one_cycle(self, total_iters=10000, max_lr=1e-3,
                        eval_iters=None, save_path=None):
        
        #if no eval iters are given, use 1 epoch as evaluation period
        if eval_iters is None:
            eval_iters = len(self.train)
        
        #wrap the optimizer in the OneCycleLR policy
        scheduler = OneCycleLR(self.optimizer, max_lr, total_steps=total_iters)
        
        rl = 0
        avg_mom = 0.98
        for ix in tqdm(range(1, total_iters + 1), file=sys.stdout):
            #train a batch of data
            loss = self._train_1_batch()
            
            #update the optimizer schedule
            scheduler.step()

            rl = rl * avg_mom + loss * (1 - avg_mom)
            debias_loss = rl / (1 - avg_mom ** (ix + 1))
            
            if (ix  % eval_iters == 0):
                self.trn_losses.append(debias_loss)
                print('Loss {}'.format(debias_loss))

                if self.trn_metrics is not None:
                    self.trn_metrics.close_epoch()
                    self.trn_metrics.print()
                
                if self.logging is not None:
                    self.log_metrics(ix, dataset='train')

                if self.val_data is not None:
                    rl = self.evaluate()

                    self.val_losses.append(rl)
                    print('Val loss {}'.format(rl))

                    if self.val_metrics is not None:
                        self.val_metrics.close_epoch()
                        self.val_metrics.print()
                        
                    if self.logging is not None:
                        self.log_metrics(ix, dataset='valid')

                if save_path is not None:
                    self.save_model(save_path)
                    print('State saved to {}'.format(save_path))
                    
    def train_poly(self, total_iters=30e3, lr=1e-2, power=0.9,
                   eval_iters=None, save_path=None):
        
        #if no eval iters are given, use 1 epoch as evaluation period
        if eval_iters is None:
            eval_iters = len(self.train)
        
        #wrap the optimizer in the OneCycleLR policy
        lambda1 = lambda iteration: (1 - (iteration / total_iters)) ** power
        scheduler = LambdaLR(self.optimizer, lambda1)
        
        rl = 0
        avg_mom = 0.98
        for ix in tqdm(range(1, total_iters + 1), file=sys.stdout):
            #train a batch of data
            loss = self._train_1_batch()
            
            #update the optimizer schedule
            scheduler.step()

            rl = rl * avg_mom + loss * (1 - avg_mom)
            debias_loss = rl / (1 - avg_mom ** (ix + 1))
            
            if (ix  % eval_iters == 0):
                self.trn_losses.append(debias_loss)
                print('Loss {}'.format(debias_loss))

                if self.trn_metrics is not None:
                    self.trn_metrics.close_epoch()
                    self.trn_metrics.print()
                
                if self.logging is not None:
                    self.log_metrics(ix, dataset='train')

                if self.val_data is not None:
                    rl = self.evaluate()

                    self.val_losses.append(rl)
                    print('Val loss {}'.format(rl))

                    if self.val_metrics is not None:
                        self.val_metrics.close_epoch()
                        self.val_metrics.print()
                        
                    if self.logging is not None:
                        self.log_metrics(ix, dataset='valid')

                if save_path is not None:
                    self.save_model(save_path)
                    print('State saved to {}'.format(save_path))
    
    def train_step_decay(self, epochs, iters_per_epoch, decay_epochs, decay_factor=0.1, save_path=None):
        for e in range(epochs):
            print(f'Epoch {e}')
            if e in decay_epochs:
                for pg in self.optimizer.param_groups:
                    pg['lr'] *= decay_factor
                    
            self._one_epoch(train_iters_per_epoch=iters_per_epoch)
            
            if self.val_data is not None:
                if save_path:
                    self.save_model(save_path)
                    print('State saved to {}'.format(save_path))
        
    def log_metrics(self, step, dataset='train'):
        #get the corresponding losses and metrics dict
        if dataset == 'train':
            losses = self.trn_losses
            metric_dict = self.trn_metrics.metrics_dict
        elif dataset == 'valid':
            losses = self.val_losses
            metric_dict = self.val_metrics.metrics_dict
            
        #log the last loss, using the dataset name as a prefix
        mlflow.log_metric(dataset + '_loss', losses[-1], step=step)
        #log all the metrics in our dict, using dataset as a prefix
        metrics = {}
        for k,v in metric_dict.items():
            for ix,class_name in enumerate(self.trn_metrics.class_names):
                metrics[dataset + '_' + class_name + '_' + k] = float(v.history[-1][ix])
        mlflow.log_metrics(metrics, step=step)
            
        
    def _one_epoch(self, train_iters_per_epoch):
        rl = 0.0
        avg_mom = 0.98
        
        if train_iters_per_epoch is None:
            train_iters_per_epoch = len(self.trn_data)
        
        for it in tqdm(range(train_iters_per_epoch)):
            loss = self._train_1_batch()
            rl = rl * avg_mom + loss * (1 - avg_mom)
            debias_loss = rl / (1 - avg_mom ** (it + 1))
        
        self.trn_losses.append(debias_loss)
        print('Loss {}'.format(debias_loss))
        
        if self.trn_metrics is not None:
            self.trn_metrics.close_epoch()
            self.trn_metrics.print()
            
        #log loss and metrics if logging is turned on
        if self.logging is not None:
            self.log_metrics(None, dataset='train')
        
        if self.val_data is not None:
            rl = self.evaluate()

            self.val_losses.append(rl)
            print('Val loss {}'.format(rl))

            if self.val_metrics is not None:
                self.val_metrics.close_epoch()
                self.val_metrics.print()
                
            #log loss and metrics if logging is turned on
            if self.logging is not None:
                self.log_metrics(None, dataset='valid')
        
    #@gpu_mem_restore
    def evaluate(self):
        """
        Evaluation method used at the end of each epoch. Not intended to
        generate predictions for validation dataset, it only returns average loss
        and stores metrics for validaiton dataset.
        
        Use Validator class for generating masks on a dataset.
        """
        rl = 0.0
        
        val_iter = Prefetcher(self.val_data)
        
        for _ in range(len(self.val_data)):
            with torch.no_grad(): #necessary to prevent CUDA memory errors
                im, msk = val_iter.next()
                output = self.model.eval()(im)
                l = self.loss(output, msk)
                rl += l.item()
            
            if self.val_metrics is not None:
                self.val_metrics.evaluate(output.detach().cpu(), msk.detach().cpu())
                
        #pred = nn.Sigmoid()(output) > 0.5
        #f, ax = plt.subplots(1, 4, figsize=(12, 6))
        #for i in range(4):
        #    ax[i].imshow(im.detach().cpu().numpy()[-i, 0], cmap='gray')
        #    ax[i].imshow(msk.detach().cpu().numpy()[-i, 0], alpha=0.3, cmap='Blues')
        #    ax[i].imshow(pred.detach().cpu().numpy()[-i, 0], alpha=0.3, cmap='Oranges')
                
        return rl / len(self.val_data)
    
    def _train_1_batch(self, use_metrics=True):
        im, msk = self._get_batch(self.trn_data)
        
        self.model.train()
        
        if self.freeze_bn:
            def set_bn_eval(module):
                if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                    for param in module.parameters():
                        requires_grad = param.requires_grad
                    if not requires_grad:
                        module.eval()

            self.model.encoder.apply(set_bn_eval)
        
        self.optimizer.zero_grad()
        output = self.model(im)
        #print(output.size(), msk.size())
        l = self.loss(output, msk)
        l.backward()
        self.optimizer.step()
        
        #if msk.ndim == 4:
        #    msk = msk[..., 0].long()
        
        if use_metrics:
            if self.trn_metrics is not None:
                self.trn_metrics.evaluate(output.detach().cpu(), msk.detach().cpu())
        
        return l.item()
            
    def _get_batch(self, dataloader):
        #data = iter(dataloader).next()
        #im = data['im'].to(self.device)
        #msk = data['msk'].to(self.device)
        
        #return im, msk
        return dataloader.next()
    
    def lr_finder(self, start_lr, end_lr, wd=0, rate=0.2, beta=0.5):
        """
        Creates a copy of the model and optimizer and then exponentially increases 
        the optimizer learning rate after each iteration, recording the loss as a
        smoothened moving average.
        
        Arguments:
        ------------
        
        start_lr: Lower bound learning rate
        end_lr: Upper bound learning rate
        wd: Value of weight decay for testing, if applicable. Default is 0.
        TODO: Allow wd to be a list or values as well
        rate: Rate of exponential increase. Should be in range (0, 1). Smaller
        rate will give better graph resolution, but cost extra time. Default is 0.2.
        beta: Factor for smoothing the moving average, between (0, 1). If results are jagged,
        consider increasing the value of beta. Default is 0.5.
        
        Example:
        ------------
        
        trainer = Trainer(...)
        lrs, losses = trainer.lr_finder(1e-8, 1e-1)
        plt.semilogx(lrs, losses)
        """
        lrs = [start_lr]
        mla = [] #loss moving average at each step
        
        for pg in self.optimizer.param_groups:
            pg['lr'] = start_lr
            #not all optimizers have weight decay
            try:
                pg['weight_decay'] = wd
            except:
                pass
            
        smoothener = SmoothenValue(beta)
        
        iters = int((1. / rate) * np.log(end_lr / start_lr))
        
        l = self._train_1_batch(use_metrics=False)
        smooth_loss = smoothener.add_value(l)
        mla.append(smoothener.smooth)
        
        for i in tqdm(range(iters)):
            for pg in self.optimizer.param_groups:
                pg['lr'] *= np.exp(rate)
                lrs.append(pg['lr'])
        
            l = self._train_1_batch(use_metrics=False)
            smooth_loss = smoothener.add_value(l)
            mla.append(smoothener.smooth)
            
            if l > (mla[0] * 2):
                print("Stopping, loss doubled from first iteration!!")
                break
            
        return lrs, mla
    
    def save_model(self, save_path):
        """
        Saves the self.model state dict
        
        Arguments:
        ------------
        
        save_path: Path of .pt file for saving
        
        Example:
        ----------
        
        trainer = Trainer(...)
        trainer.save_model(model_path + 'new_model.pt')
        """
        state = {'state_dict': self.model.state_dict(),
                 'run_id': mlflow.active_run().info.run_id if self.logging is not None else None,
                 'experiment': self.logging['experiment_name'] if self.logging is not None else None}
        
        torch.save(state, save_path)
        
    def load_model(self, model_path):
        """
        Loads the saved state dict into self.model object
        
        Arguments:
        ------------
        
        model_path: Path of .pt file for loading
        
        Example:
        -----------
        
        learner = Trainer(...)
        learner.load_model(model_path + 'new_model.pt')
        """
        if self.device.type == 'cpu':
            self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        else:
            self.model.load_state_dict(torch.load(model_path))
            
    def log(self, logging_params):
        #this function assumes that we already checked
        #if logging params exist or not
        assert (logging_params is not None), 'Cannot perform logging without parameters!'
        
        #first we need to extract the experiment name so that
        #we know where to save our files, if experiment name
        #already exists, we'll use it, otherwise we create a
        #new experiment
        mlflow.set_experiment(logging_params['experiment_name'])
        
        #to make things easy, we'll log all the parameters in the
        #logging_params dict (including experiment_name)
        mlflow.log_params(logging_params)
    
class Validator:
    """
    Given a model, performs validation on a given dataset
    """
    
    def __init__(self, model, dataset, norms, metric):
        self.model = model
        self.dataset = dataset
        self.norms = norms
        self.metric = metric
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        #calculate metrics for whole dataset
        self.metric.clear_history()
        self.calc_metric()
        
        #there should only be one metric anyway
        #key = list(self.metric.metrics_dict.keys())[0]

        #each history is a torch tensor of size (n_classes, )
        #we'll save the results as (n_samples, n_classes)
        self.results = torch.stack(self.metric.history, dim=0)
        self.n_classes = self.results.shape[1]
        
    def calc_metric(self):
        #dataset is NOT a dataloader
        for data in tqdm(self.dataset):
            with torch.no_grad(): #necessary to prevent CUDA memory errors
                #unqueeze to add batch dimension that model expects
                im = data['im'].unsqueeze(0).to(self.device)
                msk = data['msk'].unsqueeze(0).to(self.device)
                output = self.model.eval()(im)

            self.metric.calculate(output.detach().cpu(), msk.detach().cpu())
            self.metric.epoch_end()
            
    def predict_masks(self, image_batch, threshold=0.5, class_id=0):
        #dataset is NOT a dataloader
        with torch.no_grad(): #necessary to prevent CUDA memory errors
            #unqueeze to add batch dimension that model expects
            im = image_batch.to(self.device)
            output = self.model.eval()(im).detach().cpu()
            
            #if there's only 1 class then apply the sigmoid
            #else apply softmax and only return the mask for the
            #given class_id
            if self.n_classes == 1:
                prediction = nn.Sigmoid()(output) > threshold
            else:
                prediction = nn.Softmax(dim=1)(output)#[:, class_id].unsqueeze(1)
                
                #argmax the prediction
                prediction = torch.argmax(prediction, 1)
                
        
        return prediction
    
    def plot_masked_images(self, images, masks, preds):
        
        num = len(images)
        cols = 3
        
        if num % 3 == 0:
            rows = int(num / cols)
        else:
            rows = int(num / cols) + 1
        
        f, ax = plt.subplots(rows, cols, figsize=(16, 16))
        c = 0
        for y in range(rows):
            for x in range(cols):
                #if image has 3 channels, then only take the first
                img = images[c]
                if (img.size(0) == 3):
                    img = img[0]
                else:
                    img = img.squeeze()
                
                pred = preds[c].squeeze() * (masks[c].squeeze() > 0)
                ax[y, x].imshow(img, cmap='gray')
                ax[y, x].imshow(masks[c].squeeze(), cmap='rainbow', alpha=0.3)
                ax[y, x].imshow(pred.squeeze(), cmap='cool', alpha=0.3)
                c += 1
    
    def show_best(self, n=6, threshold=0.5, class_id=0):
        if self.n_classes == 1:
            best_idx = np.argsort(self.results[:, 0])[-n:]
        else:
            best_idx = np.argsort(self.results.mean(1))[-n:]
        
        images, masks = [], []
        for idx in best_idx:
            data = self.dataset[idx]
            images.append(data['im'])
            masks.append(data['msk'])
            
        image_batch = torch.stack(images, dim=0)
        mask_batch = torch.stack(masks, dim=0)
        preds_batch = self.predict_masks(image_batch, threshold, class_id)
        
        #preds_batch = preds_batch * (mask_batch > 0).type(preds_batch.dtype)
        
        self.plot_masked_images(image_batch, mask_batch, preds_batch)
        
    def show_worst(self, n=6, exclude_null=True, threshold=0.5, class_id=0):
        if (exclude_null) & (self.n_classes == 1):
            indices = np.where(self.results[:, 0] > 0)
            worst_idx = np.argsort(self.results[:, 0][indices])[:n]
        elif (exclude_null):
            indices = np.where(self.results.mean(1) > 0)
            worst_idx = np.argsort(self.results.mean(1)[indices])[:n]
        elif (self.n_classes == 1):
            worst_idx = np.argsort(self.results[:, 0][:n])
        else:
            worst_idx = np.argsort(self.results.mean(1)[:n])
        
        images, masks = [], []
        for idx in worst_idx:
            data = self.dataset[idx]
            images.append(data['im'])
            masks.append(data['msk'])
            
        image_batch = torch.stack(images, dim=0)
        mask_batch = torch.stack(masks, dim=0)
        preds_batch = self.predict_masks(image_batch, threshold, class_id)
        
        #preds_batch = preds_batch * (mask_batch > 0).type(preds_batch.dtype)
        
        self.plot_masked_images(image_batch, mask_batch, preds_batch)
        
    def show_random(self, n=6, threshold=0.5, class_id=0):
        if (self.n_classes == 1):
            random_idx = np.random.choice(range(len(self.results[:, 0])), n)
        else:
            random_idx = np.random.choice(range(len(self.results.mean(1))), n)
        
        images, masks = [], []
        for idx in random_idx:
            data = self.dataset[idx]
            images.append(data['im'])
            masks.append(data['msk'])
                        
        image_batch = torch.stack(images, dim=0)
        mask_batch = torch.stack(masks, dim=0)
        preds_batch = self.predict_masks(image_batch, threshold, class_id)
        
        #preds_batch = preds_batch * (mask_batch > 0).type(preds_batch.dtype)
        
        self.plot_masked_images(image_batch, mask_batch, preds_batch)
        
        
    def show_augmented(self, augmentations, image_index, class_id):
        data = self.dataset[image_index]
        image = data['im']
        mask = data['msk']
        pred = self.predict_masks(image.unsqueeze(0))
        
        #return image and mask to original state
        aug_img = self.denormalize(image) * 255
        aug_img = np.rollaxis(aug_img.numpy().astype(np.uint8), 0, 3)
        aug_msk = mask.squeeze().numpy().astype(np.uint8)
        
        auged = augmentations(image=aug_img, mask=aug_msk)
        aug_img = auged['image']
        aug_msk = auged['mask'] == class_id
        
        aug_pred = self.predict_masks(aug_img.unsqueeze(0), class_id=class_id)
        
        f, ax = plt.subplots(1, 2, figsize=(12, 8))
        
        ax[0].imshow(image.squeeze(), cmap='gray')
        ax[0].imshow(mask.squeeze(), cmap='Blues', alpha=0.3)
        ax[0].imshow(pred.squeeze(), cmap='Oranges', alpha=0.3)
        ax[1].imshow(aug_img.squeeze(), cmap='gray')
        ax[1].imshow(aug_msk.squeeze(), cmap='Blues', alpha=0.3)
        ax[1].imshow(aug_pred.squeeze(), cmap='Oranges', alpha=0.3)
            
            
    def denormalize(self, img):
        """
        Standard transform on each dataset is to normalize by training dataset mean and stdev pixel.
        This normalization needs to be reversed before convert a torch tensor into a PIL image.
        
        Most methods in this class assume denormalization is required. The easiest way to 
        bypass this is to set self.norms to mean=0 and stdev=1.
        
        """
        
        return img * self.norms[1] + self.norms[0]
        
            
        
class TTA:
    """
    Given a model and dataset, performs test time augmentation and returns averaged results.
    
    Arguments:
    ------------
    
    norms: Mean and standard deviation of pixel intensity from training dataset. Generally, 
    normalization is applied as a transform on the Pytorch dataset. If no normalization 
    occurs on dataset, use mean=0 and stdev=1.
    hflip: Horizontally flip image and predict mask. True or False, default is False.
    vflip: Vertically flip image and predict mask. True or False, default is False.
    angles: List of angles at which to rotate image and predict mask. Any integer in range [0, 360].
    Default is None.
    scales: List of image scales at which to make predictions. Image is resized to a new dimension
    equal to scale * image_size. In order to prevent errors, the actual scaled dimensions are set to
    the nearest factor of 16.
    
    """
    
    def __init__(self, norms=None, hflip=False, vflip=False, scales=None):
        self.norms = norms
        self.hflip = hflip
        self.vflip = vflip
        self.scales = scales
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    #@gpu_mem_restore
    def batch_predict(self, model, img_batch):
        pmasks = []
        with torch.no_grad(): #necessary to prevent CUDA memory errors

            #first we get a regular prediction for the image batch and append it to the 
            #predicted mask list
            msk = self.predict_mask(model, img_batch)
            pmasks.append(msk.squeeze().numpy())
            
            #need to put image in range [0,1] or pil image will be all noise
            #because pil will just zero negative numbers
            #to do that we need to denormalize the image
            if self.norms is not None:
                denorm_im = self.denormalize(img_batch)
                #now we have our pil_image object that we can use in each augmentation step
                batch_im = [TF.to_pil_image(im) for im in denorm_im]

            #procedure for each augmentation is:
            #apply_augmentation, predict_mask, invert_augmentation, append to pmasks
            if self.hflip:
                batch_hflip_im = torch.stack(
                    [self.apply_augmentation(im, TF.hflip) for im in batch_im], dim=0
                )
                batch_hflip_msk = self.predict_mask(model, batch_hflip_im)
                if batch_hflip_msk.size(0) == 1: 
                    batch_hflip_msk = batch_hflip_msk.unsqueeze(0)
                    
                msk = torch.stack(
                    [self.invert_augmentation(hflip_msk, TF.hflip) 
                     for hflip_msk in batch_hflip_msk], dim=0
                )
                pmasks.append(msk.squeeze().numpy())  
                
            if self.vflip:
                batch_vflip_im = torch.stack(
                    [self.apply_augmentation(im, TF.vflip) for im in batch_im], dim=0
                )
                batch_vflip_msk = self.predict_mask(model, batch_vflip_im)
                if batch_vflip_msk.size(0) == 1: 
                    batch_vflip_msk = batch_vflip_msk.unsqueeze(0)
                    
                msk = torch.stack(
                    [self.invert_augmentation(vflip_msk, TF.vflip) 
                     for vflip_msk in batch_vflip_msk], dim=0
                )
                pmasks.append(msk.squeeze().numpy())
                
            if self.scales is not None:
                im = batch_im[0]
                for scale in self.scales:
                    ideal_size = (int(scale * im.size[1]), int(scale * im.size[0]))
                    scaled_size = (int(ideal_size[0] / 16) * 16, int(ideal_size[1] / 16) * 16)
                    original_size = (im.size[1], im.size[0])
                    
                    batch_scaled_im = torch.stack(
                        [self.apply_augmentation(im, TF.resize, size=scaled_size) 
                         for im in batch_im], dim=0
                    )
                    batch_scaled_msk = self.predict_mask(model, batch_scaled_im)
                    if batch_scaled_msk.size(0) == 1: 
                        batch_scaled_msk = batch_scaled_msk.unsqueeze(0)
                        
                    msk = torch.stack(
                        [self.invert_augmentation(scaled_msk, TF.resize, size=original_size) 
                         for scaled_msk in batch_scaled_msk], dim=0
                    )
                    pmasks.append(msk.squeeze().numpy())
                
        return np.array(pmasks)
    
    #@gpu_mem_restore
    def predict(self, model, img):
        """
        The main TTA prediction method. First makes a prediction on unaugmented img and then
        performs each augmentation specified when TTA object was created. 
        After each augmentation, it performs a prediction using the given model
        and inverts the augmentation so that all prediction masks are correctly aligned.
        
        model: A Pytorch model object
        img: A torch tensor image with dimensions (channels, height, width).
        
        Returns:
        --------------
        
        A numpy array of Sigmoid activated prediction masks (i.e. pixels are in range [0, 1]). 
        Shape is (num_augmentations+1, height, width).
        
        Example:
        --------------
        
        tta = TTA(hflip=True, scales=[0.8, 1.2])
        img = dataset[index]['im']
        pmasks = tta.predict(model, img)
        
        img.shape:
        (1, 224, 224)
        pmasks.shape
        (4, 224, 224)
        
        
        """
        
        pmasks = []
        with torch.no_grad(): #necessary to prevent CUDA memory errors

            #first we get a regular prediction for the image and append it to the 
            #predicted mask list
            msk = self.predict_mask(model, img)
            pmasks.append(msk.squeeze().numpy())
            
            #need to put image in range [0,1] or pil image will be all noise
            #because pil will just zero negative numbers
            #to do that we need to denormalize the image
            #this method works on image data with a batch dimension
            if self.norms is not None:
                denorm_im = self.denormalize(img)
                #now we have our pil_image objects that we can use in each augmentation step
                im = TF.to_pil_image(denorm_im)

            #procedure for each augmentation is:
            #apply_augmentation, predict_mask, invert_augmentation, append to pmasks
            if self.hflip:
                hflip_im = self.apply_augmentation(im, TF.hflip)
                hflip_msk = self.predict_mask(model, hflip_im)
                msk = self.invert_augmentation(hflip_msk, TF.hflip)
                pmasks.append(msk.squeeze().numpy())

            if self.vflip:
                vflip_im = self.apply_augmentation(im, TF.vflip)
                vflip_msk = self.predict_mask(model, vflip_im)
                msk = self.invert_augmentation(vflip_msk, TF.vflip)
                pmasks.append(msk.squeeze().numpy())
                
            if self.scales is not None:
                for scale in self.scales:
                    ideal_size = (int(scale * im.size[1]), int(scale * im.size[0]))
                    scaled_size = (int(ideal_size[0] / 16) * 16, int(ideal_size[1] / 16) * 16)
                    original_size = (im.size[1], im.size[0])
                    
                    scaled_im = self.apply_augmentation(im, TF.resize, size=scaled_size)
                    scaled_msk = self.predict_mask(model, scaled_im)
                    msk = self.invert_augmentation(scaled_msk, TF.resize, size=original_size)
                    pmasks.append(msk.squeeze().numpy())
                
        return np.array(pmasks)
    
    def average_masks(self, pmasks):
        """
        Given a numpy array of predicted masks, averages the predictions and rounds.
        
        pmasks: Numpy array of mask predictions in range [0, 1]
        
        Example:
        ------------------
        
        pmasks = np.array([[[0.9, 0.5, 0.1]], [[0.8, 0.8, 0.8]]]) #size is 2x1x3
        avg_mask = self.average_masks(pmasks)
        
        avg_mask:
        np.array([
        [0.85, 0.65, 0.45]
        ]) #size is 1x3

        """
        
        avg_mask = np.sum(pmasks, axis=0) / len(pmasks)
        
        return avg_mask
    
    def predict_mask(self, model, img):
        """
        This method makes a prediction on a single image and applies the Sigmoid activation
        
        model: A pytorch model object
        img: A torch tensor image with dimensions (channels, height, width)
        keep_batch_dim: If True, the first dimension is not squeezed, this option
        should only be true if called by the batch_predict method.
        
        NOTE: This function should only be called within a torch.no_grad() scope. This prevents
        memory errors that arise from storing gradients.
        
        Example:
        ---------------
        
        with torch.no_grad():
            mask = self.predict_mask(self.model, img)
        
        """
        
        
        if len(img.size()) == 3:
            img = img.unsqueeze(0)
        
        im = img.to(self.device)
        output = model.eval()(im)
        msk = nn.Sigmoid()(output).squeeze(0)
        
        return msk.detach().cpu()
    
    def apply_augmentation(self, pil_image, pil_func_transform, **kwargs):
        """
        This method takes a pil image and applies the given transform using
        the given arguments:
        
        pil_image: A pil_image object
        pil_func_transform: A torchvision functional transform that operates on a pil image. 
        Only pass the function name (e.g. TF.resize, TF.rotate), arguments for 
        the transform should be included in **kwargs. To find arguments for the transform, 
        see torchvision documentation.
        **kwargs: The keyword arguments for the given transform.
        
        Example:
        --------------
        
        aug_image = self.apply_augmentation(pil_image, TF.resize, size=(256, 256))
        
        or:
        
        aug_image = self.apply_augmentation(pil_image, TF.rotate, angle=angle, expand=True)
        
        Returns:
        --------------
        
        aug_image: The torch tensor of the augmented image. To be fed directly into the model.
        
        
        """
        
        #first let's define our augmentation as a lambda function
        aug = lambda x: pil_func_transform(x, **kwargs)
        
        #next we will apply the augmentation
        aug_img = aug(pil_image)
        
        #then we need to convert the image to a torch tensor
        aug_img = TF.to_tensor(aug_img)
        
        #lastly return the renormalized image
        return self.renormalize(aug_img)
    
    def invert_augmentation(self, torch_image_tensor, pil_func_transform, **kwargs):
        """
        This method takes a torch image tensor and applies the given transform using
        the given arguments. Key difference between apply and invert augmentation methods
        is that invert accepts a torch_image_tensor and does not perform renormalization.
        
        NOTE: torch_image_tensor must be in range [0, 1] or pil transforms will result in noise.
        This can be accomplished with the denormalize method. Mask images generated by the 
        predict_mask method are already in the range [0, 1] because of the Sigmoid activation.
        
        torch_image_tensor: A torch tensor with shape (channels, height, width). 
        Note: There cannot be a batch dimension. For this class, the torch_image_tensor 
        will generally be the predicted mask.
        
        Intensities must be scaled from [0, 1] (see note above).
        pil_func_transform: A torchvision functional transform that operates on a pil image. 
        Only pass the function name (e.g. TF.resize, TF.rotate), arguments for the transform 
        should be included in **kwargs. 
        To find arguments for the transform, see torchvision documentation.
        **kwargs: The keyword arguments for the given transform.
        
        Example:
        --------------
        
        inv_aug_image = self.invert_augmentation(torch_image_tensor, TF.resize, size=(256, 256))
        
        or:
        
        inv_aug_image = self.invert_augmentation(torch_image_tensor, TF.rotate, 
        angle=angle, expand=True)
        
        Returns:
        --------------
        
        inv_aug_image: The torch tensor of the augmented image.
        
        
        """
        
        #first let's define our augmentation as a lambda function
        aug = lambda x: pil_func_transform(x, **kwargs)
        
        #next convert our torch image tensor into a pil image
        pil_image = TF.to_pil_image(torch_image_tensor)
        
        #now we apply the augmentation
        aug_img = aug(pil_image)
        
        #then we need to convert the image to a torch tensor
        aug_img = TF.to_tensor(aug_img)
        
        #lastly return augmented torch image tensor
        return aug_img
        
    
    def denormalize(self, img):
        """
        Standard transform on each dataset is to normalize by training dataset mean and stdev pixel.
        This normalization needs to be reversed before convert a torch tensor into a PIL image.
        
        Most methods in this class assume denormalization is required. The easiest way to bypass 
        this is to set self.norms to mean=0 and stdev=1.
        
        """
        
        return img * self.norms[1] + self.norms[0]
    
    def renormalize(self, img):
        """
        Standard transform on each dataset is to normalize by training dataset mean and stdev pixel.
        This normalization needs to be done after a torch tensor is created from a PIL image.
        
        Most methods in this class assume renormalization is required. The easiest way to 
        bypass this is to set self.norms to mean=0 and stdev=1.
        
        """
        
        return (img - self.norms[0]) / self.norms[1]
    
    
class SmoothenValue():
    "Create a smooth moving average for a value (loss, etc) using `beta`."
    def __init__(self, beta:float):
        self.beta,self.n,self.mov_avg = beta,0,0

    def add_value(self, val:float)->None:
        "Add `val` to calculate updated smoothed value."
        self.n += 1
        self.mov_avg = self.beta * self.mov_avg + (1 - self.beta) * val
        self.smooth = self.mov_avg / (1 - self.beta ** self.n)
        
class OneCycle(object):
    """
    In paper (https://arxiv.org/pdf/1803.09820.pdf), author suggests to do one cycle during 
    whole run with 2 steps of equal length. During first step, increase the learning rate 
    from lower learning rate to higher learning rate. And in second step, decrease it from 
    higher to lower learning rate. This is Cyclic learning rate policy. Author suggests one 
    addition to this. - During last few hundred/thousand iterations of cycle reduce the 
    learning rate to 1/100th or 1/1000th of the lower learning rate.

    Also, Author suggests that reducing momentum when learning rate is increasing. So, we make 
    one cycle of momentum also with learning rate - Decrease momentum when learning rate is 
    increasing and increase momentum when learning rate is decreasing.

    Args:
        nb              Total number of iterations including all epochs

        max_lr          The optimum learning rate. This learning rate will be used as highest 
                        learning rate. The learning rate will fluctuate between max_lr to
                        max_lr/div and then (max_lr/div)/div.

        momentum_vals   The maximum and minimum momentum values between which momentum will
                        fluctuate during cycle.
                        Default values are (0.95, 0.85)

        prcnt           The percentage of cycle length for which we annihilate learning rate
                        way below the lower learnig rate.
                        The default value is 10

        div             The division factor used to get lower boundary of learning rate. This
                        will be used with max_lr value to decide lower learning rate boundary.
                        This value is also used to decide how much we annihilate the learning 
                        rate below lower learning rate.
                        The default value is 10.
    """
    def __init__(self, nb, max_lr, momentum_vals=(0.95, 0.85), prcnt= 10 , div=10):
        self.nb = nb
        self.div = div
        self.step_len =  int(self.nb * (1- prcnt/100)/2)
        self.high_lr = max_lr
        self.low_mom = momentum_vals[1]
        self.high_mom = momentum_vals[0]
        self.prcnt = prcnt
        self.iteration = 0
        self.lrs = []
        self.moms = []
        
    def calc(self):
        self.iteration += 1
        lr = self.calc_lr()
        mom = self.calc_mom()
        return (lr, mom)
        
    def calc_lr(self):
        if self.iteration==self.nb:
            self.iteration = 0
            self.lrs.append(self.high_lr/self.div)
            return self.high_lr/self.div
        if self.iteration > 2 * self.step_len:
            ratio = (self.iteration - 2 * self.step_len) / (self.nb - 2 * self.step_len)
            lr = self.high_lr * ( 1 - 0.99 * ratio)/self.div
        elif self.iteration > self.step_len:
            ratio = 1- (self.iteration -self.step_len)/self.step_len
            lr = self.high_lr * (1 + ratio * (self.div - 1)) / self.div
        else :
            ratio = self.iteration/self.step_len
            lr = self.high_lr * (1 + ratio * (self.div - 1)) / self.div
        self.lrs.append(lr)
        return lr
    
    def calc_mom(self):
        if self.iteration==self.nb:
            self.iteration = 0
            self.moms.append(self.high_mom)
            return self.high_mom
        if self.iteration > 2 * self.step_len:
            mom = self.high_mom
        elif self.iteration > self.step_len:
            ratio = (self.iteration -self.step_len)/self.step_len
            mom = self.low_mom + ratio * (self.high_mom - self.low_mom)
        else :
            ratio = self.iteration/self.step_len
            mom = self.high_mom - ratio * (self.high_mom - self.low_mom)
        self.moms.append(mom)
        return mom

def average_scores(prediction_volumes):
    """
    Given a set of prediction volumes, performs score based reconstruction.
    
    Arguments:
    ----------
    prediction_volume: A 3D numpy array of prediction volumes. Shape is (volume_shape).
    
    Returns:
    --------
    A 3D numpy array containing the rounded, averaged voxel values. 
    If prediction_volumes contained 3 volumes, the returned volume will be True 
    at each voxel for which the sum of the 3 volumes is greater than 1.5. 
    Any voxel greater than exactly half of the number of volumes is marked as True.
    """
    
    #prediction volumes first need to be scaled to the range 0-1 and then rounded
    n = len(prediction_volumes)
    prediction_volumes = np.mean(prediction_volumes, axis=0)
    
    return prediction_volumes.astype(np.uint8)
        
def reconstruct(model, tta, dataloaders, average=True):
    """
    Performs Test Time Augmention predictions using the given model, 
    for each of the given dataloaders.
    Volumes generated from the dataloaders are averaged together based on the averaging_mode.
    
    Arguments:
    ----------
    model: A trained pytorch model object.
    
    tta: A TTA object, if no test time augmentations are required, 
    instantiate the class with all Falses and Nones. See TTA class in model.py.
    
    dataloaders: A list of pytorch dataloaders. Each dataloader will load 2d 
    images that are to be stacked sequentially in order to produce a 3D reconstruction. 
    NOTE: Slices will be stacked in the same order they are loaded,
    when creating the dataset use a sorted list of file names with the 
    correct stacking order. The volumes generated from each dataloader will be averaged 
    together before a final reconstructed volume is returned.
    
    average: Boolean
    
    """
    #volumes is where we'll put each completed volume mask by direction
    volumes = []
    
    for ix,dataloader in enumerate(dataloaders):
        image_path = dataloader.dataset.impath
        sfname = dataloader.dataset.fnames[0]
        d = dataloader.dataset.fnames[0].split('_')[-1].split('.')[0]
        orig_size = pil.open(image_path + '/' + sfname).size
        
        #dmask where we'll put our completed masks for each direction
        dmask = []
        print('Segmenting in ' + d + ' plane...')
        for batch in tqdm(dataloader, total=len(dataloader)):
            batch_img = batch['im']
            
            pmasks = tta.batch_predict(model, batch_img)
            fmsk = (tta.average_masks(pmasks) * 255).astype(np.uint8)
            
            if len(fmsk.shape) != 4:
                fmsk = np.expand_dims(fmsk, axis=0)
            
            #need to have 3 channels or pillow does weird things
            fmsk = np.pad(fmsk, ((0, 0), (0, 1), (0, 0), (0, 0)), mode='constant')

            pim = [pil.fromarray(np.rollaxis(f, 0, 3)).resize(orig_size) for f in fmsk]

            if d == 'xz':
                pim = [pil_image.transpose(pil.ROTATE_90) for pil_image in pim]
                mnp = np.stack([np.array(pil_image)[::-1] for pil_image in pim])
            else:
                mnp = np.stack([np.array(pil_image) for pil_image in pim], axis=0)
            
            dmask.append(mnp[..., :-1])
        
        if d == 'xy':
            dmask = np.concatenate(dmask, axis=0)
            dmask = np.rollaxis(dmask, 0, 3)
        elif d == 'xz':
            dmask = np.concatenate(dmask, axis=0)
            dmask = np.rollaxis(dmask, 0, 2)
        else:
            dmask = np.concatenate(dmask, axis=0)
            
        volumes.append(dmask)
        
        
    volumes = np.array(volumes)
    if average:
        volumes = average_scores(volumes)
        
    return volumes


class Prefetcher():
    def __init__(self, loader, fields=['im', 'msk']):
        self.loader = loader
        self.loader_iter = iter(loader)
        self.stream = torch.cuda.Stream()
        self.fields = fields
        self.preload()

    def __len__(self):
        return len(self.loader)

    def preload(self):
        try:
            batch = next(self.loader_iter)
            self.next_data = [batch[f] for f in self.fields]
        except StopIteration:
            self.loader_iter = iter(self.loader)
            batch = next(self.loader_iter)
            self.next_data = [batch[f] for f in self.fields]
            
        # if record_stream() doesn't work, another option is to make sure device inputs are created
        # on the main stream.
        # self.next_input_gpu = torch.empty_like(self.next_input, device='cuda')
        # self.next_target_gpu = torch.empty_like(self.next_target, device='cuda')
        # Need to make sure the memory allocated for next_* is not still in use by the main stream
        # at the time we start copying to next_*:
        # self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            self.next_data = [data.cuda(non_blocking=True) for data in self.next_data]
            
    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        
        #get the next set of data
        next_data = self.next_data
        
        #record data on current stream
        for data in next_data:
            data.record_stream(torch.cuda.current_stream())
            
        #load the next batch
        self.preload()
        
        #return all data as floats
        return [data.float() for data in next_data]
    
class SelfTrainer(Trainer):
    def __init__(self, model, optimizer, loss, ss_loss, trn_data, unlabeled_data, val_data=None,
                 metrics=None, logging=None, temp=0.5, alpha=0.25):
        #calling super will handling instantiating the regular 
        #Trainer class with all the correct attributes
        super().__init__(model, optimizer, loss, trn_data, val_data, metrics, logging)
        
        #we just need to add handling for the new attributes
        #specific to the SelfTrainer
        self.ss_loss = ss_loss
        
        #define a prefetcher for the unlabeled data, only loading images
        #and not masks
        self.unlabeled_data = Prefetcher(unlabeled_data, fields=['im'])
        
        #instantiate a sharpener object with given temp(s)
        self.sharpener = Sharpener(temp)
        self.alpha = alpha
    
    #we need to rewrite how we train 1 batch for all
    #the other methods to work
    def _train_1_batch(self, use_metrics=True):
        #first get a labeled batch of data and record
        #the number of examples in the batch
        im, msk = self._get_batch(self.trn_data)
        n_labeled = len(im)
        
        #next get an unlabeled batch of image data
        im_un = self._get_batch(self.unlabeled_data)[0]
        
        #concatenate the images for labeled and unlabeled 
        #data on batch dim
        im = torch.cat([im, im_un], dim=0)
        
        #run the model in eval_mode to get pseudo-labeled masks
        with torch.no_grad():
            msk_un = self.model.eval()(im_un)
            
            #detach pseudo masks and sharpen
            #(B, 1, H, W) --> (B, 2, H, W)
            #(B, N, H, W) --> (B, N, H, W)
            msk_un = self.sharpener.sharpen(msk_un.detach())
        
        #zero all gradients
        self.optimizer.zero_grad()
        
        #run forward training pass
        output = self.model.train()(im)
        l1 = self.loss(output[:n_labeled], msk)
        l2 = self.ss_loss(output[n_labeled:], msk_un)
        l = (1 - self.alpha) * l1 + self.alpha * l2
        
        l.backward()
        self.optimizer.step()
        
        if use_metrics:
            #when we compute metrics, we only want to use the labeled
            #data, so we can make sure that the model is actually learning
            if self.trn_metrics is not None:
                self.trn_metrics.evaluate(output[:n_labeled].detach().cpu(), msk[:n_labeled].detach().cpu())
        
        return l.item()
        
class Sharpener:
    
    """
    Sharpener apply a sharpening function to logits in order to get more
    confident prediction probabilities. The amount of increase in confidence
    is dependent on the temperature, temperatures close to 1 will not change
    the prediction probability at all
    """
    
    def __init__(self, temp, iterations=None):
        #if temp is a single value, use it as the temp
        if hasattr(temp, '__getitem__'):
            self.min_temp = temp[0]
            self.T = temp[1]
        else:
            self.min_temp = None
            self.T = temp
            
    def set_step(self, iterations):
        assert(self.min_temp is not None), \
        "Only 1 temperature was given at instantiation, use a tuple for temp (min_temp, max_temp)"
        self.step = (self.T - self.min_temp) / iterations

    def decrease_temp(self):
        #move the temp down by step
        self.T -= self.step
        
    def sharpen(self, logits):
        #apply the sharpening function to logit predictions
        #get the number of classes
        n_classes = logits.size(1)
        
        #if there's only 1 single class, then apply
        #the sigmoid function to get probabilities
        #otherwise apply softmax
        if n_classes == 1:
            #to sharpen, we need a 2 class distribution
            #get the probabilities and inverses and stack them
            pos_proba = nn.Sigmoid()(logits)
            neg_proba = (1 - pos_proba)
            #2 x (B, 1, H, W) --> (B, 2, H, W)
            proba = torch.cat([neg_proba, pos_proba], dim=1)
        else:
            proba = nn.Softmax(dim=1)(logits)
        
        #sharpen probabilities
        #f, ax = plt.subplots(1, 2, figsize=(8, 8))
        #ax[0].imshow(proba[0, 1].cpu().numpy())
        
        proba_t = proba ** (1 / self.T)
        proba_t = proba_t / proba_t.sum(dim=1, keepdim=True)
        #ax[1].imshow(proba_t[0, 1].cpu().numpy())
        
        #in the binary case, we will still return both pos and neg
        #probabilities because we assume that we will use DiceLoss
        #and not BCE loss, if that changes, we would get an error
        #because BCE expects 1 channel for output and targets
        return proba_t
    
