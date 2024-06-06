import warnings
warnings.filterwarnings("ignore")

import os
import sys
import json
import numpy as np
import torch
import torchvision
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import pdb
import pandas as pd
import time

"""
-  Import from Utils: Functions and classes required for model architecture ('resnet18'),
   logging ('Logger'), and utility functions ('AverageMeter', 'calculate_accuracy').
-  Transforms: Various data transformation functions for preprocessing videos, such as
   normalization, cropping, and flipping.
-  Data Loader: Custom data loader for loading Cichlid fish video data.
"""

# Import the necessary modules and functions from utility files
from Utils.model import resnet18
from Utils.utils import Logger, AverageMeter, calculate_accuracy
from Utils.transforms import (Compose, Normalize, Scale, CenterCrop,
                        RandomHorizontalFlip, RandomVerticalFlip,
                        FixedScaleRandomCenterCrop, MultiScaleRandomCenterCrop,
                        ToTensor, TemporalCenterCrop, TemporalCenterRandomCrop,
                        ClassLabel, VideoID,TargetCompose)
from Utils.data_loader import cichlids

class ML_model():
    def __init__(self, args):
        self.args = args
        # Path to the source JSON file for results
        self.source_json_file = os.path.join(args.Results_directory, 'cichlids.json')
    
    def work(self):
        opt = self.args
        # Log file for recording results
        log_file = os.path.join(opt.Results_directory, 'log')
        
        # Dumping args into the log file for reference
        with open(log_file, 'w') as output:
            json.dump(vars(opt), output)
        
        # Initialize the resnet18 model with the given parameters
        model = resnet18(num_classes = opt.n_classes,
                 shortcut_type = opt.resnet_shortcut,
                 sample_size = opt.sample_size,
                 sample_duration = opt.sample_duration)
        model = model.cuda() # Move the model to the GPU
        model = nn.DataParallel(model, device_ids = None) # Enable data parallelism
        
        parameters = model.parameters() # Get model parameters
        criterion = nn.CrossEntropyLoss().cuda() # Define the loss function
        
        # Load annotated data depending on the purpose
        if opt.Purpose == "classify":
            source_annotatedData = pd.read_csv(opt.Videos_to_project_file, sep = ',', header = 0)
        else:
            source_annotatedData = pd.read_csv(opt.ML_models, sep = ',', header = 0)
        
        # Create a dictionary for video annotations
        source_annotation_dict = dict(zip(source_annotatedData['VideoFile'], source_annotatedData['ProjectID']))
        
        """
        Different spatial and temporal transformations are set up for the training, validation, and test datasets.
        """
        
        # Training data transformations
        crop_method = MultiScaleRandomCenterCrop([0.99, 0.97, 0.95, 0.93, 0.91], opt.sample_size)
        spatial_transforms = {}
        mean_file = os.path.join(opt.Results_directory, 'Means.csv')
        
        with open(mean_file) as f:
            for i, line in enumerate(f):
                if i == 0:
                    continue
                # Reads mean values from a file and sets up normalization and other transformations
                tokens = line.rstrip().split(',')
                norm_method = Normalize([float(x) for x in tokens[1:4]], [float(x) for x in tokens[4:7]])
                spatial_transforms[tokens[0]] = Compose(
                    [crop_method, RandomVerticalFlip(), RandomHorizontalFlip(), ToTensor(1), norm_method])
       
        # Sets up the temporal center random crop transformation
        temporal_transform = TemporalCenterRandomCrop(opt.sample_duration)
        
        # Sets up the target transformation to convert class labels
        target_transform = ClassLabel()
        
        # Load the training data using custom 'cichlids' data loader with specified transformations
        training_data = cichlids(opt.Temporary_clips_directory,
                                 self.source_json_file,
                                 'training',
                                 spatial_transforms = spatial_transforms,
                                 temporal_transform = temporal_transform,
                                 target_transform = target_transform,
                                 annotationDict = source_annotation_dict)
        
        # Create an iterable over the training dataset, with batching, shuffling, and multi-threading
        if len(training_data) != 0:
            train_loader = torch.utils.data.DataLoader(training_data,
                                                       batch_size = opt.batch_size,
                                                       shuffle = True,
                                                       num_workers = opt.n_threads,
                                                       pin_memory = True)
            
            # Initializes loggers for training and batch-level logging
            train_logger = Logger(os.path.join(opt.Results_directory, 'train.log'),
                                  ['epoch', 'loss', 'acc', 'lr'])
            train_batch_logger = Logger(os.path.join(opt.Results_directory, 'train_batch.log'),
                                        ['epoch', 'batch', 'iter', 'loss', 'acc', 'lr'])
            
        # Validation data transformations
        crop_method = CenterCrop(opt.sample_size)
        spatial_transforms = {}
        
        with open(mean_file) as f:
            for i, line in enumerate(f):
                if i == 0:
                    continue
                # Reads mean values and sets up normalization parameters
                tokens = line.rstrip().split(',')
                norm_method = Normalize([float(x) for x in tokens[1:4]], [float(x) for x in tokens[4:7]])
                spatial_transforms[tokens[0]] = Compose([crop_method, ToTensor(1), norm_method])
        
        temporal_transform = TemporalCenterCrop(opt.sample_duration)
        
        # Load validation data using custom 'cichlids' data loader
        validation_data = cichlids(opt.Temporary_clips_directory,
                                   self.source_json_file,
                                   'validation',
                                   spatial_transforms = spatial_transforms,
                                   temporal_transform = temporal_transform,
                                   target_transform = target_transform, 
                                   annotationDict = source_annotation_dict)
        
        # Creates an iterable over the validation set, without shuffling
        val_loader = torch.utils.data.DataLoader(validation_data,
                                                 batch_size = opt.batch_size,
                                                 shuffle = False,
                                                 num_workers = opt.n_threads,
                                                 pin_memory = True)
        
        # Initializes a logger for validation
        val_logger = Logger(os.path.join(opt.Results_directory, 'val.log'), ['epoch', 'loss', 'acc'])
        
        # Test data transformations
        crop_method = CenterCrop(opt.sample_size)
        spatial_transforms = {}
        
        with open(mean_file) as f:
            for i, line in enumerate(f):
                if i == 0:
                    continue
                # Reads mean values and sets-up normalization transformations
                tokens = line.rstrip().split(',')
                norm_method = Normalize([float(x) for x in tokens[1:4]], [float(x) for x in tokens[4:7]])
                spatial_transforms[tokens[0]] = Compose([crop_method, ToTensor(1), norm_method])
        
        temporal_transform = TemporalCenterCrop(opt.sample_duration)
        
        # Load the test data using custom 'cichlids' data loader
        test_data = cichlids(opt.Temporary_clips_directory,
                             self.source_json_file,
                             'testing',
                             spatial_transforms = spatial_transforms,
                             temporal_transform = temporal_transform,
                             target_transform = target_transform,
                             annotationDict = source_annotation_dict)
        
        # Creates an iterable over the test dataset, with shuffling
        if len(test_data) != 0:
            test_loader = torch.utils.data.DataLoader(test_data,
                                                      batch_size = opt.batch_size,
                                                      shuffle = True,
                                                      num_workers = opt.n_threads,
                                                      pin_memory = True)
            # Initializes a logger for testing
            test_logger = Logger(os.path.join(opt.Results_directory, 'test.log'),
                                 ['epoch', 'loss', 'acc'])
        
        # Sets the dampening parameter based on whether the Nesterov momentum is used
        if opt.nesterov:
            dampening = 0
        else:
            dampening = opt.dampening
        
        # Initializing the SGD optimizer
        optimizer = optim.SGD(parameters, lr = opt.momentum, dampening = dampening,
                              weight_decay = opt.weight_decay, nesterov = opt.nesterov)
        
        # Load pre-trained weights from the checkpoint if specified
        if opt.Purpose in ['finetune', 'classify']:
            # Loads model and optimizer states, and sets the initial epoch
            checkpoint = torch.load(opt.Trained_model_file)
            begin_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            begin_epoch = 0
        
        # Classification task
        if opt.Purpose == 'classify':
            # Model performs inference and outputs confidence matrices (saved to a CSV)
            _, confusion_matrix, confidence_matrix = self.val_epoch(i, val_loader, model,
                                                                    criterion, opt, val_logger)
            with open(self.source_json_file, 'r') as input_f:
                source_json = json.load(input_f)
            # Updating the confidence matrix with predicted labels
            confidence_matrix.columns = source_json['labels']
            confidence_matrix['predicted_label'] = confidence_matrix.idxmax(axis = "columns")
            # Saving it to a CSV file
            confidence_matrix.to_csv(self.args.Output_file)
            return
        
        print('Time to run the model!')
        
        """
        Training and Validation: The main training loop runs for a specified number of epochs. Each
        epoch involves training on the training set and validating on the validation
        set. Confusion matrices and other metrics are saved and logged.
        """
        
        # Training loop
        for i in range(begin_epoch, opt.n_epochs + 1):
            # Calls the 'train_epoch' method for training
            self.train_epoch(i, train_loader, model, criterion, optimizer, 
                             opt, train_logger, train_batch_logger)
            
            # Calls the 'val_epoch' method for validation
            validation_loss, confusion_matrix, _ = self.val_epoch(i, val_loader, model, 
                                                                  criterion, opt, val_logger)
            
            # Saves the confusion matrix for each epoch
            confusion_matrix_file = os.path.join(self.args.Results_directory,
                                                 'epoch_{epoch}_confusion_matrix.csv'.format(epoch=i))
            confusion_matrix.to_csv(confusion_matrix_file)
            
            # Learning rate is adjusted dynamically, scheduler stepped based on validation loss
            scheduler.step(validation_loss)
            
            # Perform testing every 5 epochs if test data are available
            if i % 5 == 0 and len(test_data) != 0:
                _ = self.val_epoch(i, test_loader, model, criterion, opt, test_logger)
                
    
    def train_epoch(self, epoch, data_loader, model, criterion, optimizer, opt, epoch_logger, batch_logger):
        print('Train at epoch {}'.format(epoch))
        model.train()
        
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        accuracies = AverageMeter()
        
        end_time = time.time()
        for i, (inputs, targets, _) in enumerate(data_loader):
            data_time.update(time.time() - end_time())
            
            targets = targets.cuda(non_blocking=True)
            inputs = Variable(inputs)
            targets = Variable(targets)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            acc = calculate_accuracy(outputs, targets)
            
            losses.update(loss.data, inputs.size(0))
            accuracies.update(acc, inputs.size(0))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            batch_time.update(time.time() - end_time)
            end_time = time.time()
            
            batch_logger.log({
                'epoch': epoch,
                'batch': i + 1,
                'iter': (epoch-1) * len(data_loader) + (i+1),
                'loss': losses.val,
                'acc': accuracies.val,
                'lr': optimizer.param_groups[0]['lr']
            })
            
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data: {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch,
                      i + 1,
                      len(data_loader),
                      batch_time = batch_time,
                      data_time = data_time,
                      loss = losses,
                      acc = accuracies))
            
        epoch_logger.log({
            'epoch': epoch,
            'loss': losses.avg,
            'acc': accuracies.avg,
            'lr': optimizer.param_groups[0]['lr']
        })
        
        if epoch % opt.checkpoint == 0:
            save_file_path = os.path.join(opt.Results_directory,
                                          'save_{}.pth'.format(epoch))
            states = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(states, save_file_path)
        
    def val_epoch(self, epoch, data_loader, model, criterion, opt, logger):
        print('Validation at epoch {}'.format(epoch))
        
        model.eval()
        
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        accuracies = AverageMeter()
        
        end_time = time.time()
        confusion_matrix = np.zeros((opt.n_classes, opt.n_classes))
        confidence_for_each_validation = {}
        
        for i, (inputs, targets, paths) in enumerate(data_loader):
            data_time.update(time.time() - end_time)
            targets = targets.cuda(non_blocking=True)
            with torch.no_grad():
                inputs = Variable(inputs)
                targets = Variable(targets)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                acc = calculate_accuracy(outputs, targets)
                
                for j in range(len(targets)):
                    key = paths[j].split('/')[-1]
                    confidence_for_each_validation[key] = [x.item() for x in outputs[j]]
                rows = [int(x) for x in targets]
                columns = [int(x) for x in np.argmax(outputs.data.cpu(), 1)]
                assert len(rows) == len(columns)
                for idx in range(len(rows)):
                    confusion_matrix[rows[idx]][columns[idx]] += 1
                
                losses.update(loss.data, inputs.size(0))
                accuracies.update(acc, inputs.size(0))
                
                batch_time.update(time.time() - end_time)
                end_time = time.time()
                
                print('Epoch: [{0}][{1}/{2}]\t'
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data: {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch,
                      i + 1,
                      len(data_loader),
                      batch_time = batch_time,
                      data_time = data_time,
                      loss = losses,
                      acc = accuracies))
         
        confusion_matrix = pd.DataFrame(confusion_matrix)
        confidence_matrix = pd.DataFrame.from_dict(confidence_for_each_validation, orient='index')
        logger.log({'epoch': epoch, 'loss': losses.avg, 'acc': accuracies.avg})
        return losses.avg, confusion_matrix, confidence_matrix
                
    def test_epoch(self, epoch, data_loader, model, criterion, opt, logger):
        print('Test at epoch {}'.format(epoch))
        
        model.eval()
        
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        accuracies = AverageMeter()
        
        end_time = time.time()
        
        for i, (inputs, targets, _) in enumerate(data_loader):
            data_time.update(time.time() - end.time)
            if not opt.no_cuda:
                targets = targets.cuda(non_blocking=True)
                with torch.no_grad():
                    inputs = Variable(inputs)
                    targets = Variable(targets)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    losses.update(loss.data, inputs.size(0))
                    accuracies.update(acc, inputs.size(0))
                    
                    batch_time.update(time.time() - end_time)
                    end_time = time.time()
                    
                    print('Epoch: [{0}][{1}/{2}]\t'
                      'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data: {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                          epoch,
                          i + 1,
                          len(data_loader),
                          batch_time = batch_time,
                          data_time = data_time,
                          loss = losses,
                          acc = accuracies))
                    
        logger.log({'epoch': epoch, 'loss': losses.avg, 'acc': accuracies.avg})
        return losses.avg