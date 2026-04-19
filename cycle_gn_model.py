import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks

import random
import numpy as np
import os


class CycleGNModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        parser.add_argument('--cycle_step', type=int, default=200)
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = [ 'cycle_A', 'cycle_B']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A', 'G_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']

        self.cycle_step = opt.cycle_step

        self.loss_cycle_A, self.loss_cycle_B = 0,0
        self.cnt = 0
          
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain)

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G_A = torch.optim.Adam(itertools.chain(self.netG_A.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_G_B = torch.optim.Adam(itertools.chain(self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.extend([self.optimizer_G_A,  self.optimizer_G_B])

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))
        self.fake_A = self.netG_B(self.real_B)  # G_B(B)
        self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B))


    def backward_G(self, netG_A, rec_A, rec_B, real_A, real_B, is_idt = False):
        """Calculate the loss for generators G_A and G_B"""
        
        lambda_idt = self.opt.lambda_identity
        # Identity loss
        if is_idt and lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            idt_A = netG_A(real_B)
            loss_idt = self.criterionIdt(idt_A, real_B) * lambda_idt
        else:
            loss_idt = 0

        loss_cycle = self.criterionCycle(rec_B, real_B) 
        #Notice! not rec_A and real_A, but rec_B and real_B for training net_A.
    
        loss_G =  loss_cycle  + loss_idt
        
        loss_G.backward()
        return loss_G

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        torch.autograd.set_detect_anomaly(True)
        
        self.cnt = self.cnt + 1
        is_idt = True
        
        if (self.cnt // self.cycle_step ) % 2 == 0:
        # Train netG_A
            self.forward()
            self.set_requires_grad([self.netG_A], True)
            self.set_requires_grad([self.netG_B], False)
            self.optimizer_G_A.zero_grad()
            self.loss_cycle_A = \
                self.backward_G(self.netG_A, self.rec_A, self.rec_B, self.real_A, self.real_B, is_idt)
            self.optimizer_G_A.step()
            
        else:
        # Train netG_B
            self.forward()
            self.set_requires_grad([self.netG_A], False)
            self.set_requires_grad([self.netG_B], True)
            self.optimizer_G_B.zero_grad()       
            self.loss_cycle_B = \
                self.backward_G(self.netG_B, self.rec_B, self.rec_A, self.real_B, self.real_A, is_idt)
            self.optimizer_G_B.step()     
        


