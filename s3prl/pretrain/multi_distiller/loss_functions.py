"""
    Barlow Twins Loss Functions for Multi-Distiller
    Ported from robust-superb codebase
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import gc


class BarlowTwinsLoss(nn.Module):
    """
    Barlow Twins Loss with cross-correlation and self-correlation
    Code motivated from: https://pytorch-lightning.readthedocs.io/en/stable/notebooks/lightning_examples/barlow-twins.html
    """
    def __init__(self, lambda_coeff=5e-3, number_of_predictions=3, scale=1,
                 off_diag_cross_cor_scale=100, diag_scale=10,
                 self_correlation=False, lambda_coeff_self_corr=1, use_snr_info=False,
                 off_diag_self_cor_scale=50000):
        super().__init__()
        print(f"[BarlowTwinsLoss] init barlow loss with scale {scale}")
        print(f"[BarlowTwinsLoss] init barlow loss with off diag frame scale {off_diag_cross_cor_scale}")
        print(f"[BarlowTwinsLoss] init barlow loss with diag frame scale {diag_scale}")
        print(f"[BarlowTwinsLoss] init barlow loss with off diag self cor scale {off_diag_self_cor_scale}")
        print(f"[BarlowTwinsLoss] lambda_coeff_self_corr is {lambda_coeff_self_corr}")
        print(f"[BarlowTwinsLoss] lambda_coeff is {lambda_coeff}")

        self.lambda_coeff = lambda_coeff
        self.number_of_predictions = number_of_predictions
        self.loss_for_predictions = None
        self.scale = scale
        self.off_diag_cross_cor_scale = off_diag_cross_cor_scale
        self.diag_scale = diag_scale
        self.self_correlation = self_correlation
        self.lambda_coeff_self_corr = lambda_coeff_self_corr
        self.use_snr_info = use_snr_info
        self.off_diag_self_cor_scale = off_diag_self_cor_scale

        # Loss component monitoring
        self.last_on_diag = None
        self.last_off_diag_cross = None
        self.last_off_diag_self = None

    def off_diagonal_ele(self, x):
        """Return a flattened view of the off-diagonal elements of a square matrix"""
        # taken from: https://github.com/facebookresearch/barlowtwins/blob/main/main.py
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def off_diagonal_ele_3d(self, M):
        """Zero out diagonal elements of a 3D+ tensor"""
        try:
            M.diagonal(dim1=-1, dim2=-2).zero_()
        except RuntimeError as e:
            if 'CUDA out of memory' in str(e):
                print(f"off diag using cpu instead...")
                M.cpu().diagonal(dim1=-1, dim2=-2).zero_()
        return M

    def forward(self, z1, z2, **kwargs):
        """
        Compute Barlow Twins loss
        Args:
            z1: Student predictions (B x N x T x D) or (B x N x D)
            z2: Teacher targets (B x N x T x D) or (B x N x D)
        Returns:
            Loss tensor
        """
        if len(z1.shape) == 4:
            # Frame-level loss: B x N x T x D
            self.loss_for_predictions = torch.zeros(z1.shape[1:-1])
            orig_dtype = z1.dtype
            z1_f = z1.float()
            z2_f = z2.float()
            z1_norm = ((z1_f - torch.mean(z1_f, dim=0, keepdim=True)) / (torch.std(z1_f, dim=0, keepdim=True) + 1e-4)).to(orig_dtype)
            z2_norm = ((z2_f - torch.mean(z2_f, dim=0, keepdim=True)) / (torch.std(z2_f, dim=0, keepdim=True) + 1e-4)).to(orig_dtype)
            batch_size = z1.shape[0]

            if len(kwargs.get("teacher_snrs", [])) > 0 and self.use_snr_info:
                # SNR-aware loss computation
                coefficients_teacher = torch.tensor(kwargs.get("teacher_snrs")).unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4).cuda()
                coefficients_student = torch.tensor(kwargs.get("student_snrs")).unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4).cuda()
                cross_corr_entire = torch.matmul(z1_norm.unsqueeze(-1), z2_norm.unsqueeze(-2))
                coefficients_teacher = 9.9 * coefficients_teacher - 98
                coefficients_student = 9.9 * coefficients_student - 98
                off_diag = self.off_diagonal_ele_3d(cross_corr_entire.clone()).pow(2) / self.off_diag_cross_cor_scale
                off_diag = off_diag / coefficients_teacher
                off_diag = (torch.sum(off_diag, dim=0)) / (z1.shape[0]-1)
                cross_corr_entire = (torch.sum(cross_corr_entire, dim=0)) / (z1.shape[0]-1)
                on_diag = (torch.diagonal(cross_corr_entire, dim1=-2, dim2=-1) - 1).pow(2).sum(dim=2) / self.diag_scale
                off_diag = off_diag.sum((2, 3))
                if self.self_correlation:
                    self_corr = torch.matmul(z1_norm.unsqueeze(-1), z1_norm.unsqueeze(-2))
                    off_diag_self_corr = self.off_diagonal_ele_3d(self_corr.clone()).pow(2) / self.off_diag_self_cor_scale
                    off_diag_self_corr = off_diag_self_corr / coefficients_student
                    off_diag_self_corr = (torch.sum(off_diag_self_corr, dim=0)) / (z1.shape[0]-1)
                    off_diag_self_corr = off_diag_self_corr.sum((2, 3))
                else:
                    self_corr = 0
                    off_diag_self_corr = 0
            else:
                # Standard loss computation (no SNR info)
                cross_corr = (torch.sum(torch.matmul(z1_norm.unsqueeze(-1), z2_norm.unsqueeze(-2)), dim=0)) / (z1.shape[0]-1)
                on_diag = (torch.diagonal(cross_corr.clone(), dim1=-2, dim2=-1) - 1).pow(2).sum(dim=2) / self.diag_scale

                if self.lambda_coeff == float(0):
                    off_diag = 0
                else:
                    off_diag = self.off_diagonal_ele_3d(cross_corr.clone()).pow(2) / self.off_diag_cross_cor_scale
                    off_diag = off_diag.sum((2, 3))

                if self.self_correlation:
                    self_corr = torch.sum(torch.matmul(z1_norm.unsqueeze(-1), z1_norm.unsqueeze(-2)), dim=0) / (z1.shape[0] - 1)
                    off_diag_self_corr = self.off_diagonal_ele_3d(self_corr.clone()).pow(2) / self.off_diag_self_cor_scale
                    off_diag_self_corr = off_diag_self_corr.sum((2, 3))
                else:
                    self_corr = 0
                    off_diag_self_corr = 0

            # Store loss components for monitoring
            self.last_on_diag = on_diag.detach()
            self.last_off_diag_cross = off_diag.detach() if isinstance(off_diag, torch.Tensor) else off_diag
            self.last_off_diag_self = off_diag_self_corr.detach() if isinstance(off_diag_self_corr, torch.Tensor) else off_diag_self_corr

            loss_value = on_diag + self.lambda_coeff * off_diag + self.lambda_coeff_self_corr * off_diag_self_corr
            self.loss_for_predictions = loss_value.cuda()
        else:
            # Utterance-level loss: B x N x D
            self.loss_for_predictions = torch.zeros(z1.shape[1])
            for i in range(0, self.number_of_predictions):
                orig_dtype = z1.dtype
                z1_i = z1[:, i, :].float()
                z2_i = z2[:, i, :].float()
                z1_norm = ((z1_i - torch.mean(z1_i, dim=0)) / (torch.std(z1_i, dim=0) + 1e-4)).to(orig_dtype)
                z2_norm = ((z2_i - torch.mean(z2_i, dim=0)) / (torch.std(z2_i, dim=0) + 1e-4)).to(orig_dtype)
                cross_corr = torch.matmul(z1_norm.T, z2_norm) / (z1.shape[0]-1)
                on_diag = (torch.diagonal(cross_corr) - 1).pow(2).sum()
                off_diag = self.off_diagonal_ele(cross_corr).pow(2).sum()
                loss_value = on_diag + self.lambda_coeff * off_diag
                self.loss_for_predictions[i] = loss_value.cuda()

        return self.scale * self.loss_for_predictions


class BarlowTwinsLoss_old(nn.Module):
    """
    Old Barlow Twins Loss implementation.
    Key differences from current BarlowTwinsLoss:
      - No fp32 casting (normalizes in original dtype)
      - Default off_diag_self_cor_scale=50000 (was hardcoded, now configurable)
      - Epsilon 1e-5 instead of 1e-4
      - In-place add_(-1) operations in on_diag computation
      - No loss component monitoring (last_on_diag, etc.)
    """
    # code motivated from:
    # https://pytorch-lightning.readthedocs.io/en/stable/notebooks/lightning_examples/barlow-twins.html
    def __init__(self, lambda_coeff=5e-3, number_of_predictions=3, scale=1, off_diag_cross_cor_scale=100, diag_scale=10, self_correlation=False, lambda_coeff_self_corr=1, use_snr_info=False, off_diag_self_cor_scale=50000):
        #torch.autograd.set_detect_anomaly(True)
        super().__init__()
        print(f"[BarlowTwinsLoss_old] init barlow loss with scale {scale}")
        print(f"[BarlowTwinsLoss_old] init barlow loss with off diag frame scale {off_diag_cross_cor_scale}")
        print(f"[BarlowTwinsLoss_old] init barlow loss with off diag self-cor scale {off_diag_self_cor_scale}")
        print(f"[BarlowTwinsLoss_old] init barlow loss with diag frame scale {diag_scale}")
        print(f"[BarlowTwinsLoss_old] lambda_coeff_self_corr is {lambda_coeff_self_corr}")
        print(f"[BarlowTwinsLoss_old] lambda_coeff is {lambda_coeff}")

        self.lambda_coeff = lambda_coeff
        self.number_of_predictions = number_of_predictions
        self.loss_for_predictions = None
        self.scale = scale
        self.off_diag_cross_cor_scale = off_diag_cross_cor_scale
        self.off_diag_self_cor_scale = off_diag_self_cor_scale
        self.diag_scale = diag_scale
        self.self_correlation = self_correlation
        self.lambda_coeff_self_corr = lambda_coeff_self_corr
        self.use_snr_info = use_snr_info

    def off_diagonal_ele(self, x):
        # taken from: https://github.com/facebookresearch/barlowtwins/blob/main/main.py
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def off_diagonal_ele_3d(self, M):
        try:
            M.diagonal(dim1=-1, dim2=-2).zero_()
        except RuntimeError as e:
                    if 'CUDA out of memory' in str(e):
                        print(f"off diag using cpu instead...")
                        M.cpu().diagonal(dim1=-1, dim2=-2).zero_()
        return M



    def forward(self, z1, z2, **kwargs):
        if len(z1.shape) == 4:
            self.loss_for_predictions = torch.zeros(z1.shape[1:-1])
            z1_norm = (z1 - torch.mean(z1, dim=0, keepdim=True)) / (torch.std(z1, dim=0, keepdim=True) + 1e-5)
            z2_norm = (z2 - torch.mean(z2, dim=0, keepdim=True)) / (torch.std(z2, dim=0, keepdim=True) + 1e-5)
            batch_size = z1.shape[0]
            if len(kwargs.get("teacher_snrs", [])) > 0 and self.use_snr_info:
                ###### I still have not implemented the scaling equation properly!!!!!!!.....
                coefficients_teacher = torch.tensor(kwargs.get("teacher_snrs")).unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4).cuda()
                coefficients_student = torch.tensor(kwargs.get("student_snrs")).unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4).cuda()
                #print(f"coefficients_teacher shape {coefficients_teacher.shape}")
                #print(f"coefficients_teacher coefficients {coefficients_teacher}")
                cross_corr_entire = torch.matmul(z1_norm.unsqueeze(-1), z2_norm.unsqueeze(-2))
                coefficients_teacher = 9.9 * coefficients_teacher - 98
                coefficients_student = 9.9 * coefficients_student - 98
                off_diag = self.off_diagonal_ele_3d(cross_corr_entire.clone()).pow(2) / self.off_diag_cross_cor_scale
                off_diag = off_diag / coefficients_teacher
                off_diag = (torch.sum(off_diag, dim=0)) / (z1.shape[0]-1)
                cross_corr_entire = (torch.sum(cross_corr_entire, dim=0)) / (z1.shape[0]-1)
                on_diag = torch.diagonal(cross_corr_entire ,dim1=-2,dim2=-1).add_(-1).pow(2).sum(dim=2) / self.diag_scale #compare how this one differs from a standard cosine similarity or l1 loss, do the math and simple examples and provide it to prof Chng.
                off_diag = off_diag.sum((2,3)) #revise....
                if self.self_correlation:
                    self_corr = torch.matmul(z1_norm.unsqueeze(-1), z1_norm.unsqueeze(-2))
                    off_diag_self_corr = self.off_diagonal_ele_3d(self_corr.clone()).pow(2) / self.off_diag_self_cor_scale
                    off_diag_self_corr = off_diag_self_corr / coefficients_student
                    off_diag_self_corr = (torch.sum(off_diag_self_corr, dim=0)) / (z1.shape[0]-1)
                    off_diag_self_corr = off_diag_self_corr.sum((2,3))
                else:
                    self_corr = 0 #torch.zeros(1)
                    off_diag_self_corr = 0 #torch.zeros(1)
            else:
                cross_corr = (torch.sum( torch.matmul(z1_norm.unsqueeze(-1), z2_norm.unsqueeze(-2))  , dim=0) ) / (z1.shape[0]-1)
                on_diag = torch.diagonal(cross_corr.clone(),dim1=-2,dim2=-1).add_(-1).pow(2).sum(dim=2) / self.diag_scale #compare how this one differs from a standard cosine similarity or l1 loss, do the math and simple examples and provide it to prof Chng.
                if self.lambda_coeff == float(0):
                    off_diag = 0
                else:
                    off_diag = self.off_diagonal_ele_3d(cross_corr.clone()).pow(2) / self.off_diag_cross_cor_scale  #before this one was set to 100. Now it is set to 10.
                    off_diag = off_diag.sum((2,3))

                if self.self_correlation:
                    self_corr =  torch.sum(torch.matmul(z1_norm.unsqueeze(-1), z1_norm.unsqueeze(-2)),dim=0) / (z1.shape[0] -1)
                    off_diag_self_corr = self.off_diagonal_ele_3d(self_corr.clone()).pow(2) / self.off_diag_self_cor_scale
                    #print(off_diag_self_corr[0,0])
                    #print(off_diag_self_corr[0,0].shape)
                    off_diag_self_corr = off_diag_self_corr.sum((2,3))
                else:
                    self_corr = 0 #torch.zeros(1)
                    off_diag_self_corr = 0 #torch.zeros(1)


            loss_value = on_diag + self.lambda_coeff * off_diag +  self.lambda_coeff_self_corr * off_diag_self_corr # before this one was multiplied by lambda.......hence it is   lambda_coefficient: 5e-3 #coefficient for BarlowTwins Loss Function.
            #del on_diag
            self.loss_for_predictions = loss_value.cuda()
        else:
            self.loss_for_predictions = torch.zeros(z1.shape[1]) # size 3?
            # output: N x D, where N is the predictions and D is output dim of representations
            for i in range(0,self.number_of_predictions):
                #z1_norm = (z1[:,i,:,:] - torch.mean(z1[:,i,:,:], dim=0) ) / torch.std(z1[:,i,:,:], dim=0)
                z1_norm = (z1[:,i,:] - torch.mean(z1[:,i,:], dim=0)) / torch.std(z1[:,i,:], dim=0)
                z2_norm = (z2[:,i,:] - torch.mean(z2[:,i,:], dim=0)) / torch.std(z2[:,i,:], dim=0)
                cross_corr = torch.matmul(z1_norm.T, z2_norm) / (z1.shape[0]-1) #batch size.
                #del z1_norm , z2_norm
                #torch.cuda.empty_cache()
                on_diag = torch.diagonal(cross_corr).add_(-1).pow_(2).sum()
                off_diag = self.off_diagonal_ele(cross_corr).pow_(2).sum()
                loss_value = on_diag + self.lambda_coeff * off_diag
                self.loss_for_predictions[i] = loss_value.cuda()

        #del cross_corr
        #torch.cuda.empty_cache()
        return self.scale * self.loss_for_predictions
