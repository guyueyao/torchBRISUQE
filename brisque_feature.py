
import torch
import torch.nn.functional as F
import math
from scipy.special import gamma
import numpy as np
from typing import Type, Any, Callable, Union, List, Optional
def gaussian_2d_kernel(kernel_size, sigma):
    kernel = torch.zeros((kernel_size, kernel_size))
    center = kernel_size // 2
    if sigma == 0:
        sigma = ((kernel_size - 1) * 0.5 - 1) * 0.3 + 0.8
    s = 2 * (sigma ** 2)
    sum_val = 0
    for i in range(0, kernel_size):
        for j in range(0, kernel_size):
            x = i - center
            y = j - center
            kernel[i, j] = math.exp(-(x ** 2 + y ** 2) / s)
            sum_val += kernel[i, j]
    sum_val = 1 / sum_val
    return kernel * sum_val

def estimate_GGD_parameters(vec):
    vec=vec.view(-1)
    gam =np.arange(0.2,10.0,0.001)
    r_gam = (gamma(1/gam)*gamma(3/gam))/((gamma(2/gam))**2)#根据候选的γ计算r(γ)
    sigma_sq=torch.mean((vec)**2).to('cpu').numpy()
    E=torch.mean(torch.abs(vec)).to('cpu').numpy()
    r=sigma_sq/(E**2)#根据sigma^2和E计算r(γ)
    diff=np.abs(r-r_gam)
    gamma_param=gam[np.argmin(diff, axis=0)]
    return gamma_param,sigma_sq

def estimate_AGGD_parameters(vec):
    vec = vec.to('cpu').numpy()
    alpha =np.arange(0.2,10.0,0.001)#产生候选的α
    r_alpha=((gamma(2/alpha))**2)/(gamma(1/alpha)*gamma(3/alpha))#根据候选的γ计算r(α)
    sigma_l=np.sqrt(np.mean(vec[vec<0]**2))
    sigma_r=np.sqrt(np.mean(vec[vec>0]**2))
    gamma_=sigma_l/sigma_r
    u2=np.mean(vec**2)
    m1=np.mean(np.abs(vec))
    r_=m1**2/u2
    R_=r_*(gamma_**3+1)*(gamma_+1)/((gamma_**2+1)**2)
    diff=(R_-r_alpha)**2
    alpha_param=alpha[np.argmin(diff, axis=0)]
    const1 = np.sqrt(gamma(1 / alpha_param) / gamma(3 / alpha_param))
    const2 = gamma(2 / alpha_param) / gamma(1 / alpha_param)
    eta =(sigma_r-sigma_l)*const1*const2
    return alpha_param,eta,sigma_l**2,sigma_r**2


def brisque_feature(imdist:Type[Union[torch.Tensor,np.ndarray]],device='cuda'):

    #算法需要输入为灰度图像，像素值0-255
    if type(imdist)==np.ndarray:
        assert imdist.ndim==2 or imdist.ndim==3
        if imdist.ndim==2:
            imdist=torch.from_numpy(imdist).unsqueeze(0).unsqueeze(0)
        else:
            imdist = torch.from_numpy(imdist).unsqueeze(0)
    # input (Batch,1,H,W)
    assert imdist.dim()==4
    assert imdist.shape[1]==1 or imdist.shape[1]==3

    if torch.max(imdist)<=1:
        imdist = imdist * 255
    # RGB to Gray
    if imdist.shape[1]==3:
        imdist=imdist[:,0,:]*0.299+imdist[:,1,:]*0.587+imdist[:,2,:]*0.114
    # GPU is much much faster
    if 'cuda' in device:
        imdist=imdist.half().to(device)
    elif device=='cpu':
        imdist=imdist.float().to(device)
    else:
        raise ValueError('cpu or cuda',device)

    # 算法主体
    scalenum = 2
    window=gaussian_2d_kernel(7,7/6).unsqueeze(0).unsqueeze(0).float().to(device)
    if 'cuda' in device:
        window=window.half()

    feat=np.zeros((18*scalenum,))
    for i in range(scalenum):
        mu=F.conv2d(imdist,window,stride=1,padding=3)
        mu_sq=mu*mu
        sigma=torch.sqrt(torch.abs(F.conv2d(imdist*imdist,window,stride=1,padding=3)-mu_sq))
        structdis = (imdist - mu) / (sigma + 1)
        del mu, mu_sq,sigma
        feat[i*18],feat[i*18+1] = estimate_GGD_parameters(structdis)

        shifts = [(0,1),(1,0),(1,1),(-1,1)]
        for itr_shift in range(4):
            shifted_structdis=structdis.roll(shifts[itr_shift],(2,3))
            pair=structdis*shifted_structdis
            pair=pair.view(-1)
            feat[i*18+2+itr_shift*4],feat[i*18+3+itr_shift*4],feat[i*18+4+itr_shift*4],feat[i*18+5+itr_shift*4]=estimate_AGGD_parameters(pair)

        imdist=F.interpolate(imdist,scale_factor=(0.5,0.5),mode='bilinear')
    return feat


