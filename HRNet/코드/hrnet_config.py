# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Create by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Ke Sun (sunk@mail.ustc.edu.cn), Rainbowsecret (yuyua@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from yacs.config import CfgNode as CN


# configs for HRNet32
HRNET_32 = CN()
HRNET_32.FINAL_CONV_KERNEL = 1
# main body contains four stages
# 1/4 resolution
HRNET_32.STAGE1 = CN()
HRNET_32.STAGE1.NUM_MODULES = 1
HRNET_32.STAGE1.NUM_BRANCHES = 1
HRNET_32.STAGE1.NUM_BLOCKS = [4]  # first stage contains 4 residual unit
HRNET_32.STAGE1.BLOCK = 'BOTTLENECK' # where each unit is formed by a bottleneck
HRNET_32.STAGE1.NUM_CHANNELS = [64] #  with width 64
HRNET_32.STAGE1.FUSE_METHOD = 'SUM'

# each branch in multi-resolution parallel convolution of modularized block contains 4 residual units
"""
(HighResolutionModule) : modularized blocks

Each stage consists of modularized blocks, repeated 1, 1, 4, and 3 times, respectively for the four stages. 
The modularized block consists of 1 (2, 3 and 4) branches for the 1st (2nd, 3rd and 4th) stages.
"""

# 1/8 resolution
HRNET_32.STAGE2 = CN() # followed by one 3x3 conv changing the width of featuremap to C
HRNET_32.STAGE2.NUM_MODULES = 1 # 2nd stage contain 1 modularized block
HRNET_32.STAGE2.NUM_BRANCHES = 2
HRNET_32.STAGE2.NUM_BLOCKS = [4, 4] # each branch in multi-resolution parallel convolution of modularized block contains 4 residual units 
                                    # --> each unit contains two 3x3 conv (+BN, ReLU) for each resolution 
                                            # width of convolution of four resolution C, 2C, 4C, 8C
HRNET_32.STAGE2.NUM_CHANNELS = [32, 64]
HRNET_32.STAGE2.BLOCK = 'BASIC'
HRNET_32.STAGE2.FUSE_METHOD = 'SUM'
# 1/16 resolution
HRNET_32.STAGE3 = CN()
HRNET_32.STAGE3.NUM_MODULES = 4 # 3rd stage contain 4 modularized blocks
HRNET_32.STAGE3.NUM_BRANCHES = 3
HRNET_32.STAGE3.NUM_BLOCKS = [4, 4, 4] # each branch in multi-resolution parallel convolution of modularized block contains 4 residual units
HRNET_32.STAGE3.NUM_CHANNELS = [32, 64, 128] 
HRNET_32.STAGE3.BLOCK = 'BASIC'
HRNET_32.STAGE3.FUSE_METHOD = 'SUM'  # each output representation is the sum of the transformed representations of three inputs
# 1/32 resolution
HRNET_32.STAGE4 = CN()
HRNET_32.STAGE4.NUM_MODULES = 3 # 4th stage contain 3 modularized blocks
HRNET_32.STAGE4.NUM_BRANCHES = 4
HRNET_32.STAGE4.NUM_BLOCKS = [4, 4, 4, 4] # each branch in multi-resolution parallel convolution of modularized block contains 4 residual units
HRNET_32.STAGE4.NUM_CHANNELS = [32, 64, 128, 256]
HRNET_32.STAGE4.BLOCK = 'BASIC'
HRNET_32.STAGE4.FUSE_METHOD = 'SUM'




MODEL_CONFIGS = {
    'hrnet32': HRNET_32,
}