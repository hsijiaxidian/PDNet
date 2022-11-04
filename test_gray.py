"""
Trains a FFDNet model

By default, the training starts with a learning rate equal to 1e-3 (--lr).
After the number of epochs surpasses the first milestone (--milestone), the
lr gets divided by 100. Up until this point, the orthogonalization technique
described in the FFDNet paper is performed (--no_orthog to set it off).

Copyright (C) 2018, Matias Tassano <matias.tassano@parisdescartes.fr>

This program is free software: you can use, modify and/or
redistribute it under the terms of the GNU General Public
License as published by the Free Software Foundation, either
version 3 of the License, or (at your option) any later
version. You should have received a copy of this license along
this program. If not, see <http://www.gnu.org/licenses/>.
"""
import os
import argparse
import numpy
import torch
import cv2
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.utils as utils
from tensorboardX import SummaryWriter
from UniformNet import UniformNet
from dataset import Dataset
from utils import weights_init_orthogonal, batch_psnr, init_logger, \
			svd_orthogonalization
import warnings
warnings.filterwarnings("ignore")

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main(args):
	r"""Performs the main training loop
	"""
	# Load dataset
	print('> Loading dataset ...')
	dataset_val = Dataset(train=False, gray_mode=args.gray, shuffle=False)

	in_ch = 1
	net = UniformNet(in_ch)

	# Move to GPU
	device_ids = [0]
	model = nn.DataParallel(net, device_ids=device_ids).cuda()

	modeltld = torch.load('logs/ckpt.pth')
	model.load_state_dict(modeltld['state_dict'])
	model.eval()

	# Validation
	outputpath = "Results/Set12_25/"
	psnr_val = 0
	num = 1
	numpy.random.seed(0)
	for valimg in dataset_val:
		img_val = torch.unsqueeze(valimg, 0)
		noise = torch.FloatTensor(img_val.size()).\
					normal_(mean=0, std=args.val_noiseL)
		imgn_val = img_val + noise
		img_val, imgn_val = Variable(img_val.cuda()), Variable(imgn_val.cuda())
		ratio = (args.val_noiseL - 15/255)/(35/255)
		with torch.no_grad():
			out_val = torch.clamp(model(imgn_val, ratio), 0., 1.)
		cur_psnr = batch_psnr(out_val,img_val,1.)
		imgna = outputpath + str(num) + '_' + str(cur_psnr) + '.png'
		output = out_val.clone()
		output = torch.squeeze(output).cpu()
		estimg = output.data.numpy()
		cv2.imwrite(imgna, estimg*255)
		psnr_val += batch_psnr(out_val, img_val, 1.)
		num +=1
	psnr_val /= len(dataset_val)
	print("\n PSNR_val: %.4f" % ( psnr_val))
		# writer.add_scalar('PSNR on validation data', psnr_val, epoch)

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description="FFDNet")
	parser.add_argument("--gray", action='store_true',\
						default= True, help='train grayscale image denoising instead of RGB')
	parser.add_argument("--val_noiseL", type=float, default=45, \
						help='noise level used on validation set')
	argspar = parser.parse_args()
	# Normalize noise between [0, 1]
	argspar.val_noiseL /= 255.

	print("\n### Training FFDNet model ###")
	print("> Parameters:")
	for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
		print('\t{}: {}'.format(p, v))
	print('\n')

	main(argspar)
