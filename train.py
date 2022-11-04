
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.utils as utils
from tensorboardX import SummaryWriter
from ReactionNet_StochasticLoss import ReactionNet
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
	dataset_train = Dataset(train=True, gray_mode=args.gray, shuffle=True)
	dataset_val = Dataset(train=False, gray_mode=args.gray, shuffle=False)
	loader_train = DataLoader(dataset=dataset_train, num_workers=6, \
							   batch_size=args.batch_size, shuffle=True)
	print("\t# of training samples: %d\n" % int(len(dataset_train)))

	# Init loggers
	if not os.path.exists(args.log_dir):
		os.makedirs(args.log_dir)
	writer = SummaryWriter(args.log_dir)
	logger = init_logger(args)
	remark_psnr = 0
	ratio = 10

	# Create model
	if not args.gray:
		in_ch = 3
	else:
		in_ch = 1
	net = ReactionNet(num_input_channels=in_ch)

	# Initialize model with He init
	#net.apply(weights_init_orthogonal)

	# Define loss
	#criterion = nn.MSELoss(size_average=False)
	criterion = nn.L1Loss()

	# Move to GPU
	device_ids = [0]
	model = nn.DataParallel(net, device_ids=device_ids).cuda()
	criterion.cuda()

	# Optimizer
	optimizer = optim.Adam(model.parameters(), lr=args.lr)
	scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40, 60, 80], gamma=0.5)
	# resume = os.path.join(args.log_dir, 'ckpt_50.pth')# Training noise_level 25 by fine tuning nL 15
	# loadmodel = torch.load(resume)
	# model.load_state_dict(loadmodel['state_dict'])

	## Resume training or start anew
	if args.resume_training:
		resumef = os.path.join(args.log_dir, 'ckpt.pth')
		if os.path.isfile(resumef):
			checkpoint = torch.load(resumef)
			print("> Resuming previous training")
			model.load_state_dict(checkpoint['state_dict'])
			optimizer.load_state_dict(checkpoint['optimizer'])
			new_epoch = args.epochs
			new_milestone = args.milestone
			current_lr = args.lr
			args = checkpoint['args']
			training_params = checkpoint['training_params']
			start_epoch = training_params['start_epoch']
			args.epochs = new_epoch
			args.milestone = new_milestone
			args.lr = current_lr
			print("=> loaded checkpoint '{}' (epoch {})"\
				  .format(resumef, start_epoch))
			print("=> loaded parameters :")
			print("==> checkpoint['optimizer']['param_groups']")
			print("\t{}".format(checkpoint['optimizer']['param_groups']))
			print("==> checkpoint['training_params']")
			for k in checkpoint['training_params']:
				print("\t{}, {}".format(k, checkpoint['training_params'][k]))
			argpri = vars(checkpoint['args'])
			print("==> checkpoint['args']")
			for k in argpri:
				print("\t{}, {}".format(k, argpri[k]))

			args.resume_training = False
		else:
			raise Exception("Couldn't resume training with checkpoint {}".\
				   format(resumef))
	else:
		start_epoch = 0
		training_params = {}
		training_params['step'] = 0
		training_params['current_lr'] = 0
		training_params['no_orthog'] = args.no_orthog

	# Training
	for epoch in range(start_epoch, args.epochs):
		optimizer.step()

		for param_group in optimizer.param_groups:
			print('learning rate %f' % param_group["lr"])
		train_loss = 0

		# train
		for i, data in enumerate(loader_train, 0):
			# Pre-training step
			model.train()
			model.zero_grad()
			optimizer.zero_grad()

			# inputs: noise and noisy image
			img_train = data
			noise = torch.zeros(img_train.size())
			stdn = args.noiseL
			for nx in range(noise.size()[0]):
				sizen = noise[0, :, :, :].size()
				noise[nx, :, :, :] = torch.FloatTensor(sizen).\
									normal_(mean=0, std=float(stdn))
			imgn_train = img_train + noise

			# Create input Variables
			img_train = Variable(img_train.cuda(), volatile=True)
			imgn_train = Variable(imgn_train.cuda(), volatile=True)
			label = Variable(img_train.cuda(), volatile=True)

			out_train = model(imgn_train)
			loss = criterion(out_train, label) / (imgn_train.size()[0]/2)
			loss.backward()
			optimizer.step()
			train_loss += loss.data

			# Results
			model.eval()
			est_res = model(imgn_train)
			out_train = torch.clamp(est_res, 0., 1.)
			psnr_train = batch_psnr(out_train, img_train, 1.)
			emp_loss = train_loss/(i+1)
			emp_loss=emp_loss.cpu()

			if training_params['step'] % args.save_every == 0:
				# Apply regularization by orthogonalizing filters
				#if not training_params['no_orthog']:
				#	model.apply(svd_orthogonalization)
				# Log the scalar values
				writer.add_scalar('loss', loss.data, training_params['step'])
				writer.add_scalar('PSNR on training data', psnr_train, \
					  training_params['step'])
				print("[epoch %d][%d/%d] loss: %.4f PSNR_train: %.4f" %\
					(epoch+1, i+1, len(loader_train), emp_loss, psnr_train))
			training_params['step'] += 1
		# At the end of each epoch
		with open(os.path.join(args.log_dir, 'loss.txt'), 'a') as f:
			f.write(str(emp_loss))
			f.write('\n')
		scheduler.step()

		# Validation
		model.eval()

		psnr_val = 0
		for valimg in dataset_val:
			img_val = torch.unsqueeze(valimg, 0)
			noise = torch.FloatTensor(img_val.size()).\
					normal_(mean=0, std=args.val_noiseL)
			imgn_val = img_val + noise
			img_val, imgn_val = Variable(img_val.cuda()), Variable(imgn_val.cuda())
			with torch.no_grad():
				vest_res = model(imgn_val)
				out_val = torch.clamp(vest_res,0.,1,)
			psnr_val += batch_psnr(out_val, img_val, 1.)
		psnr_val /= len(dataset_val)
		print("\n[epoch %d] PSNR_val: %.4f" % (epoch+1, psnr_val))
		writer.add_scalar('PSNR on validation data', psnr_val, epoch)
		# writer.add_scalar('Learning rate', current_lr, epoch)
		with open(os.path.join(args.log_dir,'val_psnr.txt'),'a') as f:
			f.write(str(psnr_val))
			f.write('\n')

		# Log val images
		try:
			if epoch == 0:
				# Log graph of the model
				writer.add_graph(model, (imgn_val), )
				# Log validation images
				for idx in range(2):
					imclean = utils.make_grid(img_val.data[idx].clamp(0., 1.), \
											nrow=2, normalize=False, scale_each=False)
					imnsy = utils.make_grid(imgn_val.data[idx].clamp(0., 1.), \
											nrow=2, normalize=False, scale_each=False)
					writer.add_image('Clean validation image {}'.format(idx), imclean, epoch)
					writer.add_image('Noisy validation image {}'.format(idx), imnsy, epoch)
			for idx in range(2):
				imrecons = utils.make_grid(out_val.data[idx].clamp(0., 1.), \
										nrow=2, normalize=False, scale_each=False)
				writer.add_image('Reconstructed validation image {}'.format(idx), \
								imrecons, epoch)
			# Log training images
			imclean = utils.make_grid(img_train.data, nrow=8, normalize=True, \
						 scale_each=True)
			writer.add_image('Training patches', imclean, epoch)

		except Exception as e:
			logger.error("Couldn't log results: {}".format(e))

		# save model and checkpoint
		training_params['start_epoch'] = epoch + 1
		torch.save(model.state_dict(), os.path.join(args.log_dir, 'net.pth'))
		save_dict = { \
			'state_dict': model.state_dict(), \
			'optimizer' : optimizer.state_dict(), \
			'training_params': training_params, \
			'args': args\
			}
		if remark_psnr<=psnr_val:
			remark_psnr=psnr_val
			print("\n Update the best result with PSNR: %.4f" % psnr_val)
			torch.save(save_dict, os.path.join(args.log_dir, 'ckpt.pth')) # ckpt.pth里面存储了目前最好的训练结果
		else:
			print("\n The best PSNR is %.4f" % remark_psnr)
		del save_dict

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description="FFDNet")
	parser.add_argument("--gray", action='store_true',\
						default= True, help='train grayscale image denoising instead of RGB')

	parser.add_argument("--log_dir", type=str, default="logs", \
					 help='path of log files')
	#Training parameters
	parser.add_argument("--batch_size", type=int, default=16, 	\
					 help="Training batch size")
	parser.add_argument("--num_iters", type=int, default=5, 	\
					 help="number of iterations")
	parser.add_argument("--epochs", "--e", type=int, default=100, \
					 help="Number of total training epochs")
	parser.add_argument("--resume_training", "--r", action='store_true',\
						default = False, help="resume training from a previous checkpoint")
	parser.add_argument("--milestone", nargs=2, type=int, default=[30, 50], \
						help="When to decay learning rate; should be lower than 'epochs'")
	parser.add_argument("--lr", type=float, default=1e-4, \
					 help="Initial learning rate")
	parser.add_argument("--no_orthog", action='store_true',\
						default = False, help="Don't perform orthogonalization as regularization")
	parser.add_argument("--save_every", type=int, default=100,\
						help="Number of training steps to log psnr and perform \
						orthogonalization")
	parser.add_argument("--save_every_epochs", type=int, default=2,\
						help="Number of training epochs to save state")
	parser.add_argument("--noiseL", nargs=1, type=int, default=50, \
					 help="Noise training interval")
	parser.add_argument("--val_noiseL", type=float, default=25, \
						help='noise level used on validation set')
	argspar = parser.parse_args()
	# Normalize noise between [0, 1]
	argspar.val_noiseL /= 255.
	argspar.noiseIntL[0] /= 255.
	argspar.noiseIntL[1] /= 255.

	print("\n### Training FFDNet model ###")
	print("> Parameters:")
	for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
		print('\t{}: {}'.format(p, v))
	print('\n')

	main(argspar)
