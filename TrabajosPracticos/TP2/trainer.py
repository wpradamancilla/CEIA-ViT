"""
trainer.py

Author: Abraham Rodriguez
DATE: 24/5/2023
"""
import copy

import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm



class EarlyStopping():
	"""
	EarlyStopping serves as a mechanism to check if the loss does not have a considerable change, this can help to prevent overfitting
	and reduce the number of epochs (training time).
	"""
	def __init__(self, patience:int=5, min_delta :float=0, restore_best_weights:bool=True):
		"""

		Class constructor, sets mechanism to a certain quantity of patience, and a defined min_delta,
		and the best weights of the trained model.

		:param patience : patience to stop
		:type patience : int

		:param min_delta : minimum difference between losses per epoch.
		:type min_delta : float

		:param restore_best_weights :  restore best model
		:type restore_best_weights : bool

		"""
		self.patience = patience
		self.min_delta = min_delta
		self.restore_best_weights = restore_best_weights
		self.best_model = None
		self.best_loss = None
		self.counter = 0
		self.status = ""

	def __call__(self, model:torch.nn.Module, val_loss: float):
		"""
		Excutes logic when calling EarlyStopping object e.g
		es = EarlyStopping(patience=5)
		es(model,val_loss)
		"""
		if self.best_loss is None:
			self.best_loss = val_loss
			self.best_model = copy.deepcopy(model)

		elif self.best_loss - val_loss > self.min_delta:
			self.best_loss = val_loss
			self.counter = 0
			self.best_model.load_state_dict(model.state_dict())

		elif self.best_loss - val_loss < self.min_delta:
			self.counter += 1
			if self.counter >= self.patience:
				self.status = f"Stopped on {self.counter}"
				if self.restore_best_weights:
					model.load_state_dict(self.best_model.state_dict())
				return True

		self.status = f"{self.counter}/{self.patience}"
		return False

class Trainer():
	"""
	Custom trainer Class that wraps the training and evaluation of a model, using torch autocast
	"""
	def __init__(self, model : torch.nn.Module, train_data_loader: DataLoader,
							test_data_loader: DataLoader ,loss_fn:torch.nn.Module,
								optimizer: torch.optim.Optimizer, device: str):
		"""

		Class constructor, sets mechanism to a certain quantity of patience, and a defined min_delta,
		and the best weights of the trained model.

		:param model : patience to stop
		:type model : torch.nn.Module

		:param train_data_loader : minimum difference between losses per epoch.
		:type train_data_loader : torch.utils.data.DataLoader

		:param test_data_loader :  restore best model
		:type test_data_loader : torch.utils.data.DataLoader

		:param loss_fn :  restore best model
		:type loss_fn : torch.nn.Module

		:param optimizer :  restore best model
		:type optimizer: torch.optim.Optimizer

		:param device :  restore best model
		:type device: str

		"""
		self.model = model
		self.train_data_loader = train_data_loader
		self.test_data_loader = test_data_loader
		self.loss_fn = loss_fn
		self.metrics = None
		self.optimizer = optimizer
		self.device = device

	# @property
	# def device(self):
	# 	return self.device

	# @device.setter
	# def device(self, new_device : str ):
	# 	self.device = new_device

	# @device.deleter
	# def device(self):
	# 	del self.device

	def train_model(self,use_amp = False, dtype : torch.dtype = torch.bfloat16):

		model = self.model.train()
		#scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
		losses = []
		bar = tqdm(self.train_data_loader)
		for train_input, train_mask in bar:
	
				train_mask = train_mask.to(self.device)
				train_input=train_input.to(self.device)
				with torch.autocast(device_type='cuda', dtype=dtype, enabled=use_amp):
					output = model(train_input)
					loss = self.loss_fn(output, train_mask)
				# if isinstance(dtype, type(torch.float16)):
				# 	scaler.scale(loss).backward()
				# 	scaler.step(self.optimizer)
				# 	scaler.update()
				# else:
					
				loss.backward()
				self.optimizer.step()

				# outputs=model(train_input.float())
				# loss = loss_fn(outputs.float(), train_mask.float())
				losses.append(loss.item())
				#loss.backward()
				#optimizer.step()
				#optimizer.zero_grad()
				for param in model.parameters():
					param.grad = None
				bar.set_description(f"loss {loss:.5f}")
		return np.mean(losses)

	def eval_model(self):
		model = self.model.eval()

		losses = []
		bar = tqdm(self.test_data_loader)
		with torch.no_grad():
				for val_input, val_mask in bar:

						val_mask = val_mask.to(self.device)
						val_input=val_input.to(self.device)
						outputs=model(val_input)

						loss = self.loss_fn(outputs, val_mask)
						losses.append(loss.item())
						bar.set_description(f"val_loss {loss:.5f}")

		return np.mean(losses)