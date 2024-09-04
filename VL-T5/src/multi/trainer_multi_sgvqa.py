from tqdm import *

from src.sgvqa import Trainer

from Question_type import Sg_task, show_results_matrix
from utils import load_state_dict, LossMeter

class SGTrainerMulti(Trainer):
	def train(self, load=False):
		from src.multi.datasets_sgvqa import get_loader_multi, get_loader_test_multi
		args = self.args
		task = 'knowledge'
		print(f"Loading {task}")
		print(f'Building train loader at GPU {args.gpu}')

		if load:
			self.load(self.args.checkpoint)

		train_loader, total_num_Q = get_loader_multi(
			args,
			split=args.train, scenario=args.scenario, 
			batch_size=args.batch_size,
			workers=args.num_workers,
			task=task,
		)
		print(f'Building val loader at GPU {args.gpu}')

		val_loader, _ = get_loader_multi(
			args,
			split=args.valid, scenario=args.scenario, 
			batch_size=args.valid_batch_size,
			workers=4,
			task=task,
		)

		print(f'Building test loader at GPU {args.gpu}')
		test_loader = get_loader_test_multi(
				args,
				split=args.test, scenario=args.scenario, 
				batch_size=args.valid_batch_size,
				workers=4,
				task=task,
			)

		# if self.verbose:
		self.optim, self.lr_scheduler = self.create_optimizer_and_scheduler(None)
		All_examplar = []
		each_memory = 0
		loss_meter = LossMeter()
		loss_meter_mem = LossMeter()
		best_valid = 0.
		best_epoch = 0
		valid_score_raw_best = 0.0
		if self.args.distributed:
			dist.barrier()
		start_epoch = 0
		
		patience_counter = 0
		patience = 5
		task_idx = 0
		for epoch in range(start_epoch, self.args.epochs):
			self.model.train()
			if self.args.distributed:
				train_loader.sampler.set_epoch(epoch)

			if args.show_train_progress:
				pbar = tqdm(total=len(train_loader), ncols=120)
			epoch_results = {
				'loss': 0.,
				}

			for batch in train_loader:
				results, lr = self.train_step(batch, epoch_results, 0, each_memory)

				if self.args.distributed:
					# Sum the loss across all processes
					distributed_loss = 0.
					distributed_loss = results['loss'].detach()
					dist.all_reduce(distributed_loss, op=dist.ReduceOp.SUM)
					distributed_loss = distributed_loss / self.args.world_size  # Average the loss
					loss_meter.update(distributed_loss.item())
				else:
					# Non-distributed, business as usual
					loss_meter.update(results['loss'].item())

				desc_str = f'Epoch {epoch} | LR {lr:.6f} | Loss {loss_meter.val:.4f}'
				loss_meter_mem.update(-1)

				if args.show_train_progress:
					pbar.set_description(desc_str)
					pbar.update(1)

				if self.args.distributed:
					dist.barrier()
				
			if args.show_train_progress:
				pbar.close()

			if args.gpu == 0:
				print(f"Epoch {epoch}| Loss: {loss_meter.val}, Loss_mem: {loss_meter_mem.val}")
				score_dict = self.evaluate(val_loader, task)
				# valid_score = score_dict['topk_score'] * 100.
				valid_score_raw = score_dict['overall']
				log_str = ''
				# log_str += "\nEpoch %d: Valid Raw %0.2f Topk %0.2f" % (epoch, valid_score_raw, valid_score)
				log_str += "\nEpoch %d: Valid Raw %0.2f" % (epoch, valid_score_raw)
				print(log_str)
			
			if valid_score_raw > valid_score_raw_best:
				valid_score_raw_best = valid_score_raw
				patience_counter = 0  # Reset the patience counter
				print("Saving Best")
				self.save(task + "_BEST")
			else:
				patience_counter += 1  # Increment the patience counter
				print(f"No improvement for {patience_counter} epochs.")
				# self.save(task + "_LAST")
			if patience_counter > patience:
				print("Early stopping triggered.")
				print("Saving Last")
				break  # Break out of the training loop
			# self.save(task + f"{epoch}")
			
			if self.args.distributed:
				dist.barrier()
		print("Saving Last")
		self.save(task + "_LAST")
