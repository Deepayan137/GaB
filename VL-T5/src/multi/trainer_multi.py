from tqdm import *

from src.vqacl import Trainer
from utils import LossMeter


class TrainerMulti(Trainer):
    def train(self, load=False):
        from src.multi.datasets import get_loader_multi, get_loader_test_multi

        args = self.args
        task = "q_causal"
        print(f"Loading {task}")
        print(f"Building Multi train loader at GPU {args.gpu}")

        train_loader, total_num_Q = get_loader_multi(
            args,
            self.coco_Ours,
            [],
            self.train_dset,
            split=args.train,
            mode="train",
            batch_size=args.batch_size,
            distributed=args.distributed,
            gpu=args.gpu,
            workers=args.num_workers,
            topk=args.train_topk,
            task=task,
        )
        print(f"Building Multi val loader at GPU {args.gpu}")
        val_loader, _ = get_loader_multi(
            args,
            self.coco_Ours,
            [],
            self.val_dset,
            split=args.valid,
            mode="val",
            batch_size=args.valid_batch_size,
            distributed=args.distributed,
            gpu=args.gpu,
            workers=4,
            topk=args.valid_topk,
            task=task,
        )

        print(f"Building test loader at GPU {args.gpu}")
        test_loader = get_loader_test_multi(
            args,
            self.coco_Ours,
            [],
            self.test_dset,
            split=args.test,
            mode="val",
            batch_size=args.valid_batch_size,
            distributed=args.distributed,
            gpu=args.gpu,
            workers=4,
            topk=args.valid_topk,
            task=task,
        )

        # if self.verbose:
        self.optim, self.lr_scheduler = self.create_optimizer_and_scheduler(None)
        All_examplar = []
        each_memory = 0
        loss_meter = LossMeter()
        loss_meter_mem = LossMeter()
        best_valid = 0.0
        best_epoch = 0
        valid_score_raw_best = 0.0
        if self.args.distributed:
            dist.barrier()
        start_epoch = 0

        valid_score_raw_best = 0.0
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
                "loss": 0.0,
            }
            for batch in train_loader:
                results, lr = self.train_step(batch, epoch_results, 0, each_memory)

                loss_meter.update(results["loss"].item())

                desc_str = f"Epoch {epoch} | LR {lr:.6f} | Loss {loss_meter.val:.4f}"
                loss_meter_mem.update(-1)

                if args.show_train_progress:
                    pbar.set_description(desc_str)
                    pbar.update(1)

            if args.show_train_progress:
                pbar.close()

            # if args.gpu == 0:
            print(f"Epoch {epoch}| Loss: {loss_meter.val}, Loss_mem: {loss_meter_mem.val}")
            score_dict = self.evaluate(val_loader, task)
            valid_score = score_dict["topk_score"] * 100.0
            valid_score_raw = score_dict["overall"]
            log_str = ""
            log_str += "\nEpoch %d: Valid Raw %0.2f Topk %0.2f" % (epoch, valid_score_raw, valid_score)
            if valid_score_raw > valid_score_raw_best:
                valid_score_raw_best = valid_score_raw
                patience_counter = 0  # Reset the patience counter
                print("Saving Best")
                self.save(task + "_BEST")
            else:
                patience_counter += 1  # Increment the patience counter
                print(f"No improvement for {patience_counter} epochs.")
            if patience_counter > patience:
                print("Early stopping triggered.")
                break  # Break out of the training loop
            # self.save(task + f"{epoch}")
        print("Saving Last")
        self.save(task + "_LAST")
