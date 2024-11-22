import sys
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from sgvqa_data_blip import SGVQA, SGVQAEvaluator

sys.path.append("..")
from Question_type import Sg_task


def get_loader_multi(args, split="train", scenario="scene", batch_size=32, workers=4, task="object"):
    verbose = True
    dataset_num = 0
    all_datasets = []
    for task in Sg_task["function"]["oarlks"]:
        dataset = SGVQA(split=split, verbose=verbose, args=args, scenario=scenario, task=task)
        all_datasets.append(dataset)
        dataset_num += len(dataset)

    concat_dataset = ConcatDataset(all_datasets)
    print(f"No. of samples in train data is {len(concat_dataset)}")
    if args.distributed:
        sampler = DistributedSampler(concat_dataset)
    else:
        sampler = None
    if split == "train":
        loader = DataLoader(
            concat_dataset,
            batch_size=batch_size,
            shuffle=None if (sampler is not None) else True,
            num_workers=workers,
            pin_memory=True,
            sampler=sampler,
            collate_fn=dataset.collate_fn,
        )
    else:
        loader = DataLoader(
            concat_dataset,
            batch_size=batch_size,
            num_workers=workers,
            pin_memory=True,
            sampler=sampler,
            shuffle=None if (sampler is not None) else False,
            collate_fn=dataset.collate_fn,
            drop_last=False,
        )
    if verbose:
        loader.evaluator = SGVQAEvaluator()

    loader.task = "vqa"
    total_num = dataset_num
    return loader, total_num


def get_loader_test_multi(args, split="train", scenario="scene", batch_size=32, workers=4, task="what"):
    verbose = True
    dataset_num = 0
    all_datasets = []
    for task in Sg_task["function"]["oarlks"]:
        dataset = SGVQA(split=split, verbose=verbose, args=args, scenario=scenario, task=task)
        all_datasets.append(dataset)
        dataset_num += len(dataset)

    concat_dataset = ConcatDataset(all_datasets)
    print(f"No. of samples in test data is {len(concat_dataset)}")
    if args.distributed:
        sampler = DistributedSampler(concat_dataset)
    else:
        sampler = None

    if split == "train":
        loader = DataLoader(
            concat_dataset,
            batch_size=batch_size,
            shuffle=(sampler is None),
            num_workers=workers,
            pin_memory=True,
            sampler=sampler,
            collate_fn=dataset.collate_fn,
        )
    else:
        loader = DataLoader(
            concat_dataset,
            batch_size=batch_size,
            num_workers=workers,
            pin_memory=True,
            sampler=sampler,
            shuffle=None if (sampler is not None) else False,
            collate_fn=dataset.collate_fn,
            drop_last=False,
        )

    if verbose:
        loader.evaluator = SGVQAEvaluator()

    loader.task = "vqa"
    return loader
