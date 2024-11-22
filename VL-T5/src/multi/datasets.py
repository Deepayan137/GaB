import sys
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
from torch.utils.data.distributed import DistributedSampler


from src.vqa_data_blip import VQAFineTuneDataset, VQAEvaluator

sys.path.append("..")
from Question_type import All_task_list


def get_loader_multi(
    args,
    coco_Ours,
    Examplar_set,
    _dset,
    split="karpathy_train",
    mode="train",
    batch_size=32,
    workers=4,
    distributed=False,
    gpu=0,
    topk=-1,
    task="what",
):
    verbose = gpu == 0
    dataset_num = 0
    all_datasets = []
    cates = list(range(80))
    for task in All_task_list:
        dataset = VQAFineTuneDataset(
            coco_Ours,
            Examplar_set,
            split,
            raw_dataset=_dset,
            rank=gpu,
            topk=topk,
            verbose=verbose,
            args=args,
            mode=mode,
            task=task,
            cates=cates,
        )
        all_datasets.append(dataset)
        dataset_num += len(dataset)

    concat_dataset = ConcatDataset(all_datasets)
    print(f"Number of samples in the dataset:{len(concat_dataset)}")
    if distributed:
        sampler = DistributedSampler(concat_dataset)
    else:
        sampler = None
    if mode == "train":
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
        loader.evaluator = VQAEvaluator(_dset)

    loader.task = "vqa"
    total_num = dataset_num
    return loader, total_num


def get_loader_test_multi(
    args,
    coco_Ours,
    Examplar_set,
    _dset,
    split="karpathy_train",
    mode="train",
    batch_size=32,
    workers=4,
    distributed=False,
    gpu=0,
    topk=-1,
    task="what",
):
    verbose = gpu == 0
    dataset_num = 0
    all_datasets = []
    cates = list(range(80))
    for task in All_task_list:
        dataset = VQAFineTuneDataset(
            coco_Ours,
            Examplar_set,
            split,
            raw_dataset=_dset,
            rank=gpu,
            topk=topk,
            verbose=verbose,
            args=args,
            mode=mode,
            task=task,
            cates=[i for i in range(80)],
        )  # all categories
        all_datasets.append(dataset)
        dataset_num += len(dataset)

    concat_dataset = ConcatDataset(all_datasets)
    print(f"Number of samples in the dataset:{len(concat_dataset)}")
    if distributed:
        sampler = DistributedSampler(concat_dataset)
    else:
        sampler = None

    if mode == "train":
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
        loader.evaluator = VQAEvaluator(_dset)

    loader.task = "vqa"
    return loader
