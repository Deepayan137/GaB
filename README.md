# One VLM to Keep it Learning: Generation and Balancing for Data-free Continual Visual Question Answering
[![Paper](https://img.shields.io/badge/paper-arxiv.2310.02835-B31B1B.svg)](https://arxiv.org/abs/2411.02210)

<img src="media/method.png" alt="Paper" width="1200">
<div align="left">
> **Abstract:** Vision-Language Models (VLMs) have shown significant promise in Visual Question Answering (VQA) tasks by leveraging web-scale multimodal datasets. However, these models often struggle with continual learning due to catastrophic forgetting when adapting to new tasks. As an effective remedy to mitigate catastrophic forgetting, rehearsal strategy uses the data of past tasks upon learning new task. However, such strategy incurs the need of storing past data, which might not be feasible due to hardware constraints or privacy concerns. In this work, we propose the first data-free method that leverages the language generation capability of a VLM, instead of relying on external models, to produce pseudo-rehearsal data for addressing continual VQA. Our proposal, named as GaB, generates pseudo-rehearsal data by posing previous task questions on new task data. Yet, despite being effective, the distribution of generated questions skews towards the most frequently posed questions due to the limited and task-specific training data. To mitigate this issue, we introduce a pseudo-rehearsal balancing module that aligns the generated data towards the ground-truth data distribution using either the question meta-statistics or an unsupervised clustering method. We evaluate our proposed method on two recent benchmarks, \ie VQACL-VQAv2 and CLOVE-function benchmarks. GaB outperforms all the data-free baselines with substantial improvement in maintaining VQA performance across evolving tasks, while being on-par with methods with access to the past data.
   
## Setup

```bash
# Create python environment (optional)
conda create -n vqacl python=3.7
source activate vqacl

# Install python dependencies
pip install -r requirements.txt
```


## Code structure
```bash
# Store images, features, and annotations
./datasets
    COCO/
        images/
        features/
    vqa/
        Paritition_Q/
    nextqa/
        Paritition_Q/
    ...


# Training and testing in the VQACL setting
./VL-T5/
    src/
        blip2/
        	modeling_blip.py                                  <= Our Blip2 model classes
        analysis/                                             <= question generationa and sampling
        vqacl.py vqa_data_blip.py vqa_model_blip.py ...   	  <= Testing in the VQACL setting
        param.py                                              <= (argparse) configuration
        utils.py                            				  <= utility functions
    snap/                                                     <= store weight checkpoints
    scripts/                                                  <= bash scripts for evaluation
```

## Dataset Preparation / Model checkpoint
- Download the VQACL partition of VQA v2 from [Google Drive](https://drive.google.com/file/d/11gx7AxyeMP1KVuzHErIfNKCLeBWGq3pE/view?usp=share_link) and put it into datasets/nextqa/Partition_Q.
- Download `datasets/COCO` from [Google Drive](https://drive.google.com/drive/folders/1MBBhlkP83VMKS2Qe0SmFfzkHhMpIG5wf?usp=sharing)
- Download model checkpoints from [Google Drive](https://drive.google.com/drive/folders/1Wuqnyo5Rf807te8CXjcgpbMH8vkdeb6M?usp=sharing)

## Usage

```bash
# Training with 1 gpu for VQA v2
cd VL-T5/
bash scripts/train_blip.sh path/to/ckpt_dir balance_strategy # cluster, classifier or unbalanced
# Testing with 1 gpu for VQA v2
cd VL-T5/
bash scripts/test_blip.sh path/to/ckpt_dir
```

**Note:** 
- Download the checkpoint for the first task (recognition) from (recognition from [recognition_ckpt](https://drive.google.com/file/d/1e-s4byIImEEgPlQITw19PtsfHj-mN4Br/view?usp=drive_link)), place it in `snap` folder, and start training or evaluating by specifying the checkpoint's path as a command line argument.
- Before training our model on any subsequent tasks (beyond the first), we need to generate QA pairs as part of our data-free rehearsal strategy.
- While the next section details how to generate and balance data, pre-generated questions and their balanced versions are already available for download at [data link](https://drive.google.com/drive/folders/1LZJ1SDMcl_Rz12pvbBJQwli-dRgulIt5?usp=drive_link). Unzip the files and place the directory inside `datasets/vqa/`


## Question Generation and Balancing strategy

* To train the question generator model, adjust the following settings in the `train_blip.sh` script:
    - Enable caption loss: `--use_cap_loss True`
    - Disable using generated data: `--use_gen_data False`
    - Ensure training starts from scratch: `--train_from_scratch True`
    - Remove the memory flag: remove `--memory`
    - Use `--epochs 1`
    - We provide the trained question generation heads in [link](https://drive.google.com/file/d/1RIn7UjOrIh87Zfgw7Z6BoLV8IouGHGBg/view?usp=sharing). Unzip the folder and place it inside `snap`.

**Note:** It is feasible to train the question generation and answering heads simultaneously, but this approach demands reducing the batch size from `80` to `32` to prevent CUDA out-of-memory errors, significantly slowing down the training process.

* Question Generation:
    - Execute the command: `python -m src.analysis.vqacl_gen_ques`
* Storing Question Category Statistics:
    - Obtain the classifier and sentence representations [here](https://drive.google.com/file/d/1YD3HoHWT7HBzZCDdcnJrUCA30t7oCjno/view?usp=drive_link).
    - Ensure to download and position these files within the `../ckpt_vqacl` directory.
    - Run: `python -m src.analysis.vqacl_question_distribution`
    - This will generate a `../metrics` folder to store all distributions.
    - Note: Classifier training and clustering are conducted exclusively on the training set questions.
 * Balanced Data Generation::
    - After acquiring the question category statistics, generate balanced data using:
    - `python -m src.analysis.vqacl_create_balanced_rehearsal`
    - The default balancing strategy utilized is `cluster`.

* We provide the balanced data files in the links below the usage section.

