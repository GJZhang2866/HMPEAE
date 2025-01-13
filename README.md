Source code for  ACL 2024 paper: [Hyperspherical Multi-Prototype with Optimal Transport for Event Argument Extraction].

 Our code is based on TabEAE (Revisiting Event Argument Extraction: Can EAE Models Learn Better When Being Aware of Event Co-occurrences? ) [here](https://github.com/Stardust-hyx/TabEAE) and thanks for their implement.

## How to use?

### 1. Dependencies
In addition to the environment required by TabEAE, you will need to install the following additional packages:

- rich
- POT==0.9.5


### 2. Train Prototype
We provide prototypes of the precomputations used in the paper in the folders [prototypes_rams](./prototypes_rams) and [prototypes_wiki](./prototypes_wiki)

To create your own prototypes, use the ```built_prototypes.py``` script. 

An example run for 66 classes, each with 2 prototypes and 1024 dimensions:

```bash
>>  python built_prototypes.py -c 66 -d 1024 -nppt 2 -hd 768 -w Semantic_vectors/rams_bert_sem.npy
```
[Semantic_vectors/rams_bert_sem](./Semantic_vectors/rams_bert_sem) is the semantic vector of role labels obtained by bert. You can use```get_label_semnatic.py``` to get your Semantic_vectors file. You can also use other encoders like Word2Vec or Sentenc-Bert.

### 3. Training and Evaluation

The training scripts are provided.

```bash
>> bash scripts/train_rams.sh
>> bash scripts/train_wikievent.sh
```

You can change the settings in the corresponding scripts.

And you can evaluate the model by the following scripts.

```bash
>> bash scripts/infer_rams.sh
>> bash scripts/infer_wikievent.sh
```

You can download our best model checkpoint [here](https://pan.baidu.com/s/1D9ig-CfbHoXjYSoU85o4MQ?pwd=epgm).

If you have any questions, pls contact us via zgj2866@gmail.com. Thanks!



