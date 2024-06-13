# HiFormer - LITS: Hierarchical Multi-scale Representations Using Transformers for Medical Image Segmentation, implemented on the LITS Dataset

This is a fork of the original [HiFormer repository](https://github.com/amirhossein-kz/HiFormer/tree/main) which I have modified to implement the Liver Tumor Segmentation (LITS) dataset, instead of the original Synapse Dataset.

All model credits and original implementation credits go to the original authors of the paper.

This code has been implemented in Python 3 using PyTorch, and tested on Ubuntu OS, and popular online data science platforms Kaggle and Google Colab.

## Prepare data, Train/Test/Val
The train.py and test.py scripts require a root path directory.
The structure of this directory should be as follows:

```text
root_path
├── liver_0
│   ├── images
│   │   └── 45.jpg
│   │   └── 46.jpg
|   |   └── ---
│   └── masks
│       ├── liver
|       │   └── 46.jpg
|       │   └── 47.jpg
|       |   └── ---
│       └── cancer
|           └── 46.jpg
|           └── 47.jpg
|           └── ---
|
|
├── liver_130
│   ├── images
│   │   └── 45.jpg
│   │   └── 46.jpg
|   |   └── ---
│   └── masks
│       ├── liver
|       │   └── 46.jpg
|       │   └── 47.jpg
|       |   └── ---
│       └── cancer
|           └── 46.jpg
|           └── 47.jpg
|           └── ---
```
Note that separate folders for train, test, and split are not required!

Based on the random seed given to the train and test script, it will automatically divide the 131 folders into train and split on a 0.8,0.1,0.1 basis.

MAKE SURE TO GIVE THE SAME SEED TO TRAIN AND TEST SCRIPT, else the testing split for test.py may include a folder present in the training split of train.py.

Please follow the instructions below to run the script -

1) Obtain the dataset in jpg format and create the root structure as defined above

1) Run the following code to install the Requirements.

    `pip install -r requirements.txt`

2) Run the below code to train HiFormer on the synapse dataset.

    ```bash
    python train.py --root_path /LITS_Data/ --model_name hiformer-b --batch_size 16 --eval_interval 10 --max_epochs 400 --seed 69 
   ```
    **--root_path**     [Root data path]

    **--eval_interval** [Evaluation epoch]

    **--model_name**    [Choose from [hiformer-s, hiformer-b, hiformer-l]]

    **--is_liver**      [If added, model will train/test using the liver images. If not added, it will train/test on the cancer(tumor) images]

    **--seed**          [Numpy and Torch random seed for reproducibility, and to ensure consistent train-test-val split amongst training and testing]

4) Run the below code to test HiFormer on the synapse dataset.
    ```bash
    python test.py --test_path /LITS_Data/ --model_name hiformer-b --model_weight './hiformer-b_best.pth' --seed 69
    ```
    **--root_path**     [Root data path]
   
    **--model_name**    [Choose from [hiformer-s, hiformer-b, hiformer-l]]

    **--is_liver**      [If added, model will train/test using the liver images. If not added, it will train/test on the cancer(tumor) images]

    **--seed**          [Numpy and Torch random seed for reproducibility, and to ensure consistent train-test-val split amongst training and testing]

    **--model_weight**  [Path to model weights (.pth) file for testing]

## Query

The LITS Implementation has been done by Jasmer Singh Sanjotra.

[*jasmer.sanjotra@gmail.com*](mailto:jasmer.sanjotra@gmail.com)

------

Original implementations are done by Amirhossein Kazerouni, Milad Soltany and Moein Heidari.

[*amirhossein477@gmail.com*](mailto:amirhossein477@gmail.com)

[*soltany.m.99@gmail.com*](mailto:soltany.m.99@gmail.com)

[*moeinheidari7829@gmail.com*](mailto:moeinheidari7829@gmail.com)

Original Authors -
</br>
> [Moein Heidari](https://scholar.google.com/citations?user=mir8D5UAAAAJ&hl=en&oi=sra)\*,
[Amirhossein Kazerouni](https://scholar.google.com/citations?user=aKDCc3MAAAAJ&hl=en)\*, [Milad Soltany](https://scholar.google.com/citations?view_op=list_works&hl=en&hl=en&user=Gm23tVgAAAAJ)\*, [R. Azad](https://scholar.google.com/citations?hl=en&user=Qb5ildMAAAAJ&view_op=list_works&sortby=pubdate), [Ehsan Khodapanah Aghdam](https://scholar.google.com/citations?user=a4DcyOYAAAAJ&hl=en), [Julien Cohen-Adad](https://scholar.google.ca/citations?user=6cAZ028AAAAJ&hl=en) and [Dorit Merhof
](https://scholar.google.com/citations?user=JH5HObAAAAAJ&sortby=pubdate), "HiFormer: Hierarchical Multi-scale Representations Using Transformers for Medical Image Segmentation", download [link](https://arxiv.org/pdf/2207.08518).

