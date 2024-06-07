# [HiFormer: Hierarchical Multi-scale Representations Using Transformers for Medical Image Segmentation](https://arxiv.org/pdf/2207.08518.pdf), [WACV 2023](https://wacv2023.thecvf.com/home)

This is a fork of the original HiFormer repository which I have modified to add implementation of the Liver Tumor Segmentation (LITS) dataset, instead of the original Synapse Dataset.
All model credits and original implementation credits go to the original authors of the paper. I have just an implementation of the LITS dataset:

Original Authors -
</br>
> [Moein Heidari](https://scholar.google.com/citations?user=mir8D5UAAAAJ&hl=en&oi=sra)\*,
[Amirhossein Kazerouni](https://scholar.google.com/citations?user=aKDCc3MAAAAJ&hl=en)\*, [Milad Soltany](https://scholar.google.com/citations?view_op=list_works&hl=en&hl=en&user=Gm23tVgAAAAJ)\*, [R. Azad](https://scholar.google.com/citations?hl=en&user=Qb5ildMAAAAJ&view_op=list_works&sortby=pubdate), [Ehsan Khodapanah Aghdam](https://scholar.google.com/citations?user=a4DcyOYAAAAJ&hl=en), [Julien Cohen-Adad](https://scholar.google.ca/citations?user=6cAZ028AAAAJ&hl=en) and [Dorit Merhof
](https://scholar.google.com/citations?user=JH5HObAAAAAJ&sortby=pubdate), "HiFormer: Hierarchical Multi-scale Representations Using Transformers for Medical Image Segmentation", download [link](https://arxiv.org/pdf/2207.08518).

This code has been implemented in Python using Pytorch library and tested in ubuntu OS, though should be compatible with related environment. following Environement and Library needed to run the code:

- Python 3
- Pytorch

## Prepare data, Train/Test
[ALERT: THE FOLLOWING IS FROM THE ORIGINAL REPO. TO BE UPDATED SOON]
Please go to ["Instructions.ipynb"](https://github.com/amirhossein-kz/HiFormer/blob/main/Instructions.ipynb) for complete detail on dataset preparation and Train/Test procedure or follow the instructions below. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/amirhossein-kz//HiFormer/blob/main/Instructions.ipynb)


1) Download the Synapse Dataset from [here](https://drive.google.com/uc?export=download&id=18I9JHH_i0uuEDg-N6d7bfMdf7Ut6bhBi).

2) Run the following code to install the Requirements.

    `pip install -r requirements.txt`

3) Run the below code to train HiFormer on the synapse dataset.

    ```bash
    python train.py --root_path ./data/Synapse/train_npz --test_path ./data/Synapse/test_vol_h5  --model_name hiformer-b --batch_size 10 --eval_interval 20 --max_epochs 400 
   ```
    **--root_path**     [Train data path]

    **--test_path**     [Test data path]

    **--eval_interval** [Evaluation epoch]

    **--model_name**    [Choose from [hiformer-s, hiformer-b, hiformer-l]]
    
4) Run the below code to test HiFormer on the synapse dataset.
    ```bash
    python test.py --test_path ./data/Synapse/test_vol_h5 --model_name hiformer-b --is_savenii --model_weight './hiformer-b_best.pth'
    ```
    **--test_path**     [Test data path]
    
    **--model_name**    [choose from [hiformer-s, hiformer-b, hiformer-l]]
    
    **--is_savenii**    [Whether to save results during inference]
    
    **--model_weight**  [HiFormer trained model path]


## Query
All implementations are done by Amirhossein Kazerouni, Milad Soltany and Moein Heidari. For any query, please contact us for more information.

[*amirhossein477@gmail.com*](mailto:amirhossein477@gmail.com)

[*soltany.m.99@gmail.com*](mailto:soltany.m.99@gmail.com)

[*moeinheidari7829@gmail.com*](mailto:moeinheidari7829@gmail.com)


## Citation
```python
@inproceedings{heidari2023hiformer,
  title={Hiformer: Hierarchical multi-scale representations using transformers for medical image segmentation},
  author={Heidari, Moein and Kazerouni, Amirhossein and Soltany, Milad and Azad, Reza and Aghdam, Ehsan Khodapanah and Cohen-Adad, Julien and Merhof, Dorit},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={6202--6212},
  year={2023}
}
```
