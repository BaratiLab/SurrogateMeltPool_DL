# SurrogateMeltPool_DL
Pytorch Implementation for Surrogate Modeling of Melt Pool Temperature Field using Deep Learning

## Installing
Clone the repository on your local machine
```
git clone https://github.com/BaratiLab/SurrogateMeltPool_DL.git
```
Install the required packages. It is recommended to proceed in a new environment.
```
pip install -r requirements.txt
```
## Datasets
The paper uses three datasets, one of which is publicly available as an example.
You can find the first dataset (Ti64-5 in the paper) [here](https://drive.google.com/file/d/1QkKCXeMPpXUZOrJnbJ-xTfV2oQFG_-y8/view?usp=share_link).
Please download the zip file and unzip in  the directory Datasets/Ti64-5_cropped/

## Using the trained model
Open the jupyter notebook Main.ipynb and follow the instructions. You can try the model for samples from the dataset (if you have already downloaded it) and also try the model for an arbitrary input.

## Training your own model
You can train a new model by running
```
python Train.py
```
and answering the prompts. Choose the first dataset as it is the only one available. A reasonable choice for the number of epochs would be 100.

## Reference
Please use the following reference in case you find this repository useful.

```
@article{hemmasian2023surrogate,
  title={Surrogate modeling of melt pool temperature field using deep learning},
  author={Hemmasian, AmirPouya and Ogoke, Francis and Akbari, Parand and Malen, Jonathan and Beuth, Jack and Farimani, Amir Barati},
  journal={Additive Manufacturing Letters},
  volume={5},
  pages={100123},
  year={2023},
  publisher={Elsevier}
}
```
