## PDATC-NCPMKL: a novel approach based on network consistency projection and multi-kernel learning to predict of drug's Anatomical Therapeutic Chemical (ATC) code


This code is an implementation of our paper
"**PDATC-NCPMKL: a novel approach based on network consistency projection and multi-kernel learning to predict of drug's Anatomical Therapeutic Chemical (ATC) code**"

We proposed a model PDATC-NCPMKL based on multi-kernel learning and network consistency projection algorithm. By integrating multi-source information of drugs (drug and target protein, drug and side effect, drug and interaction, drug and fingerprint, and drug and ATC information), the similarity was calculated from drug kernel space and ATC kernel space by the multi-kernel learning algorithm. A network consistency projection was then used by pretreating the drug and ATC label matrix to predict the ATC code of the compound in the **second, third, and fourth levels**.

In our problem setting of ATC prediction,
The input is 5 matrices:
1. Drug-atc matrix.
2. Drug-target protein matrix.
3. Drug-fingerprint matrix.
4. Drug-side effect matrix.
5. Drug-interaction matrix.

The output is the predicted score matrix and the true label matrix.
Drug fingerprint information 1024 dimensional vectors were obtained using SMILES information of drugs using RDKit, an open-source chemical framework.
Our PDATC-NCPMKL forecast is outlined below:

![](model.png)

For each module of the above process please read our paper.

## Requirements

- python = 3.8
- cvxopt = 1.3.0
- cvxpy = 1.2.0
- ecos = 2.0.10
- fastcache = 1.1.0
- numpy+mkl = 1.22.4
- osqp = 0.6.2
- pandans = 1.3.5
- scikit-learn = 1.0.2
- scipy = 1.7.3
- scs = 3.2.0

## Files:
new_2930_second_ATC.csv: The adjacency matrix of drug and second level ATC code.

new_2930_third_ATC.csv: The adjacency matrix of drug and third level ATC code.

new_2930_fourth_ATC.csv: The adjacency matrix of drug and fourth level ATC code.

2930_fingerprint.csv: The drug-fingerprint adjacency matrix.

side_effects.csv: The drug-side effects kernel matrix.

uniprot.csv: The drug-protein kernel matrix.

interaction_kernel.csv: The drug-interaction kernel matrix.

Among them, drug SMILES information, drug ATC information, and drug target protein information were obtained from **DrugBank** database (https://go.drugbank.com/), drug side effects were obtained from **SIDER** database (http://sideeffects.emblde/), and drug scope was obtained from **STITCH** website (http://stitch4.embl.de/).



## Usage
### How to use it?
### 1. Use the data set we provide
### 1.1 train
```python
 python main.py
```
### 1.2 Modify model parameters
```python
 drug_atc_path = 'data/drug_ATC/new_2930_fourth_ATC.csv'
 op = Options(drug_atc_path, 4, 0.9)
 op.train(10)
```
- where **drug_atc_path** is the file path where the adjacency matrix of drug and ATC code resides.
- Parameter **4** represents the training prediction of the **fourth level ATC code**. You can train different level models, such as **2, 3, 4**
- The parameter **0.9** means that the weight of the adjacency matrix preprocessing algorithm WKNKN is 0.9. You can adjust the parameter value to any decimal **between 0.0 and 1.0**
- **op.train(10)** stands for **ten** fold cross training

### 2. Use your own data set
### 2.1 Preprocessed data set
You need to re-prepare the above 7 files, the file format is CSV
For other normal dataset, whose format is like below:
1. The adjacency matrix of drug-ATC code
```
        code1   code2   code3   code4   ...    codem  
drug1     0       1       1       0     ...      0 
drug2     1       0       0       1     ...      0 
drug3     1       1       0       0     ...      1 
...      ...     ...     ...     ...    ...     ...
drugn     0       0       1       1     ...      0 
```
2. The matrix of drug with other information
```
          0       1       2       3     ... 
drug1     0       1       1       0     ... 
drug2     1       0       0       1     ... 
drug3     1       1       0       0     ... 
...      ...     ...     ...     ...    ... 
drugn     0       0       1       1     ... 
```
### 2.2 train
```python
def file_path(self):
    drug_fingerprint_path = 'your own drug fingerprint file path'
    drug_side_effects_path = 'your own drug side effect file path'
    drug_target_protein_path = 'your own drug target protein file path'
    drug_interaction_path = 'your own drug interaction file path'
    return drug_fingerprint_path, drug_side_effects_path, drug_target_protein_path, drug_interaction_path
```
```python
if __name__ == "__main__":
     drug_atc_path = 'your file path'
     op = Options(drug_atc_path, 4, 0.9)
     op.train(10)
```

### The results predicted by the model
The prediction results of the model are saved in a file named **PDATC-NCPMKL_omega_o_level_num_predict.csv**, 
where **PDATC-NCPMKL_omega_o_level_num_actual.csv** stores the true values and **PDATC-NCPMKL_omega_o_level_num_predict.csv** stores the predicted values.
**o** indicates the weight value of WKNKN, and **num** indicates the ATC code of the predicted num level.


## Result
The PR curve and AUC curve predicted by our model on the dataset are shown below:
1. The PR curve
![](PR.png)
2. The AUC curve
![](AUC.png)

