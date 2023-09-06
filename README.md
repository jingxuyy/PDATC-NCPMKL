## PDATC-NCPMKL: Predicting drug’s Anatomical Therapeutic Chemical (ATC) codes based on network consistency projection and multiple kernel learning

This code is an implementation of our paper
"**PDATC-NCPMKL: Predicting drug’s Anatomical Therapeutic Chemical (ATC) codes based on network consistency projection and multiple kernel learning**"

We proposed a model PDATC-NCPMKL based on multiple kernel learning and network consistency projection algorithm. By integrating multi-source information of drugs (drug target protein, drug side effect, drug interaction, drug fingerprint, and drug-ATC code association), several drug kernels were constructed. In the same way, the ATC code kernels were set up. The drug and ATC code kernels were fused into a unified drug kernel and ATC code kernel by a multiple kernel learning algorithm and a kernel integrated scheme. On the other hand, the drug-ATC code association adjacency matrix was reformulated by a variant of **weighted K nearest known neighbors (WKNKN)**. Above kernels and matrix were fed into the network consistency projection to generate the association score matrix. The model was tested on the ATC codes at **the second, third and fourth levels** using **ten-fold cross-validation**. For detailed descriptions on the model and results, please refer to our article.

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

|filename|explain|
|:---|:---|
|new_2930_second_ATC.csv| The adjacency matrix of drugs and ATC codes at the second level. |
|new_2930_third_ATC.csv| The adjacency matrix of drugs and ATC codes at the third level. |
|new_2930_fourth_ATC.csv| The adjacency matrix of drugs and ATC codes at the fourth level. |
|2930_fingerprint.csv| The drug representation based on their fingerprints. |
|side_effects.csv| The drug representation based on their side effects. |
|uniprot.csv| The drug representation based on their target proteins. |
|interaction_kernel.csv| The drug kernel using the interaction information collected in STITCH. |

Drug SMILES information, drug ATC code information, and drug target protein information were obtained from **DrugBank** database (https://go.drugbank.com/), drug side effect information were obtained from **SIDER** database (http://sideeffects.emblde/), and drug interaction information was obtained from **STITCH** website (http://stitch4.embl.de/).



## Usage

### How to use it?

### 1. Use the data set we provide
### 1.1 Cross verification
If you use our dataset for cross-validation, all you need to do is enter the following command in the terminal:
```python
 python main.py
```
### 1.2 Modify model parameters
You just need to adjust the following code in the **main.py** file.
~~~python
if __name__ == "__main__":
    drug_atc_path = 'data/drug_ATC/new_2930_fourth_ATC.csv'
    op = Options(drug_atc_path=drug_atc_path, level=4, omega=0.9)
    op.train(k=10)
~~~
- where **drug_atc_path** is the file path where the adjacency matrix of drug and ATC code resides.
- The value of the **level** parameter represents the number of layers of ATC code predicted by the PDATC-NCPMKL model. You can try different layers, for example: **level=2, level=3, level=4**.
- The parameter **omega** represents the value of WKNKN in the Reformulation of adjacency matrix for drug-ATC association. You can set it to any decimal **between 0.0 and 1.0**.
- The value of parameter **k** represents **k-fold cross-validation**, and k is set to 10 in our experiment. You can set it to other integers.

### 2. Use your own data set

### 2.1 Preprocessed data set

You need to re-prepare the above 7 files, the file format is CSV
For other normal dataset, whose format is like below:

1. The adjacency matrix of drug-ATC code

|DrugBankID|code***1***|code***2***|code***3***|code***4***|...|code***m***|
|:----|:----|:----|:----|:----|:----|:----|
|drugID***1***|0|1|1|0|...|0|
|drugID***2***|1|0|0|1|...|0|
|drugID***3***|1|1|0|0|...|1|
|...|...|...|...|...|...|...|
|drugID***n***|0|0|1|1|...|0|

2. The adjacency matrix of drug-fingerprint

|DrugBankID|F*1*|F*2*|F*3*|F*4*|...| 
|:---|:---|:---|:---|:---|:---|
|drugID***1***|0|1|1|0|...| 
|drugID***2***|1|0|0|1|...| 
|drugID***3***|1|1|0|0|...| 
|...|...|...|...|...|...| 
|drugID***n***|0|0|1|1|...|

3. The matrix of drug-interaction

|DrugBankID|drugID***1***|drugID***2***|drugID***3***|drugID***4***|...|drugID***n***|
|:----|:----|:----|:----|:----|:----|:----|
|drugID***1***|1|0.3|0|0.75|...|0.33|
|drugID***2***|0.3|1|0.9|0.22|...|0.68|
|drugID***3***|0|0.9|1|0|...|0.47|
|drugID***4***|0.75|0.22|0|1|...|0.92|
|...|...|...|...|...|...|...|
|drugID***n***|0.33|0.68|0.47|0.92|...|1|

4. The adjacency matrix of drug-side effects

|DrugBankID|side*1*|side*2*|side*3*|side*4*|...| 
|:---|:---|:---|:---|:---|:---|
|drugID***1***|1|0|0|1|...| 
|drugID***2***|1|1|0|0|...| 
|drugID***3***|0|0|0|1|...| 
|...|...|...|...|...|...| 
|drugID***n***|0|0|1|1|...|

5. The adjacency matrix of drug-side effects

|DrugBankID|target*1*|target*2*|target*3*|target*4*|...| 
|:---|:---|:---|:---|:---|:---|
|drugID***1***|0|1|0|1|...| 
|drugID***2***|1|0|0|0|...| 
|drugID***3***|0|0|1|1|...| 
|...|...|...|...|...|...| 
|drugID***n***|0|0|1|0|...|

### 2.2 Preprocessing ATC code shortest path
Because it involves ATC code tree structure to find the shortest path, different data sets involve different ATC, so you should prepare different levels of ATC code shortest path file, which is also CSV format, as follows:

||ATCcode***1***|ATCcode***2***|ATCcode***3***|ATCcode***4***|...|ATCcode***m***|
|:----|:----|:----|:----|:----|:----|:----|
|ATCcode***1***|0|2|2|4|...|4|
|ATCcode***2***|2|0|2|8|...|6|
|ATCcode***3***|2|2|0|4|...|8|
|ATCcode***4***|4|8|4|0|...|2|
|...|...|...|...|...|...|...|
|ATCcode***m***|4|6|8|2|...|0|
- You should put this file in the **PDATC-NCPMKL/shortest_path/** folder, and it should have **the same file name as mine**. (**For example, the second level ATC code file is named new_2ATC_shortest_path_length_matrix.csv**)
- In addition, in order to prevent the accuracy of SPro kernel matrix calculation, ensure that the **order of ATCcode** here is consistent with that in **the adjacency matrix of drug-ATC code**.
### 2.3 Cross verification
You just need to modify the following code in the **main.py** file to run it:
~~~python
def file_path(self):
    drug_fingerprint_path = 'your own drug fingerprint file path'
    drug_side_effects_path = 'your own drug side effect file path'
    drug_target_protein_path = 'your own drug target protein file path'
    drug_interaction_path = 'your own drug interaction file path'
    return drug_fingerprint_path, drug_side_effects_path, drug_target_protein_path, drug_interaction_path
~~~
~~~python
if __name__ == "__main__":
    drug_atc_path = 'your drug-ATC code adjacency matrix file path'
    op = Options(drug_atc_path=drug_atc_path, level=4, omega=0.9)
    op.train(k=10)
~~~

### The results predicted by the model
After running our model, the **PDATC-NCPMKL_predict.csv** file and **PDATC-NCPMKL_actual.csv** file will be generated, where the **PDATC-NCPMKL_predict.csv** file will store the **predicted score**, the **actual value** is saved in the **PDATC-NCPMKL_actual.csv** file.


## Result
The PR curve and ROC curve predicted by our model on the dataset are shown below:
1. The PR curve
![](PR.png)
2. The ROC curve
![](AUC.png)

