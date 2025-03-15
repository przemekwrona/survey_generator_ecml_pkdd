# Survey Generator

We present the tool with a proposition for injecting generated instances into the data stream of surveys.
Our assumption is to investigate the proportion of unlabelled instances to improve the classification task in the prediction of Travel Mode Choice (TMC).
This tool generates datasets that differ in the number of instances used in the initial learning process and the percentage of unlabelled instances.

To initialize the project, you need Poetry in version 1.8.4.

In the resources directory, you can see the structure of five files: CITIZENSW1.csv, CITIZENSW2.csv, CITIZENS_W1_W2.csv, PARENTSW1.csv, and PARENTSW2.csv.
These files contain only headers, and their content has been removed because they originate from real surveys.
Files with the prefix `warsaw_generated_all_ones-part-` and `warsaw_generated_traffic_matrix-part-` are split ZIP files due to their size.
To revert this process, go to the `Merge synthetic data` section.

Additionally, this files present two data sources with synthetic journeys.
The first is a collection of 100k synthetic journeys based on the All Ones Matrix, while the second is based on the Traffic Matrix.

## Init project

Init project with Poetry `1.8.4`

```shell
poetry install
```

## Merge synthetic data
Due to the fact that the ZIP file with the synthetic journey is larger than 100MB, we were obligated to divide the dataset into smaller portions.

Files was split by command

```shell
split -b 40M warsaw_generated_all_ones.csv.zip warsaw_generated_all_ones-part-
split -b 40M warsaw_generated_traffic_matrix.csv.zip warsaw_generated_traffic_matrix-part-
```
To revert this proces you have to call 

```shell
cat warsaw_generated_all_ones-part-* > warsaw_generated_all_ones.csv.zip
cat warsaw_generated_traffic_matrix-part-* > warsaw_generated_traffic_matrix.csv.zip
```

## Generate surveys
```shell
poetry run arff
```

## Example

### Case 1
Generate Collection of dataset based on Traffic Matrix
```shell
poetry run arff --randomized False
```

### Case 2
Generate Collection of dataset based on All Ones Matrix
```shell
poetry run arff --randomized True
```
