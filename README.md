# Survey Generator

We present the tool with a proposition for injecting generated instances into the data stream of surveys.
Our assumption is to investigate the proportion of unlabelled instances to improve the classification task in the prediction of Travel Mode Choice (TMC).
This tool generates datasets that differ in the number of instances used in the initial learning process and the percentage of unlabelled instances.

To initialize the project, you need Poetry in version `1.8.4`.

In the resources directory, you can see the structure of five files: `CITIZENSW1.csv`, `CITIZENSW2.csv`, `CITIZENS_W1_W2.csv`, `PARENTSW1.csv`, and `PARENTSW2.csv`.
These files contain only headers, and their content has been removed because they originate from real surveys.
Files with the prefix `warsaw_generated_all_ones-part-` and `warsaw_generated_traffic_matrix-part-` are split ZIP files due to their size.
To revert this process, go to the `Merge synthetic data` section.

The first is a collection of 100k synthetic journeys based on the UNIFORM_FREQ method, while the second one is based on the REAL_FREQ method,
Additionally, these files present two data sources with synthetic journeys. 
Files with the prefix `warsaw_generated_all_ones-part-` and `warsaw_generated_traffic_matrix-part-` are split ZIP files due to their size and contain SU data generated in this work based on `UNIFORM_FREQ` and `REAL_FREQ` methods, respectively.

Calling the `arff` function initiates the process of generating training datasets in the ARFF format.
This process involves injecting synthetic data into a stream of real data.
The procedure consists of two loops that require defining:
 * The number of instances needed in the initial learning phase (q).
 * The percentage of synthetic data in the entire dataset (p).

The parameter p takes values from 0 to 90 with a step of 10.
The value q ranges from 0 to 1600 for CW* datasets and from 0 to 800 for PW* datasets, with a step of 200 in both cases.

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

## Generate synthetic unlabelled instances
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
