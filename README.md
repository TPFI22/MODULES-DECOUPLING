# LEVERAGING TEMPORAL GRAPH NETWORKS USING MODULES DECOUPLING

## Running the experiments


### Setup

Install the requirements:

```pip install -r requirements.txt```

Download the datasets from [here](https://github.com/fpour/DGB) and place them in the ```data``` folder.

### Models training

To train the models, run the following command:

```python train.py --dataset <dataset_name> --model <model_name> --normalization <normalization_type>```

To train the models for inductive setting use the --inductive flag:

```python train.py --dataset <dataset_name> --model <model_name> --normalization <normalization_type> --inductive```

### Models evaluation

To evaluate the models, run the following command:

```python evaluate.py --dataset <dataset_name> --model <model_name> --normalization <normalization_type>```

To evaluate the models for inductive setting use the --inductive flag:

```python evaluate.py --dataset <dataset_name> --model <model_name> --normalization <normalization_type> --inductive```

To evaluate the models for the online-learning setting use the --online flag:

```python evaluate.py --dataset <dataset_name> --model <model_name> --normalization <normalization_type> --learn```