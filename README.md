# Medical Insurance Cost Prediction

A small deep-learning regression project that predicts individual medical insurance charges from patient characteristics. The work is documented in [`sprint12_deeplearning_project.ipynb`](sprint12_deeplearning_project.ipynb).

## Dataset

The notebook loads the public insurance dataset from [TripleTen-DS](https://github.com/TripleTen-DS/Dataset), containing 1,338 records. Features include age, sex, BMI, number of children, smoking status, and region; the target is insurance charges.

## Method

- Inspect the data and charge distribution.
- One-hot encode categorical features.
- Split into training and test sets, then scale features and target using `StandardScaler` fitted on the training data.
- Train a Keras dense neural network with ReLU hidden layers and He initialization.
- Compare the baseline with models using dropout and L2 regularization; evaluate on a held-out test set and inspect prediction and residual plots.

The notebook reports that the baseline performed best in its experiments: mild regularization did not improve validation error, while stronger regularization reduced performance. Its saved test result is a mean absolute error of about **$2,868**. Results may vary between runs because the random seed is not fixed.

## Run

Use Python with Jupyter, then install the notebook dependencies:

```bash
python -m pip install jupyter pandas matplotlib scikit-learn tensorflow
```

Open `sprint12_deeplearning_project.ipynb` in Jupyter or VS Code and run the cells from top to bottom. The dataset is downloaded by the notebook when it runs.
