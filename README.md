# Fashion MNIST Classification with CNN

This repository contains Python and R implementations of a 6-layer CNN for classifying Fashion MNIST images.

## Files
- `fashion_mnist_classification.py`: Python implementation
- `fashion_mnist_cnn.R`: R implementation
- `fashion_minst_cnn.ipynp` : Jupyter Notebook
- `fashion_minst_test` : Test Data set
- `fashion_minst_train` : Training data set

## Requirements
- Python: TensorFlow, Keras, NumPy, Pandas, Matplotlib
- R: keras, tensorflow, tidyverse packages

## Usage

### Python
1. Ensure CSV files are in the same directory
2. Run: `python fashion_mnist_classification.py`
3. Outputs:
   - Model file: `fashion_mnist_cnn.h5`
   - Training history plot: `training_history.png`
   - Prediction visualization: `predictions.png`

### R
1. Set the correct data path in the script
2. Run the script in RStudio or R console
3. Uncomment visualization sections as needed

## Results
Both implementations achieve ~90% accuracy on the test set. The Python script automatically saves visualizations, while the R script has visualization functions that can be uncommented.

________________________________________________________________________________________________________________________________________________________________________________________________________________
                                                                                    Thanks
