# Backpropagation

**PSYCH420 final project**

To see this .md file in preview mode on VSCode, `ctrl + shift + v` or for Mac, `cmd + shift + v`

## Set up

To run code: `python3 backprop.py`

To see help: `python3 backprop.py -h`

### Set up virtual environment (may not be necessary if you have matplotlib installed already)

1. Go into your project directory in the terminal
2. Create a virtual environment (this will create a directory called env)`python3 -m venv env`
3. Activate the virtual environment `source env/bin/activate`
4. Check list of packages installed in virtual env `pip list`
5. Install required dependenciesi `pip install -r requirements.txt`
6. Run program `python3 backprop.py`
7. To deactivate the virtual env, `deactivate`

Note: to update requirements.txt, `pip freeze > requirements.txt`

## Definitions & Terminology

- **Bias**: a constant value added to sum of weighted inputs at each neuron, to shift activation function’s output
  - AKA pre-setting a level of “on” or “off”-ness to make neuron more likely to fire even with small input (via large positive bias), or less likely (via large negative bias)
- **Epoch**: a complete pass through an entire training dataset
- **Delta**: how much each neuron’s output needs to change to reduce error
- **Activation**: the output of a neuron after applying an activation function
- **Activation function**: used to introduce non-linearity to the network. Without it, each layer would simply apply another linear transformation, only modeling linear relationships.
  - There are many types of activation functions, some being:
    - **Sigmoid**: squashes values between 0 and 1
    - **ReLU**: used in hidden layers of deep networks to speed up training
    - **Tanh**: squashes values between -1 and 1, zero-centred
    - **Softmax**: used in output layer for classification to convert outputs to probabilities
- **Loss/cost function**: measures the difference between expeted and actual output to understand how accurate the network is
- **MSE**: mean squared error, a loss function to measures error
- **Overfitting**: memorizing training data instead of generalizing
