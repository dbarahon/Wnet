import numpy as np
from tensorflow.keras.models import load_model, Model
import tensorflow as tf 

#program to test Wnet using the same approach as in the fortran module. 
#Written by Donifan Barahona
means= [243.9, 0.6, 6.3, 0.013, 0.0002, 5.04, 21.8, 0.002, 9.75e-7, 7.87e-6, 0.6,  5.04, 21.8, 0.002 ] #hardcoded from G5NR #hardcoded from G5NR based on 100 time steps
stds =[30.3, 0.42, 16.1, 7.9, 0.05, 20.6, 20.8, 0.0036, 7.09e-6, 2.7e-5, 0.42,  20.6, 20.8, 0.0036]
# Fixed seed for reproducibility
np.random.seed(12345)
tf.random.set_seed(12345)
Ns =  1
NF=14
# ---------------------------
# Generate Perturbed Inputs (Range [0.7, 1.3])
# ---------------------------
def generate_perturbed_inputs(Ns):
    """
    Generate Ns samples by perturbing the mean to be in the range [0.6, 1.4].
    """
    perturbed_inputs = []
    for i in range(Ns):
        rand = np.random.rand(NF).astype(np.float32)  # Uniform [0,1]
        perturbation = 0.6 + (1.4 - 0.6) * rand
        print (perturbation)
        if False: #reproduces exactly the first values from F90
            perturbation  =  np.array([1.39688196505185,
                                  0.955320040044469,
                                  0.750406903766369,
                                   1.35267724021224,
                                  0.664087335694313,
                                  0.865826942117554,
                                  0.634150574590451,
                                  0.829104645305264,
                                  0.650579362874499,
                                  0.922793588897890,
                                  0.958937425962500,
                                  0.787562824014034,
                                   1.36635846660494,
                                  0.708843879612037])

            print (perturbation)
        perturbed_mean = perturbation*means  # Directly set to the range
        perturbed_inputs.append(perturbed_mean)
    return np.array(perturbed_inputs)


# Standardize inputs
def standardize_inputs(inputs):
    """
    Standardize inputs using prescribed mean and stddev.
    """
    return (inputs - means) / stds

# Load the Keras model
path  = '/discover/nobackup/dbarahon/ML_param/W_NET//single_level/response_and_final/GAN/best_generator.h5'
model = load_model(path)
model.summary()


# Generate perturbed input samples
inputs = generate_perturbed_inputs(Ns)
standardized_inputs = standardize_inputs(inputs)
 
# ---------------------------
# Run Model Predictions
# ---------------------------
outputs = model.predict(standardized_inputs)

# ---------------------------
# Display Results
# ---------------------------
print("Running Wnet with Ns =", Ns)
print("Perturbed Inputs (Range 0.7–1.3):")
for i in range(Ns):
    print(f"Sample {i + 1} Input: {inputs[i]}")

print("\nModel Outputs:")
for i, output in enumerate(outputs):
    print(f"Sample {i + 1}: {output[0]}")
