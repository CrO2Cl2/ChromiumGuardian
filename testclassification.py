#TESTCLASSIFICATION.PY
#Script to test the prediction
#This is not actually necessary for actual usage, it is only needed for testing

import time
import ChromiumGuardian as crg
while True:
    # Define the message to classify
    message = input("message:") # INPUT THE MESSAGES TO CLASSIFY HERE. the input is only for testing purposes

    start_time = time.perf_counter()

    # Classify the message
    prediction = crg.classify_message(message)

    end_time = time.perf_counter()

    # Print the prediction and time needed for the operation
    print(f"Prediction: {prediction}")
    print(f"Time needed for the operation: {end_time - start_time:.4f} seconds")
