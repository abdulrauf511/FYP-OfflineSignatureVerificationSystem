from io import BytesIO
import matplotlib.pyplot as plt
import PySimpleGUI as sg
import mysql.connector
import zipfile
import pandas as pd
import os
import numpy as np
from PIL import Image
#import cv2
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from skimage.transform import resize
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
import itertools
from tensorflow.keras.models import load_model
import keras
import csv
from PIL import Image
from keras import backend as K
from tensorflow.keras.utils import plot_model
def euclidean_distance(vectors):
    x, y = vectors
    sum_square = tf.reduce_sum(tf.square(x - y), axis=1, keepdims=True)
    return tf.sqrt(tf.maximum(sum_square, K.epsilon()))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)
from tensorflow.keras.layers import Input, Flatten, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
import tensorflow as tf

def create_model(input_shape):
    Depth = 64

    # Load the ResNet50 model without the top classification layers
    resnet_base = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

    # Freeze the pre-trained weights so they are not updated during training
    for layer in resnet_base.layers:
        layer.trainable = False

    # Create the multiple input branches
    one_input = Input(shape=input_shape, name='one_input')
    one_branch = resnet_base(one_input)
    one_output = Flatten()(one_branch)

    two_input = Input(shape=input_shape, name='two_input')
    two_branch = resnet_base(two_input)
    two_output = Flatten()(two_branch)

    three_input = Input(shape=input_shape, name='three_input')
    three_branch = resnet_base(three_input)
    three_output = Flatten()(three_branch)

    four_input = Input(shape=input_shape, name='four_input')
    four_branch = resnet_base(four_input)
    four_output = Flatten()(four_branch)

    five_input = Input(shape=input_shape, name='five_input')
    five_branch = resnet_base(five_input)
    five_output = Flatten()(five_branch)

    # Compute the average of the output branches
    average = Lambda(lambda x: tf.reduce_mean(x, axis=0))([one_output, two_output, three_output, four_output])

    # Compute the distance between the average and the fifth output branch
    distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([average, five_output])

    dense = Dense(64, activation='relu')(distance)
    output = Dense(1, activation='sigmoid', name='output')(dense)

    # Create the final model
    model = Model(inputs=[one_input, two_input, three_input, four_input, five_input], outputs=[output])

    return model

# Define the input shape
input_shape = (118, 118, 3)

# Create the model
model = create_model(input_shape)

# Print the model summary
model.summary()

# Plot the model architecture
plot_model(model, to_file='modelres.png', show_shapes=True)
threshold1=0.48
model.load_weights("C:/Users/User/Downloads/siamese-sigcom-020.h5")
def predict(img_A,img_B,img_C,img_D,img_E):
    result = model.predict([img_A.reshape((1, 118, 118, 3)),
                            img_B.reshape((1, 118, 118, 3)),
                            img_C.reshape((1, 118, 118, 3)),
                            img_D.reshape((1, 118, 118, 3)),
                            img_E.reshape((1, 118, 118, 3))])

    difference=result[0][0]
    return difference

    
imageA=imread(r"D:\FYP Data\FINAL_SIGNS\01_045.png")
imageB=imread(r"D:\FYP Data\FINAL_SIGNS\02_045.png")
imageC=imread(r"D:\FYP Data\FINAL_SIGNS\03_045.png")
imageD=imread(r"D:\FYP Data\FINAL_SIGNS\04_045.png")
imageE=imread(r"D:\FYP Data\FINAL_SIGNS\09_045.png")

imageA=resize(imageA,(118,118,3))
imageB=resize(imageB,(118,118,3))
imageC=resize(imageC,(118,118,3))
imageD=resize(imageD,(118,118,3))
imageE=resize(imageE,(118,118,3))

# Establish a connection to your MySQL database
db = mysql.connector.connect(
    host='localhost',
    user='root',
    password='',
    database='SVS2'
)
cursor = db.cursor()

# Define the layout of the login screen
login_layout = [
    [sg.Text('Signature Verification App', font=('Helvetica', 20))],
    [sg.Button('Login as User', key='-LOGIN_USER-', size=(15, 1))],
    [sg.Button('Login as Admin', key='-LOGIN_ADMIN-', size=(15, 1))],
]

# Create the login window
login_window = sg.Window('Signature Verification App', login_layout, size=(800, 600), element_justification='center')

while True:
    login_event, _ = login_window.read()

    # If user closes the window, exit the program
    if login_event == sg.WINDOW_CLOSED:
        break

    if login_event == '-LOGIN_USER-':
        # TODO: Handle user login logic
        sg.popup('User login selected', title='Login')

        # Close the login window
        login_window.close()
        break

    if login_event == '-LOGIN_ADMIN-':
        # TODO: Handle admin login logic
        sg.popup('Admin login selected', title='Login')

        # Prepare the SQL statement to insert into the Employee table
        admin_insert_sql = "INSERT INTO Employee (Fname, Lname, emp_id, is_admin, Address, Sex, designation) VALUES (%s, %s, %s, %s, %s, %s, %s)"

        # Prompt the admin for employee details
        admin_layout = [
            [sg.Text('Admin Employee Registration', font=('Helvetica', 20))],
            [sg.Text('First Name:', justification='r', size=(12, 1)), sg.InputText(key='-ADMIN_FNAME-', justification='left')],
            [sg.Text('Last Name:', justification='r', size=(12, 1)), sg.InputText(key='-ADMIN_LNAME-', justification='left')],
            [sg.Text('Employee ID:', justification='r', size=(12, 1)), sg.InputText(key='-ADMIN_EMP_ID-', justification='left')],
            [sg.Text('Address:', justification='r', size=(12, 1)), sg.InputText(key='-ADMIN_ADDRESS-', justification='left')],
            [sg.Text('Sex:', justification='r', size=(12, 1)), sg.InputText(key='-ADMIN_SEX-', justification='left')],
            [sg.Text('Designation:', justification='r', size=(12, 1)), sg.InputText(key='-ADMIN_DESIGNATION-', justification='left')],
            [sg.Button('Add Employee', key='-ADD_EMPLOYEE-', size=(15, 1))],
        ]

        # Create the admin employee registration window
        admin_window = sg.Window('Admin Employee Registration', admin_layout, size=(800, 600), element_justification='center')

        # Event loop to process events and get user input on the admin employee registration screen
        while True:
            admin_event, admin_values = admin_window.read()

            # If user closes the window or clicks the 'Exit' button, exit the program
            if admin_event == sg.WINDOW_CLOSED:
                break

            # Handle the 'Add Employee' button click event
            if admin_event == '-ADD_EMPLOYEE-':
                # Retrieve input values from the GUI
                admin_fname = admin_values['-ADMIN_FNAME-']
                admin_lname = admin_values['-ADMIN_LNAME-']
                admin_emp_id = admin_values['-ADMIN_EMP_ID-']
                admin_address = admin_values['-ADMIN_ADDRESS-']
                admin_sex = admin_values['-ADMIN_SEX-']
                admin_designation = admin_values['-ADMIN_DESIGNATION-']

                # Prepare the values for the SQL statement
                admin_insert_values = (admin_fname, admin_lname, admin_emp_id, 'Y', admin_address, admin_sex, admin_designation)

                try:
                    # Execute the SQL statement to insert into the Employee table
                    cursor.execute(admin_insert_sql, admin_insert_values)
                    db.commit()
                except mysql.connector.Error as err:
                    # Display an error message
                    sg.popup(f'Error occurred while adding admin employee:\n{err}', title='Error')

                # Print the input values (for testing purposes)
                print(f'Admin First Name: {admin_fname}')
                print(f'Admin Last Name: {admin_lname}')
                print(f'Admin Employee ID: {admin_emp_id}')
                print(f'Admin Address: {admin_address}')
                print(f'Admin Sex: {admin_sex}')
                print(f'Admin Designation: {admin_designation}')

                # Close the admin employee registration window
                admin_window.close()

                # Proceed to customer registration screen
                break

        # Close the login window
        login_window.close()

        # Proceed to customer registration screen
        break

# Define the layout of your GUI
layout = [
    [sg.Text('CHOOSE AN OPTION', font=('Helvetica', 20))],
    [sg.Button('Register Customer', key='-REGISTER_CUSTOMER-', size=(15, 1))],
    [sg.Button('Delete Customer', key='-DELETE_CUSTOMER-', size=(15, 1))],
    [sg.Button('Start Verifying', key='-START_VERIFYING-', size=(15, 1))],
    [sg.Button('Exit', size=(15, 1))]
]

# Create the window
window = sg.Window('CHOOSE AN OPTION', layout, size=(800, 600), element_justification='center')

# Event loop to process events and get user input on the database frontend screen
while True:
    event, values = window.read()

    # If user closes the window or clicks the 'Exit' button, exit the program
    if event == sg.WINDOW_CLOSED or event == 'Exit':
        break

    # Handle the 'Register Customer' button click event
    if event == '-REGISTER_CUSTOMER-':
        # Define the layout of the customer registration screen
        customer_layout = [
            [sg.Text('CUSTOMER REGISTRATION', font=('Helvetica', 20))],
            [sg.Text('First Name:', justification='r', size=(12, 1)), sg.InputText(key='-FNAME-', justification='left')],
            [sg.Text('Last Name:', justification='r', size=(12, 1)), sg.InputText(key='-LNAME-', justification='left')],
            [sg.Text('Customer ID:', justification='r', size=(12, 1)), sg.InputText(key='-CUSTOMER_ID-', justification='left')],
            [sg.Text('Birthdate:', justification='r', size=(12, 1)), sg.InputText(key='-BDATE-', justification='left')],
            [sg.Text('Address:', justification='r', size=(12, 1)), sg.InputText(key='-ADDRESS-', justification='left')],
            [sg.Text('Sex:', justification='r', size=(12, 1)), sg.InputText(key='-SEX-', justification='left')],
            [sg.Button('Add Customer', key='-ADD_CUSTOMER-', size=(15, 1))],
            [sg.Button('Cancel', size=(15, 1))]
        ]

        # Create the customer registration window
        customer_window = sg.Window('Customer Registration', customer_layout, size=(800, 600),
                                    element_justification='center')

        # Event loop to process events and get user input on the customer registration screen
        while True:
            customer_event, customer_values = customer_window.read()

            # If user closes the window or clicks the 'Cancel' button, break out of the loop
            if customer_event == sg.WINDOW_CLOSED or customer_event == 'Cancel':
                break

            # Handle the 'Add Customer' button click event
            if customer_event == '-ADD_CUSTOMER-':
                # Retrieve input values from the GUI
                fname = customer_values['-FNAME-']
                lname = customer_values['-LNAME-']
                customer_id = customer_values['-CUSTOMER_ID-']
                bdate = customer_values['-BDATE-']
                address = customer_values['-ADDRESS-']
                sex = customer_values['-SEX-']

                # Prepare the SQL statement
                sql = "INSERT INTO Customer (Fname, Lname, Customer_id, Bdate, Address, Sex) VALUES (%s, %s, %s, %s, %s, %s)"
                values = (fname, lname, customer_id, bdate, address, sex)

                try:
                    # Execute the SQL statement
                    cursor.execute(sql, values)
                    db.commit()

                    # Display a success message
                    sg.popup('Customer added successfully!', title='Success')
                    signature_layout = [
                        [sg.Text('SIGNATURE REGISTRATION', font=('Helvetica', 20))],
                        [sg.Text('Customer ID:', justification='r', size=(12, 1)), sg.InputText(key='-SIG_CUST_ID-', justification='left')],
                        [sg.Text('Sign ID:', justification='r', size=(12, 1)), sg.InputText(key='-SIGN_ID-', justification='left')],
                        [sg.Text('Sign 1:', justification='r', size=(12, 1)), sg.Input(key='-SIGN1-', justification='left'), sg.FileBrowse()],
                        [sg.Text('Sign 2:', justification='r', size=(12, 1)), sg.Input(key='-SIGN2-', justification='left'), sg.FileBrowse()],
                        [sg.Text('Sign 3:', justification='r', size=(12, 1)), sg.Input(key='-SIGN3-', justification='left'), sg.FileBrowse()],
                        [sg.Text('Sign 4:', justification='r', size=(12, 1)), sg.Input(key='-SIGN4-', justification='left'), sg.FileBrowse()],
                        [sg.Button('Add Signature', key='-ADD_SIGNATURE-', size=(15, 1))],
                        [sg.Button('Exit', size=(15, 1))]
                    ]

                    # Create the signature registration window
                    signature_window = sg.Window('Signature Registration', signature_layout, size=(800, 600), element_justification='center')

                    # Event loop to process events and get user input on the signature registration screen
                    while True:
                        signature_event, signature_values = signature_window.read()

                        # If user closes the window or clicks the 'Exit' button, exit the program
                        if signature_event == sg.WINDOW_CLOSED or signature_event == 'Exit':
                            break

                        # Handle the 'Add Signature' button click event
                        if signature_event == '-ADD_SIGNATURE-':
                            # Retrieve input values from the GUI
                            cust_id = signature_values['-SIG_CUST_ID-']
                            sign_id = signature_values['-SIGN_ID-']
                            sign1 = signature_values['-SIGN1-']
                            sign2 = signature_values['-SIGN2-']
                            sign3 = signature_values['-SIGN3-']
                            sign4 = signature_values['-SIGN4-']

                            # Prepare the SQL statement
                            sql = "INSERT INTO Signatures (c_id, sign_id,sign1,sign2,sign3,sign4) VALUES (%s, %s,%s,%s,%s,%s)"
                            values = (cust_id, sign_id, sign1, sign2, sign3, sign4)
                            try:
                                # Execute the SQL statement
                                cursor.execute(sql, values)
                                db.commit()

                                # Display a success message
                                sg.popup('Signature added successfully!', title='Success')
                            except mysql.connector.Error as err:
                                # Display an error message
                                sg.popup(f'Error occurred while adding signature:\n{err}', title='Error')

                            # Print the input values (for testing purposes)
                            print(f'Customer ID: {cust_id}')
                            print(f'Sign ID: {sign_id}')
                            print(f'Sign 1: {sign1}')
                            print(f'Sign 2: {sign2}')
                            print(f'Sign 3: {sign3}')
                            print(f'Sign 4: {sign4}')

                    # Close the signature registration window
                    signature_window.close()
                except mysql.connector.Error as err:
                    # Display an error message
                    sg.popup(f'Error occurred while adding customer:\n{err}', title='Error')

                # Print the input values (for testing purposes)
                print(f'First Name: {fname}')
                print(f'Last Name: {lname}')
                print(f'Customer ID: {customer_id}')
                print(f'Birthdate: {bdate}')
                print(f'Address: {address}')
                print(f'Sex: {sex}')

        # Close the customer registration window
        customer_window.close()

    # Handle the 'Delete Customer' button click event
    if event == '-DELETE_CUSTOMER-':
        # Define the layout of the customer deletion screen
        delete_layout = [
            [sg.Text('CUSTOMER DELETION', font=('Helvetica', 20))],
            [sg.Text('Customer ID:', justification='r', size=(12, 1)), sg.InputText(key='-DELETE_ID-', justification='left')],
            [sg.Button('Delete', key='-DELETE-', size=(15, 1))],
            [sg.Button('Cancel', size=(15, 1))]
        ]

        # Create the customer deletion window
        delete_window = sg.Window('Customer Deletion', delete_layout, size=(800, 600),
                                  element_justification='center')

        # Event loop to process events and get user input on the customer deletion screen
        while True:
            delete_event, delete_values = delete_window.read()

            # If user closes the window or clicks the 'Cancel' button, break out of the loop
            if delete_event == sg.WINDOW_CLOSED or delete_event == 'Cancel':
                break

            # Handle the 'Delete' button click event
            if delete_event == '-DELETE-':
                # Retrieve input value from the GUI
                delete_id = delete_values['-DELETE_ID-']

                # Prepare the SQL statement
                delete_sql = "DELETE FROM Customer WHERE Customer_id = %s"
                delete_value = (delete_id,)

                try:
                    # Execute the SQL statement
                    cursor.execute(delete_sql, delete_value)
                    db.commit()

                    # Display a success message
                    sg.popup('Customer deleted successfully!', title='Success')
                except mysql.connector.Error as err:
                    # Display an error message
                    sg.popup(f'Error occurred while deleting customer:\n{err}', title='Error')

                # Print the input value (for testing purposes)
                print(f'Delete ID: {delete_id}')

        # Close the customer deletion window
        delete_window.close()
    if event == '-START_VERIFYING-':
        delete_layout = [
            [sg.Text('VERIFICATION', font=('Helvetica', 20))],
            [sg.Text('Customer ID:', justification='r', size=(12, 1)), sg.InputText(key='-CUSTOMER_ID-', justification='left')],
            [sg.Button('VERIFY', key='-DELETE-', size=(15, 1))]
        ]

        # Create the customer deletion window
        verify_window = sg.Window('Customer sign verification', delete_layout, size=(800, 600),
                                  element_justification='center')

        # Event loop to process events and get user input on the customer deletion screen
        while True:
            verify_event, verify_values = verify_window.read()

            # If user closes the window or clicks the 'Cancel' button, break out of the loop
            if verify_event == sg.WINDOW_CLOSED:
                break
            if verify_event == '-DELETE-':
                # Retrieve images from the database
                try:
                    cursor = db.cursor()
                    query = "SELECT sign1, sign2, sign3, sign4 FROM signatures WHERE c_id = %s"
                    cursor.execute(query, (verify_values['-CUSTOMER_ID-'],))

                    # Fetch all rows returned by the query
                    rows = cursor.fetchall()

                    # Create a dictionary to store the images
                    images_dict = {}

                    # Iterate over the rows and extract the images for each ID
                    for row in rows:
                        id = verify_values['-CUSTOMER_ID-']

                        # Get the image data from the row
                        signature_layout = [
                            [sg.Text('SIGNATURE UPLOADING', font=('Helvetica', 20))],
                            [sg.Text('Customer ID:', justification='r', size=(12, 1)), sg.InputText(key='-SIG_CUST_ID-', justification='left')],
                            [sg.Text('Sign ID:', justification='r', size=(12, 1)), sg.InputText(key='-SIGN_ID-', justification='left')],
                            [sg.Text('Sign 1:', justification='r', size=(12, 1)), sg.Input(key='-SIGN1-', justification='left'), sg.FileBrowse()],
                            [sg.Text('Sign 2:', justification='r', size=(12, 1)), sg.Input(key='-SIGN2-', justification='left'), sg.FileBrowse()],
                            [sg.Text('Sign 3:', justification='r', size=(12, 1)), sg.Input(key='-SIGN3-', justification='left'), sg.FileBrowse()],
                            [sg.Text('Sign 4:', justification='r', size=(12, 1)), sg.Input(key='-SIGN4-', justification='left'), sg.FileBrowse()],
                            [sg.Text('TESST SIGN:', justification='r', size=(12, 1)), sg.Input(key='-SIGN5-', justification='left'), sg.FileBrowse()],
                            [sg.Button('UPLOAD Signature', key='-UPLOAD_SIGNATURE-', size=(15, 1))],
                            [sg.Button('Exit', size=(15, 1))]
                        ]
                        signature_window = sg.Window('SIGNATURE UPLOAD', signature_layout, size=(800, 600), element_justification='center')

                        # Event loop to process events and get user input on the signature registration screen
                        while True:
                            signature_event, signature_values = signature_window.read()

                            # If user closes the window or clicks the 'Exit' button, exit the program
                            if signature_event == sg.WINDOW_CLOSED or signature_event == 'Exit':
                                break

                            # Handle the 'Add Signature' button click event
                            if signature_event == '-UPLOAD_SIGNATURE-':
                                # Retrieve input values from the GUI
                                cust_id = signature_values['-SIG_CUST_ID-']
                                sign_id = signature_values['-SIGN_ID-']
                                sign1 = signature_values['-SIGN1-']
                                sign2 = signature_values['-SIGN2-']
                                sign3 = signature_values['-SIGN3-']
                                sign4 = signature_values['-SIGN4-']
                                sign5 = signature_values['-SIGN5-']
                            imageA=imread(sign1)
                            imageB=imread(sign2)
                            imageC=imread(sign3)
                            imageD=imread(sign4)
                            imageE=imread(sign5)
                            imageA=resize(imageA,(118,118,3))
                            imageB=resize(imageB,(118,118,3))
                            imageC=resize(imageC,(118,118,3))
                            imageD=resize(imageD,(118,118,3))
                            imageE=resize(imageE,(118,118,3))
                            diff=predict(imageA,imageB,imageC,imageD,imageE)
                            if(diff>threshold1):
                                print("its a forged signature")
                                sg.popup('ITS A FORGED SIGNATURE', title='RESULT')
                            else:
                                sg.popup('ITS A GENUINE SIGNATURE', title='RESULT')
                                

                except mysql.connector.Error as err:
                    # Display an error message
                    sg.popup(f'Error occurred while fetching signatures:\n{err}', title='Error')

        

    
