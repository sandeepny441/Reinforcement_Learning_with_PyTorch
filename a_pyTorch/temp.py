import streamlit as st

# Title and Instructions
st.title("Simple Calculator")
st.write("Select an operation and enter two numbers.")

# Input fields for numbers
num1 = st.number_input("Enter first number", format="%.2f")
num2 = st.number_input("Enter second number", format="%.2f")

# Dropdown for selecting operation
operation = st.selectbox("Choose an operation", ["Add", "Subtract", "Multiply", "Divide"])

# Button to calculate
if st.button("Calculate"):
    if operation == "Add":
        result = num1 + num2
        st.write(f"The result of adding {num1} and {num2} is: {result}")
    elif operation == "Subtract":
        result = num1 - num2
        st.write(f"The result of subtracting {num2} from {num1} is: {result}")
    elif operation == "Multiply":
        result = num1 * num2
        st.write(f"The result of multiplying {num1} and {num2} is: {result}")
    elif operation == "Divide":
        if num2 == 0:
            st.write("Error: Division by zero is undefined.")
        else:
            result = num1 / num2
            st.write(f"The result of dividing {num1} by {num2} is: {result}")
