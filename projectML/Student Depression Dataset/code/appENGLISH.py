import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import joblib



model = joblib.load(r'C:\Users\97ngu\OneDrive\Desktop\course\ML\projectML\Student Depression Dataset\testAPPmodelStudentDepression.pkl')


# Preprocessing function
def preprocess_input():
    try:
        gender = 1 if gender_var.get().strip().lower() == "male" else 0
        sleep_map = {"healthy": 2, "moderate": 1, "unhealthy": 0}
        diet_map = {"healthy": 0, "moderate": 1, "unhealthy": 2}
        sleep = sleep_map[sleep_var.get().strip().lower()]
        diet = diet_map[diet_var.get().strip().lower()]
        suicidal = 1 if suicidal_var.get().strip().lower() == "yes" else 0
        family_history = 1 if family_var.get().strip().lower() == "yes" else 0

        inputs = [
            gender,
            float(age_var.get()),
            float(ap_var.get()),
            float(wp_var.get()),
            float(cgpa_var.get()),
            float(ss_var.get()),
            float(js_var.get()),
            sleep,
            diet,
            suicidal,
            float(hours_var.get()),
            float(financial_var.get()),
            family_history
        ]
        return np.array(inputs).reshape(1, -1)
    except Exception as e:
        messagebox.showerror("Input Error", f"Please check your entries.\n{e}")
        return None

def predict():
    X = preprocess_input()
    if X is not None:
        result = model.predict(X)[0]
        message = "⚠️ Depression likely detected." if result == 1 else "✅ No signs of depression."
        messagebox.showinfo("Prediction", message)

# GUI setup
root = tk.Tk()
root.title("Student Depression Prediction")
root.geometry("500x700")

tk.Label(root, text="Enter the required student information", font=("Arial", 14)).pack(pady=10)
form = tk.Frame(root)
form.pack()

def add_field(label, var, row, widget=tk.Entry, **kwargs):
    tk.Label(form, text=label).grid(row=row, column=0, padx=10, pady=5, sticky='w')
    input_widget = widget(form, textvariable=var, **kwargs)
    input_widget.grid(row=row, column=1, padx=10, pady=5, sticky='ew')

# Variables
gender_var = tk.StringVar(value="Male")
sleep_var = tk.StringVar(value="Moderate")
diet_var = tk.StringVar(value="Moderate")
suicidal_var = tk.StringVar(value="No")
family_var = tk.StringVar(value="No")
age_var = tk.StringVar()
ap_var = tk.StringVar()
wp_var = tk.StringVar()
cgpa_var = tk.StringVar()
ss_var = tk.StringVar()
js_var = tk.StringVar()
hours_var = tk.StringVar()
financial_var = tk.StringVar()

# Fields
add_field("Gender (male/female):", gender_var, 0, ttk.Combobox, values=["Male", "Female"], state="readonly")
add_field("Age:", age_var, 1)
add_field("Academic Pressure (1-5):", ap_var, 2)
add_field("Work Pressure (0-5):", wp_var, 3)
add_field("CGPA (0-10):", cgpa_var, 4)
add_field("Study Satisfaction (0-5):", ss_var, 5)
add_field("Job Satisfaction (0-4):", js_var, 6)
add_field("Sleep Duration:", sleep_var, 7, ttk.Combobox, values=["Healthy", "Moderate", "Unhealthy"], state="readonly")
add_field("Dietary Habits:", diet_var, 8, ttk.Combobox, values=["Healthy", "Moderate", "Unhealthy"], state="readonly")
add_field("Have suicidal thoughts? (yes/no):", suicidal_var, 9, ttk.Combobox, values=["Yes", "No"], state="readonly")
add_field("Work/Study Hours:", hours_var, 10)
add_field("Financial Stress (0-5):", financial_var, 11)
add_field("Family Mental Illness? (yes/no):", family_var, 12, ttk.Combobox, values=["Yes", "No"], state="readonly")

tk.Button(root, text="🧠 Predict", command=predict, bg="darkblue", fg="white", font=("Arial", 12)).pack(pady=20)

root.mainloop()