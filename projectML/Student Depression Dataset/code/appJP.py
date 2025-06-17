import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import joblib

# モデルの読み込み
model = joblib.load("xgboost_model.pkl")

def normalize_input(value):
    return value.strip()

def preprocess_input():
    try:
        gender_map = {"男性": "male", "女性": "female"}
        sleep_map_ja = {"健康的": "healthy", "普通": "moderate", "不健康": "unhealthy"}
        diet_map_ja = {"健康的": "healthy", "普通": "moderate", "不健康": "unhealthy"}
        yn_map = {"はい": "yes", "いいえ": "no"}

        gender = 1 if gender_map[normalize_input(gender_var.get())] == "male" else 0
        sleep = {"healthy": 2, "moderate": 1, "unhealthy": 0}[sleep_map_ja[normalize_input(sleep_var.get())]]
        diet = {"healthy": 0, "moderate": 1, "unhealthy": 2}[diet_map_ja[normalize_input(diet_var.get())]]
        suicidal = 1 if yn_map[normalize_input(suicidal_var.get())] == "yes" else 0
        family_history = 1 if yn_map[normalize_input(family_var.get())] == "yes" else 0

        inputs = [
            gender,
            float(age_var.get()),
            int(profession_var.get()),
            float(academic_var.get()),
            float(work_var.get()),
            float(cgpa_var.get()),
            float(study_sat_var.get()),
            float(job_sat_var.get()),
            sleep,
            diet,
            int(degree_var.get()),
            suicidal,
            float(hours_var.get()),
            float(finance_var.get()),
            family_history
        ]
        return np.array(inputs).reshape(1, -1)
    except Exception as e:
        messagebox.showerror("入力エラー", f"入力を確認してください。\n{e}")
        return None

def predict():
    X = preprocess_input()
    if X is not None:
        prediction = model.predict(X)[0]
        result = "⚠️ うつの兆候があります。" if prediction == 1 else "✅ うつの兆候はありません。"
        messagebox.showinfo("予測結果", result)

# GUI構築
root = tk.Tk()
root.title("うつ予測アプリ（学生用）")
root.geometry("520x740")
tk.Label(root, text="学生の情報を入力してください", font=("Arial", 14)).pack(pady=10)
frame = tk.Frame(root)
frame.pack()

def add_field(label, var, row, widget=tk.Entry, **kwargs):
    tk.Label(frame, text=label).grid(row=row, column=0, padx=10, pady=5, sticky="w")
    field = widget(frame, textvariable=var, **kwargs)
    field.grid(row=row, column=1, padx=10, pady=5, sticky="ew")

# 変数定義
gender_var = tk.StringVar(value="男性")
sleep_var = tk.StringVar(value="普通")
diet_var = tk.StringVar(value="普通")
suicidal_var = tk.StringVar(value="いいえ")
family_var = tk.StringVar(value="いいえ")

age_var = tk.StringVar()
profession_var = tk.StringVar()
academic_var = tk.StringVar()
work_var = tk.StringVar()
cgpa_var = tk.StringVar()
study_sat_var = tk.StringVar()
job_sat_var = tk.StringVar()
degree_var = tk.StringVar()
hours_var = tk.StringVar()
finance_var = tk.StringVar()

# 入力項目
add_field("性別（男性／女性）:", gender_var, 0, ttk.Combobox, values=["男性", "女性"], state="readonly")
add_field("年齢:", age_var, 1)
add_field("職業コード:", profession_var, 2)
add_field("学業のプレッシャー (1-5):", academic_var, 3)
add_field("仕事のプレッシャー (0-5):", work_var, 4)
add_field("GPA (0-10):", cgpa_var, 5)
add_field("学習の満足度 (0-5):", study_sat_var, 6)
add_field("仕事の満足度 (0-4):", job_sat_var, 7)
add_field("睡眠の質:", sleep_var, 8, ttk.Combobox, values=["健康的", "普通", "不健康"], state="readonly")
add_field("食生活の質:", diet_var, 9, ttk.Combobox, values=["健康的", "普通", "不健康"], state="readonly")
add_field("学位コード:", degree_var, 10)
add_field("自殺願望がありますか？", suicidal_var, 11, ttk.Combobox, values=["はい", "いいえ"], state="readonly")
add_field("学習／仕事時間 (時間):", hours_var, 12)
add_field("経済的ストレス (0-5):", finance_var, 13)
add_field("家族にメンタルの病歴がありますか？", family_var, 14, ttk.Combobox, values=["はい", "いいえ"], state="readonly")

# 予測ボタン
tk.Button(root, text="予測する", command=predict, bg="darkgreen", fg="white", font=("Arial", 12)).pack(pady=20)

root.mainloop()