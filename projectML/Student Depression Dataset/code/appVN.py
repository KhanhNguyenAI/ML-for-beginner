import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import joblib

# Tải mô hình đã huấn luyện (đã huấn luyện với 13 đặc trưng)
model = joblib.load(r"C:\Users\97ngu\OneDrive\Desktop\course\ML\projectML\Student Depression Dataset\testAPPmodelStudentDepression.pkl")

def lam_sach(text):
    return text.strip().lower()

def xu_ly_du_lieu():
    try:
        map_gioi_tinh = {"nam": "male", "nữ": "female"}
        map_ngu = {"khỏe mạnh": "healthy", "trung bình": "moderate", "không khỏe": "unhealthy"}
        map_an = {"khỏe mạnh": "healthy", "trung bình": "moderate", "không khỏe": "unhealthy"}
        map_yesno = {"có": "yes", "không": "no"}

        # Ánh xạ giá trị
        gioi_tinh = 1 if map_gioi_tinh[lam_sach(gioi_tinh_var.get())] == "male" else 0
        ngu = {"healthy": 2, "moderate": 1, "unhealthy": 0}[map_ngu[lam_sach(ngu_var.get())]]
        an = {"healthy": 0, "moderate": 1, "unhealthy": 2}[map_an[lam_sach(an_var.get())]]
        tu_tu = 1 if map_yesno[lam_sach(tu_tu_var.get())] == "yes" else 0
        tien_su = 1 if map_yesno[lam_sach(tien_su_var.get())] == "yes" else 0

        dau_vao = [
            gioi_tinh,
            float(tuoi_var.get()),
            float(ap_luc_hoc_var.get()),
            float(ap_luc_lam_var.get()),
            float(diem_tb_var.get()),
            float(hl_hoc_var.get()),
            float(hl_lam_var.get()),
            ngu,
            an,
            tu_tu,
            float(gio_lam_var.get()),
            float(tai_chinh_var.get()),
            tien_su
        ]
        return np.array(dau_vao).reshape(1, -1)
    except Exception as e:
        messagebox.showerror("Lỗi", f"Vui lòng kiểm tra lại dữ liệu nhập.\n{e}")
        return None

def du_doan():
    x = xu_ly_du_lieu()
    if x is not None:
        ket_qua = model.predict(x)[0]
        msg = "⚠️ Có dấu hiệu trầm cảm." if ket_qua == 1 else "✅ Không có dấu hiệu trầm cảm."
        messagebox.showinfo("Kết quả dự đoán", msg)

# Giao diện chính
root = tk.Tk()
root.title("Ứng dụng Dự đoán Trầm cảm Sinh viên")
root.geometry("540x700")
tk.Label(root, text="Vui lòng nhập thông tin sinh viên:", font=("Arial", 14)).pack(pady=10)
khung = tk.Frame(root)
khung.pack()

def tao_truong(nhan, bien, hang, kieu=tk.Entry, **kwargs):
    tk.Label(khung, text=nhan).grid(row=hang, column=0, padx=10, pady=5, sticky='w')
    o = kieu(khung, textvariable=bien, **kwargs)
    o.grid(row=hang, column=1, padx=10, pady=5, sticky='ew')

# Khởi tạo biến
gioi_tinh_var = tk.StringVar(value="Nam")
ngu_var = tk.StringVar(value="Trung bình")
an_var = tk.StringVar(value="Trung bình")
tu_tu_var = tk.StringVar(value="Không")
tien_su_var = tk.StringVar(value="Không")
tuoi_var = tk.StringVar()
ap_luc_hoc_var = tk.StringVar()
ap_luc_lam_var = tk.StringVar()
diem_tb_var = tk.StringVar()
hl_hoc_var = tk.StringVar()
hl_lam_var = tk.StringVar()
gio_lam_var = tk.StringVar()
tai_chinh_var = tk.StringVar()

# Giao diện nhập liệu
tao_truong("Giới tính (Nam/Nữ):", gioi_tinh_var, 0, ttk.Combobox, values=["Nam", "Nữ"], state="readonly")
tao_truong("Tuổi:", tuoi_var, 1)
tao_truong("Áp lực học tập (1-5):", ap_luc_hoc_var, 2)
tao_truong("Áp lực công việc (0-5):", ap_luc_lam_var, 3)
tao_truong("Điểm GPA (0-10):", diem_tb_var, 4)
tao_truong("Hài lòng học tập (0-5):", hl_hoc_var, 5)
tao_truong("Hài lòng công việc (0-4):", hl_lam_var, 6)
tao_truong("Chất lượng giấc ngủ:", ngu_var, 7, ttk.Combobox, values=["Khỏe mạnh", "Trung bình", "Không khỏe"], state="readonly")
tao_truong("Thói quen ăn uống:", an_var, 8, ttk.Combobox, values=["Khỏe mạnh", "Trung bình", "Không khỏe"], state="readonly")
tao_truong("Từng có ý nghĩ tự tử? (Có/Không):", tu_tu_var, 9, ttk.Combobox, values=["Có", "Không"], state="readonly")
tao_truong("Giờ học/làm mỗi ngày:", gio_lam_var, 10)
tao_truong("Áp lực tài chính (0-5):", tai_chinh_var, 11)
tao_truong("Tiền sử bệnh tâm lý trong gia đình? (Có/Không):", tien_su_var, 12, ttk.Combobox, values=["Có", "Không"], state="readonly")

# Nút dự đoán
tk.Button(root, text="🔍 Dự đoán", command=du_doan, bg="green", fg="white", font=("Arial", 12)).pack(pady=20)

root.mainloop()