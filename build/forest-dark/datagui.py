import tkinter as tk
from tkinter import ttk, messagebox
import openpyxl 
import os


def create_new_window(parent, callback):
    def load_data():
        path = "C:/Users/user/Desktop/build/ecg.xlsx"
        workbook = openpyxl.load_workbook(path)
        sheet = workbook.active

        list_values = list(sheet.values)
        print(list_values)
        for col_name in list_values[0]:
            treeview.heading(col_name, text=col_name)

        for value_tuple in list_values[1:]:
            treeview.insert('', tk.END, values=value_tuple)


    def insert_row():
        first_name = first_name_entry.get()
        last_name = last_name_entry.get()
        age_value = age.get()
        sex = status_combobox.get()
        note = note_entry.get()
        print(first_name, last_name, age_value, sex, note)

        # Insert row into Excel sheet
        path = "C:/Users/user/Desktop/build/ecg.xlsx"
        workbook = openpyxl.load_workbook(path)
        sheet = workbook.active
        row_values = [first_name, last_name, int(age_value), sex, note]
        sheet.append(row_values)
        workbook.save(path)

        # Insert row into treeview
        treeview.insert('', tk.END, values=row_values)

        # Clear the values
        first_name_entry.delete(0, "end")
        first_name_entry.insert(0, "First Name")
        last_name_entry.delete(0, "end")
        last_name_entry.insert(0, "Last Name")
        age.delete(0, "end")
        age.insert(0, "Age")
        status_combobox.set(combo_list[0])
        note_entry.delete(0, "end")
        note_entry.insert(0, "Note")

    def find_folder_by_index(index):
        folder_path = os.path.join("C:/Users/user/Desktop/build/Database/", index)
        if os.path.exists(folder_path):
            return folder_path
        else:
            return None

    def get_selected_row():
        selected_item = treeview.focus()
        if selected_item:
            values = treeview.item(selected_item, 'values')
            index = values[0]  # the first column is the index
            folder_path = find_folder_by_index(index)
            if folder_path:
                if callback:
                    callback(folder_path)
                tk.messagebox.showinfo("Folder Found", f"Folder found: {folder_path}")
                return folder_path
            else:
                tk.messagebox.showerror("Folder Not Found", f"Folder not found for index: {index}")


    root = tk.Tk()

    style = ttk.Style(root)
    root.tk.call("source", "C:/Users/user/Desktop/build/forest-light.tcl")
    root.tk.call("source", "C:/Users/user/Desktop/build/forest-dark.tcl")
    style.theme_use("forest-light")

    combo_list = ["Male", "Female"]

    frame = ttk.Frame(root)
    frame.pack()

    widgets_frame = ttk.LabelFrame(frame, text="Insert Row")
    widgets_frame.grid(row=0, column=0, padx=20, pady=10)

    first_name_entry = ttk.Entry(widgets_frame)
    first_name_entry.insert(0, "First Name")
    first_name_entry.bind("<FocusIn>", lambda e: first_name_entry.delete('0', 'end'))
    first_name_entry.grid(row=0, column=0, padx=5, pady=(0, 5), sticky="ew")

    last_name_entry = ttk.Entry(widgets_frame)
    last_name_entry.insert(0, "Last Name")
    last_name_entry.bind("<FocusIn>", lambda e: last_name_entry.delete('0', 'end'))
    last_name_entry.grid(row=1, column=0, padx=5, pady=(0, 5), sticky="ew")

    age = ttk.Entry(widgets_frame) 
    age.insert(0, "Age") 
    age.bind("<FocusIn>", lambda e: age.delete('0', 'end')) 
    age.grid(row=2, column=0, padx=5, pady=(0, 5), sticky="ew")

    status_combobox = ttk.Combobox(widgets_frame, values=combo_list)
    status_combobox.current(0)
    status_combobox.grid(row=3, column=0, padx=5, pady=5,  sticky="ew")

    note_entry = ttk.Entry(widgets_frame)
    note_entry.insert(0, "Note")
    note_entry.bind("<FocusIn>", lambda e: note_entry.delete('0', 'end'))
    note_entry.grid(row=4, column=0, padx=5, pady=(0, 5), sticky="ew")

    button = ttk.Button(widgets_frame, text="Insert", command=insert_row)
    button.grid(row=5, column=0, padx=5, pady=5, sticky="nsew")

    separator = ttk.Separator(widgets_frame)
    separator.grid(row=6, column=0, padx=(20, 10), pady=10, sticky="ew")

    button = ttk.Button(widgets_frame, text="Get Selected Row", command=get_selected_row)
    button.grid(row=7, column=0, padx=5, pady=5, sticky="nsew")

    treeFrame = ttk.Frame(frame)
    treeFrame.grid(row=0, column=1, pady=10)
    treeScroll = ttk.Scrollbar(treeFrame)
    treeScroll.pack(side="right", fill="y")

    cols = ("First Name", "Last Name", "Age", "Sex", "Note")
    treeview = ttk.Treeview(treeFrame, show="headings",
                            yscrollcommand=treeScroll.set, columns=cols, height=13)
    treeview.column("First Name", width=70)
    treeview.column("Last Name", width=70)
    treeview.column("Age", width=50)
    treeview.column("Sex", width=50)
    treeview.column("Note", width=200)
    treeview.pack()
    treeScroll.config(command=treeview.yview)
    load_data()


    root.mainloop()