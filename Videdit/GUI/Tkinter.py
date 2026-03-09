import tkinter as tk
from tkinter import *

root= tk.Tk()
root.title("Experimentation")
timeline = tk.Canvas(root, width=800, height=900, bg="purple")
timeline.pack()

top_frame =tk.Frame(root,background="orange")
top_frame.pack(padx=10,pady=10, side="right")


root.mainloop()