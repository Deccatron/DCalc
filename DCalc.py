from customtkinter import * #Import a fuck ton of stuff because i am NOT programming graphing logic...
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sympy import *
from sympy import Symbol, solve, diff, integrate, limit, Matrix, sin, cos, tan, exp, log, sqrt, pi, E, oo, I
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import threading
import time
import math
import io
import sys
from contextlib import redirect_stdout
from tkinter import filedialog, messagebox
import tkinter as tk
from tkinter import ttk
import re
import builtins
import webbrowser
import subprocess
import os

init_printing()

set_appearance_mode("dark")
set_default_color_theme("blue")

class PythonInterpreter:
    def __init__(self, parent):
        self.parent = parent
        self.history = []
        self.history_index = -1
        
        # Create a new module for the interpreter
        self.module = type(sys)('__main__')
        
        # Add all necessary imports to the module
        self.module.np = np
        self.module.plt = plt
        self.module.math = math
        self.module.sympy = sys.modules['sympy']
        self.module.Symbol = Symbol
        self.module.solve = solve
        self.module.diff = diff
        self.module.integrate = integrate
        self.module.limit = limit
        self.module.Matrix = Matrix
        self.module.sin = sin
        self.module.cos = cos
        self.module.tan = tan
        self.module.exp = exp
        self.module.log = log
        self.module.sqrt = sqrt
        self.module.pi = pi
        self.module.E = E
        self.module.oo = oo
        self.module.I = I
        self.module.init_printing = init_printing
        
        # Initialize the module's __builtins__
        self.module.__builtins__ = builtins.__dict__
        
        # Initialize printing
        init_printing()
        
    def execute(self, code):
        try:
            # Create a buffer to capture stdout
            buffer = io.StringIO()
            
            # Execute the code and capture output
            with redirect_stdout(buffer):
                # Execute in the module's namespace
                exec(code, self.module.__dict__)
            
            # Get the output
            output = buffer.getvalue()
            return output, None
        except Exception as e:
            return None, str(e)
            
    def get_variables(self):
        """Get all non-private variables from the module"""
        return {k: v for k, v in self.module.__dict__.items() 
                if not k.startswith('_') and k not in ['np', 'plt', 'math', 'sympy']}

class SyntaxHighlighter:
    def __init__(self, text_widget):
        self.text_widget = text_widget
        
        # Define colors for different syntax elements
        self.colors = {
            'keyword': '#FF79C6',  # Pink
            'string': '#F1FA8C',   # Yellow
            'comment': '#6272A4',  # Gray
            'function': '#8BE9FD', # Cyan
            'number': '#BD93F9',   # Purple
            'operator': '#FF79C6', # Pink
            'builtin': '#50FA7B'   # Green
        }
        
        # Define Python keywords
        self.keywords = [
            'and', 'as', 'assert', 'break', 'class', 'continue', 'def', 'del',
            'elif', 'else', 'except', 'False', 'finally', 'for', 'from', 'global',
            'if', 'import', 'in', 'is', 'lambda', 'None', 'nonlocal', 'not', 'or',
            'pass', 'raise', 'return', 'True', 'try', 'while', 'with', 'yield'
        ]
        
        # Define Python built-in functions
        self.builtins = [
            'abs', 'all', 'any', 'bin', 'bool', 'bytearray', 'bytes', 'callable',
            'chr', 'classmethod', 'compile', 'complex', 'delattr', 'dict', 'dir',
            'divmod', 'enumerate', 'eval', 'exec', 'filter', 'float', 'format',
            'frozenset', 'getattr', 'globals', 'hasattr', 'hash', 'help', 'hex',
            'id', 'input', 'int', 'isinstance', 'issubclass', 'iter', 'len', 'list',
            'locals', 'map', 'max', 'memoryview', 'min', 'next', 'object', 'oct',
            'open', 'ord', 'pow', 'print', 'property', 'range', 'repr', 'reversed',
            'round', 'set', 'setattr', 'slice', 'sorted', 'staticmethod', 'str',
            'sum', 'super', 'tuple', 'type', 'vars', 'zip', '__import__'
        ]
        
        # Configure tags
        for name, color in self.colors.items():
            self.text_widget.tag_configure(name, foreground=color)
            
    def highlight(self):
        # Remove existing tags
        for tag in self.colors.keys():
            self.text_widget.tag_remove(tag, "1.0", "end")
            
        # Get the content
        content = self.text_widget.get("1.0", "end-1c")
        
        # Highlight keywords
        for keyword in self.keywords:
            start = "1.0"
            while True:
                start = self.text_widget.search(r'\y' + keyword + r'\y', start, "end", regexp=True)
                if not start:
                    break
                end = f"{start}+{len(keyword)}c"
                self.text_widget.tag_add("keyword", start, end)
                start = end
                
        # Highlight built-ins
        for builtin in self.builtins:
            start = "1.0"
            while True:
                start = self.text_widget.search(r'\y' + builtin + r'\y', start, "end", regexp=True)
                if not start:
                    break
                end = f"{start}+{len(builtin)}c"
                self.text_widget.tag_add("builtin", start, end)
                start = end
                
        # Highlight strings
        start = "1.0"
        while True:
            start = self.text_widget.search(r'["\'].*?["\']', start, "end", regexp=True)
            if not start:
                break
            line, col = map(int, start.split('.'))
            content = self.text_widget.get(start, f"{line}.end")
            end_col = col
            in_string = False
            string_char = None
            
            for i, char in enumerate(content):
                if char in '"\'':
                    if not in_string:
                        in_string = True
                        string_char = char
                    elif char == string_char:
                        end_col = col + i + 1
                        break
                        
            if end_col > col:
                self.text_widget.tag_add("string", start, f"{line}.{end_col}")
            start = f"{line}.{end_col}"
            
        # Highlight comments
        start = "1.0"
        while True:
            start = self.text_widget.search(r'#.*$', start, "end", regexp=True)
            if not start:
                break
            line = start.split('.')[0]
            self.text_widget.tag_add("comment", start, f"{line}.end")
            start = f"{int(line) + 1}.0"
            
        # Highlight numbers
        start = "1.0"
        while True:
            start = self.text_widget.search(r'\b\d+\.?\d*\b', start, "end", regexp=True)
            if not start:
                break
            line, col = map(int, start.split('.'))
            content = self.text_widget.get(start, f"{line}.end")
            match = re.match(r'\d+\.?\d*', content)
            if match:
                end_col = col + len(match.group())
                self.text_widget.tag_add("number", start, f"{line}.{end_col}")
            start = f"{line}.{end_col}"
            
        # Highlight operators
        operators = ['+', '-', '*', '/', '=', '==', '!=', '<', '>', '<=', '>=', '+=', '-=', '*=', '/=', '%', '**', '//']
        for operator in operators:
            start = "1.0"
            while True:
                start = self.text_widget.search(re.escape(operator), start, "end")
                if not start:
                    break
                end = f"{start}+{len(operator)}c"
                self.text_widget.tag_add("operator", start, end)
                start = end

class DCalc:
    def __init__(self, root):
        self.root = root
        self.root.title("DCalc | A Fun Lil Project By Deccatron!")
        self.root.geometry("1200x800")
        
        self.calculation_history = []
        self.graph_history = []
        
        self.interpreter = PythonInterpreter(self)
        
        self.create_main_ui()
        
        self.theme_var = StringVar(value="dark")
        
        self.bind_shortcuts()
        
        self.desmos_process = None

    def bind_shortcuts(self):
        self.root.bind('<Control-s>', lambda e: self.save_python_code())
        self.root.bind('<Control-o>', lambda e: self.load_python_code())
        self.root.bind('<Control-r>', lambda e: self.run_python_code())
        self.root.bind('<Control-l>', lambda e: self.clear_output())
        self.root.bind('<Control-Key-1>', lambda e: self.tabview.set("Calculator"))
        self.root.bind('<Control-Key-2>', lambda e: self.tabview.set("2D Graph"))
        self.root.bind('<Control-Key-3>', lambda e: self.tabview.set("3D Graph"))
        self.root.bind('<Control-Key-4>', lambda e: self.tabview.set("Function Analysis"))
        self.root.bind('<Control-Key-5>', lambda e: self.tabview.set("Python Code"))

    def create_main_ui(self):
        # Create sidebar for common functions and history
        self.sidebar_frame = CTkFrame(self.root, width=250, corner_radius=0)
        self.sidebar_frame.pack(side="left", fill="y", padx=0, pady=0)
        
        # Add logo with modern styling
        self.logo_label = CTkLabel(self.sidebar_frame, text="DCalc", 
                                 font=("Segoe UI", 24, "bold"), text_color="#4F8AF9")
        self.logo_label.pack(pady=(20, 10))
        
        # Add Desmos button with modern styling
        self.desmos_btn = CTkButton(
            self.sidebar_frame,
            text="🌐 Open Desmos",
            command=self.open_desmos,
            fg_color="#4F8AF9",
            hover_color="#3B7AD9",
            height=35,
            font=("Segoe UI", 12)
        )
        self.desmos_btn.pack(pady=(0, 20), padx=20, fill="x")
        
        # Theme selector with modern styling
        self.theme_frame = CTkFrame(self.sidebar_frame, fg_color="transparent")
        self.theme_frame.pack(fill="x", padx=20, pady=(0, 20))
        
        self.theme_label = CTkLabel(self.theme_frame, text="Theme:", 
                                  font=("Segoe UI", 12))
        self.theme_label.pack(side="left", padx=(0, 10))
        
        self.theme_menu = CTkOptionMenu(
            self.theme_frame,
            values=["dark", "light", "system"],
            command=self.change_theme,
            fg_color="#4F8AF9",
            button_color="#3B7AD9",
            button_hover_color="#2B6AD9",
            font=("Segoe UI", 12)
        )
        self.theme_menu.pack(side="left", fill="x", expand=True)
        
        # Common functions section with modern styling
        self.functions_label = CTkLabel(self.sidebar_frame, text="Quick Functions", 
                                      font=("Segoe UI", 16, "bold"), text_color="#4F8AF9")
        self.functions_label.pack(pady=(20, 10))
        
        common_functions = ["sin(x)", "cos(x)", "tan(x)", "exp(x)", "log(x)", "x**2", "sqrt(x)", "1/x"]
        
        self.functions_frame = CTkFrame(self.sidebar_frame, fg_color="transparent")
        self.functions_frame.pack(fill="x", padx=20, pady=(0, 20))
        
        for i, func in enumerate(common_functions):
            if i % 2 == 0:
                row_frame = CTkFrame(self.functions_frame, fg_color="transparent")
                row_frame.pack(fill="x", pady=2)
                
            btn = CTkButton(row_frame, text=func, width=100, height=30,
                          font=("Segoe UI", 12),
                          fg_color="#4F8AF9",
                          hover_color="#3B7AD9",
                          command=lambda f=func: self.insert_function(f))
            btn.pack(side="left", padx=2, expand=True)
        
        # History section with modern styling
        self.history_label = CTkLabel(self.sidebar_frame, text="History", 
                                    font=("Segoe UI", 16, "bold"), text_color="#4F8AF9")
        self.history_label.pack(pady=(20, 10))
        
        self.history_listbox = CTkTextbox(self.sidebar_frame, width=210, height=200,
                                        font=("Consolas", 12),
                                        fg_color="#1E1E1E",
                                        text_color="#F8F8F2",
                                        corner_radius=8)
        self.history_listbox.pack(padx=20, pady=(0, 10))
        self.history_listbox.configure(state="disabled")
        
        # Clear history button with modern styling
        self.clear_history_btn = CTkButton(self.sidebar_frame, text="Clear History",
                                         font=("Segoe UI", 12),
                                         fg_color="#4F8AF9",
                                         hover_color="#3B7AD9",
                                         height=35,
                                         command=self.clear_history)
        self.clear_history_btn.pack(pady=(0, 20), padx=20, fill="x")
        
        # Main content area with tabs
        self.main_frame = CTkFrame(self.root, fg_color="transparent")
        self.main_frame.pack(side="right", fill="both", expand=True, padx=20, pady=20)
        
        # Create modern tabview
        self.tabview = CTkTabview(self.main_frame, fg_color="#1E1E1E", corner_radius=8)
        self.tabview.pack(fill="both", expand=True)
        
        # Add tabs with modern styling
        self.tabview.add("Calculator")
        self.tabview.add("2D Graph")
        self.tabview.add("3D Graph")
        self.tabview.add("Function Analysis")
        self.tabview.add("Python Code")
        self.tabview.add("Unit Converter")
        self.tabview.add("Physics & Chemistry")
        
        # Configure tab appearance
        self.tabview._segmented_button.configure(
            font=("Segoe UI", 12),
            fg_color="#2B2B2B",
            selected_color="#4F8AF9",
            selected_hover_color="#3B7AD9",
            unselected_color="#2B2B2B",
            unselected_hover_color="#3B3B3B"
        )
        
        # Setup each tab
        self.setup_calculator_tab()
        self.setup_2d_graph_tab()
        self.setup_3d_graph_tab()
        self.setup_analysis_tab()
        self.setup_python_code_tab()
        self.setup_unit_converter_tab()
        self.setup_physics_chem_tab()
        
        # Status bar with modern styling
        self.status_bar = CTkFrame(self.root, height=30, corner_radius=0)
        self.status_bar.pack(side="bottom", fill="x")
        
        self.status_label = CTkLabel(self.status_bar, text="Ready",
                                   font=("Segoe UI", 12),
                                   text_color="#858585")
        self.status_label.pack(side="left", padx=20)
        
        # Version info with modern styling
        self.version_label = CTkLabel(self.status_bar, text="v2.0",
                                    font=("Segoe UI", 12),
                                    text_color="#858585")
        self.version_label.pack(side="right", padx=20)
        
    def setup_calculator_tab(self):
        calc_tab = self.tabview.tab("Calculator")
        
        # Expression input with modern styling
        expr_frame = CTkFrame(calc_tab, fg_color="transparent")
        expr_frame.pack(fill="x", padx=20, pady=(20, 10))
        
        expr_label = CTkLabel(expr_frame, text="Expression:", 
                             font=("Segoe UI", 16, "bold"),
                             text_color="#4F8AF9")
        expr_label.pack(side="left", padx=(0, 10))
        
        self.expr_entry = CTkEntry(expr_frame, font=("Segoe UI", 16),
                                 height=40, corner_radius=8,
                                 fg_color="#2B2B2B",
                                 border_color="#4F8AF9")
        self.expr_entry.pack(side="left", padx=(0, 10), expand=True, fill="x")
        
        eval_button = CTkButton(expr_frame, text="Evaluate",
                               font=("Segoe UI", 12),
                               fg_color="#4F8AF9",
                               hover_color="#3B7AD9",
                               height=40,
                               command=self.eval_expr)
        eval_button.pack(side="left")
        
        # Scientific keypad with modern styling
        keypad_frame = CTkFrame(calc_tab, fg_color="transparent")
        keypad_frame.pack(fill="x", padx=20, pady=10)
        
        # Create keypad buttons
        keypad_buttons = [
            ['7', '8', '9', '+', 'sin', 'cos'],
            ['4', '5', '6', '-', 'tan', 'log'],
            ['1', '2', '3', '*', 'exp', 'sqrt'],
            ['0', '.', '=', '/', 'π', '^']
        ]
        
        for row_idx, row in enumerate(keypad_buttons):
            row_frame = CTkFrame(keypad_frame, fg_color="transparent")
            row_frame.pack(fill="x", pady=2)
            
            for btn_text in row:
                if btn_text == '=':
                    btn = CTkButton(row_frame, text=btn_text, width=60, height=40,
                                  font=("Segoe UI", 14),
                                  fg_color="#4F8AF9",
                                  hover_color="#3B7AD9",
                                  command=self.eval_expr)
                else:
                    btn = CTkButton(row_frame, text=btn_text, width=60, height=40,
                                  font=("Segoe UI", 14),
                                  fg_color="#2B2B2B",
                                  hover_color="#3B3B3B",
                                  command=lambda t=btn_text: self.handle_keypad(t))
                btn.pack(side="left", padx=2, expand=True)
        
        # Matrix operations with modern styling
        matrix_frame = CTkFrame(calc_tab, fg_color="transparent")
        matrix_frame.pack(fill="x", padx=20, pady=10)
        
        matrix_label = CTkLabel(matrix_frame, text="Matrix Operations:",
                              font=("Segoe UI", 16, "bold"),
                              text_color="#4F8AF9")
        matrix_label.pack(anchor="w", pady=(0, 10))
        
        matrix_buttons = [
            ("Create Matrix", self.create_matrix),
            ("Determinant", lambda: self.insert_function("Matrix.det(")),
            ("Inverse", lambda: self.insert_function("Matrix.inv(")),
            ("Transpose", lambda: self.insert_function("Matrix.T")),
            ("Eigenvalues", lambda: self.insert_function("Matrix.eigenvals("))
        ]
        
        matrix_btn_frame = CTkFrame(matrix_frame, fg_color="transparent")
        matrix_btn_frame.pack(fill="x")
        
        for btn_text, cmd in matrix_buttons:
            btn = CTkButton(matrix_btn_frame, text=btn_text,
                          font=("Segoe UI", 12),
                          fg_color="#4F8AF9",
                          hover_color="#3B7AD9",
                          height=35,
                          command=cmd)
            btn.pack(side="left", padx=2, expand=True)
        
        # Results display with modern styling
        self.result_frame = CTkFrame(calc_tab)
        self.result_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        result_label = CTkLabel(self.result_frame, text="Results:",
                              font=("Segoe UI", 16, "bold"),
                              text_color="#4F8AF9")
        result_label.pack(anchor="w", padx=20, pady=10)
        
        self.result_text = CTkTextbox(self.result_frame,
                                    font=("Consolas", 14),
                                    fg_color="#1E1E1E",
                                    text_color="#F8F8F2",
                                    corner_radius=8)
        self.result_text.pack(fill="both", expand=True, padx=20, pady=(0, 20))
        
    def setup_2d_graph_tab(self):
        graph2d_tab = self.tabview.tab("2D Graph")
        
        # Graph control panel
        control_frame = CTkFrame(graph2d_tab)
        control_frame.pack(fill="x", padx=10, pady=10)
        
        # Function entry
        fx_frame = CTkFrame(control_frame)
        fx_frame.pack(fill="x", pady=5)
        
        fx_label = CTkLabel(fx_frame, text="f(x) = ", font=("Arial", 16))
        fx_label.pack(side="left", padx=5)
        
        self.fx_entry = CTkEntry(fx_frame, width=300)
        self.fx_entry.pack(side="left", padx=5, expand=True, fill="x")
        
        # Add function button to support multiple plots
        self.add_function_btn = CTkButton(fx_frame, text="+", width=30, 
                                         command=self.add_2d_function)
        self.add_function_btn.pack(side="left", padx=5)
        
        # Functions list
        self.functions_list_frame = CTkFrame(control_frame)
        self.functions_list_frame.pack(fill="x", pady=5)
        
        self.functions_list_label = CTkLabel(self.functions_list_frame, text="Functions to plot:")
        self.functions_list_label.pack(anchor="w", padx=5)
        
        self.functions_listbox = CTkTextbox(self.functions_list_frame, height=80)
        self.functions_listbox.pack(fill="x", padx=5, pady=5)
        self.functions_listbox.configure(state="disabled")
        
        clear_functions_btn = CTkButton(self.functions_list_frame, text="Clear All", 
                                       command=self.clear_2d_functions)
        clear_functions_btn.pack(pady=5)
        
        # Range settings
        range_frame = CTkFrame(control_frame)
        range_frame.pack(fill="x", pady=5)
        
        x_range_label = CTkLabel(range_frame, text="X Range:")
        x_range_label.pack(side="left", padx=5)
        
        self.x_from_entry = CTkEntry(range_frame, width=80)
        self.x_from_entry.insert(0, "-10")
        self.x_from_entry.pack(side="left", padx=5)
        
        x_to_label = CTkLabel(range_frame, text="to")
        x_to_label.pack(side="left")
        
        self.x_to_entry = CTkEntry(range_frame, width=80)
        self.x_to_entry.insert(0, "10")
        self.x_to_entry.pack(side="left", padx=5)
        
        y_range_label = CTkLabel(range_frame, text="Y Range:")
        y_range_label.pack(side="left", padx=(20, 5))
        
        self.y_from_entry = CTkEntry(range_frame, width=80)
        self.y_from_entry.insert(0, "auto")
        self.y_from_entry.pack(side="left", padx=5)
        
        y_to_label = CTkLabel(range_frame, text="to")
        y_to_label.pack(side="left")
        
        self.y_to_entry = CTkEntry(range_frame, width=80)
        self.y_to_entry.insert(0, "auto")
        self.y_to_entry.pack(side="left", padx=5)
        
        # Plot options
        options_frame = CTkFrame(control_frame)
        options_frame.pack(fill="x", pady=5)
        
        # Plot style
        style_label = CTkLabel(options_frame, text="Plot Style:")
        style_label.pack(side="left", padx=5)
        
        self.plot_style_var = StringVar(value="solid")
        plot_styles = ["solid", "dashed", "dotted", "dashdot"]
        
        self.plot_style_menu = CTkOptionMenu(
            options_frame,
            values=plot_styles,
            variable=self.plot_style_var
        )
        self.plot_style_menu.pack(side="left", padx=5)
        
        # Grid option
        self.grid_var = IntVar(value=1)
        grid_check = CTkCheckBox(options_frame, text="Grid", variable=self.grid_var)
        grid_check.pack(side="left", padx=20)
        
        # Legend option
        self.legend_var = IntVar(value=1)
        legend_check = CTkCheckBox(options_frame, text="Legend", variable=self.legend_var)
        legend_check.pack(side="left", padx=20)
        
        # Plot button
        plot_btn = CTkButton(control_frame, text="Generate Plot", command=self.plot_2d)
        plot_btn.pack(pady=10)
        
        # Plot frame
        self.plot2d_frame = CTkFrame(graph2d_tab)
        self.plot2d_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Function list for 2D plotting
        self.plot_functions_2d = []
        
    def setup_3d_graph_tab(self):
        graph3d_tab = self.tabview.tab("3D Graph")
        
        # Control panel
        control_frame = CTkFrame(graph3d_tab)
        control_frame.pack(fill="x", padx=10, pady=10)
        
        # Function entry
        fx3d_frame = CTkFrame(control_frame)
        fx3d_frame.pack(fill="x", pady=5)
        
        fx3d_label = CTkLabel(fx3d_frame, text="f(x,y) = ", font=("Arial", 16))
        fx3d_label.pack(side="left", padx=5)
        
        self.fx3d_entry = CTkEntry(fx3d_frame, width=400)
        self.fx3d_entry.pack(side="left", padx=5, expand=True, fill="x")
        
        # Surface type selection
        surface_frame = CTkFrame(control_frame)
        surface_frame.pack(fill="x", pady=5)
        
        surface_label = CTkLabel(surface_frame, text="Plot Type:")
        surface_label.pack(side="left", padx=5)
        
        self.surface_type_var = StringVar(value="surface")
        surface_types = ["surface", "wireframe", "contour"]
        
        self.surface_type_menu = CTkOptionMenu(
            surface_frame,
            values=surface_types,
            variable=self.surface_type_var
        )
        self.surface_type_menu.pack(side="left", padx=5)
        
        # Colormap selection
        colormap_label = CTkLabel(surface_frame, text="Colormap:")
        colormap_label.pack(side="left", padx=(20, 5))
        
        self.colormap_var = StringVar(value="viridis")
        colormaps = ["viridis", "plasma", "inferno", "magma", "cividis", "jet", "rainbow"]
        
        self.colormap_menu = CTkOptionMenu(
            surface_frame,
            values=colormaps,
            variable=self.colormap_var
        )
        self.colormap_menu.pack(side="left", padx=5)
        
        # Range settings
        range3d_frame = CTkFrame(control_frame)
        range3d_frame.pack(fill="x", pady=5)
        
        x_range_label = CTkLabel(range3d_frame, text="X Range:")
        x_range_label.pack(side="left", padx=5)
        
        self.x3d_from_entry = CTkEntry(range3d_frame, width=60)
        self.x3d_from_entry.insert(0, "-5")
        self.x3d_from_entry.pack(side="left", padx=5)
        
        x_to_label = CTkLabel(range3d_frame, text="to")
        x_to_label.pack(side="left")
        
        self.x3d_to_entry = CTkEntry(range3d_frame, width=60)
        self.x3d_to_entry.insert(0, "5")
        self.x3d_to_entry.pack(side="left", padx=5)
        
        y_range_label = CTkLabel(range3d_frame, text="Y Range:")
        y_range_label.pack(side="left", padx=(20, 5))
        
        self.y3d_from_entry = CTkEntry(range3d_frame, width=60)
        self.y3d_from_entry.insert(0, "-5")
        self.y3d_from_entry.pack(side="left", padx=5)
        
        y_to_label = CTkLabel(range3d_frame, text="to")
        y_to_label.pack(side="left")
        
        self.y3d_to_entry = CTkEntry(range3d_frame, width=60)
        self.y3d_to_entry.insert(0, "5")
        self.y3d_to_entry.pack(side="left", padx=5)
        
        # Resolution settings
        resolution_frame = CTkFrame(control_frame)
        resolution_frame.pack(fill="x", pady=5)
        
        resolution_label = CTkLabel(resolution_frame, text="Resolution:")
        resolution_label.pack(side="left", padx=5)
        
        self.resolution_var = StringVar(value="60")
        resolutions = ["20", "40", "60", "80", "100"]
        
        self.resolution_menu = CTkOptionMenu(
            resolution_frame,
            values=resolutions,
            variable=self.resolution_var
        )
        self.resolution_menu.pack(side="left", padx=5)
        
        # Plot button
        plot3d_btn = CTkButton(control_frame, text="Generate 3D Plot", command=self.plot_3d)
        plot3d_btn.pack(pady=10)
        
        # Plot frame
        self.plot3d_frame = CTkFrame(graph3d_tab)
        self.plot3d_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
    def setup_analysis_tab(self):
        analysis_tab = self.tabview.tab("Function Analysis")
        
        # Control panel
        control_frame = CTkFrame(analysis_tab)
        control_frame.pack(fill="x", padx=10, pady=10)
        
        # Function entry
        func_frame = CTkFrame(control_frame)
        func_frame.pack(fill="x", pady=5)
        
        func_label = CTkLabel(func_frame, text="Function f(x) = ", font=("Arial", 16))
        func_label.pack(side="left", padx=5)
        
        self.analysis_func_entry = CTkEntry(func_frame, width=400)
        self.analysis_func_entry.pack(side="left", padx=5, expand=True, fill="x")
        
        # Analysis options
        options_frame = CTkFrame(control_frame)
        options_frame.pack(fill="x", pady=5)
        
        # Checkboxes for different analyses
        self.derivative_var = IntVar(value=1)
        derivative_check = CTkCheckBox(options_frame, text="Derivative", variable=self.derivative_var)
        derivative_check.pack(side="left", padx=10)
        
        self.integral_var = IntVar(value=1)
        integral_check = CTkCheckBox(options_frame, text="Integral", variable=self.integral_var)
        integral_check.pack(side="left", padx=10)
        
        self.critical_var = IntVar(value=1)
        critical_check = CTkCheckBox(options_frame, text="Critical Points", variable=self.critical_var)
        critical_check.pack(side="left", padx=10)
        
        self.limit_var = IntVar(value=0)
        limit_check = CTkCheckBox(options_frame, text="Limits", variable=self.limit_var)
        limit_check.pack(side="left", padx=10)
        
        # Range settings
        range_frame = CTkFrame(control_frame)
        range_frame.pack(fill="x", pady=5)
        
        x_range_label = CTkLabel(range_frame, text="X Range:")
        x_range_label.pack(side="left", padx=5)
        
        self.analysis_x_from = CTkEntry(range_frame, width=80)
        self.analysis_x_from.insert(0, "-10")
        self.analysis_x_from.pack(side="left", padx=5)
        
        x_to_label = CTkLabel(range_frame, text="to")
        x_to_label.pack(side="left")
        
        self.analysis_x_to = CTkEntry(range_frame, width=80)
        self.analysis_x_to.insert(0, "10")
        self.analysis_x_to.pack(side="left", padx=5)
        
        # Limit point (for limit analysis)
        limit_point_label = CTkLabel(range_frame, text="Limit at x =")
        limit_point_label.pack(side="left", padx=(20, 5))
        
        self.limit_point_entry = CTkEntry(range_frame, width=80)
        self.limit_point_entry.insert(0, "0")
        self.limit_point_entry.pack(side="left", padx=5)
        
        # Analysis button
        analyze_btn = CTkButton(control_frame, text="Analyze Function", command=self.analyze_function)
        analyze_btn.pack(pady=10)
        
        # Results frame
        self.analysis_results_frame = CTkFrame(analysis_tab)
        self.analysis_results_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Split into graph and text results
        self.analysis_split_frame = CTkFrame(self.analysis_results_frame)
        self.analysis_split_frame.pack(fill="both", expand=True)
        
        # Graph area
        self.analysis_graph_frame = CTkFrame(self.analysis_split_frame)
        self.analysis_graph_frame.pack(side="left", fill="both", expand=True)
        
        # Text results
        self.analysis_text_frame = CTkFrame(self.analysis_split_frame, width=300)
        self.analysis_text_frame.pack(side="right", fill="y")
        
        analysis_text_label = CTkLabel(self.analysis_text_frame, text="Analysis Results:")
        analysis_text_label.pack(anchor="w", padx=5, pady=5)
        
        self.analysis_textbox = CTkTextbox(self.analysis_text_frame, width=300, height=500)
        self.analysis_textbox.pack(padx=5, pady=5, fill="both", expand=True)
        
    def setup_python_code_tab(self):
        python_tab = self.tabview.tab("Python Code")
        
        # Create main container with padding
        main_container = CTkFrame(python_tab)
        main_container.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create top bar with buttons
        top_bar = CTkFrame(main_container)
        top_bar.pack(fill="x", pady=(0, 10))
        
        # Add buttons with icons and tooltips
        buttons = [
            ("Run (Ctrl+R)", self.run_python_code, "▶️"),
            ("Clear (Ctrl+L)", self.clear_output, "🗑️"),
            ("Save (Ctrl+S)", self.save_python_code, "💾"),
            ("Load (Ctrl+O)", self.load_python_code, "📂"),
            ("New", self.new_python_file, "📄"),
            ("Help", self.show_python_help, "❓")
        ]
        
        for text, command, icon in buttons:
            btn = CTkButton(top_bar, text=f"{icon} {text}", command=command)
            btn.pack(side="left", padx=5)
        
        # Create split view for editor and output
        split_frame = CTkFrame(main_container)
        split_frame.pack(fill="both", expand=True)
        
        # Left side - Code editor (60% width)
        editor_frame = CTkFrame(split_frame, width=600)
        editor_frame.pack(side="left", fill="both", expand=True, padx=(0, 5))
        editor_frame.pack_propagate(False)  # Prevent frame from shrinking
        
        # Add line numbers using tkinter Text widget for better alignment
        self.line_numbers = tk.Text(editor_frame, width=4, font=("Consolas", 14),
                                  bg="#1E1E1E", fg="#858585", padx=5, pady=5,
                                  highlightthickness=0, borderwidth=0)
        self.line_numbers.pack(side="left", fill="y")
        self.line_numbers.configure(state="disabled")
        
        # Code editor with syntax highlighting using tkinter Text widget
        self.code_editor = tk.Text(editor_frame, font=("Consolas", 14), bg="#2B2B2B", fg="#F8F8F2",
                                 insertbackground="white", selectbackground="#44475A",
                                 selectforeground="white", undo=True, width=60,
                                 padx=5, pady=5, highlightthickness=0, borderwidth=0)
        self.code_editor.pack(side="left", fill="both", expand=True)
        
        # Initialize syntax highlighter
        self.highlighter = SyntaxHighlighter(self.code_editor)
        
        # Right side - Output and console (40% width)
        output_frame = CTkFrame(split_frame, width=400)
        output_frame.pack(side="right", fill="both", expand=True, padx=(5, 0))
        output_frame.pack_propagate(False)  # Prevent frame from shrinking
        
        # Output area with tabs
        self.output_tabs = CTkTabview(output_frame)
        self.output_tabs.pack(fill="both", expand=True)
        
        # Console output tab
        console_tab = self.output_tabs.add("Console")
        self.output_text = CTkTextbox(console_tab, font=("Consolas", 14))
        self.output_text.pack(fill="both", expand=True)
        
        # Variables tab
        vars_tab = self.output_tabs.add("Variables")
        self.vars_text = CTkTextbox(vars_tab, font=("Consolas", 14))
        self.vars_text.pack(fill="both", expand=True)
        
        # Add example code
        example_code = """# Welcome to the Python Interpreter!
# Try these examples:

# 1. Basic math
x = 5
y = 10
print(f"Sum: {x + y}")

# 2. Using numpy
arr = np.array([1, 2, 3, 4, 5])
print(f"Mean: {np.mean(arr)}")

# 3. Using sympy
x = Symbol('x')
expr = x**2 + 2*x + 1
print(f"Expression: {expr}")
print(f"Derivative: {diff(expr, x)}")

# 4. Plotting
plt.figure(figsize=(8, 6))
x = np.linspace(0, 10, 100)
plt.plot(x, np.sin(x))
plt.title("Sine Wave")
plt.show()"""
        
        self.code_editor.insert("1.0", example_code)
        self.update_line_numbers()
        self.highlighter.highlight()
        
        # Bind events
        self.code_editor.bind('<KeyRelease>', self.on_code_change)
        self.code_editor.bind('<Control-Return>', lambda e: self.run_python_code())

    def on_code_change(self, event=None):
        self.update_line_numbers()
        self.highlighter.highlight()

    def update_line_numbers(self, event=None):
        # Get the number of lines
        line_count = self.code_editor.get("1.0", "end-1c").count('\n') + 1
        
        # Update line numbers
        self.line_numbers.configure(state="normal")
        self.line_numbers.delete("1.0", "end")
        
        # Add padding to align with code
        for i in range(1, line_count + 1):
            self.line_numbers.insert("end", f"{i:3d}\n")
            
        self.line_numbers.configure(state="disabled")
        
        # Sync scrolling between editor and line numbers
        self.line_numbers.yview_moveto(self.code_editor.yview()[0])

    def new_python_file(self):
        self.code_editor.delete("1.0", "end")
        self.update_line_numbers()
        self.set_status("New file created")
        
    def show_python_help(self):
        help_text = """Python Interpreter Help:

Keyboard Shortcuts:
- Ctrl+R: Run code
- Ctrl+S: Save code
- Ctrl+O: Load code
- Ctrl+L: Clear output
- Ctrl+Enter: Run current line

Available Libraries:
- numpy (as np)
- matplotlib.pyplot (as plt)
- math
- sympy
- Symbol, solve, diff, integrate, limit

Tips:
1. Use print() to see output
2. Variables are preserved between runs
3. Use plt.show() to display plots
4. Check the Variables tab to see current variables"""
        
        help_window = CTkToplevel(self.root)
        help_window.title("Python Interpreter Help")
        help_window.geometry("600x400")
        
        help_textbox = CTkTextbox(help_window, font=("Consolas", 14))
        help_textbox.pack(fill="both", expand=True, padx=10, pady=10)
        help_textbox.insert("1.0", help_text)
        help_textbox.configure(state="disabled")
        
    def run_python_code(self):
        try:
            # Get the code from the editor
            code = self.code_editor.get("1.0", "end-1c")
            
            # Clear previous output
            self.output_text.delete("1.0", "end")
            
            # Execute the code
            output, error = self.interpreter.execute(code)
            
            if error:
                self.output_text.insert("1.0", f"Error: {error}")
                self.set_status("Error executing code")
            else:
                if output:
                    self.output_text.insert("1.0", output)
                else:
                    self.output_text.insert("1.0", "Code executed successfully. No output.")
                self.set_status("Code executed successfully")
            
            # Update variables display
            self.update_variables_display()
            
        except Exception as e:
            self.output_text.delete("1.0", "end")
            self.output_text.insert("1.0", f"Error: {str(e)}")
            self.set_status("Error executing code")
            
    def update_variables_display(self):
        self.vars_text.delete("1.0", "end")
        
        # Get all variables from the interpreter
        vars_text = "Current Variables:\n\n"
        for name, value in self.interpreter.get_variables().items():
            try:
                # Try to get a string representation
                value_str = str(value)
                # Truncate long values
                if len(value_str) > 100:
                    value_str = value_str[:100] + "..."
                vars_text += f"{name} = {value_str}\n"
            except:
                vars_text += f"{name} = <complex object>\n"
        
        self.vars_text.insert("1.0", vars_text)

    def clear_output(self):
        self.output_text.delete("1.0", "end")
        self.set_status("Output cleared")

    def save_python_code(self):
        try:
            code = self.code_editor.get("1.0", "end-1c")
            
            # Open file dialog
            from tkinter import filedialog
            file_path = filedialog.asksaveasfilename(
                defaultextension=".py",
                filetypes=[("Python files", "*.py"), ("All files", "*.*")]
            )
            
            if file_path:
                with open(file_path, 'w') as f:
                    f.write(code)
                self.set_status(f"Code saved to {file_path}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save code: {str(e)}")
            self.set_status("Error saving code")

    def load_python_code(self):
        try:
            # Open file dialog
            from tkinter import filedialog
            file_path = filedialog.askopenfilename(
                filetypes=[("Python files", "*.py"), ("All files", "*.*")]
            )
            
            if file_path:
                with open(file_path, 'r') as f:
                    code = f.read()
                
                self.code_editor.delete("1.0", "end")
                self.code_editor.insert("1.0", code)
                self.set_status(f"Code loaded from {file_path}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load code: {str(e)}")
            self.set_status("Error loading code")

    # Helper functions
    def change_theme(self, theme):
        set_appearance_mode(theme)
        
    def insert_function(self, func):
        current_pos = self.expr_entry.index("insert")
        self.expr_entry.insert(current_pos, func)
        
    def create_matrix(self):
        # Open a dialog to create a matrix
        matrix_dialog = CTkToplevel(self.root)
        matrix_dialog.title("Create Matrix")
        matrix_dialog.geometry("400x300")
        matrix_dialog.grab_set()  # Modal dialog
        
        # Row and column selection
        dim_frame = CTkFrame(matrix_dialog)
        dim_frame.pack(padx=10, pady=10, fill="x")
        
        rows_label = CTkLabel(dim_frame, text="Rows:")
        rows_label.pack(side="left", padx=5)
        
        rows_var = StringVar(value="2")
        rows_entry = CTkEntry(dim_frame, width=50, textvariable=rows_var)
        rows_entry.pack(side="left", padx=5)
        
        cols_label = CTkLabel(dim_frame, text="Columns:")
        cols_label.pack(side="left", padx=5)
        
        cols_var = StringVar(value="2")
        cols_entry = CTkEntry(dim_frame, width=50, textvariable=cols_var)
        cols_entry.pack(side="left", padx=5)
        
        # Create matrix UI based on dimensions
        def create_matrix_ui():
            try:
                rows = int(rows_var.get())
                cols = int(cols_var.get())
                
                if rows > 5 or cols > 5:
                    messagebox.showwarning("Warning", "Maximum matrix size is 5x5")
                    return
                
                # Clear previous entries if any
                for widget in matrix_entries_frame.winfo_children():
                    widget.destroy()
                    
                # Create matrix entries
                entries = []
                for i in range(rows):
                    row_entries = []
                    row_frame = CTkFrame(matrix_entries_frame)
                    row_frame.pack(pady=2)
                    
                    for j in range(cols):
                        entry = CTkEntry(row_frame, width=50)
                        entry.insert(0, "0")
                        entry.pack(side="left", padx=2)
                        row_entries.append(entry)
                    
                    entries.append(row_entries)
                
                return entries
            except ValueError:
                return None
        
        # Create button
        create_btn = CTkButton(dim_frame, text="Create", 
                              command=lambda: create_matrix_ui())
        create_btn.pack(side="left", padx=10)
        
        # Matrix entries frame
        matrix_entries_frame = CTkFrame(matrix_dialog)
        matrix_entries_frame.pack(padx=10, pady=10, fill="both", expand=True)
        
        # Initial matrix UI
        entries = create_matrix_ui()
        
        # Insert matrix button
        def insert_matrix_to_expr():
            try:
                rows = int(rows_var.get())
                cols = int(cols_var.get())
                
                # Get values from entries
                matrix_data = []
                for i in range(rows):
                    row_data = []
                    for j in range(cols):
                        val = entries[i][j].get()
                        if not val:
                            val = "0"
                        row_data.append(val)
                    matrix_data.append(row_data)
                
                # Format matrix string
                matrix_str = "Matrix(["
                for i, row in enumerate(matrix_data):
                    if i > 0:
                        matrix_str += ", "
                    row_str = "[" + ", ".join(row) + "]"
                    matrix_str += row_str
                matrix_str += "])"
                
                # Insert into expression entry
                current_pos = self.expr_entry.index("insert")
                self.expr_entry.insert(current_pos, matrix_str)
                
                matrix_dialog.destroy()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to create matrix: {str(e)}")
        
        insert_btn = CTkButton(matrix_dialog, text="Insert to Expression", 
                              command=insert_matrix_to_expr)
        insert_btn.pack(padx=10, pady=10)
        
    def handle_keypad(self, btn_text):
        if btn_text == 'π':
            self.insert_function("pi")
        elif btn_text == '^':
            self.insert_function("**")
        else:
            self.insert_function(btn_text)
            
    def clear_history(self):
        self.calculation_history = []
        self.history_listbox.configure(state="normal")
        self.history_listbox.delete("1.0", "end")
        self.history_listbox.configure(state="disabled")
        
    def add_to_history(self, expr, result):
        # Add to history list
        history_item = f"{expr} = {result}\n"
        self.calculation_history.append(history_item)
        
        # Update history display
        self.history_listbox.configure(state="normal")
        self.history_listbox.delete("1.0", "end")
        
        # Show most recent entries first
        for item in reversed(self.calculation_history[-10:]):
            self.history_listbox.insert("1.0", item + "\n")
            
        self.history_listbox.configure(state="disabled")
        
    def add_2d_function(self):
        func = self.fx_entry.get()
        if func.strip():
            self.plot_functions_2d.append(func)
            self.fx_entry.delete(0, "end")
            
            # Update functions list
            self.functions_listbox.configure(state="normal")
            self.functions_listbox.delete("1.0", "end")
            
            for i, f in enumerate(self.plot_functions_2d):
                self.functions_listbox.insert("end", f"{i+1}. {f}\n")
                
            self.functions_listbox.configure(state="disabled")
            
    def clear_2d_functions(self):
        self.plot_functions_2d = []
        self.functions_listbox.configure(state="normal")
        self.functions_listbox.delete("1.0", "end")
        self.functions_listbox.configure(state="disabled")
        
    def set_status(self, message):
        self.status_label.configure(text=message)
        self.root.update_idletasks()
        
    def eval_expr(self):
        try:
            expr = self.expr_entry.get()
            if not expr.strip():
                return
                
            # Convert expression to sympy format
            x = Symbol('x')
            expr_sympy = parse_expr(expr)
            
            # Evaluate the expression
            result = expr_sympy.evalf()
            
            # Display result
            self.result_text.delete("1.0", "end")
            self.result_text.insert("1.0", f"Expression: {expr}\nResult: {result}")
            
            # Add to history
            self.add_to_history(expr, result)
            
            # Update status
            self.set_status("Expression evaluated successfully")
            
        except Exception as e:
            self.result_text.delete("1.0", "end")
            self.result_text.insert("1.0", f"Error: {str(e)}")
            self.set_status("Error evaluating expression")
            
    def plot_2d(self):
        try:
            if not self.plot_functions_2d:
                self.show_message("Warning", "Please add at least one function to plot")
                return
                
            # Clear previous plot
            for widget in self.plot2d_frame.winfo_children():
                widget.destroy()
                
            # Create figure
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111)
            
            # Get range settings
            x_from = float(self.x_from_entry.get())
            x_to = float(self.x_to_entry.get())
            
            # Generate x values
            x = np.linspace(x_from, x_to, 1000)
            
            # Plot each function
            for i, func_str in enumerate(self.plot_functions_2d):
                try:
                    # Convert string to function
                    x_sym = Symbol('x')
                    func = parse_expr(func_str)
                    func_lambda = lambdify(x_sym, func, 'numpy')
                    
                    # Calculate y values
                    y = func_lambda(x)
                    
                    # Plot with different styles
                    style = self.plot_style_var.get()
                    ax.plot(x, y, label=f"f(x) = {func_str}", linestyle=style)
                    
                except Exception as e:
                    self.show_message("Error", f"Error plotting function {func_str}: {str(e)}")
                    continue
            
            # Set y range if specified
            if self.y_from_entry.get() != "auto" and self.y_to_entry.get() != "auto":
                ax.set_ylim(float(self.y_from_entry.get()), float(self.y_to_entry.get()))
            
            # Add grid if enabled
            if self.grid_var.get():
                ax.grid(True)
            
            # Add legend if enabled
            if self.legend_var.get():
                ax.legend()
            
            # Add labels
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title('2D Function Plot')
            
            # Create canvas
            canvas = FigureCanvasTkAgg(fig, master=self.plot2d_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)
            
            self.set_status("2D plot generated successfully")
            
        except Exception as e:
            self.show_message("Error", f"Error generating plot: {str(e)}")
            self.set_status("Error generating plot")
            
    def plot_3d(self):
        try:
            func_str = self.fx3d_entry.get()
            if not func_str.strip():
                self.show_message("Warning", "Please enter a function f(x,y)", "warning")
                return
                
            # Clear previous plot
            for widget in self.plot3d_frame.winfo_children():
                widget.destroy()
                
            # Create figure
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')
            
            # Get range settings
            x_from = float(self.x3d_from_entry.get())
            x_to = float(self.x3d_to_entry.get())
            y_from = float(self.y3d_from_entry.get())
            y_to = float(self.y3d_to_entry.get())
            
            # Get resolution
            resolution = int(self.resolution_var.get())
            
            # Generate mesh
            x = np.linspace(x_from, x_to, resolution)
            y = np.linspace(y_from, y_to, resolution)
            X, Y = np.meshgrid(x, y)
            
            # Convert string to function
            x_sym, y_sym = symbols('x y')
            func = parse_expr(func_str)
            func_lambda = lambdify((x_sym, y_sym), func, 'numpy')
            
            # Calculate Z values
            Z = func_lambda(X, Y)
            
            # Plot based on selected type
            plot_type = self.surface_type_var.get()
            if plot_type == "surface":
                surf = ax.plot_surface(X, Y, Z, cmap=self.colormap_var.get())
                fig.colorbar(surf)
            elif plot_type == "wireframe":
                ax.plot_wireframe(X, Y, Z, cmap=self.colormap_var.get())
            else:  # contour
                ax.contour3D(X, Y, Z, 50, cmap=self.colormap_var.get())
            
            # Add labels
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            ax.set_title('3D Function Plot')
            
            # Create canvas
            canvas = FigureCanvasTkAgg(fig, master=self.plot3d_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)
            
            self.set_status("3D plot generated successfully")
            
        except Exception as e:
            self.show_message("Error", f"Error generating 3D plot: {str(e)}", "error")
            self.set_status("Error generating 3D plot")
            
    def analyze_function(self):
        try:
            func_str = self.analysis_func_entry.get()
            if not func_str.strip():
                messagebox.showwarning("Warning", "Please enter a function to analyze", "warning")
                return
                
            # Clear previous results
            for widget in self.analysis_graph_frame.winfo_children():
                widget.destroy()
            self.analysis_textbox.delete("1.0", "end")
            
            # Convert string to function
            x = Symbol('x')
            func = parse_expr(func_str)
            
            # Create figure for plotting
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111)
            
            # Get range settings
            x_from = float(self.analysis_x_from.get())
            x_to = float(self.analysis_x_to.get())
            
            # Generate x values for plotting
            x_vals = np.linspace(x_from, x_to, 1000)
            func_lambda = lambdify(x, func, 'numpy')
            y_vals = func_lambda(x_vals)
            
            # Plot original function
            ax.plot(x_vals, y_vals, label=f"f(x) = {func_str}")
            
            results = []
            
            # Calculate derivative if selected
            if self.derivative_var.get():
                deriv = diff(func, x)
                deriv_lambda = lambdify(x, deriv, 'numpy')
                y_deriv = deriv_lambda(x_vals)
                ax.plot(x_vals, y_deriv, '--', label=f"f'(x) = {deriv}")
                results.append(f"Derivative: {deriv}")
            
            # Calculate integral if selected
            if self.integral_var.get():
                integral = integrate(func, x)
                results.append(f"Indefinite Integral: {integral}")
                
                # Calculate definite integral
                def_integral = integrate(func, (x, x_from, x_to))
                results.append(f"Definite Integral from {x_from} to {x_to}: {def_integral}")
            
            # Find critical points if selected
            if self.critical_var.get():
                deriv = diff(func, x)
                critical_points = solve(deriv, x)
                results.append("\nCritical Points:")
                for point in critical_points:
                    # Evaluate second derivative at critical point
                    second_deriv = diff(deriv, x)
                    second_deriv_val = second_deriv.subs(x, point)
                    
                    if second_deriv_val > 0:
                        point_type = "local minimum"
                    elif second_deriv_val < 0:
                        point_type = "local maximum"
                    else:
                        point_type = "saddle point"
                        
                    results.append(f"x = {point}: {point_type}")
            
            # Calculate limit if selected
            if self.limit_var.get():
                limit_point = float(self.limit_point_entry.get())
                try:
                    limit_val = limit(func, x, limit_point)
                    results.append(f"\nLimit as x → {limit_point}: {limit_val}")
                except:
                    results.append(f"\nLimit as x → {limit_point}: Does not exist")
            
            # Add grid and legend
            ax.grid(True)
            ax.legend()
            
            # Add labels
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title('Function Analysis')
            
            # Create canvas
            canvas = FigureCanvasTkAgg(fig, master=self.analysis_graph_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)
            
            # Display results in text box
            self.analysis_textbox.insert("1.0", "\n".join(results))
            
            self.set_status("Function analysis completed successfully")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error analyzing function: {str(e)}", "error")
            self.set_status("Error analyzing function")

    def show_message(self, title, message, type="info"):
        """Show a message box with the given title and message"""
        if type == "warning":
            messagebox.showwarning(title, message)
        elif type == "error":
            messagebox.showerror(title, message)
        else:
            messagebox.showinfo(title, message)

    def open_desmos(self):
        """Open Desmos in a separate window using a subprocess"""
        try:
            if self.desmos_process is None or self.desmos_process.poll() is not None:
                # Create a temporary Python script to run Desmos
                script = """
import webview
webview.create_window('Desmos Calculator', 'https://www.desmos.com/calculator',
                     width=1000, height=800, resizable=True, min_size=(800, 600))
webview.start()
"""
                # Write the script to a temporary file
                script_path = os.path.join(os.path.dirname(__file__), 'desmos_window.py')
                with open(script_path, 'w') as f:
                    f.write(script)
                
                # Run the script in a separate process
                self.desmos_process = subprocess.Popen([sys.executable, script_path])
                self.set_status("Opening Desmos in new window...")
            else:
                self.set_status("Desmos window already open")
        except Exception as e:
            self.show_message("Error", f"Failed to open Desmos: {str(e)}")
            self.set_status("Error opening Desmos")

    def setup_unit_converter_tab(self):
        unit_tab = self.tabview.tab("Unit Converter")
        
        # Create main container
        main_container = CTkFrame(unit_tab)
        main_container.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Conversion type selection
        type_frame = CTkFrame(main_container)
        type_frame.pack(fill="x", pady=10)
        
        type_label = CTkLabel(type_frame, text="Conversion Type:", font=("Segoe UI", 14, "bold"))
        type_label.pack(side="left", padx=10)
        
        self.conversion_type = StringVar(value="length")
        conversion_types = [
            "length", "mass", "temperature", "time", "area", 
            "volume", "speed", "pressure", "energy", "power"
        ]
        
        self.type_menu = CTkOptionMenu(
            type_frame,
            values=conversion_types,
            variable=self.conversion_type,
            command=self.update_unit_options
        )
        self.type_menu.pack(side="left", padx=10)
        
        # Conversion frame
        conv_frame = CTkFrame(main_container)
        conv_frame.pack(fill="x", pady=10)
        
        # From unit
        from_frame = CTkFrame(conv_frame)
        from_frame.pack(side="left", fill="x", expand=True, padx=5)
        
        from_label = CTkLabel(from_frame, text="From:", font=("Segoe UI", 12))
        from_label.pack(anchor="w", padx=5)
        
        self.from_unit = StringVar()
        self.from_menu = CTkOptionMenu(from_frame, variable=self.from_unit)
        self.from_menu.pack(fill="x", padx=5)
        
        self.from_value = CTkEntry(from_frame, placeholder_text="Enter value")
        self.from_value.pack(fill="x", padx=5, pady=5)
        
        # To unit
        to_frame = CTkFrame(conv_frame)
        to_frame.pack(side="left", fill="x", expand=True, padx=5)
        
        to_label = CTkLabel(to_frame, text="To:", font=("Segoe UI", 12))
        to_label.pack(anchor="w", padx=5)
        
        self.to_unit = StringVar()
        self.to_menu = CTkOptionMenu(to_frame, variable=self.to_unit)
        self.to_menu.pack(fill="x", padx=5)
        
        self.to_value = CTkEntry(to_frame, state="readonly")
        self.to_value.pack(fill="x", padx=5, pady=5)
        
        # Convert button
        convert_btn = CTkButton(
            main_container,
            text="Convert",
            command=self.perform_conversion,
            font=("Segoe UI", 12),
            height=35
        )
        convert_btn.pack(pady=10)
        
        # Initialize unit options
        self.update_unit_options()
        
        # Bind value change to conversion
        self.from_value.bind('<KeyRelease>', lambda e: self.perform_conversion())
        self.from_unit.trace('w', lambda *args: self.perform_conversion())
        self.to_unit.trace('w', lambda *args: self.perform_conversion())
        
    def update_unit_options(self, *args):
        """Update the available units based on the selected conversion type"""
        conversion_type = self.conversion_type.get()
        
        # Define unit mappings
        unit_mappings = {
            "length": {
                "units": ["meters", "kilometers", "centimeters", "millimeters", 
                         "inches", "feet", "yards", "miles"],
                "conversions": {
                    "meters": 1.0,
                    "kilometers": 1000.0,
                    "centimeters": 0.01,
                    "millimeters": 0.001,
                    "inches": 0.0254,
                    "feet": 0.3048,
                    "yards": 0.9144,
                    "miles": 1609.344
                }
            },
            "mass": {
                "units": ["kilograms", "grams", "milligrams", "pounds", "ounces"],
                "conversions": {
                    "kilograms": 1.0,
                    "grams": 0.001,
                    "milligrams": 0.000001,
                    "pounds": 0.45359237,
                    "ounces": 0.028349523125
                }
            },
            "temperature": {
                "units": ["celsius", "fahrenheit", "kelvin"],
                "conversions": {
                    "celsius": lambda x: x,
                    "fahrenheit": lambda x: (x - 32) * 5/9,
                    "kelvin": lambda x: x - 273.15
                }
            },
            "time": {
                "units": ["seconds", "minutes", "hours", "days", "weeks", "months", "years"],
                "conversions": {
                    "seconds": 1.0,
                    "minutes": 60.0,
                    "hours": 3600.0,
                    "days": 86400.0,
                    "weeks": 604800.0,
                    "months": 2592000.0,
                    "years": 31536000.0
                }
            },
            "area": {
                "units": ["square meters", "square kilometers", "square centimeters", 
                         "square millimeters", "square inches", "square feet", 
                         "square yards", "square miles", "acres", "hectares"],
                "conversions": {
                    "square meters": 1.0,
                    "square kilometers": 1000000.0,
                    "square centimeters": 0.0001,
                    "square millimeters": 0.000001,
                    "square inches": 0.00064516,
                    "square feet": 0.09290304,
                    "square yards": 0.83612736,
                    "square miles": 2589988.110336,
                    "acres": 4046.8564224,
                    "hectares": 10000.0
                }
            },
            "volume": {
                "units": ["cubic meters", "liters", "milliliters", "cubic centimeters",
                         "cubic inches", "cubic feet", "gallons", "quarts", "pints"],
                "conversions": {
                    "cubic meters": 1.0,
                    "liters": 0.001,
                    "milliliters": 0.000001,
                    "cubic centimeters": 0.000001,
                    "cubic inches": 0.000016387064,
                    "cubic feet": 0.028316846592,
                    "gallons": 0.003785411784,
                    "quarts": 0.000946352946,
                    "pints": 0.000473176473
                }
            },
            "speed": {
                "units": ["meters per second", "kilometers per hour", "miles per hour",
                         "feet per second", "knots"],
                "conversions": {
                    "meters per second": 1.0,
                    "kilometers per hour": 0.277778,
                    "miles per hour": 0.44704,
                    "feet per second": 0.3048,
                    "knots": 0.514444
                }
            },
            "pressure": {
                "units": ["pascals", "kilopascals", "megapascals", "bar", "psi",
                         "atmosphere", "torr", "millimeters of mercury"],
                "conversions": {
                    "pascals": 1.0,
                    "kilopascals": 1000.0,
                    "megapascals": 1000000.0,
                    "bar": 100000.0,
                    "psi": 6894.76,
                    "atmosphere": 101325.0,
                    "torr": 133.322,
                    "millimeters of mercury": 133.322
                }
            },
            "energy": {
                "units": ["joules", "kilojoules", "calories", "kilocalories",
                         "watt-hours", "kilowatt-hours", "electronvolts"],
                "conversions": {
                    "joules": 1.0,
                    "kilojoules": 1000.0,
                    "calories": 4.184,
                    "kilocalories": 4184.0,
                    "watt-hours": 3600.0,
                    "kilowatt-hours": 3600000.0,
                    "electronvolts": 1.602176634e-19
                }
            },
            "power": {
                "units": ["watts", "kilowatts", "megawatts", "horsepower",
                         "foot-pounds per second", "BTU per hour"],
                "conversions": {
                    "watts": 1.0,
                    "kilowatts": 1000.0,
                    "megawatts": 1000000.0,
                    "horsepower": 745.7,
                    "foot-pounds per second": 1.355818,
                    "BTU per hour": 0.2930711
                }
            }
        }
        
        # Update menus with new units
        self.from_menu.configure(values=unit_mappings[conversion_type]["units"])
        self.to_menu.configure(values=unit_mappings[conversion_type]["units"])
        
        # Set default values
        if not self.from_unit.get() or self.from_unit.get() not in unit_mappings[conversion_type]["units"]:
            self.from_unit.set(unit_mappings[conversion_type]["units"][0])
        if not self.to_unit.get() or self.to_unit.get() not in unit_mappings[conversion_type]["units"]:
            self.to_unit.set(unit_mappings[conversion_type]["units"][1])
            
        # Store the conversion mappings
        self.current_conversions = unit_mappings[conversion_type]["conversions"]
        
    def perform_conversion(self):
        """Perform the unit conversion"""
        try:
            # Get the input value
            value = float(self.from_value.get())
            
            # Get the conversion type
            conv_type = self.conversion_type.get()
            
            # Handle temperature conversion separately
            if conv_type == "temperature":
                # Convert to Celsius first
                from_unit = self.from_unit.get()
                to_unit = self.to_unit.get()
                
                if from_unit == "celsius":
                    celsius = value
                elif from_unit == "fahrenheit":
                    celsius = (value - 32) * 5/9
                else:  # kelvin
                    celsius = value - 273.15
                    
                # Convert from Celsius to target unit
                if to_unit == "celsius":
                    result = celsius
                elif to_unit == "fahrenheit":
                    result = (celsius * 9/5) + 32
                else:  # kelvin
                    result = celsius + 273.15
            else:
                # For other units, use the conversion factors
                from_factor = self.current_conversions[self.from_unit.get()]
                to_factor = self.current_conversions[self.to_unit.get()]
                
                # Convert to base unit, then to target unit
                base_value = value * from_factor
                result = base_value / to_factor
            
            # Update the result
            self.to_value.configure(state="normal")
            self.to_value.delete(0, "end")
            self.to_value.insert(0, f"{result:.6g}")
            self.to_value.configure(state="readonly")
            
        except ValueError:
            # Handle invalid input
            self.to_value.configure(state="normal")
            self.to_value.delete(0, "end")
            self.to_value.insert(0, "Invalid input")
            self.to_value.configure(state="readonly")
        except Exception as e:
            self.show_message("Error", f"Conversion error: {str(e)}")

    def setup_physics_chem_tab(self):
        phys_tab = self.tabview.tab("Physics & Chemistry")
        
        # Create main container
        main_container = CTkFrame(phys_tab)
        main_container.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Calculator type selection
        type_frame = CTkFrame(main_container)
        type_frame.pack(fill="x", pady=10)
        
        type_label = CTkLabel(type_frame, text="Calculator Type:", font=("Segoe UI", 14, "bold"))
        type_label.pack(side="left", padx=10)
        
        self.calc_type = StringVar(value="kinematics")
        calc_types = [
            "kinematics", "projectile", "ohms_law", "gas_laws", 
            "molar_mass", "dilution", "ph_calculator"
        ]
        
        self.calc_menu = CTkOptionMenu(
            type_frame,
            values=calc_types,
            variable=self.calc_type,
            command=self.update_calculator
        )
        self.calc_menu.pack(side="left", padx=10)
        
        # Calculator frame
        self.calc_frame = CTkFrame(main_container)
        self.calc_frame.pack(fill="both", expand=True, pady=10)
        
        # Initialize the first calculator
        self.update_calculator()
        
    def update_calculator(self, *args):
        """Update the calculator interface based on the selected type"""
        # Clear previous calculator
        for widget in self.calc_frame.winfo_children():
            widget.destroy()
            
        calc_type = self.calc_type.get()
        
        if calc_type == "kinematics":
            self.setup_kinematics_calculator()
        elif calc_type == "projectile":
            self.setup_projectile_calculator()
        elif calc_type == "ohms_law":
            self.setup_ohms_law_calculator()
        elif calc_type == "gas_laws":
            self.setup_gas_laws_calculator()
        elif calc_type == "molar_mass":
            self.setup_molar_mass_calculator()
        elif calc_type == "dilution":
            self.setup_dilution_calculator()
        elif calc_type == "ph_calculator":
            self.setup_ph_calculator()
            
    def setup_kinematics_calculator(self):
        title = CTkLabel(self.calc_frame, text="Kinematics", font=("Segoe UI", 16, "bold"))
        title.pack(pady=10)
        
        # Input frame
        input_frame = CTkFrame(self.calc_frame)
        input_frame.pack(fill="x", padx=20, pady=10)
        
        # Variables
        self.kin_vars = {
            "initial_velocity": StringVar(),
            "final_velocity": StringVar(),
            "acceleration": StringVar(),
            "time": StringVar(),
            "displacement": StringVar()
        }
        
        # Create input fields
        for i, (name, var) in enumerate(self.kin_vars.items()):
            frame = CTkFrame(input_frame)
            frame.pack(fill="x", pady=5)
            
            label = CTkLabel(frame, text=f"{name.replace('_', ' ').title()}:")
            label.pack(side="left", padx=5)
            
            entry = CTkEntry(frame, textvariable=var)
            entry.pack(side="left", padx=5, expand=True, fill="x")
            
        # Calculate button
        calc_btn = CTkButton(
            self.calc_frame,
            text="Calculate",
            command=self.calculate_kinematics,
            font=("Segoe UI", 12),
            height=35
        )
        calc_btn.pack(pady=10)
        
        # Results
        self.kin_results = CTkTextbox(self.calc_frame, height=100)
        self.kin_results.pack(fill="x", padx=20, pady=10)
        
    def calculate_kinematics(self):
        """Calculate kinematics values"""
        try:
            # Get values
            v0 = float(self.kin_vars["initial_velocity"].get() or 0)
            v = float(self.kin_vars["final_velocity"].get() or 0)
            a = float(self.kin_vars["acceleration"].get() or 0)
            t = float(self.kin_vars["time"].get() or 0)
            s = float(self.kin_vars["displacement"].get() or 0)
            
            # Count known variables
            known = sum(1 for x in [v0, v, a, t, s] if x != 0)
            
            if known < 3:
                self.kin_results.delete("1.0", "end")
                self.kin_results.insert("1.0", "Please provide at least 3 known values")
                return
                
            # Calculate missing values
            results = []
            
            # If time is unknown
            if t == 0:
                if v != 0 and v0 != 0 and a != 0:
                    t = (v - v0) / a
                    results.append(f"Time = {t:.2f} s")
                elif s != 0 and v0 != 0 and a != 0:
                    # Solve quadratic equation
                    t = (-v0 + (v0**2 + 2*a*s)**0.5) / a
                    results.append(f"Time = {t:.2f} s")
                    
            # If final velocity is unknown
            if v == 0:
                if v0 != 0 and a != 0 and t != 0:
                    v = v0 + a*t
                    results.append(f"Final Velocity = {v:.2f} m/s")
                elif s != 0 and v0 != 0 and t != 0:
                    v = 2*s/t - v0
                    results.append(f"Final Velocity = {v:.2f} m/s")
                    
            # If displacement is unknown
            if s == 0:
                if v0 != 0 and a != 0 and t != 0:
                    s = v0*t + 0.5*a*t**2
                    results.append(f"Displacement = {s:.2f} m")
                elif v != 0 and v0 != 0 and a != 0:
                    s = (v**2 - v0**2) / (2*a)
                    results.append(f"Displacement = {s:.2f} m")
                    
            # Display results
            self.kin_results.delete("1.0", "end")
            if results:
                self.kin_results.insert("1.0", "\n".join(results))
            else:
                self.kin_results.insert("1.0", "Unable to calculate with given values")
                
        except ValueError:
            self.kin_results.delete("1.0", "end")
            self.kin_results.insert("1.0", "Please enter valid numbers")
        except Exception as e:
            self.kin_results.delete("1.0", "end")
            self.kin_results.insert("1.0", f"Error: {str(e)}")
            
    def setup_projectile_calculator(self):
        title = CTkLabel(self.calc_frame, text="Projectile Motion", font=("Segoe UI", 16, "bold"))
        title.pack(pady=10)
        
        # Input frame
        input_frame = CTkFrame(self.calc_frame)
        input_frame.pack(fill="x", padx=20, pady=10)
        
        # Variables
        self.proj_vars = {
            "initial_velocity": StringVar(),
            "angle": StringVar(),
            "height": StringVar(value="0"),
            "gravity": StringVar(value="9.81")
        }
        
        # Create input fields
        for i, (name, var) in enumerate(self.proj_vars.items()):
            frame = CTkFrame(input_frame)
            frame.pack(fill="x", pady=5)
            
            label = CTkLabel(frame, text=f"{name.replace('_', ' ').title()}:")
            label.pack(side="left", padx=5)
            
            entry = CTkEntry(frame, textvariable=var)
            entry.pack(side="left", padx=5, expand=True, fill="x")
            
        # Calculate button
        calc_btn = CTkButton(
            self.calc_frame,
            text="Calculate",
            command=self.calculate_projectile,
            font=("Segoe UI", 12),
            height=35
        )
        calc_btn.pack(pady=10)
        
        # Results
        self.proj_results = CTkTextbox(self.calc_frame, height=150)
        self.proj_results.pack(fill="x", padx=20, pady=10)
        
    def calculate_projectile(self):
        """Calculate projectile motion values"""
        try:
            # Get values
            v0 = float(self.proj_vars["initial_velocity"].get())
            angle = float(self.proj_vars["angle"].get())
            h0 = float(self.proj_vars["height"].get())
            g = float(self.proj_vars["gravity"].get())
            
            # Convert angle to radians
            angle_rad = math.radians(angle)
            
            # Calculate components
            v0x = v0 * math.cos(angle_rad)
            v0y = v0 * math.sin(angle_rad)
            
            # Calculate time of flight
            # Solve quadratic equation: h0 + v0y*t - 0.5*g*t^2 = 0
            a = -0.5 * g
            b = v0y
            c = h0
            
            # Use quadratic formula
            t1 = (-b + math.sqrt(b**2 - 4*a*c)) / (2*a)
            t2 = (-b - math.sqrt(b**2 - 4*a*c)) / (2*a)
            t = max(t1, t2)  # Use the positive time
            
            # Calculate range
            range_x = v0x * t
            
            # Calculate maximum height
            t_max = v0y / g
            h_max = h0 + v0y * t_max - 0.5 * g * t_max**2
            
            # Display results
            results = [
                f"Time of Flight: {t:.2f} s",
                f"Range: {range_x:.2f} m",
                f"Maximum Height: {h_max:.2f} m",
                f"Horizontal Velocity: {v0x:.2f} m/s",
                f"Vertical Velocity: {v0y:.2f} m/s"
            ]
            
            self.proj_results.delete("1.0", "end")
            self.proj_results.insert("1.0", "\n".join(results))
            
        except ValueError:
            self.proj_results.delete("1.0", "end")
            self.proj_results.insert("1.0", "Please enter valid numbers")
        except Exception as e:
            self.proj_results.delete("1.0", "end")
            self.proj_results.insert("1.0", f"Error: {str(e)}")
            
    def setup_ohms_law_calculator(self):
        title = CTkLabel(self.calc_frame, text="Ohm's Law", font=("Segoe UI", 16, "bold"))
        title.pack(pady=10)
        
        # Input frame
        input_frame = CTkFrame(self.calc_frame)
        input_frame.pack(fill="x", padx=20, pady=10)
        
        # Variables
        self.ohm_vars = {
            "voltage": StringVar(),
            "current": StringVar(),
            "resistance": StringVar(),
            "power": StringVar()
        }
        
        # Create input fields
        for i, (name, var) in enumerate(self.ohm_vars.items()):
            frame = CTkFrame(input_frame)
            frame.pack(fill="x", pady=5)
            
            label = CTkLabel(frame, text=f"{name.title()} (V/A/Ω/W):")
            label.pack(side="left", padx=5)
            
            entry = CTkEntry(frame, textvariable=var)
            entry.pack(side="left", padx=5, expand=True, fill="x")
            
        # Calculate button
        calc_btn = CTkButton(
            self.calc_frame,
            text="Calculate",
            command=self.calculate_ohms_law,
            font=("Segoe UI", 12),
            height=35
        )
        calc_btn.pack(pady=10)
        
        # Results
        self.ohm_results = CTkTextbox(self.calc_frame, height=100)
        self.ohm_results.pack(fill="x", padx=20, pady=10)
        
    def calculate_ohms_law(self):
        """Calculate Ohm's Law values"""
        try:
            # Get values
            v = float(self.ohm_vars["voltage"].get() or 0)
            i = float(self.ohm_vars["current"].get() or 0)
            r = float(self.ohm_vars["resistance"].get() or 0)
            p = float(self.ohm_vars["power"].get() or 0)
            
            # Count known variables
            known = sum(1 for x in [v, i, r, p] if x != 0)
            
            if known < 2:
                self.ohm_results.delete("1.0", "end")
                self.ohm_results.insert("1.0", "Please provide at least 2 known values")
                return
                
            # Calculate missing values
            results = []
            
            # If voltage is unknown
            if v == 0:
                if i != 0 and r != 0:
                    v = i * r
                    results.append(f"Voltage = {v:.2f} V")
                elif p != 0 and i != 0:
                    v = p / i
                    results.append(f"Voltage = {v:.2f} V")
                    
            # If current is unknown
            if i == 0:
                if v != 0 and r != 0:
                    i = v / r
                    results.append(f"Current = {i:.2f} A")
                elif p != 0 and v != 0:
                    i = p / v
                    results.append(f"Current = {i:.2f} A")
                    
            # If resistance is unknown
            if r == 0:
                if v != 0 and i != 0:
                    r = v / i
                    results.append(f"Resistance = {r:.2f} Ω")
                elif p != 0 and i != 0:
                    r = p / (i**2)
                    results.append(f"Resistance = {r:.2f} Ω")
                    
            # If power is unknown
            if p == 0:
                if v != 0 and i != 0:
                    p = v * i
                    results.append(f"Power = {p:.2f} W")
                elif i != 0 and r != 0:
                    p = i**2 * r
                    results.append(f"Power = {p:.2f} W")
                    
            # Display results
            self.ohm_results.delete("1.0", "end")
            if results:
                self.ohm_results.insert("1.0", "\n".join(results))
            else:
                self.ohm_results.insert("1.0", "Unable to calculate with given values")
                
        except ValueError:
            self.ohm_results.delete("1.0", "end")
            self.ohm_results.insert("1.0", "Please enter valid numbers")
        except Exception as e:
            self.ohm_results.delete("1.0", "end")
            self.ohm_results.insert("1.0", f"Error: {str(e)}")
            
    def setup_gas_laws_calculator(self):
        title = CTkLabel(self.calc_frame, text="Gas Laws", font=("Segoe UI", 16, "bold"))
        title.pack(pady=10)
        
        # Law selection
        law_frame = CTkFrame(self.calc_frame)
        law_frame.pack(fill="x", padx=20, pady=10)
        
        law_label = CTkLabel(law_frame, text="Gas Law:")
        law_label.pack(side="left", padx=5)
        
        self.gas_law = StringVar(value="boyles")
        laws = ["boyles", "charles", "gay_lussac", "combined", "ideal"]
        
        self.law_menu = CTkOptionMenu(
            law_frame,
            values=laws,
            variable=self.gas_law,
            command=self.update_gas_law
        )
        self.law_menu.pack(side="left", padx=5)
        
        # Input frame
        self.gas_input_frame = CTkFrame(self.calc_frame)
        self.gas_input_frame.pack(fill="x", padx=20, pady=10)
        
        # Variables
        self.gas_vars = {
            "pressure1": StringVar(),
            "volume1": StringVar(),
            "temperature1": StringVar(),
            "pressure2": StringVar(),
            "volume2": StringVar(),
            "temperature2": StringVar(),
            "moles": StringVar(),
            "gas_constant": StringVar(value="0.0821")
        }
        
        # Calculate button
        calc_btn = CTkButton(
            self.calc_frame,
            text="Calculate",
            command=self.calculate_gas_law,
            font=("Segoe UI", 12),
            height=35
        )
        calc_btn.pack(pady=10)
        
        # Results
        self.gas_results = CTkTextbox(self.calc_frame, height=100)
        self.gas_results.pack(fill="x", padx=20, pady=10)
        
        # Initialize the first gas law
        self.update_gas_law()
        
    def update_gas_law(self, *args):
        """Update the gas law calculator interface"""
        # Clear previous inputs
        for widget in self.gas_input_frame.winfo_children():
            widget.destroy()
            
        law = self.gas_law.get()
        
        # Create input fields based on the selected law
        if law == "boyles":
            self.create_gas_input("pressure1", "Initial Pressure (atm):")
            self.create_gas_input("volume1", "Initial Volume (L):")
            self.create_gas_input("pressure2", "Final Pressure (atm):")
            self.create_gas_input("volume2", "Final Volume (L):")
        elif law == "charles":
            self.create_gas_input("volume1", "Initial Volume (L):")
            self.create_gas_input("temperature1", "Initial Temperature (K):")
            self.create_gas_input("volume2", "Final Volume (L):")
            self.create_gas_input("temperature2", "Final Temperature (K):")
        elif law == "gay_lussac":
            self.create_gas_input("pressure1", "Initial Pressure (atm):")
            self.create_gas_input("temperature1", "Initial Temperature (K):")
            self.create_gas_input("pressure2", "Final Pressure (atm):")
            self.create_gas_input("temperature2", "Final Temperature (K):")
        elif law == "combined":
            self.create_gas_input("pressure1", "Initial Pressure (atm):")
            self.create_gas_input("volume1", "Initial Volume (L):")
            self.create_gas_input("temperature1", "Initial Temperature (K):")
            self.create_gas_input("pressure2", "Final Pressure (atm):")
            self.create_gas_input("volume2", "Final Volume (L):")
            self.create_gas_input("temperature2", "Final Temperature (K):")
        elif law == "ideal":
            self.create_gas_input("pressure1", "Pressure (atm):")
            self.create_gas_input("volume1", "Volume (L):")
            self.create_gas_input("temperature1", "Temperature (K):")
            self.create_gas_input("moles", "Moles (mol):")
            self.create_gas_input("gas_constant", "Gas Constant (L·atm/mol·K):")
            
    def create_gas_input(self, var_name, label_text):
        """Create an input field for gas law calculations"""
        frame = CTkFrame(self.gas_input_frame)
        frame.pack(fill="x", pady=5)
        
        label = CTkLabel(frame, text=label_text)
        label.pack(side="left", padx=5)
        
        entry = CTkEntry(frame, textvariable=self.gas_vars[var_name])
        entry.pack(side="left", padx=5, expand=True, fill="x")
        
    def calculate_gas_law(self):
        """Calculate gas law values"""
        try:
            law = self.gas_law.get()
            results = []
            
            if law == "boyles":
                p1 = float(self.gas_vars["pressure1"].get() or 0)
                v1 = float(self.gas_vars["volume1"].get() or 0)
                p2 = float(self.gas_vars["pressure2"].get() or 0)
                v2 = float(self.gas_vars["volume2"].get() or 0)
                
                # Count known variables
                known = sum(1 for x in [p1, v1, p2, v2] if x != 0)
                
                if known < 3:
                    self.gas_results.delete("1.0", "end")
                    self.gas_results.insert("1.0", "Please provide at least 3 known values")
                    return
                    
                # Calculate missing value
                if p1 == 0:
                    p1 = (p2 * v2) / v1
                    results.append(f"Initial Pressure = {p1:.2f} atm")
                elif v1 == 0:
                    v1 = (p2 * v2) / p1
                    results.append(f"Initial Volume = {v1:.2f} L")
                elif p2 == 0:
                    p2 = (p1 * v1) / v2
                    results.append(f"Final Pressure = {p2:.2f} atm")
                elif v2 == 0:
                    v2 = (p1 * v1) / p2
                    results.append(f"Final Volume = {v2:.2f} L")
                    
            elif law == "charles":
                v1 = float(self.gas_vars["volume1"].get() or 0)
                t1 = float(self.gas_vars["temperature1"].get() or 0)
                v2 = float(self.gas_vars["volume2"].get() or 0)
                t2 = float(self.gas_vars["temperature2"].get() or 0)
                
                # Count known variables
                known = sum(1 for x in [v1, t1, v2, t2] if x != 0)
                
                if known < 3:
                    self.gas_results.delete("1.0", "end")
                    self.gas_results.insert("1.0", "Please provide at least 3 known values")
                    return
                    
                # Calculate missing value
                if v1 == 0:
                    v1 = (v2 * t1) / t2
                    results.append(f"Initial Volume = {v1:.2f} L")
                elif t1 == 0:
                    t1 = (v1 * t2) / v2
                    results.append(f"Initial Temperature = {t1:.2f} K")
                elif v2 == 0:
                    v2 = (v1 * t2) / t1
                    results.append(f"Final Volume = {v2:.2f} L")
                elif t2 == 0:
                    t2 = (v2 * t1) / v1
                    results.append(f"Final Temperature = {t2:.2f} K")
                    
            # Display results
            self.gas_results.delete("1.0", "end")
            if results:
                self.gas_results.insert("1.0", "\n".join(results))
            else:
                self.gas_results.insert("1.0", "Unable to calculate with given values")
                
        except ValueError:
            self.gas_results.delete("1.0", "end")
            self.gas_results.insert("1.0", "Please enter valid numbers")
        except Exception as e:
            self.gas_results.delete("1.0", "end")
            self.gas_results.insert("1.0", f"Error: {str(e)}")
            
    def setup_molar_mass_calculator(self):
        title = CTkLabel(self.calc_frame, text="Molar Mass", font=("Segoe UI", 16, "bold"))
        title.pack(pady=10)
        
        # Input frame
        input_frame = CTkFrame(self.calc_frame)
        input_frame.pack(fill="x", padx=20, pady=10)
        
        # Chemical formula input
        formula_frame = CTkFrame(input_frame)
        formula_frame.pack(fill="x", pady=5)
        
        formula_label = CTkLabel(formula_frame, text="Chemical Formula:")
        formula_label.pack(side="left", padx=5)
        
        self.formula_entry = CTkEntry(formula_frame)
        self.formula_entry.pack(side="left", padx=5, expand=True, fill="x")
        
        # Calculate button
        calc_btn = CTkButton(
            self.calc_frame,
            text="Calculate",
            command=self.calculate_molar_mass,
            font=("Segoe UI", 12),
            height=35
        )
        calc_btn.pack(pady=10)
        
        # Results
        self.molar_results = CTkTextbox(self.calc_frame, height=150)
        self.molar_results.pack(fill="x", padx=20, pady=10)
        
    def calculate_molar_mass(self):
        """Calculate molar mass of a chemical formula"""
        try:
            formula = self.formula_entry.get().strip()
            if not formula:
                self.molar_results.delete("1.0", "end")
                self.molar_results.insert("1.0", "Please enter a chemical formula")
                return
                
            # Parse the formula and calculate molar mass
            # This is a simplified version - in reality, you'd want a more robust parser
            total_mass = 0
            elements = {}
            
            # Split the formula into elements and their counts
            current_element = ""
            current_count = ""
            
            for char in formula:
                if char.isupper():
                    if current_element:
                        count = int(current_count) if current_count else 1
                        elements[current_element] = elements.get(current_element, 0) + count
                    current_element = char
                    current_count = ""
                elif char.islower():
                    current_element += char
                elif char.isdigit():
                    current_count += char
                    
            # Add the last element
            if current_element:
                count = int(current_count) if current_count else 1
                elements[current_element] = elements.get(current_element, 0) + count
                
            # Calculate total mass
            atomic_masses = {
                "H": 1.008, "He": 4.003, "Li": 6.941, "Be": 9.012, "B": 10.811,
                "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998, "Ne": 20.180,
                "Na": 22.990, "Mg": 24.305, "Al": 26.982, "Si": 28.086, "P": 30.974,
                "S": 32.065, "Cl": 35.453, "Ar": 39.948, "K": 39.098, "Ca": 40.078,
                "Fe": 55.845, "Cu": 63.546, "Zn": 65.380, "Ag": 107.868, "Au": 196.967
            }
            
            results = []
            for element, count in elements.items():
                if element in atomic_masses:
                    mass = atomic_masses[element] * count
                    total_mass += mass
                    results.append(f"{element}{count if count > 1 else ''}: {mass:.3f} g/mol")
                else:
                    self.molar_results.delete("1.0", "end")
                    self.molar_results.insert("1.0", f"Unknown element: {element}")
                    return
                    
            # Display results
            self.molar_results.delete("1.0", "end")
            self.molar_results.insert("1.0", f"Formula: {formula}\n\n")
            self.molar_results.insert("end", "Elemental Composition:\n")
            self.molar_results.insert("end", "\n".join(results))
            self.molar_results.insert("end", f"\n\nTotal Molar Mass: {total_mass:.3f} g/mol")
            
        except Exception as e:
            self.molar_results.delete("1.0", "end")
            self.molar_results.insert("1.0", f"Error: {str(e)}")
            
    def setup_dilution_calculator(self):
        title = CTkLabel(self.calc_frame, text="Dilution", font=("Segoe UI", 16, "bold"))
        title.pack(pady=10)
        
        # Input frame
        input_frame = CTkFrame(self.calc_frame)
        input_frame.pack(fill="x", padx=20, pady=10)
        
        # Variables
        self.dilution_vars = {
            "initial_concentration": StringVar(),
            "initial_volume": StringVar(),
            "final_concentration": StringVar(),
            "final_volume": StringVar()
        }
        
        # Create input fields
        for i, (name, var) in enumerate(self.dilution_vars.items()):
            frame = CTkFrame(input_frame)
            frame.pack(fill="x", pady=5)
            
            label = CTkLabel(frame, text=f"{name.replace('_', ' ').title()} (M/L):")
            label.pack(side="left", padx=5)
            
            entry = CTkEntry(frame, textvariable=var)
            entry.pack(side="left", padx=5, expand=True, fill="x")
            
        # Calculate button
        calc_btn = CTkButton(
            self.calc_frame,
            text="Calculate",
            command=self.calculate_dilution,
            font=("Segoe UI", 12),
            height=35
        )
        calc_btn.pack(pady=10)
        
        # Results
        self.dilution_results = CTkTextbox(self.calc_frame, height=100)
        self.dilution_results.pack(fill="x", padx=20, pady=10)
        
    def calculate_dilution(self):
        """Calculate dilution values"""
        try:
            # Get values
            c1 = float(self.dilution_vars["initial_concentration"].get() or 0)
            v1 = float(self.dilution_vars["initial_volume"].get() or 0)
            c2 = float(self.dilution_vars["final_concentration"].get() or 0)
            v2 = float(self.dilution_vars["final_volume"].get() or 0)
            
            # Count known variables
            known = sum(1 for x in [c1, v1, c2, v2] if x != 0)
            
            if known < 3:
                self.dilution_results.delete("1.0", "end")
                self.dilution_results.insert("1.0", "Please provide at least 3 known values")
                return
                
            # Calculate missing value
            results = []
            
            if c1 == 0:
                c1 = (c2 * v2) / v1
                results.append(f"Initial Concentration = {c1:.3f} M")
            elif v1 == 0:
                v1 = (c2 * v2) / c1
                results.append(f"Initial Volume = {v1:.3f} L")
            elif c2 == 0:
                c2 = (c1 * v1) / v2
                results.append(f"Final Concentration = {c2:.3f} M")
            elif v2 == 0:
                v2 = (c1 * v1) / c2
                results.append(f"Final Volume = {v2:.3f} L")
                
            # Display results
            self.dilution_results.delete("1.0", "end")
            if results:
                self.dilution_results.insert("1.0", "\n".join(results))
            else:
                self.dilution_results.insert("1.0", "Unable to calculate with given values")
                
        except ValueError:
            self.dilution_results.delete("1.0", "end")
            self.dilution_results.insert("1.0", "Please enter valid numbers")
        except Exception as e:
            self.dilution_results.delete("1.0", "end")
            self.dilution_results.insert("1.0", f"Error: {str(e)}")
            
    def setup_ph_calculator(self):
        title = CTkLabel(self.calc_frame, text="pH", font=("Segoe UI", 16, "bold"))
        title.pack(pady=10)
        
        # Input frame
        input_frame = CTkFrame(self.calc_frame)
        input_frame.pack(fill="x", padx=20, pady=10)
        
        # Variables
        self.ph_vars = {
            "concentration": StringVar(),
            "ph": StringVar(),
            "poh": StringVar()
        }
        
        # Create input fields
        for i, (name, var) in enumerate(self.ph_vars.items()):
            frame = CTkFrame(input_frame)
            frame.pack(fill="x", pady=5)
            
            label = CTkLabel(frame, text=f"{name.title()} (M):")
            label.pack(side="left", padx=5)
            
            entry = CTkEntry(frame, textvariable=var)
            entry.pack(side="left", padx=5, expand=True, fill="x")
            
        # Calculate button
        calc_btn = CTkButton(
            self.calc_frame,
            text="Calculate",
            command=self.calculate_ph,
            font=("Segoe UI", 12),
            height=35
        )
        calc_btn.pack(pady=10)
        
        # Results
        self.ph_results = CTkTextbox(self.calc_frame, height=100)
        self.ph_results.pack(fill="x", padx=20, pady=10)
        
    def calculate_ph(self):
        """Calculate pH values"""
        try:
            # Get values
            conc = float(self.ph_vars["concentration"].get() or 0)
            ph = float(self.ph_vars["ph"].get() or 0)
            poh = float(self.ph_vars["poh"].get() or 0)
            
            # Count known variables
            known = sum(1 for x in [conc, ph, poh] if x != 0)
            
            if known < 1:
                self.ph_results.delete("1.0", "end")
                self.ph_results.insert("1.0", "Please provide at least 1 known value")
                return
                
            # Calculate missing values
            results = []
            
            if conc > 0:
                ph_calc = -math.log10(conc)
                poh_calc = 14 - ph_calc
                results.append(f"pH = {ph_calc:.3f}")
                results.append(f"pOH = {poh_calc:.3f}")
            elif ph > 0:
                conc_calc = 10**(-ph)
                poh_calc = 14 - ph
                results.append(f"Concentration = {conc_calc:.3e} M")
                results.append(f"pOH = {poh_calc:.3f}")
            elif poh > 0:
                ph_calc = 14 - poh
                conc_calc = 10**(-ph_calc)
                results.append(f"pH = {ph_calc:.3f}")
                results.append(f"Concentration = {conc_calc:.3e} M")
                
            # Display results
            self.ph_results.delete("1.0", "end")
            if results:
                self.ph_results.insert("1.0", "\n".join(results))
            else:
                self.ph_results.insert("1.0", "Unable to calculate with given values")
                
        except ValueError:
            self.ph_results.delete("1.0", "end")
            self.ph_results.insert("1.0", "Please enter valid numbers")
        except Exception as e:
            self.ph_results.delete("1.0", "end")
            self.ph_results.insert("1.0", f"Error: {str(e)}")

if __name__ == "__main__":
    root = CTk()
    app = DCalc(root)
    root.mainloop()
        