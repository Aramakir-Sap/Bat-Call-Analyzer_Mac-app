import tkinter as tk
from pathlib import Path
from tkinter import ttk, messagebox, PhotoImage
from tkinter.scrolledtext import ScrolledText

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.patches import Polygon


def get_bat_call_data():
    """Load bat call measurements from the project CSV file."""
    csv_path = Path(__file__).resolve().parent / "Madabat_tool.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Could not find bat call data file: {csv_path}")

    df = pd.read_csv(csv_path)
    rename_map = {
        "Species": "species",
        "Family": "family",
        "Signal_Type": "signal_type",
        "Fppeak": "Fppeak",
        "Fp_mean_K": "Fp_mean_K",
        "Fp_max": "Fp_max",
        "Fp_min": "Fp_min",
        "Dur": "Dur",
        "IPI": "IPI",
        "Data": "data",
    }
    df = df.rename(columns=rename_map)

    if "species" not in df.columns:
        raise ValueError(
            "The CSV does not contain the expected 'Species' column.")

    for col in ["signal_type", "bandwidth", "Fmean_K", "Fmean_manual", "TBC", "Fmax", "Fmin"]:
        if col not in df.columns:
            df[col] = np.nan

    df["Fmax"] = pd.to_numeric(df["Fp_max"], errors="coerce")
    df["Fmin"] = pd.to_numeric(df["Fp_min"], errors="coerce")
    df["Fppeak"] = pd.to_numeric(df["Fppeak"], errors="coerce")
    df["Fp_mean_K"] = pd.to_numeric(df["Fp_mean_K"], errors="coerce")
    df["Fp_max"] = pd.to_numeric(df["Fp_max"], errors="coerce")
    df["Fp_min"] = pd.to_numeric(df["Fp_min"], errors="coerce")
    df["Dur"] = pd.to_numeric(df["Dur"], errors="coerce")
    df["IPI"] = pd.to_numeric(df["IPI"], errors="coerce")
    df["TBC"] = pd.to_numeric(df.get("TBC", np.nan), errors="coerce")
    df["bandwidth"] = pd.to_numeric(df["bandwidth"], errors="coerce")
    df["Fmean_K"] = pd.to_numeric(df["Fmean_K"], errors="coerce")
    df["Fmean_manual"] = pd.to_numeric(df["Fmean_manual"], errors="coerce")

    expected_columns = [
        "species",
        "signal_type",
        "Fppeak",
        "Fp_mean_K",
        "Fmean_K",
        "Fmean_manual",
        "Fp_max",
        "Fmax",
        "Fp_min",
        "Fmin",
        "Dur",
        "TBC",
        "IPI",
        "bandwidth",
        "family",
        "data",
    ]
    for col in expected_columns:
        if col not in df.columns:
            df[col] = np.nan

    return df[expected_columns]


class BatCallAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Bat Call Analyzer")
        self.root.geometry("1200x900")

        # Set window icon
        try:
            icon_path = Path(__file__).resolve().parent / "bat_icon.png"
            if icon_path.exists():
                icon = PhotoImage(file=str(icon_path))
                self.root.iconphoto(False, icon)
        except Exception as e:
            print(f"Could not load icon: {e}")

        self.all_data = None
        self.data_by_signal_type = {}
        self.current_data = None
        self.current_call_type = None
        self.data_loaded = False
        self.user_points = []
        self.param_entries = {}
        self.point_size = tk.IntVar(value=60)
        # Track which call types have shown warning
        self.warning_shown_for_calltype = set()

        self.create_widgets()
        self.root.after(100, self.load_data)

    def create_widgets(self):
        control_frame = ttk.Frame(self.root, padding="5")
        control_frame.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(control_frame, text="Call Type:", font=(
            "Arial", 9, "bold")).pack(side=tk.LEFT, padx=(5, 2))
        self.call_type_var = tk.StringVar()
        self.call_type_combo = ttk.Combobox(
            control_frame, textvariable=self.call_type_var, state="readonly", width=18)
        self.call_type_combo.pack(side=tk.LEFT, padx=3)
        self.call_type_combo.bind(
            "<<ComboboxSelected>>", self.on_call_type_selected)

        ttk.Separator(control_frame, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=8)

        ttk.Label(control_frame, text="X:").pack(side=tk.LEFT, padx=(5, 2))
        self.x_var = tk.StringVar()
        self.x_combo = ttk.Combobox(
            control_frame, textvariable=self.x_var, width=14)
        self.x_combo.pack(side=tk.LEFT, padx=2)
        self.x_combo.bind("<<ComboboxSelected>>",
                          self.on_axis_parameter_changed)

        ttk.Label(control_frame, text="Y:").pack(side=tk.LEFT, padx=(5, 2))
        self.y_var = tk.StringVar()
        self.y_combo = ttk.Combobox(
            control_frame, textvariable=self.y_var, width=14)
        self.y_combo.pack(side=tk.LEFT, padx=2)
        self.y_combo.bind("<<ComboboxSelected>>",
                          self.on_axis_parameter_changed)

        ttk.Button(control_frame, text="Generate Plot",
                   command=self.generate_plot).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Clear",
                   command=self.clear_user_calls).pack(side=tk.LEFT, padx=2)

        self.size_label = ttk.Label(control_frame, text="60", width=3)
        self.size_label.pack(side=tk.LEFT, padx=2)
        ttk.Label(control_frame, text="Size:").pack(side=tk.LEFT, padx=(5, 2))
        ttk.Scale(
            control_frame,
            from_=20,
            to=150,
            orient=tk.HORIZONTAL,
            variable=self.point_size,
            command=self.on_settings_changed,
            length=100,
        ).pack(side=tk.LEFT, padx=2)

        io_container = ttk.Frame(self.root)
        io_container.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        io_container.columnconfigure(0, weight=1)
        io_container.columnconfigure(1, weight=1)

        input_frame = ttk.LabelFrame(
            io_container, text="Your Call Parameters", padding="8")
        input_frame.grid(row=0, column=0, sticky=tk.NSEW, padx=(0, 5))

        param_names = [
            "Fppeak",
            "Fp_mean_K",
            "Fmean_K",
            "Fmean_manual",
            "Fp_max",
            "Fp_min",
            "Dur",
            "IPI",
            "TBC",
            "bandwidth",
        ]
        for index, param_name in enumerate(param_names):
            row = index // 2
            col = (index % 2) * 2
            ttk.Label(input_frame, text=f"{param_name}:").grid(
                row=row, column=col, padx=5, pady=3, sticky=tk.E)
            entry = ttk.Entry(input_frame, width=12)
            entry.grid(row=row, column=col + 1, padx=5, pady=3)
            self.param_entries[param_name] = entry

        button_frame = ttk.Frame(input_frame)
        button_frame.grid(row=(len(param_names) + 1) // 2,
                          column=0, columnspan=6, pady=8)
        ttk.Button(button_frame, text="Add Call Point",
                   command=self.add_user_call).pack(side=tk.LEFT, padx=3)
        ttk.Button(button_frame, text="Clear Points",
                   command=self.clear_user_calls).pack(side=tk.LEFT, padx=3)
        ttk.Button(button_frame, text="Identify Species",
                   command=self.identify_all_user_points).pack(side=tk.LEFT, padx=3)

        calls_frame = ttk.LabelFrame(
            io_container, text="Added Calls", padding="8")
        calls_frame.grid(row=0, column=1, sticky=tk.NSEW, padx=(5, 0))
        calls_frame.columnconfigure(0, weight=1)
        calls_frame.rowconfigure(0, weight=1)

        self.calls_tree = ttk.Treeview(calls_frame, columns=(
            "idx", "params"), show="headings", height=8)
        self.calls_tree.heading("idx", text="#")
        self.calls_tree.heading("params", text="Parameters")
        self.calls_tree.column("idx", width=25, anchor="center", stretch=False)
        self.calls_tree.column("params", minwidth=100,
                               anchor="w", stretch=True)
        self.calls_tree.grid(row=0, column=0, sticky=tk.NSEW)

        scrollbar = ttk.Scrollbar(
            calls_frame, orient=tk.VERTICAL, command=self.calls_tree.yview)
        scrollbar.grid(row=0, column=1, sticky=tk.NS)
        self.calls_tree.configure(yscrollcommand=scrollbar.set)

        btn_frame = ttk.Frame(calls_frame)
        btn_frame.grid(row=1, column=0, columnspan=2,
                       sticky=tk.EW, pady=(5, 0))
        ttk.Button(btn_frame, text="Remove Selected",
                   command=self.remove_selected_call, width=18).pack()

        self.status_label = ttk.Label(
            self.root, text="Loading data...", relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

        self.plot_frame = ttk.Frame(self.root)
        self.plot_frame.pack(side=tk.TOP, fill=tk.BOTH,
                             expand=True, padx=10, pady=10)

        self.canvas = None
        self.toolbar = None

    def load_data(self):
        try:
            self.all_data = get_bat_call_data()
            self.data_by_signal_type = {
                signal_type: group
                for signal_type, group in self.all_data.groupby("signal_type")
                if signal_type is not None and str(signal_type).strip() != ""
            }
            self.data_loaded = True
            self.populate_call_type_options()
            self.update_status("Data loaded successfully.")
        except Exception as exc:
            self.update_status(f"Error loading data: {exc}")
            messagebox.showerror("Error", str(exc))

    def populate_call_type_options(self):
        options = sorted(self.data_by_signal_type.keys())
        self.call_type_combo["values"] = options
        if options:
            self.call_type_var.set(options[0])
            self.on_call_type_selected()

    def on_call_type_selected(self, event=None):
        self.current_call_type = self.call_type_var.get()
        if self.current_call_type and self.current_call_type in self.data_by_signal_type:
            self.current_data = self.data_by_signal_type[self.current_call_type]
            # Reset warning flag for new call type
            self.warning_shown_for_calltype.discard(self.current_call_type)
            self.prepare_axis_options()
            self.clear_plot()
            self.generate_plot()

    def prepare_axis_options(self):
        if self.current_data is None:
            return

        numeric_columns = [
            col
            for col in ["Fppeak", "Fp_mean_K", "Fp_max", "Fp_min", "Dur", "IPI", "bandwidth", "Fmean_K", "Fmean_manual"]
            if col in self.current_data.columns and pd.api.types.is_numeric_dtype(self.current_data[col])
        ]
        self.x_combo["values"] = numeric_columns
        self.y_combo["values"] = numeric_columns
        if numeric_columns:
            self.x_var.set(numeric_columns[0])
            self.y_var.set(numeric_columns[1] if len(
                numeric_columns) > 1 else numeric_columns[0])

    def on_axis_parameter_changed(self, event=None):
        self.generate_plot()

    def on_settings_changed(self, *args):
        self.size_label.config(text=f"{self.point_size.get()}")
        if self.current_call_type and self.x_var.get() and self.y_var.get():
            self.generate_plot()

    def clear_param_entries(self):
        for entry in self.param_entries.values():
            entry.delete(0, tk.END)

    @staticmethod
    def _cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def _compute_convex_hull(self, points):
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2:
            return None
        points = points[np.isfinite(points).all(axis=1)]
        if len(points) < 3:
            return None
        unique_points = np.unique(points, axis=0)
        if len(unique_points) < 3:
            return None
        ordered = unique_points[np.lexsort(
            (unique_points[:, 1], unique_points[:, 0]))]

        lower = []
        for point in ordered:
            while len(lower) >= 2 and self._cross(lower[-2], lower[-1], point) <= 0:
                lower.pop()
            lower.append(point)

        upper = []
        for point in reversed(ordered):
            while len(upper) >= 2 and self._cross(upper[-2], upper[-1], point) <= 0:
                upper.pop()
            upper.append(point)

        hull = np.array(lower[:-1] + upper[:-1])
        if len(hull) < 3:
            return None
        return hull

    def _get_95_percent_points(self, points):
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2:
            return points
        if len(points) < 4:
            return points
        centroid = points.mean(axis=0)
        distances = np.linalg.norm(points - centroid, axis=1)
        cutoff = np.percentile(distances, 95)
        kept = points[distances <= cutoff]
        if len(kept) < 3:
            return points
        return kept

    def _draw_convex_hull(self, ax, points, color, alpha=0.14, linewidth=1.5, linestyle="-"):
        hull = self._compute_convex_hull(self._get_95_percent_points(points))
        if hull is None:
            return
        ax.add_patch(
            Polygon(
                hull,
                closed=True,
                fill=True,
                alpha=alpha,
                facecolor=color,
                edgecolor=color,
                linewidth=linewidth,
                linestyle=linestyle,
            )
        )

    def add_user_call(self):
        if self.current_data is None or self.current_data.empty:
            messagebox.showwarning(
                "No Data", "Load a call type first before adding points.")
            return

        point = {}
        for param_name, entry in self.param_entries.items():
            value = entry.get().strip()
            if value:
                try:
                    point[param_name] = float(value)
                except ValueError:
                    messagebox.showerror(
                        "Invalid Value", f"Please enter a valid number for {param_name}.")
                    return

        if not point:
            messagebox.showwarning(
                "Missing Values", "Enter at least one parameter value before adding a point.")
            return

        self.user_points.append(point)
        self.clear_param_entries()
        self.refresh_calls_tree()
        self.generate_plot()
        self.update_status(
            f"Added {len(self.user_points)} user call point(s).")

        # Check if parameters are significantly different from typical range
        self._check_parameter_outlier(point)

    def clear_user_calls(self):
        self.user_points.clear()
        self.clear_param_entries()
        self.refresh_calls_tree()
        self.generate_plot()
        self.update_status("User call points cleared.")

    def refresh_calls_tree(self):
        """Update the calls tree to show all added user calls."""
        for item in self.calls_tree.get_children():
            self.calls_tree.delete(item)

        for idx, point in enumerate(self.user_points, 1):
            params_list = [f"{k}:{v:.2f}" for k, v in sorted(point.items())]
            params_str = ", ".join(params_list)

            # Truncate to fit in display (approximately 70 chars) with ellipsis
            max_length = 70
            if len(params_str) > max_length:
                params_str = params_str[:max_length-3] + "..."

            self.calls_tree.insert("", tk.END, values=(str(idx), params_str))

    def _check_parameter_outlier(self, point):
        """Check if the point parameters are significantly different from the typical range."""
        # Only show warning once per call type
        if self.current_call_type in self.warning_shown_for_calltype:
            return

        x_param = self.x_var.get()
        y_param = self.y_var.get()

        # Check if x and y parameters exist in the point
        if x_param not in point or y_param not in point:
            return

        # Get max values from current data
        if self.current_data is None or self.current_data.empty:
            return

        x_max = self.current_data[x_param].max()
        y_max = self.current_data[y_param].max()

        threshold_x = x_max * 0.5  # 50% of max
        threshold_y = y_max * 0.5  # 50% of max

        # Check if user input exceeds threshold
        x_value = point[x_param]
        y_value = point[y_param]

        if x_value > threshold_x or y_value > threshold_y:
            # Show warning and mark this call type as warned
            messagebox.showwarning(
                "Warning: Unusual Parameters",
                "Warning: The parameters you have entered significantly differ from the typical known range of values for this call type. "
                "Please, ensure that your input is correct. If the parameters are correct, please contact giorgia.castiello@uniroma1.it, "
                "to help us improve our database. Thank you!"
            )
            self.warning_shown_for_calltype.add(self.current_call_type)

    def remove_selected_call(self):
        """Remove the selected call from the list."""
        selection = self.calls_tree.selection()
        if not selection:
            messagebox.showwarning(
                "No Selection", "Please select a call to remove.")
            return

        item = selection[0]
        idx = int(self.calls_tree.item(item, "values")[0]) - 1

        if 0 <= idx < len(self.user_points):
            self.user_points.pop(idx)
            self.refresh_calls_tree()
            self.generate_plot()
            self.update_status(
                f"Removed call. {len(self.user_points)} point(s) remaining.")

    def _get_identification_params(self, point):
        if self.current_data is None or self.current_data.empty:
            return []

        available_params = [
            param for param in point.keys()
            if param in self.current_data.columns and param != "species" and pd.api.types.is_numeric_dtype(self.current_data[param])
        ]
        if not available_params:
            available_params = [
                param for param in point.keys() if param != "species"]
        return available_params

    def _calc_distance(self, point_values, centroid_values):
        point_array = np.asarray(point_values, dtype=float)
        centroid_array = np.asarray(centroid_values, dtype=float)
        if point_array.shape != centroid_array.shape:
            return float("inf")
        return float(np.linalg.norm(point_array - centroid_array))

    def identify_species_for_point(self, point):
        if self.current_data is None or self.current_data.empty:
            return []

        available_params = self._get_identification_params(point)
        if not available_params:
            return []

        results = []
        for species, group in self.current_data.groupby("species"):
            if pd.isna(species) or str(species).strip() == "":
                continue
            species_data = group[available_params].apply(
                pd.to_numeric, errors="coerce")
            centroid = species_data.mean()
            user_vector = np.array([float(point.get(param, np.nan))
                                   for param in available_params], dtype=float)
            centroid_vector = np.array(
                [float(centroid[param]) for param in available_params], dtype=float)

            # Find valid parameters (both user and species have data)
            valid_mask = ~(np.isnan(user_vector) | np.isnan(centroid_vector))
            if not valid_mask.any():
                continue  # No valid parameters in common

            # Use only valid parameters for distance calculation
            valid_user = user_vector[valid_mask]
            valid_centroid = centroid_vector[valid_mask]
            valid_ranges = species_data.iloc[:, valid_mask].max(
            ) - species_data.iloc[:, valid_mask].min()
            valid_ranges = valid_ranges.replace(0, np.nan)

            if valid_ranges.isna().all():
                normalized_user = valid_user
                normalized_centroid = valid_centroid
            else:
                normalized_user = valid_user / valid_ranges.values
                normalized_centroid = valid_centroid / valid_ranges.values

            distance = self._calc_distance(
                normalized_user, normalized_centroid)
            results.append((str(species), distance, len(group)))

        results.sort(key=lambda item: item[1])
        return results[:3]

    def identify_all_user_points(self):
        if not self.user_points:
            messagebox.showwarning(
                "No Points", "Add at least one call point first.")
            return

        lines = ["Species identification report"]
        point_reports = []
        for index, point in enumerate(self.user_points, start=1):
            matches = self.identify_species_for_point(point)
            point_reports.append((index, point, matches))
            lines.append(f"\nPoint {index}: {point}")
            if matches:
                for species, distance, count in matches:
                    lines.append(
                        f"- {species} (distance={distance:.3f}, n={count})")
            else:
                lines.append("- No match found")

        centroid_matches = []
        if self.user_points:
            first_point = self.user_points[0]
            params = self._get_identification_params(first_point)
            if params:
                centroid_values = {}
                for param in params:
                    values = [float(
                        point[param]) for point in self.user_points if param in point and point[param] is not None]
                    if values:
                        centroid_values[param] = float(np.mean(values))
                if centroid_values:
                    lines.append("\nUser-point centroid")
                    centroid_matches = self.identify_species_for_point(
                        centroid_values)
                    if centroid_matches:
                        for species, distance, count in centroid_matches:
                            lines.append(
                                f"- {species} (distance={distance:.3f}, n={count})")
                    else:
                        lines.append("- No match found")

        self.root.after(0, lambda: self._show_identification_popup(
            lines, point_reports, centroid_matches))

    def _show_identification_popup(self, lines, point_reports, centroid_matches):
        popup = tk.Toplevel(self.root)
        popup.title("Species Identification Report")
        popup.transient(self.root)
        popup.grab_set()
        popup.focus_set()

        # Use pack layout for auto-sizing
        popup.columnconfigure(0, weight=0)
        popup.rowconfigure(0, weight=0)

        header = ttk.Label(
            popup,
            text="🦇 Species Identification Report",
            font=("Segoe UI", 14, "bold"),
            padding=(10, 10, 10, 8),
        )
        header.pack(fill=tk.X, padx=10, pady=(10, 8))

        notebook = ttk.Notebook(popup)
        notebook.pack(fill=tk.BOTH, expand=False, padx=10, pady=(0, 10))

        self._create_individual_points_tab(notebook, point_reports)
        self._create_cluster_check_tab(
            notebook, point_reports, centroid_matches)
        self._create_summary_tab(notebook, point_reports, centroid_matches)

        close_btn = ttk.Button(popup, text="Close",
                               command=popup.destroy, width=15)
        close_btn.pack(anchor=tk.E, padx=10, pady=(0, 10))

        # By default a ttk.Notebook sizes itself to fit the LARGEST of all
        # its tabs, not the one currently shown -- that's what left blank
        # space under the shorter "Individual Points" / "Cluster Check"
        # tabs. Instead, resize the notebook (and the popup) to match
        # whichever tab is actually selected, every time it changes.
        def fit_popup_to_selected_tab(event=None):
            popup.update_idletasks()
            selected = notebook.select()
            if not selected:
                return
            tab_frame = notebook.nametowidget(selected)
            tab_frame.update_idletasks()
            content_height = tab_frame.winfo_reqheight()

            # Explicitly setting the notebook's height overrides its
            # default "fit the largest tab" behavior.
            notebook.configure(height=content_height)
            popup.update_idletasks()

            screen_width = self.root.winfo_screenwidth()
            screen_height = self.root.winfo_screenheight()
            max_width = int(screen_width * 0.75)
            max_height = int(screen_height * 0.80)

            natural_width = popup.winfo_reqwidth()
            natural_height = popup.winfo_reqheight()
            width = min(natural_width, max_width)
            height = min(natural_height, max_height)

            popup.geometry(f"{width}x{height}")

        notebook.bind("<<NotebookTabChanged>>", fit_popup_to_selected_tab)
        fit_popup_to_selected_tab()

    def _create_scrollable_area(self, parent, max_height=420):
        """Create a canvas+scrollbar area whose height hugs its content
        (up to max_height) instead of always stretching to fill the tab,
        which is what caused the large blank space below short reports.
        The scrollbar only appears once the content actually overflows.
        """
        canvas = tk.Canvas(parent, bg="white", highlightthickness=0)
        scrollbar = ttk.Scrollbar(
            parent, orient=tk.VERTICAL, command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        canvas_window = canvas.create_window(
            (0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        def on_frame_configure(event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))
            content_height = scrollable_frame.winfo_reqheight()
            new_height = min(content_height, max_height)
            if canvas.winfo_reqheight() != new_height:
                canvas.configure(height=new_height)

            needs_scroll = content_height > max_height
            if needs_scroll and not scrollbar.winfo_ismapped():
                scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            elif not needs_scroll and scrollbar.winfo_ismapped():
                scrollbar.pack_forget()

        def on_canvas_configure(event):
            canvas.itemconfig(canvas_window, width=event.width)

        scrollable_frame.bind("<Configure>", on_frame_configure)
        canvas.bind("<Configure>", on_canvas_configure)

        def _on_mousewheel(event):
            if scrollbar.winfo_ismapped():
                canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind("<Enter>", lambda e: canvas.bind_all(
            "<MouseWheel>", _on_mousewheel))
        canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))

        canvas.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        return canvas, scrollable_frame

    def _create_individual_points_tab(self, notebook, point_reports):
        frame = ttk.Frame(notebook, padding=10)
        notebook.add(frame, text="Individual Points ({})" .format(
            len(point_reports)))
        frame.columnconfigure(0, weight=1)

        canvas, scrollable_frame = self._create_scrollable_area(frame)

        for idx, (point_idx, point, matches) in enumerate(point_reports):
            point_frame = ttk.LabelFrame(
                scrollable_frame, text="Point #{}".format(point_idx), padding=12)
            point_frame.pack(fill=tk.X, pady=10, padx=5)
            point_frame.columnconfigure(0, weight=1)

            ttk.Label(point_frame, text="Call Parameters:", font=(
                "Segoe UI", 9, "bold"), foreground="#333").pack(anchor=tk.W, pady=(0, 6))

            params_text = " | ".join(["{}: {:.2f}".format(k, v)
                                     for k, v in point.items()])
            params_label = ttk.Label(
                point_frame, text=params_text, foreground="#0066CC", font=("Segoe UI", 8, "italic"), wraplength=600, justify=tk.LEFT)
            params_label.pack(anchor=tk.W, pady=(0, 12), padx=(10, 0))

            ttk.Label(point_frame, text="Species Matches:", font=(
                "Segoe UI", 9, "bold"), foreground="#333").pack(anchor=tk.W, pady=(0, 8))

            if matches:
                for rank, (species, distance, count) in enumerate(matches, 1):
                    match_frame = ttk.Frame(point_frame)
                    match_frame.pack(fill=tk.X, pady=5, padx=(10, 0))
                    match_frame.columnconfigure(3, weight=1)

                    if rank == 1:
                        badge_bg = "#FFD700"
                        badge_text = "1st"
                        badge_fg = "black"
                    elif rank == 2:
                        badge_bg = "#C0C0C0"
                        badge_text = "2nd"
                        badge_fg = "black"
                    else:
                        badge_bg = "#FF9500"
                        badge_text = "3rd"
                        badge_fg = "white"

                    badge = tk.Label(
                        match_frame,
                        text=badge_text,
                        bg=badge_bg,
                        fg=badge_fg,
                        font=("Segoe UI", 8, "bold"),
                        width=4,
                        padx=4,
                        pady=2
                    )
                    badge.pack(side=tk.LEFT, padx=(0, 10))

                    info = ttk.Label(
                        match_frame,
                        text="{0}  distance = {1:.4f}  (n={2} samples)".format(
                            species, distance, count),
                        font=("Segoe UI", 9)
                    )
                    info.pack(side=tk.LEFT, fill=tk.X, expand=True)
            else:
                ttk.Label(point_frame, text="⚠ No match found",
                          foreground="#d32f2f", font=("Segoe UI", 9, "italic")).pack(anchor=tk.W, padx=(10, 0))

    def _create_cluster_check_tab(self, notebook, point_reports, centroid_matches):
        frame = ttk.Frame(notebook, padding=15)
        notebook.add(frame, text="Cluster Check")
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(3, weight=1)

        stats_frame = ttk.LabelFrame(
            frame, text="📊 Cluster Statistics", padding=12)
        stats_frame.pack(fill=tk.X, pady=(0, 12))
        stats_frame.columnconfigure(1, weight=1)

        ttk.Label(stats_frame, text="Total Points:", font=(
            "Segoe UI", 9, "bold")).grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        ttk.Label(stats_frame, text=str(len(point_reports)), font=(
            "Segoe UI", 10), foreground="#0066CC").grid(row=0, column=1, sticky=tk.W)

        info_label = ttk.Label(
            frame,
            text="Analysis based on the centroid (average) of all your input points. Each entered call is compared with species centroids from the selected signal type.",
            foreground="#555", font=("Segoe UI", 9), wraplength=800, justify=tk.LEFT
        )
        info_label.pack(anchor=tk.W, pady=(0, 12))

        ttk.Label(frame, text="🎯 Species Ranked by Match Quality", font=(
            "Segoe UI", 11, "bold")).pack(anchor=tk.W, pady=(0, 10))

        canvas, scrollable_frame = self._create_scrollable_area(frame)
        scrollable_frame.columnconfigure(0, weight=1)

        matches_list = centroid_matches or (
            point_reports[0][2] if point_reports else [])

        if matches_list:
            min_distance = min([m[1] for m in matches_list]
                               ) if matches_list else 0
            max_distance = max([m[1] for m in matches_list]
                               ) if matches_list else 1
            distance_range = max_distance - min_distance if max_distance > min_distance else 1

            for rank, (species, distance, count) in enumerate(matches_list[:15], 1):
                match_frame = ttk.Frame(scrollable_frame)
                match_frame.pack(fill=tk.X, pady=4, padx=5)
                match_frame.columnconfigure(3, weight=1)

                if rank == 1:
                    badge_bg = "#FFD700"
                    badge_fg = "black"
                elif rank == 2:
                    badge_bg = "#C0C0C0"
                    badge_fg = "black"
                elif rank == 3:
                    badge_bg = "#FF9500"
                    badge_fg = "white"
                else:
                    badge_bg = "#E8E8E8"
                    badge_fg = "#333"

                ordinal = str(rank) + ("th" if rank >
                                       3 else ["st", "nd", "rd"][rank-1])
                badge = tk.Label(
                    match_frame,
                    text=ordinal,
                    bg=badge_bg,
                    fg=badge_fg,
                    font=("Segoe UI", 8, "bold"),
                    width=4,
                    padx=4,
                    pady=3
                )
                badge.pack(side=tk.LEFT, padx=(0, 10))

                info = ttk.Label(
                    match_frame,
                    text="{0}  |  Distance: {1:.4f}  |  Samples: n={2}".format(
                        species, distance, count),
                    font=("Segoe UI", 9)
                )
                info.pack(side=tk.LEFT, fill=tk.X, expand=True)

    def _create_summary_tab(self, notebook, point_reports, centroid_matches):
        frame = ttk.Frame(notebook, padding=15)
        notebook.add(frame, text="Summary")
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(1, weight=1)

        canvas, scrollable_frame = self._create_scrollable_area(frame)
        scrollable_frame.columnconfigure(0, weight=1)

        ttk.Label(scrollable_frame, text="📍 Cluster Centroid Profile", font=(
            "Segoe UI", 11, "bold"), foreground="#1565C0").pack(anchor=tk.W, pady=(0, 10))

        param_frame = ttk.LabelFrame(
            scrollable_frame, text="Averaged Parameters from All Input Points", padding=12)
        param_frame.pack(fill=tk.X, pady=(0, 16), padx=5)
        param_frame.columnconfigure(1, weight=1)
        param_frame.columnconfigure(3, weight=1)

        if point_reports:
            first_point = point_reports[0][1]
            params = self._get_identification_params(first_point)
            if params:
                centroid_values = {}
                row = 0
                for param_idx, param in enumerate(params):
                    values = [float(point[1].get(param, np.nan))
                              for point in point_reports if param in point[1] and point[1][param] is not None]
                    if values:
                        centroid_values[param] = float(np.mean(values))

                        col = 0 if param_idx % 2 == 0 else 2
                        if param_idx % 2 == 0 and param_idx > 0:
                            row += 1

                        param_label = ttk.Label(param_frame, text="{0}:".format(
                            param), font=("Segoe UI", 9, "bold"), foreground="#333")
                        param_label.grid(row=row, column=col,
                                         sticky=tk.W, padx=(0, 10), pady=5)

                        value_label = ttk.Label(param_frame, text="{0:.2f}".format(
                            centroid_values[param]), foreground="#0066CC", font=("Segoe UI", 9, "italic"))
                        value_label.grid(row=row, column=col +
                                         1, sticky=tk.W, pady=5)

        ttk.Label(scrollable_frame, text="🏆 Best Matching Species", font=(
            "Segoe UI", 11, "bold"), foreground="#1565C0").pack(anchor=tk.W, pady=(12, 10))

        matches_list = centroid_matches or (
            point_reports[0][2] if point_reports else [])

        for rank, (species, distance, count) in enumerate(matches_list[:5], 1):
            match_frame = ttk.Frame(scrollable_frame)
            match_frame.pack(fill=tk.X, pady=6, padx=5)
            match_frame.columnconfigure(2, weight=1)

            if rank == 1:
                badge_bg = "#FFD700"
                badge_fg = "black"
                frame_relief = "solid"
                frame_borderwidth = 2
            elif rank == 2:
                badge_bg = "#C0C0C0"
                badge_fg = "black"
                frame_relief = "flat"
                frame_borderwidth = 1
            else:
                badge_bg = "#FF9500"
                badge_fg = "white"
                frame_relief = "flat"
                frame_borderwidth = 1

            inner_frame = tk.Frame(
                match_frame, relief=frame_relief, borderwidth=frame_borderwidth, bg="#F5F5F5")
            inner_frame.pack(fill=tk.X)
            inner_frame.columnconfigure(2, weight=1)

            badge = tk.Label(
                inner_frame,
                text=str(rank),
                bg=badge_bg,
                fg=badge_fg,
                font=("Segoe UI", 9, "bold"),
                width=3,
                padx=6,
                pady=4
            )
            badge.grid(row=0, column=0, padx=8, pady=6)

            species_label = ttk.Label(
                inner_frame,
                text=species,
                font=("Segoe UI", 10, "bold"),
                foreground="#000"
            )
            species_label.grid(row=0, column=1, sticky=tk.W, padx=(0, 15))

            info_frame = tk.Frame(inner_frame, bg="#F5F5F5")
            info_frame.grid(row=0, column=2, sticky=tk.EW, padx=10, pady=6)
            info_frame.columnconfigure(0, weight=1)

            info_text = "Distance: {0:.4f}  |  Reference Samples: n={1}".format(
                distance, count)
            info_label = ttk.Label(
                info_frame,
                text=info_text,
                font=("Segoe UI", 9),
                foreground="#666"
            )
            info_label.pack(side=tk.LEFT)

        ttk.Separator(scrollable_frame, orient="horizontal").pack(
            fill=tk.X, pady=12)

        ttk.Label(scrollable_frame, text="ℹ️  Interpretation Tips", font=(
            "Segoe UI", 11, "bold"), foreground="#1565C0").pack(anchor=tk.W, pady=(0, 10))

        notes_frame = ttk.LabelFrame(
            scrollable_frame, text="How to interpret these results", padding=12)
        notes_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=(0, 10))
        notes_frame.columnconfigure(0, weight=1)

        notes_list = [
            ("Individual Points", "Each of your input recordings is analyzed separately and compared to species centroids."),
            ("Cluster Check", "All your recordings are averaged together, then the combined profile is matched against species."),
            ("Distance Metric",
             "Uses normalized Euclidean distance. Lower values indicate better matches."),
            ("Reference Data", "Comparisons are against species centroids computed from the Madabat dataset."),
            ("Top Matches", "Consider the top 3-5 matches. If distances are very close, the distinction may be ambiguous.")
        ]

        for title, description in notes_list:
            title_label = ttk.Label(
                notes_frame, text="• " + title + ":", font=("Segoe UI", 9, "bold"), foreground="#333")
            title_label.pack(anchor=tk.W, pady=(6, 2))
            desc_label = ttk.Label(notes_frame, text=description, font=(
                "Segoe UI", 9), foreground="#666", wraplength=700, justify=tk.LEFT)
            desc_label.pack(anchor=tk.W, padx=(20, 0), pady=(0, 8))

    def generate_plot(self):
        if self.current_data is None or self.current_data.empty or not self.x_var.get() or not self.y_var.get():
            return

        x_col = self.x_var.get()
        y_col = self.y_var.get()
        fig = Figure(figsize=(8, 6), dpi=100)
        ax = fig.add_subplot(111)

        if "species" in self.current_data.columns:
            species_groups = []
            for species, group in self.current_data.groupby("species"):
                if pd.isna(species) or str(species).strip() == "":
                    continue
                species_groups.append((species, group))

            if species_groups:
                cmap = plt.cm.tab10
                for index, (species, group) in enumerate(species_groups):
                    color = cmap(index % 10)
                    points = group[[x_col, y_col]].apply(
                        pd.to_numeric, errors="coerce").dropna()
                    if len(points) >= 3:
                        self._draw_convex_hull(
                            ax, points.to_numpy(), color=color)
                    ax.scatter(
                        group[x_col],
                        group[y_col],
                        s=self.point_size.get(),
                        alpha=0.7,
                        c=[color],
                        edgecolors="black",
                        linewidths=0.4,
                        label=str(species),
                    )
            else:
                ax.scatter(self.current_data[x_col], self.current_data[y_col], s=self.point_size.get(
                ), alpha=0.7, edgecolors="black", linewidths=0.4, label="Known calls")
        else:
            ax.scatter(self.current_data[x_col], self.current_data[y_col], s=self.point_size.get(
            ), alpha=0.7, edgecolors="black", linewidths=0.4, label="Known calls")

        if self.user_points:
            user_points = [
                (point[x_col], point[y_col])
                for point in self.user_points
                if x_col in point and y_col in point
            ]
            if user_points:
                user_x = [item[0] for item in user_points]
                user_y = [item[1] for item in user_points]
                if len(user_points) >= 3:
                    self._draw_convex_hull(
                        ax,
                        np.array(user_points, dtype=float),
                        color="red",
                        alpha=0.10,
                        linewidth=1.2,
                        linestyle="--",
                    )
                ax.scatter(user_x, user_y, s=self.point_size.get() * 1.3, c="red", marker="D",
                           edgecolors="black", linewidths=1.5, label="Your calls", zorder=5)

        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f"{self.current_call_type} scatter plot")
        ax.grid(True, alpha=0.3)
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend(
                handles,
                labels,
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                fontsize=9.5,
                markerscale=0.9,
                ncol=1,
            )
        fig.subplots_adjust(right=0.72)

        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        self.canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame)
        self.toolbar.update()
        self.plot_frame.update_idletasks()

    def update_status(self, message):
        self.status_label.config(text=message)

    def clear_plot(self):
        if self.current_data is not None and not self.current_data.empty and self.x_var.get() and self.y_var.get():
            self.generate_plot()
        else:
            if hasattr(self, "plot_frame"):
                for widget in self.plot_frame.winfo_children():
                    widget.destroy()
            self.update_status("Plot cleared.")


if __name__ == "__main__":
    root = tk.Tk()
    app = BatCallAnalyzerApp(root)
    root.mainloop()
