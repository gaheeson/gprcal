#!/usr/bin/env python3
"""
gprcal_gui.py  —  GPRCal: GUI for the Gamma Pass Rate Calculator

Requires: pydicom, numpy, scipy, matplotlib, tkinter
Optional: numba (fast backend), pymedphys (library backend)
Run:
    python gprcal_gui.py
"""

import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from gprcal_backends import detect_backends, compute_gamma
from gprcal import load_rtdose, compute_gamma_2d, _MULTI_CRITERIA


# =============================================================================
# Main window
# =============================================================================

class GammaCalcGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("GPRCal — Gamma Pass Rate Calculator")
        root.resizable(True, True)
        root.minsize(580, 660)

        self._ref_path       = tk.StringVar()   # full path (internal)
        self._eval_path      = tk.StringVar()   # full path (internal)
        self._ref_display    = tk.StringVar()   # basename only (shown in UI)
        self._eval_display   = tk.StringVar()   # basename only (shown in UI)

        self._dd        = tk.DoubleVar(value=3.0)
        self._dta       = tk.DoubleVar(value=3.0)
        self._threshold = tk.DoubleVar(value=30.0)
        self._local     = tk.BooleanVar(value=False)
        self._normalize = tk.BooleanVar(value=True)
        self._interp    = tk.IntVar(value=10)
        self._mode2d    = tk.BooleanVar(value=False)
        self._multi     = tk.BooleanVar(value=False)

        # Backend
        self._backends   = detect_backends()
        self._avail      = [k for k, v in self._backends.items() if v]
        self._backend    = tk.StringVar(value=self._avail[0] if self._avail else "Pure Python")

        self._last_gamma   = None
        self._last_ref     = None
        self._last_evl     = None
        self._cancel_event = None
        self._last_dd    = None
        self._last_dta   = None
        self._last_thr   = None
        self._last_local = None

        self._build()

    # ── Layout ────────────────────────────────────────────────────────────────

    def _build(self):
        pad = dict(padx=10, pady=4)

        # ── Files ─────────────────────────────────────────────────────────────
        file_frame = ttk.LabelFrame(self.root, text="Input Files")
        file_frame.pack(fill="x", **pad)
        file_frame.columnconfigure(1, weight=1)

        ttk.Label(file_frame, text="Reference (.dcm):").grid(
            row=0, column=0, sticky="w", padx=6, pady=3)
        ttk.Entry(file_frame, textvariable=self._ref_display,
                  state="readonly").grid(
            row=0, column=1, sticky="ew", padx=4, pady=3)
        ttk.Button(file_frame, text="Browse…",
                   command=self._browse_ref).grid(row=0, column=2, padx=6, pady=3)

        ttk.Label(file_frame, text="Evaluation (.dcm):").grid(
            row=1, column=0, sticky="w", padx=6, pady=3)
        ttk.Entry(file_frame, textvariable=self._eval_display,
                  state="readonly").grid(
            row=1, column=1, sticky="ew", padx=4, pady=3)
        ttk.Button(file_frame, text="Browse…",
                   command=self._browse_eval).grid(row=1, column=2, padx=6, pady=3)

        # ── Criteria ──────────────────────────────────────────────────────────
        crit_frame = ttk.LabelFrame(self.root, text="Gamma Criteria")
        crit_frame.pack(fill="x", **pad)

        def _spin(parent, var, from_, to, inc, row, col, width=7):
            sb = ttk.Spinbox(parent, textvariable=var, from_=from_, to=to,
                             increment=inc, width=width)
            sb.grid(row=row, column=col, sticky="w", padx=4, pady=3)
            return sb

        ttk.Label(crit_frame, text="DD (%)").grid(
            row=0, column=0, sticky="w", padx=6, pady=3)
        _spin(crit_frame, self._dd, 0.1, 10.0, 0.5, 0, 1)

        ttk.Label(crit_frame, text="DTA (mm)").grid(
            row=0, column=2, sticky="w", padx=6, pady=3)
        _spin(crit_frame, self._dta, 0.1, 10.0, 0.5, 0, 3)

        ttk.Label(crit_frame, text="Threshold (%)").grid(
            row=0, column=4, sticky="w", padx=6, pady=3)
        _spin(crit_frame, self._threshold, 0.0, 50.0, 5.0, 0, 5)

        ttk.Label(crit_frame, text="Sub-voxel interp").grid(
            row=1, column=0, sticky="w", padx=6, pady=3)
        _spin(crit_frame, self._interp, 3, 20, 1, 1, 1)

        ttk.Checkbutton(crit_frame, text="Local gamma",
                        variable=self._local).grid(
            row=1, column=2, columnspan=2, sticky="w", padx=6, pady=3)
        ttk.Checkbutton(crit_frame, text="Normalize to max",
                        variable=self._normalize).grid(
            row=1, column=4, columnspan=2, sticky="w", padx=6, pady=3)

        # ── Backend ───────────────────────────────────────────────────────────
        backend_frame = ttk.LabelFrame(self.root, text="Backend")
        backend_frame.pack(fill="x", **pad)

        # Build radio buttons for each backend
        all_backends = ["Pure Python", "Numba", "pymedphys"]
        descriptions = {
            "Pure Python": "Pure Python / scipy  (slow for large 3D volumes)",
            "Numba":       "Numba  (JIT-compiled, parallel — ~50-200× faster)",
            "pymedphys":   "pymedphys  (Python library)",
        }
        for name in all_backends:
            available = self._backends.get(name, False)
            label = descriptions[name]
            if not available:
                label += "  [not installed]"
            rb = ttk.Radiobutton(
                backend_frame,
                text=label,
                variable=self._backend,
                value=name,
                state="normal" if available else "disabled",
            )
            rb.pack(anchor="w", padx=8, pady=2)

        # Numba note
        if self._backends.get("Numba"):
            import numba
            n_threads = numba.config.NUMBA_NUM_THREADS
            note = ttk.Label(
                backend_frame,
                text=f"  * Numba: {n_threads} CPU threads available."
                     f" First run compiles (~10-20s); subsequent runs are instant.",
                foreground="gray",
                font=("TkDefaultFont", 9),
            )
            note.pack(anchor="w", padx=8, pady=(0, 4))

        # ── Mode ──────────────────────────────────────────────────────────────
        mode_frame = ttk.LabelFrame(self.root, text="Mode")
        mode_frame.pack(fill="x", **pad)

        ttk.Checkbutton(
            mode_frame,
            text="2D mode  (middle Z-slice only — much faster)",
            variable=self._mode2d,
        ).pack(side="left", padx=8, pady=4)

        ttk.Checkbutton(
            mode_frame,
            text="Multi-criteria  (3/3 · 2/2 · 1/1 · 3/2)",
            variable=self._multi,
        ).pack(side="left", padx=8, pady=4)

        # ── Buttons ───────────────────────────────────────────────────────────
        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(fill="x", **pad)

        self._run_btn = ttk.Button(
            btn_frame, text="Run Gamma", command=self._run)
        self._run_btn.pack(side="left", padx=4)

        self._stop_btn = ttk.Button(
            btn_frame, text="Stop", command=self._stop, state="disabled")
        self._stop_btn.pack(side="left", padx=4)

        ttk.Button(btn_frame, text="Show Map",
                   command=self._show_map).pack(side="left", padx=4)
        ttk.Button(btn_frame, text="Show Histogram",
                   command=self._show_hist).pack(side="left", padx=4)
        ttk.Button(btn_frame, text="Save Map (.npy)…",
                   command=self._save_map).pack(side="left", padx=4)
        ttk.Button(btn_frame, text="Save Figure…",
                   command=self._save_figure).pack(side="left", padx=4)

        # ── Progress ──────────────────────────────────────────────────────────
        self._progress = ttk.Progressbar(
            self.root, mode="indeterminate", length=100)
        self._progress.pack(fill="x", padx=10, pady=2)

        # ── Results ───────────────────────────────────────────────────────────
        res_frame = ttk.LabelFrame(self.root, text="Results")
        res_frame.pack(fill="both", expand=True, **pad)

        scroll_frame = ttk.Frame(res_frame)
        scroll_frame.pack(fill="both", expand=True, padx=4, pady=4)

        self._result_text = tk.Text(
            scroll_frame, height=10, font=("Courier", 11),
            state="disabled", background="#f5f5f5",
            relief="flat", wrap="none",
        )
        sb_y = ttk.Scrollbar(scroll_frame, orient="vertical",
                              command=self._result_text.yview)
        self._result_text.configure(yscrollcommand=sb_y.set)
        sb_y.pack(side="right", fill="y")
        self._result_text.pack(side="left", fill="both", expand=True)

    # ── File dialogs ──────────────────────────────────────────────────────────

    def _browse_ref(self):
        p = filedialog.askopenfilename(
            title="Select Reference RT Dose",
            filetypes=[("DICOM files", "*.dcm"), ("All files", "*.*")],
        )
        if p:
            self._ref_path.set(p)
            self._ref_display.set(os.path.basename(p))

    def _browse_eval(self):
        p = filedialog.askopenfilename(
            title="Select Evaluation RT Dose",
            filetypes=[("DICOM files", "*.dcm"), ("All files", "*.*")],
        )
        if p:
            self._eval_path.set(p)
            self._eval_display.set(os.path.basename(p))

    # ── Run (threaded) ────────────────────────────────────────────────────────

    def _run(self):
        ref_path  = self._ref_path.get().strip()
        eval_path = self._eval_path.get().strip()
        if not ref_path:
            messagebox.showerror("Error", "Reference file is not selected.")
            return
        if not eval_path:
            messagebox.showerror("Error", "Evaluation file is not selected.")
            return

        self._cancel_event = threading.Event()
        self._run_btn.config(state="disabled")
        self._stop_btn.config(state="normal")
        self._progress.start(10)
        self._log_clear()

        thread = threading.Thread(target=self._run_worker,
                                  args=(ref_path, eval_path), daemon=True)
        thread.start()

    def _run_worker(self, ref_path, eval_path):
        import time
        try:
            self._log(f"Loading reference:  {os.path.basename(ref_path)}\n")
            ref = load_rtdose(ref_path)
            self._log(f"  {ref}\n")

            self._log(f"Loading evaluation: {os.path.basename(eval_path)}\n")
            evl = load_rtdose(eval_path)
            self._log(f"  {evl}\n\n")

            dd        = float(self._dd.get())
            dta       = float(self._dta.get())
            threshold = float(self._threshold.get())
            local     = bool(self._local.get())
            norm      = bool(self._normalize.get())
            interp    = int(self._interp.get())
            mode2d    = bool(self._mode2d.get())
            multi     = bool(self._multi.get())
            backend   = self._backend.get()

            criteria = _MULTI_CRITERIA if multi else [(dd, dta)]
            last_map = None

            for dd_i, dta_i in criteria:
                lbl = "Local" if local else "Global"
                self._log(f"{'─'*52}\n")
                self._log(f"{lbl} Gamma  {dd_i}%/{dta_i}mm/{threshold:.0f}%"
                          f"  [{backend}]\n")
                self._log(f"{'─'*52}\n")

                t0 = time.time()

                if mode2d:
                    sl = ref.shape[0] // 2
                    gmap, pr = compute_gamma_2d(
                        ref.dose[sl], ref.x, ref.y,
                        evl.dose[sl], evl.x, evl.y,
                        dd_percent=dd_i, dta_mm=dta_i,
                        threshold_percent=threshold,
                        local_gamma=local, normalize=norm,
                        interp_fraction=interp,
                    )
                else:
                    gmap, pr = compute_gamma(
                        ref, evl,
                        dd_percent=dd_i, dta_mm=dta_i,
                        threshold_percent=threshold,
                        local_gamma=local, normalize=norm,
                        interp_fraction=interp,
                        backend=backend,
                        progress_cb=self._log,
                        cancel_event=self._cancel_event,
                    )

                elapsed = time.time() - t0
                valid   = gmap[~np.isnan(gmap)]
                n_pass  = int(np.sum(valid <= 1.0))
                n_total = len(valid)

                self._log(f"  Pass rate (gamma <= 1) : {pr:.2f}%"
                          f"  ({n_pass}/{n_total} pts)\n")
                self._log(f"  Mean gamma             : {np.nanmean(gmap):.4f}\n")
                self._log(f"  Max gamma              : {np.nanmax(gmap):.4f}\n")
                self._log(f"  Time                   : {elapsed:.1f}s\n\n")
                last_map = gmap

            self._log("Done.\n")

            self._last_gamma = last_map
            self._last_ref   = ref
            self._last_evl   = evl
            self._last_dd    = dd
            self._last_dta   = dta
            self._last_thr   = threshold
            self._last_local = local

        except InterruptedError:
            self._log("\nCancelled.\n")
        except Exception as exc:
            import traceback
            self._log(f"\nERROR: {exc}\n")
            self._log(traceback.format_exc())
        finally:
            self.root.after(0, self._run_done)

    def _stop(self):
        if self._cancel_event is not None:
            self._cancel_event.set()
        self._stop_btn.config(state="disabled")
        self._log("  [Stop requested — cancelling after current batch…]\n")

    def _run_done(self):
        self._progress.stop()
        self._run_btn.config(state="normal")
        self._stop_btn.config(state="disabled")

    # ── Logging ───────────────────────────────────────────────────────────────

    def _log(self, text: str):
        def _append():
            self._result_text.config(state="normal")
            self._result_text.insert("end", text)
            self._result_text.see("end")
            self._result_text.config(state="disabled")
        self.root.after(0, _append)

    def _log_clear(self):
        def _clear():
            self._result_text.config(state="normal")
            self._result_text.delete("1.0", "end")
            self._result_text.config(state="disabled")
        self.root.after(0, _clear)

    # ── Plot / Save ───────────────────────────────────────────────────────────

    def _check_results(self) -> bool:
        if self._last_gamma is None:
            messagebox.showinfo("Info", "Run the gamma calculation first.")
            return False
        return True

    def _pass_rate(self):
        valid = self._last_gamma[~np.isnan(self._last_gamma)]
        return float(np.sum(valid <= 1.0) / len(valid) * 100) if len(valid) else 0.0

    def _show_map(self):
        if not self._check_results():
            return
        from gprcal import plot_results
        plot_results(self._last_ref, self._last_evl, self._last_gamma,
                     self._pass_rate(),
                     self._last_dd, self._last_dta, self._last_thr,
                     self._last_local)

    def _show_hist(self):
        if not self._check_results():
            return
        from gprcal import plot_histogram
        plot_histogram(self._last_gamma, self._pass_rate(),
                       self._last_dd, self._last_dta,
                       self._last_thr, local=self._last_local)

    def _save_map(self):
        if not self._check_results():
            return
        path = filedialog.asksaveasfilename(
            title="Save Gamma Map",
            defaultextension=".npy",
            filetypes=[("NumPy array", "*.npy")],
        )
        if path:
            np.save(path, self._last_gamma)
            messagebox.showinfo("Saved", f"Gamma map saved:\n{path}")

    def _save_figure(self):
        if not self._check_results():
            return
        path = filedialog.asksaveasfilename(
            title="Save Figure",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")],
        )
        if not path:
            return
        from gprcal import plot_results
        plot_results(self._last_ref, self._last_evl, self._last_gamma,
                     self._pass_rate(),
                     self._last_dd, self._last_dta, self._last_thr,
                     self._last_local, save_path=path)


# =============================================================================
# Entry point
# =============================================================================

def main():
    root = tk.Tk()
    try:
        root.tk.call("source", "azure.tcl")
        root.tk.call("set_theme", "light")
    except tk.TclError:
        pass
    GammaCalcGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
