from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

import matplotlib as mpl
import numpy as np
import cmocean
import pickle
from glob import glob
import os
from pyzfn import Pyzfn as op
from pyzfn import utils


class Data:
    def __init__(self, dset, Bmin, Bmax, fmin, fmax):
        self.dset = dset
        self.Bmin = Bmin
        self.Bmax = Bmax
        self.fmin = fmin
        self.fmax = fmax
        self.paths, self.Bs = self.get_paths_and_Bs()
        self.dx = op(self.paths[0]).dx * 1e9
        self.freqs, self.ifmin, self.ifmax = self.get_freqs()
        self.spectra = self.get_spectrum_data()
        self.mode_freqs, self.mode_ifreqs = self.get_mode_freqs()
        self.modes = self.get_branch_modes()
        self.indices_mode_peak = self.get_indices_mode_peak()
        self.radii = self.get_radii()
        self.ratios = self.get_rim_bulk_ratio()
        self.mode_sections = self.get_mode_sections()
        self.thetas, self.phases, self.amplitudes = self.get_thetas_phase_amp()
        self.mode_images = self.get_mode_images()
        self.adlmr, self.adlsmr, self.rl = self.get_mz()

    def get_paths_and_Bs(self):
        paths = sorted(
            glob(f"{self.dset}/*.zarr"),
            key=lambda x: int(x.split("/")[-1].replace(".zarr", "")),
        )
        paths = [p for p in paths if os.path.exists(f"{p}/tmodes")]
        paths_out, Bs = [], []
        for p in paths:
            if os.path.exists(f"{p}/tmodes"):
                B = int(p.split("/")[-1].replace(".zarr", ""))
                if self.Bmin <= B <= self.Bmax:
                    Bs.append(B)
                    paths_out.append(p)
        return np.array(paths_out), np.array(Bs)

    def get_freqs(self):
        freqs = np.array(op(self.paths[-1]).fft.m.freqs[:])
        ifmin = np.abs(freqs - self.fmin).argmin()
        ifmax = np.abs(freqs - self.fmax).argmin()
        return freqs[ifmin:ifmax], ifmin, ifmax

    def get_spectrum_data(self, force=False):
        spectra = np.empty((self.paths.shape[0], self.freqs.shape[0]))
        for i, p in enumerate(self.paths):
            # spectra[i] = np.sum(op(p).fft.m.spec[self.ifmin : self.ifmax, :], axis=-1)
            spectra[i] = op(p).fft.m.spec[self.ifmin : self.ifmax, 0]
        spectra = np.array(spectra).T
        spectra = spectra / spectra.max()

        return spectra

    def get_mode_freqs(self):
        mode_freqs = []
        mode_ifreqs = []
        for p, spec in zip(self.paths, self.spectra.T):
            m = op(p)
            f = self.freqs[np.argmax(spec)]
            ifreq = int(np.argmin(np.abs(m.tmodes.m.freqs[:] - f)))
            if ifreq == 0 and int(m.name) < 100:
                ifreq = 1
            mode_freqs.append(m.tmodes.m.freqs[ifreq])
            mode_ifreqs.append(ifreq)
        return np.array(mode_freqs), np.array(mode_ifreqs)

    def get_branch_modes(self):
        modes = []
        for p, ifreq in zip(self.paths, self.mode_ifreqs):
            modes.append(op(p).tmodes.m.arr[ifreq, 0, :, :, :])
        return np.array(modes)

    def get_indices_mode_peak(self):
        return np.array(
            [
                np.argmax(np.abs(np.sum(mode[127, :127, :], axis=-1)))
                for mode in self.modes
            ]
        )

    def get_radii(self):
        return np.array(
            [
                (self.modes.shape[1] - 2 * i) / 2 * self.dx
                for i in self.indices_mode_peak
            ]
        )

    def get_rim_bulk_ratio(self):
        ratios = []
        for mode in self.modes:
            mode = np.abs(np.sum(mode[128, :77], axis=-1))
            rim_area = np.trapz(mode[51:])
            bulk_area = np.trapz(mode[:51])
            ratios.append(rim_area / (rim_area + bulk_area))
        return np.array(ratios)

    def get_mode_sections(self):
        mode_sections = []
        for mode in self.modes:
            mode = mode[127, :77, :]
            mode = np.average(mode, axis=-1)
            mode = np.abs(mode)
            mode_sections.append(mode)
        return np.array(mode_sections)

    def get_thetas_phase_amp(self):
        phases, amplitudes = [], []
        for mode, modes_peak_index in zip(self.modes, self.indices_mode_peak):
            arr = np.average(mode, axis=-1)
            phase = np.angle(arr)
            phase += np.pi  # [0,2pi]
            _, phase = self.sample_circle_edge(
                phase, mode.shape[0] // 2 - modes_peak_index
            )
            phase = np.unwrap(phase)

            thetas, amplitude = self.sample_circle_edge(
                np.abs(np.average(mode, axis=-1)), mode.shape[0] // 2 - modes_peak_index
            )
            phases.append(phase)
            amplitudes.append(amplitude)
        return np.array(thetas), np.array(phases), np.array(amplitudes)

    def sample_circle_edge(self, array, radius, num_points=360):
        h, k = array.shape[0] // 2, array.shape[1] // 2
        thetas = np.linspace(0, 2 * np.pi, num_points)
        x = h + radius * np.cos(thetas)
        y = k + radius * np.sin(thetas)

        # Round to nearest integers and ensure within array bounds
        x = np.clip(np.round(x).astype(int), 0, array.shape[1] - 1)
        y = np.clip(np.round(y).astype(int), 0, array.shape[0] - 1)

        # Sample values from array
        sampled_values = array[y, x]
        return thetas, sampled_values

    def get_mode_images(self):
        mode_images = []
        for mode in self.modes:
            arr = np.average(mode, axis=-1)
            ang = np.angle(arr)
            ang += np.pi  # [0,2pi]
            ang -= ang[64, 127]
            ang -= ang.min()  # [0,2pi]
            ang -= np.pi  # [-pi,pi]
            ang = ang / np.pi / 2 + 0.5  # [0,1]
            rgbi = plt.cm.hsv(ang)[:, :, :3]
            hsli = utils.rgb2hsl(rgbi)
            hsli[:, :, 1] = np.abs(arr) / np.abs(arr).max()
            mode_images.append(utils.hsl2rgb(hsli))
        return np.array(mode_images)

    def get_mz(self):
        adlsmr, rl, adlmr = [], [], []
        for B in self.Bs:
            adlsmr.append(
                np.ma.masked_equal(
                    op(
                        f"{self.dset.replace('adl-mr','adl-s-mr')}/{B:0>4}.zarr/"
                    ).stable[0, 0, 127, :77, 2],
                    0,
                )
            )
            adlmr.append(
                np.ma.masked_equal(
                    op(f"{self.dset}/{B:0>4}.zarr/").stable[0, 0, 127, :77, 2],
                    0,
                )
            )
            rl.append(
                np.ma.masked_equal(
                    op(f"{self.dset.replace('adl-mr','rl')}/{B:0>4}.zarr/").stable[
                        0, 0, 127, :77, 2
                    ],
                    0,
                )
            )
        return (
            np.ma.masked_array(adlmr),
            np.ma.masked_array(adlsmr),
            np.ma.masked_array(rl),
        )


class Plot:
    def __init__(self, fig, data, step=40):
        self.step = step
        self.data = data

        self.fig = fig
        self.gs = self.fig.add_gridspec(12, 12)
        self.axes = self.get_axes()
        self.colors = [i["color"] for i in list(mpl.rcParams["axes.prop_cycle"])]

        self.hline, self.vline, self.vline2 = self.make_cursor_lines(
            self.axes["ax1"], self.axes["ax2"]
        )
        self.prepare_ax1(self.axes["ax1"])
        self.prepare_ax2(self.axes["ax2"])
        self.prepare_ax3(self.axes["ax3"])
        self.prepare_ax4(self.axes["ax4"], self.axes["ax4b"])
        self.prepare_ax5(self.axes["ax5"], self.axes["ax5b"])
        self.gs.update(
            left=0.07, right=0.90, top=0.98, bottom=0.10, wspace=1, hspace=0.6
        )

    def get_axes(self):
        axes = {
            "ax1": self.fig.add_subplot(self.gs[:7, :5]),
            "ax2": self.fig.add_subplot(self.gs[7:, :5]),
            "ax3": self.fig.add_subplot(self.gs[:6, 6:9]),
            "ax4": self.fig.add_subplot(self.gs[:6, 9:]),
            "ax5": self.fig.add_subplot(self.gs[7:, 6:]),
        }
        axes["ax4b"] = plt.twinx(axes["ax4"])
        axes["ax5b"] = plt.twinx(axes["ax5"])
        return axes

    def make_cursor_lines(self, axa, axb):
        hline = axa.axhline(self.data.fmin, ls="--", lw=1, c="white")
        vline = axa.axvline(self.data.Bs[0], ls="--", lw=1, c="white")
        vline2 = axb.axvline(self.data.Bmin, ls="--", lw=1, c="k")
        return hline, vline, vline2

    def prepare_ax1(self, ax):
        ax.tick_params(
            axis="x", bottom=False, top=True, labelbottom=False, labeltop=True
        )
        ax.imshow(
            self.data.spectra,
            aspect="auto",
            origin="lower",
            norm=mpl.colors.LogNorm(vmin=0.01),
            interpolation="None",
            extent=[
                self.data.Bs[0] - (self.data.Bs[1] - self.data.Bs[0]) / 2,
                self.data.Bs[-1] + (self.data.Bs[1] - self.data.Bs[0]) / 2,
                self.data.freqs.min(),
                self.data.freqs.max(),
            ],
            cmap="cmo.amp_r",
        )
        ax.scatter(
            self.data.Bs, self.data.mode_freqs, s=7, marker="x", lw=0.5, alpha=0.7
        )

    def prepare_ax2(self, ax):
        ax.set(xlabel="B0 (mT)")
        ax.plot(self.data.Bs, self.data.ratios)
        ax.set_ylabel("area ratio", c=self.colors[0])
        ax.tick_params(axis="y", colors=self.colors[0])
        axb = plt.twinx(ax)
        axb.plot(
            self.data.Bs, self.data.indices_mode_peak * self.data.dx, c=self.colors[1]
        )
        axb.axhline(100, ls="--", c=self.colors[1])
        axb.set_ylabel("Peak Location (nm)", c=self.colors[1])
        axb.tick_params(axis="y", colors=self.colors[1])
        ax.grid()
        ax.set_xlim(self.data.Bs[0], self.data.Bs[-1])

    def prepare_ax3(self, ax):
        img = self.data.mode_images[self.step]
        extent = [0, img.shape[0] * self.data.dx, 0, img.shape[1] * self.data.dx]
        ax.imshow(
            img,
            aspect="equal",
            origin="lower",
            extent=extent,
        )
        ax.contour(
            np.abs(op(self.data.paths[0]).Ku1[0, 0, :, :, 0]),
            levels=[226597.5],
            linestyles=["dashed"],
            alpha=0.5,
            extent=extent,
        )
        ax.plot(
            [0, 80 * self.data.dx],
            [127 * self.data.dx, 127 * self.data.dx],
            ls="dashed",
            c="red",
        )

        ax.add_patch(
            mpl.patches.Circle(
                (
                    self.data.modes[self.step].shape[0] // 2 * self.data.dx,
                    self.data.modes[self.step].shape[1] // 2 * self.data.dx,
                ),
                self.data.radii[self.step],
                fill=False,
                edgecolor="red",
                linestyle="--",
            ),
        )

    def prepare_ax4(self, axa, axb):
        axa.plot(
            self.data.thetas,
            self.data.phases[self.step],
            label="phase",
            c=self.colors[0],
        )
        axa.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
        axa.set_xticklabels(
            ["0", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"]
        )
        axa.grid(True)
        axa.legend(
            loc="center",
            bbox_to_anchor=(0.2, 0.1),
            facecolor="#ffffff",
            borderpad=0.1,
            labelspacing=0.1,
            handlelength=1,
            handletextpad=0.2,
            edgecolor="none",
            fancybox=False,
            fontsize=6,
        )
        axa.tick_params(axis="y", colors=self.colors[0])
        axa.set_ylabel("phase (radiant)", c=self.colors[0], labelpad=-1)
        axa.set_ylim(self.data.phases.min(), self.data.phases.max())

        axb.plot(
            self.data.thetas,
            self.data.amplitudes[self.step],
            label="amplitude",
            c=self.colors[1],
        )
        axb.tick_params(axis="y", colors=self.colors[1])
        axb.set_ylabel("Mode Amplitude (arb. units)", c=self.colors[1], labelpad=-1)
        axb.set_ylim(self.data.amplitudes.min(), self.data.amplitudes.max())

    def prepare_ax5(self, axa, axb):
        mode_section = self.data.mode_sections[self.step]
        x = np.arange(mode_section.shape[0]) * op(self.data.paths[0]).dx * 1e9
        axa.set_ylim(self.data.mode_sections.min(), self.data.mode_sections.max())
        axa.set_xlabel("$x$ (nm)", labelpad=-2)
        axa.plot(x, mode_section, lw=1, label=f"Mode", c=self.colors[0])
        axa.axvline(100, ls="--", c="k", alpha=0.4)
        axa.tick_params(axis="y", colors=self.colors[0])
        axa.set_ylabel("Mode Amplitude (arb. units)", c=self.colors[0])
        axa.grid()

        for y, label, c in zip(
            [self.data.adlmr, self.data.adlsmr, self.data.rl],
            ["ADL-MR", "ADL-S-MR", "Ring lattice"],
            ["#ff0000", "#0000ff", "#00ff00"],
        ):
            axb.plot(x, y[self.step], c=c, label=label, lw=1)
        axb.axvline(100, ls="--", c="k", alpha=0.4)

        axb.set_ylim(-1.1, 1.1)
        axb.set_ylabel("$m_{z}$", labelpad=-1)
        axb.legend(
            loc="center",
            bbox_to_anchor=(0.3, 0.6),
            facecolor="#ffffff",
            borderpad=0.1,
            labelspacing=0.1,
            handlelength=1,
            handletextpad=0.2,
            edgecolor="none",
            fancybox=False,
            fontsize=6,
        )

    def update(self, step):
        self.step = step
        self.hline.set_ydata(self.data.mode_freqs[self.step])
        self.vline.set_xdata(self.data.Bs[self.step])
        self.vline2.set_xdata(self.data.Bs[self.step])
        self.axes["ax3"].get_images()[0].set_data(self.data.mode_images[self.step])
        self.axes["ax3"].patches[0].set_radius(self.data.radii[self.step])
        self.axes["ax4"].get_lines()[0].set_ydata(self.data.phases[self.step])
        self.axes["ax4b"].get_lines()[0].set_ydata(self.data.amplitudes[self.step])
        self.axes["ax5"].get_lines()[0].set_ydata(self.data.mode_sections[self.step])
        self.axes["ax5b"].get_lines()[0].set_ydata(self.data.adlmr[self.step])
        self.axes["ax5b"].get_lines()[1].set_ydata(self.data.adlsmr[self.step])
        self.axes["ax5b"].get_lines()[2].set_ydata(self.data.rl[self.step])

    def anim(self, name):
        self.step = 0
        ani = FuncAnimation(
            self.fig,
            self.update,
            interval=100,
            frames=np.linspace(
                0, self.data.Bs.shape[0] - 1, self.data.Bs.shape[0], dtype="int"
            ),
        )
        ani.save(
            f"./{name}.mp4",
            writer="ffmpeg",
            fps=10,
            dpi=300,
            extra_args=["-vcodec", "h264", "-pix_fmt", "yuv420p"],
        )


def prepare_mpl_params():
    def is_times_new_roman_available():
        try:
            fonts = mpl.font_manager.findSystemFonts(fontpaths=None, fontext="ttf")
            for font in fonts:
                try:
                    if (
                        mpl.font_manager.FontProperties(fname=font).get_name()
                        == "Times New Roman"
                    ):
                        return True
                except Exception:
                    continue
            return False
        except Exception:
            return False

    if not is_times_new_roman_available():
        print(
            "Times New Roman was not detected, please install it for the figures to render properly"
        )

    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["font.serif"] = ["Times New Roman"]
    mpl.rcParams["font.weight"] = "100"
    mpl.rcParams["mathtext.fontset"] = "custom"
    mpl.rcParams["mathtext.cal"] = "Times New Roman:italic"
    mpl.rcParams["mathtext.rm"] = "Times New Roman"
    mpl.rcParams["mathtext.it"] = "Times New Roman:italic"
    mpl.rcParams["mathtext.bf"] = "Times New Roman:bold"
    fs = 10
    mpl.rcParams["font.size"] = fs
    mpl.rcParams["axes.labelsize"] = fs
    mpl.rcParams["xtick.labelsize"] = fs
    mpl.rcParams["ytick.labelsize"] = fs
    mpl.rcParams["legend.fontsize"] = fs
    mpl.rcParams["lines.linewidth"] = 0.8
    mpl.rcParams["axes.spines.left"] = True
    mpl.rcParams["axes.spines.bottom"] = True
    mpl.rcParams["axes.spines.top"] = True
    mpl.rcParams["axes.spines.right"] = True

    mpl.rcParams["xtick.major.size"] = 5
    mpl.rcParams["xtick.minor.size"] = 3
    mpl.rcParams["xtick.major.width"] = 0.6
    mpl.rcParams["xtick.minor.width"] = 0.4
    mpl.rcParams["xtick.direction"] = "in"
    mpl.rcParams["xtick.minor.visible"] = True
    mpl.rcParams["ytick.major.size"] = 5
    mpl.rcParams["ytick.minor.size"] = 3
    mpl.rcParams["ytick.major.width"] = 0.6
    mpl.rcParams["ytick.minor.width"] = 0.4
    mpl.rcParams["ytick.direction"] = "in"
    mpl.rcParams["ytick.minor.visible"] = True

    mpl.rcParams["figure.dpi"] = 200
    mpl.rcParams["savefig.dpi"] = 300
    mpl.rcParams["image.origin"] = "lower"


def save_cache(data, path):
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_cache(path):
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    dset = "../paper/adl-mr"
    data = Data(dset=dset, Bmin=0, Bmax=400, fmin=3, fmax=10)
    save_cache(data, f"./cached_data.pkl")
    # load_cache(data, f"./cached_data.pkl")
    prepare_mpl_params()
    fig = plt.figure(figsize=(7, 7 / 2), dpi=150)
    p = Plot(fig, data, step=172)
    ani = p.anim("adl-mr.mp4")
