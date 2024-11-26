import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from .ship.utils.hull_shape import ship_coo


class Viewer:
    def __init__(self, ship, world, logger, log_dir="./"):
        self.ship = ship
        self.world = world
        self.logger = logger
        self.log_dir = log_dir

    def get_x0y0_plot(self, prefix, x_emg_stop=None, ext_type="png"):
        df = self.logger.get_df()
        x_col = self.ship.STATE_NAME[0:6]
        #
        fig, ax = plt.subplots(1, 1, figsize=(5, 5), tight_layout=True)
        ax.set_xlabel("$y_{0} \ \mathrm{[m]}$")
        ax.set_ylabel("$x_{0} \ \mathrm{[m]}$")
        ax.axis("equal")
        ax.plot(df[x_col[2]], df[x_col[0]], color="black", ls="dashed")
        #
        steps = 100
        kwrds = {"fill": False, "lw": 0.3, "ec": "black"}
        for i in range(len(df) - 1, -1, -steps):
            ship_state = df[x_col].iloc[i].to_numpy()
            ship_poly = self.get_ship_poly(ship_state)
            polygon = plt.Polygon(ship_poly, **kwrds)
            ax.add_patch(polygon)
        #
        kwrds = {"fill": True, "lw": 0.3, "fc": "#C8CCDE", "ec": "#0A0F60"}
        for world_poly in self.world.obstacle_polygons:
            world_poly_ = world_poly[:, [1, 0]]
            polygon = plt.Polygon(world_poly_, **kwrds)
            wolrd = ax.add_patch(polygon)
        #[furuya]x_emg_stopの軌跡をプロット
        color_list = ["red", "blue", "green", ""]
        if x_emg_stop is not None:
            for i in range(len(x_emg_stop)):
                ax.plot(x_emg_stop[i, :, 2], x_emg_stop[i, :, 0], alpha=0.3, ls="--", label="n = {i}")
            # ax.plot(x_emg_stop[0, :, 2], x_emg_stop[0, :, 0], color="red", label="0s")
            # ax.plot(x_emg_stop[4, :, 2], x_emg_stop[4, :, 0], color="blue", label="50s")
            # ax.plot(x_emg_stop[9, :, 2], x_emg_stop[9, :, 0], color="green", label="100s")
            ax.legend()
            #
        fig.savefig(f"{self.log_dir}/{prefix}_traj.{ext_type}")
        plt.clf()
        plt.close()
        
    def get_x0y0_plot_2(self, prefix, csv_dir, p_array=None,  x_emg_stop=None, ext_type="png"):
        # df = self.logger.get_df()
        x_col = self.ship.STATE_NAME[0:6]
        df = pd.read_csv(csv_dir)
        #
        fig, ax = plt.subplots(1, 1, figsize=(4.8, 4.8*1.5), tight_layout=True)
        ax.set_xlabel("$y_{0} \ \mathrm{[m]}$", fontsize=22)
        ax.set_ylabel("$x_{0} \ \mathrm{[m]}$", fontsize=22)
        ax.axis("equal")
        # ax.plot(df['y_position_mid [m]'], df['x_position_mid [m]'], lw=1.5, color="black", ls="dashed", label="planning traj")
        ax.set_xlim(left=-10, right=10)
        ax.set_ylim(bottom=-20, top=70)
        #
        steps = 100
        kwrds = {"fill": False, "lw": 0.3, "ec": "black"}
        # for i in range(len(df) - 1, -1, -steps):
        #     ship_state = df[x_col].iloc[i].to_numpy()
        #     ship_poly = self.get_ship_poly(ship_state)
        #     polygon = plt.Polygon(ship_poly, **kwrds)
        #     ax.add_patch(polygon)
        #
        kwrds = {"fill": True, "lw": 0.3, "fc": "#C8CCDE", "ec": "#0A0F60"}
        for world_poly in self.world.obstacle_polygons:
            world_poly_ = world_poly[:, [1, 0]]
            polygon = plt.Polygon(world_poly_, **kwrds)
            wolrd = ax.add_patch(polygon)
        #[furuya]x_emg_stopの軌跡をプロット
        color_list = ["red", "blue", "green", ""]
        kwrds = {"fill": False, "lw": 0.1, "alpha": 1}
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        # if x_emg_stop is not None:
        #     ax.plot(x_emg_stop[-1, 0, 2], x_emg_stop[-1, 0, 0], color="red", alpha=0.8, ls=None, lw=0.7, marker='.', ms=5, label="Potential traj")
        #     for i in range(len(x_emg_stop)):
        #         ax.plot(x_emg_stop[i, 0, 2], x_emg_stop[i, 0, 0], color="red", alpha=0.8, ls=None, lw=0.7, marker='.', ms=5)
        #         ax.plot(x_emg_stop[i, :, 2], x_emg_stop[i, :, 0], color="red", alpha=0.8, ls=None, lw=0.7, marker=None, ms=None)
        #         if p_array is not None:
        #             x = p_array[i, :, 0]
        #             y = p_array[i, :, 1]
        #             ax.plot(y, x, color="black", lw=0.6,)
                    
                    
                
        #         # for j in range(len(x_emg_stop) - 1, -1, -10):
        #         #     ship_state = x_emg_stop[i, j]
        #         #     ship_poly = self.get_ship_poly(ship_state)
        #         #     polygon = plt.Polygon(ship_poly, **kwrds)
        #         #     ax.add_patch(polygon)
        #     # ax.plot(x_emg_stop[0, :, 2], x_emg_stop[0, :, 0], color="red", label="0s")
        #     # ax.plot(x_emg_stop[4, :, 2], x_emg_stop[4, :, 0], color="blue", label="50s")
        #     # ax.plot(x_emg_stop[9, :, 2], x_emg_stop[9, :, 0], color="green", label="100s")
        #     ax.legend(bbox_to_anchor=(1, 1),fontsize=20)
            #
        # ax.plot(x_emg_stop[:, -1, 2], x_emg_stop[:, -1, 0])
        fig.savefig(f"{self.log_dir}/{prefix}_traj.{ext_type}")
        plt.clf()
        plt.close()

    def get_timeseries_plot(self, TIME_NAME, NAMES, prefix, ext_type="png"):
        DIM = len(NAMES)
        fig, axes = plt.subplots(
            DIM,
            1,
            sharex=True,
            figsize=(20, 2 * DIM),
            tight_layout=True,
        )
        # label
        for i in range(DIM):
            axes[i].set_ylabel(NAMES[i])
        axes[-1].set_xlabel(TIME_NAME)
        # plot
        for i in range(DIM):
            self.set2ax_data(axes[i], NAMES[i])
        #
        fig.savefig(f"{self.log_dir}/{prefix}_timeseries.{ext_type}")
        plt.clf()
        plt.close()

    def set2ax_data(self, ax, cols):
        df = self.logger.get_df()
        t = df.index
        #
        cols = [cols] if type(cols) == str else cols
        colors = self.get_colors(N=len(cols))
        linestyles = self.get_linestyles(N=len(cols))
        for i, col in enumerate(cols):
            ax.plot(t, df[col], color=colors[i], linestyle=linestyles[i])

    def get_ship_poly(self, x):
        eta = x[[0, 2, 4]]
        L = self.ship.L
        B = self.ship.B
        ship_poly = ship_coo(eta, L, B)[:, [1, 0]]
        return ship_poly

    @staticmethod
    def get_colors(N=1):
        BASE_COLORS = ["black", "red", "blue"]
        colors = []
        for i in range(N):
            ii = i % len(BASE_COLORS)
            colors.append(BASE_COLORS[ii])
        return colors

    @staticmethod
    def get_linestyles(N=1):
        BASE_LINESTYLES = ["solid", "dashed", "dashdot", "dotted"]
        linestyles = []
        for i in range(N):
            ii = i % len(BASE_LINESTYLES)
            linestyles.append(BASE_LINESTYLES[ii])
        return linestyles
