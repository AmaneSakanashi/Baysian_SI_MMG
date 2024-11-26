import matplotlib.pyplot as plt
import shipsim

def state_plot(data_list, line_list, log_dir, prefix):
    NAMES = ['x_position_mid [m]',
            'u_velo [m/s]',
            'y_position_mid [m]',
            'vm_velo [m/s]',
            'psi [rad]',
            'r_angvelo [rad/s]',
            ]
    
    TIME_NAME = "$t$ [s]"
    
    NAME_labels = ['$x_0$ [m]',
            '$u$ [m/s]',
            '$y_0$ [m]',
            '$v_{\mathrm{m}}$ [m/s]',
            '$\psi$ [rad]',
            '$r$ [rad/s]',
            ]
    
    DIM = len(NAMES)

    fig, axes = plt.subplots(
            2,  # number of rows
            3,  # number of columns
            sharex=True,
            figsize=(4.8*3, 4.8*2),
            tight_layout=False,
        )

    # label
    for i in range(DIM):
        ax = axes[i%2, i//2]  # change here
        ax.set_ylabel(NAME_labels[i], fontsize=22)
        if i%2 == 1:  # change here
            ax.set_xlabel(TIME_NAME, fontsize=22)

    # plot
    for j, data in enumerate(data_list):
        for i in range(DIM):
            ax = axes[i%2, i//2]  # change here
            cols = NAMES[i]
            df = data
            t = df.index
            #
            cols = [cols] if type(cols) == str else cols
            for i, col in enumerate(cols):
                ax.plot(t, df[col], color="black", ls=line_list[j])
                
    fig.savefig(f"{log_dir}/{prefix}_state_timeseries.png")
    plt.clf()
    plt.close()
    
def action_plot(data_list, line_list, log_dir, prefix):
    NAMES = ['delta_rudder [rad]',
            'n_prop [rps]']
   
    NAME_labels = ['$\delta$ [rad]',
            '$n_{\mathrm{p}}$ [rps]']
    
    TIME_NAME = "$t$ [s]"
    
    DIM = len(NAMES)

    fig, axes = plt.subplots(
            1,  # number of rows
            2,  # number of columns
            sharex=True,
            figsize=(4.8*3, 4.8),
            tight_layout=False,
        )

    # label
    for i in range(DIM):
        ax = axes[i]  # change here
        ax.set_ylabel(NAME_labels[i], fontsize=22)
        if i%1 == 0:  # change here
            ax.set_xlabel(TIME_NAME, fontsize=22)

    # plot
    for j, data in enumerate(data_list):
        for i in range(DIM):
            ax = axes[i]
            cols = NAMES[i]
            df = data
            t = df.index
            #
            cols = [cols] if type(cols) == str else cols
            for i, col in enumerate(cols):
                if i == 0:
                    ax.plot(t, df[col], color="black", ls=line_list[j], label=f"case{j+1}")
                else:
                    ax.plot(t, df[col], color="black", ls=line_list[j])
    ax.legend(bbox_to_anchor=(1, 1),fontsize=20)
    fig.savefig(f"{log_dir}/{prefix}_action_timeseries.png")
    plt.clf()
    plt.close()

def delta_plot(data_list, line_list, log_dir, prefix):
    NAMES = ['delta_rudder [rad]']
   
    NAME_labels = ['$\delta$ [rad]']
    
    TIME_NAME = "$t$ [s]"
    
    DIM = len(NAMES)

    fig, axes = plt.subplots(
            1,  # number of rows
            1,  # number of columns
            sharex=True,
            figsize=(3 * 1, 3 * 1),
            tight_layout=True,
        )

    # label
    for i in range(DIM):
        ax = axes  # change here
        ax.set_ylabel(NAME_labels[i], fontsize=16)
        if i%1 == 0:  # change here
            ax.set_xlabel(TIME_NAME, fontsize=16)

    # plot
    for i in range(DIM):
        ax = axes
        df = data_list
        t = df["t [s]"]
        ax.plot(t, df[NAMES], color="black", ls=line_list[0])
                
    fig.savefig(f"{log_dir}/{prefix}_delta_timeseries.png")
    plt.clf()
    plt.close()
    
    def traj_plot(prefix, csv_dir, x_emg_stop):
        sim.viewer.get_x0y0_plot_2(prefix=prefix, csv_dir=csv_dir, p_array=None, x_emg_stop=x_emg_stop, ext_type="pdf")
