import os
import shutil
import matplotlib.pyplot as plt
import numpy as np

# Add Graphviz to PATH manually
os.environ["PATH"] += os.pathsep + r"D:\Graphviz-14.1.1-win64\bin"

from graphviz import Digraph

# ==============================================================================
# 1. ASSET GENERATION (Mini-Plots)
# ==============================================================================
ASSET_DIR = "assets"

def generate_assets():
    if not os.path.exists(ASSET_DIR):
        os.makedirs(ASSET_DIR)
    
    # Style settings for scientific plots
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({'font.size': 14, 'axes.titlesize': 16, 'axes.labelsize': 14})
    
    def save_plot(name):
        plt.tight_layout()
        plt.savefig(os.path.join(ASSET_DIR, name), dpi=100, transparent=True, bbox_inches='tight', pad_inches=0.1)
        plt.close()

    # --- Plot 1: Parameter Fitting (Curve Fit) - Theme: Sage Green ---
    x = np.linspace(0, 10, 20)
    y = 2 * x + 1 + np.random.normal(0, 2, 20)
    plt.figure(figsize=(3, 2))
    plt.plot(x, 2*x+1, color='#2E7D32', linewidth=3, label='Fit') # Dark Green
    plt.plot(x, y, 'o', color='#43A047', markersize=8, alpha=0.6, label='Data') # Medium Green
    plt.axis('off')
    save_plot("icon_fitting.png")

    # --- Plot 2: Battery Dynamics (Pulse Response) - Theme: Steel Blue ---
    t = np.linspace(0, 10, 100)
    v = 4.2 - 0.5 * (1 - np.exp(-t/2))
    plt.figure(figsize=(3, 2))
    plt.plot(t, v, color='#1565C0', linewidth=3) # Dark Blue
    plt.fill_between(t, v, 3.5, color='#BBDEFB', alpha=0.5) # Light Blue
    plt.axis('off')
    save_plot("icon_battery.png")

    # --- Plot 3: Thermal/Physics (Heatmap/Rise) - Theme: Coral/Burnt Orange ---
    x = np.linspace(0, 10, 100)
    y = np.exp(x/5)
    plt.figure(figsize=(3, 2))
    plt.plot(x, y, color='#D84315', linewidth=3) # Deep Orange
    plt.fill_between(x, y, 0, color='#FFCCBC', alpha=0.5) # Light Orange
    plt.arrow(5, 2, 0, 3, head_width=0.5, head_length=0.5, fc='#BF360C', ec='#BF360C')
    plt.axis('off')
    save_plot("icon_physics.png")

    # --- Plot 4: Simulation (Time Series) - Theme: Deep Purple ---
    t = np.linspace(0, 20, 200)
    y1 = np.sin(t)
    y2 = np.cos(t) * 0.8
    plt.figure(figsize=(3, 2))
    plt.plot(t, y1, color='#6A1B9A', linewidth=2) # Purple
    plt.plot(t, y2, color='#AB47BC', linewidth=2, linestyle='--') # Lighter Purple
    plt.axis('off')
    save_plot("icon_simulation.png")

    # --- Plot 5: Pareto Frontier (Trade-off) - Theme: Teal ---
    x = np.linspace(0.1, 1, 20)
    y = 1/x
    plt.figure(figsize=(3, 2))
    plt.plot(x, y, color='#00695C', linewidth=3) # Teal
    plt.plot(0.4, 2.5, '*', color='#FDD835', markersize=18, markeredgecolor='#F57F17') # Gold Star
    plt.fill_between(x, y, 10, color='#B2DFDB', alpha=0.3)
    plt.axis('off')
    save_plot("icon_pareto.png")

# ==============================================================================
# 2. FLOWCHART GENERATION
# ==============================================================================
def create_scientific_diagram():
    dot = Digraph('SmartphoneBatteryFlow', comment='Scientific Flowchart')
    dot.attr(compound='true', rankdir='TB', splines='polyline', nodesep='0.28', ranksep='0.26')
    dot.attr('graph', fontname='Helvetica', fontsize='22', labelloc='t',
             label='Smartphone Battery Modeling: End-to-End Scientific Workflow')
    dot.attr('graph', bgcolor='#F7F9FC')
    dot.attr('node', shape='box', style='filled,rounded,bold', fontname='Helvetica',
             fontsize='16', fontweight='bold', penwidth='1.8', margin='0.02,0.02')
    dot.attr('edge', fontname='Helvetica', fontsize='10', color='#455A64', penwidth='1.1', arrowhead='vee')
    dot.attr('graph', pad='0.01', margin='0.01', pack='true', packmode='graph')

    palette = {
        'panel_border': '#2E7D32',
        'panel_bg': '#FFFFFF',
        'panel_bg_p1': '#E3F2FD',
        'panel_bg_p2': '#E8F5E9',
        'panel_bg_p3': '#FFEBEE',
        'panel_bg_p4': '#F3E5F5',
        'panel_bg_p5': '#E0F2F1',
        'gold': {'fill': '#E8C27A', 'stroke': '#B07F17'},
        'green': {'fill': '#CDEDD5', 'stroke': '#2E7D32'},
        'pink': {'fill': '#F8C8C8', 'stroke': '#C62828'},
        'purple': {'fill': '#E4C7F2', 'stroke': '#6A1B9A'},
        'blue': {'fill': '#C7DFF6', 'stroke': '#1565C0'},
        'gray': {'fill': '#F2F2F2', 'stroke': '#9E9E9E'},
        'dark_purple': {'fill': '#6A1B9A', 'stroke': '#4A148C', 'font': 'white'},
        'orange': {'fill': '#FFCCBC', 'stroke': '#E64A19'}
    }

    def styled_node(name, label, style_key, shape='box', fontcolor='black'):
        style = palette[style_key]
        dot.node(
            name,
            label=label,
            shape=shape,
            style='filled,rounded',
            fillcolor=style['fill'],
            color=style['stroke'],
            fontcolor=style.get('font', fontcolor)
        )

    def add_footer_bar(cluster, name, label, width_px=300, fill=None):
        # Footer bars removed per request for large background blocks.
        return

    def add_mini_plot(cluster, name, img_name, caption):
        img_path = os.path.join(os.getcwd(), ASSET_DIR, img_name).replace('\\', '/')
        html = f'''<
        <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="2">
          <TR><TD><IMG SRC="{img_path}" SCALE="TRUE"/></TD></TR>
          <TR><TD><B>{caption}</B></TD></TR>
        </TABLE>>'''
        cluster.node(name, label=html, shape='plain', style='')

    # ==============================================================================
    # STAGES: Large background blocks that group panels
    # ==============================================================================
    with dot.subgraph(name='cluster_STAGE1') as stage:
        stage.attr(label='Stage 1: Initialization', style='filled,rounded',
                   color='#78909C', fillcolor='#DDEAF6', fontcolor='#455A64')

        with stage.subgraph(name='cluster_S1_P2') as c:
            c.attr(label='Data-Driven Parameter Estimation', style='filled,rounded',
                   color=palette['panel_border'], fillcolor=palette['panel_bg_p2'], fontcolor=palette['panel_border'])
            with c.subgraph(name='cluster_P2_Left') as left:
                left.attr(label='Electrical Parameters', style='filled',
                          fillcolor='#FDF5E6', color=palette['pink']['stroke'], fontcolor=palette['pink']['stroke'])
                styled_node('P2_L1', 'OCV Data', 'pink')
                styled_node('P2_L2', 'Curve Fitting', 'pink')
                styled_node('P2_L3', 'Combined Model Coeffs', 'pink')
                dot.edge('P2_L1', 'P2_L2')
                dot.edge('P2_L2', 'P2_L3')

            with c.subgraph(name='cluster_P2_Right') as right:
                right.attr(label='Thermal Parameters', style='filled',
                           fillcolor='#FFF8E1', color=palette['pink']['stroke'], fontcolor=palette['pink']['stroke'])
                styled_node('P2_R1', 'Temp Rise Data', 'pink')
                styled_node('P2_R2', 'Arrhenius Plot', 'pink')
                styled_node('P2_R3', 'Ea_R & Ea_cap', 'pink')
                dot.edge('P2_R1', 'P2_R2')
                dot.edge('P2_R2', 'P2_R3')

            styled_node('P2_Merge', 'Parameter\nValidation', 'gray', shape='diamond')
            styled_node('P2_Done', 'config.yaml', 'gray')
            dot.edge('P2_L3', 'P2_Merge')
            dot.edge('P2_R3', 'P2_Merge')
            dot.edge('P2_Merge', 'P2_Done', xlabel='RMSE < 1%')
            add_mini_plot(c, 'P2_Mini', 'icon_fitting.png', 'OCV Fit and Arrhenius')
            dot.edge('P2_Merge', 'P2_Mini', style='dashed')
            # Iterative identification loop: validation drives re-fitting (feedback arrow).
            dot.edge(
                'P2_Merge',
                'P2_L2',
                style='dashed',
                color='#8E24AA',
                penwidth='1.4',
                constraint='false',
                xlabel='iterate',
            )
            dot.edge(
                'P2_Merge',
                'P2_R2',
                style='dashed',
                color='#8E24AA',
                penwidth='1.4',
                constraint='false',
            )

        with stage.subgraph(name='cluster_S1_P1') as c:
            c.attr(label='Core Digital Twin Model', style='filled,rounded',
                   color=palette['panel_border'], fillcolor=palette['panel_bg_p1'], fontcolor=palette['panel_border'])
            styled_node('P1_In', 'Device Parameters\n(AOSP / Datasheets)', 'gold')
            styled_node('P1_Model', 'Coupled ODE System\n(dSOC/dt, dT/dt)', 'green', shape='ellipse')
            styled_node('P1_Out', 'State Vector\n(SOC, Temp, Voltage)', 'gold')
            styled_node('P1_Pred', 'Time-to-Empty\n(TTE)', 'gold')
            styled_node('P1_Final', 'Dynamic Strategy', 'orange')
            dot.edge('P1_In', 'P1_Model')
            dot.edge('P1_Model', 'P1_Model', xlabel='Feedback')
            dot.edge('P1_Model', 'P1_Out')
            dot.edge('P1_Out', 'P1_Pred')
            dot.edge('P1_Pred', 'P1_Final')
            add_mini_plot(c, 'P1_Mini', 'icon_battery.png', 'SOC/T/V Timeline')
            dot.edge('P1_Out', 'P1_Mini', style='dashed')

    with dot.subgraph(name='cluster_STAGE2') as stage:
        stage.attr(label='Stage 2: Coupling and Simulation', style='filled,rounded',
                   color='#78909C', fillcolor='#F1E6F8', fontcolor='#455A64')

        with stage.subgraph(name='cluster_S2_P3') as c:
            c.attr(label='Multi-Physics Coupling Effects', style='filled,rounded',
                   color=palette['panel_border'], fillcolor=palette['panel_bg_p3'], fontcolor=palette['panel_border'])
            styled_node('P3_Feat1', 'Temperature Rise', 'purple')
            styled_node('P3_Feat2', 'Internal Resistance Drop', 'dark_purple')
            styled_node('P3_Feat3', 'Non-linear Aging', 'purple')
            styled_node('P3_Res', 'Voltage Sag Risk', 'dark_purple')
            dot.edge('P3_Feat1', 'P3_Feat2')
            dot.edge('P3_Feat2', 'P3_Feat1', xlabel='Feedback')
            dot.edge('P3_Feat1', 'P3_Feat3', xlabel='Accelerates')
            dot.edge('P3_Feat2', 'P3_Res')
            add_mini_plot(c, 'P3_Mini', 'icon_physics.png', 'R(T,SOC) Surface')
            dot.edge('P3_Res', 'P3_Mini', style='dashed')

        with stage.subgraph(name='cluster_S2_P4') as c:
            c.attr(label='Simulation & Optimization Framework', style='filled,rounded',
                   color=palette['panel_border'], fillcolor=palette['panel_bg_p4'], fontcolor=palette['panel_border'])

            with c.subgraph(name='cluster_P4_Input') as sub:
                sub.attr(label='Scenario Input', style='filled',
                         fillcolor='#EAF2FB', color=palette['blue']['stroke'], fontcolor=palette['blue']['stroke'])
                styled_node('P4_In1', 'Video / Gaming', 'blue')
                styled_node('P4_In2', 'Grid Search Params', 'blue')

            with c.subgraph(name='cluster_P4_Core') as sub:
                sub.attr(label='Solver Engine', style='filled',
                         fillcolor='#F3E5F5', color=palette['green']['stroke'], fontcolor=palette['green']['stroke'])
                styled_node('P4_Date', 'Scenario Duration', 'orange')
                styled_node('P4_Solver', 'DOP853 Integrator', 'green')
                dot.edge('P4_Date', 'P4_Solver')

            with c.subgraph(name='cluster_P4_Out') as sub:
                sub.attr(label='Validation', style='filled',
                         fillcolor='#F5F5F5', color=palette['purple']['stroke'], fontcolor=palette['purple']['stroke'])
                styled_node('P4_Pred', 'Model Voltage', 'purple')
                styled_node('P4_Real', 'Real Measurements', 'gray')
                styled_node('P4_Diff', 'Residual Analysis', 'blue')
                dot.edge('P4_Pred', 'P4_Diff')
                dot.edge('P4_Real', 'P4_Diff')

            dot.edge('P4_In1', 'P4_Solver')
            dot.edge('P4_In2', 'P4_Solver')
            dot.edge('P4_Solver', 'P4_Pred')
            add_mini_plot(c, 'P4_Mini', 'icon_simulation.png', 'Validation Residuals')
            dot.edge('P4_Diff', 'P4_Mini', style='dashed')

    with dot.subgraph(name='cluster_STAGE3') as stage:
        stage.attr(label='Stage 3: Decision', style='filled,rounded',
                   color='#78909C', fillcolor='#E0F2F1', fontcolor='#455A64')

        with stage.subgraph(name='cluster_S3_P5') as c:
            c.attr(label='Decision Support & Trade-off', style='filled,rounded',
                   color=palette['panel_border'], fillcolor=palette['panel_bg_p5'], fontcolor=palette['panel_border'])
            styled_node('P5_In', 'Pareto Frontier', 'green')
            styled_node('P5_Method', 'K-Means / Sorting', 'gray')
            styled_node('P5_Class', 'Strategy Classification', 'green')
            styled_node('P5_Final', 'Optimal Policy', 'green')
            dot.edge('P5_In', 'P5_Method')
            dot.edge('P5_Method', 'P5_Class')
            dot.edge('P5_Class', 'P5_Final')
            add_mini_plot(c, 'P5_Mini', 'icon_pareto.png', 'Pareto and Policy Map')
            dot.edge('P5_In', 'P5_Mini', style='dashed')

    # ==============================================================================
    # CONNECTIONS BETWEEN PANELS
    # ==============================================================================
    dot.edge('P2_Done', 'P1_In', penwidth='1.6')
    dot.edge('P3_Res', 'P1_Model', style='dashed', color='#BF360C',
             minlen='2', constraint='false')
    dot.edge('P1_Final', 'P4_In1', penwidth='1.6', minlen='2')
    dot.edge('P4_Diff', 'P5_In', penwidth='1.6', minlen='2')

    # ==============================================================================
    # SNAKE LAYOUT (3 rows, narrower width)
    # Row1: P2 -> P1
    # Row2: P3 -> P4 (right-to-left visual snake)
    # Row3: P5
    # ==============================================================================
    with dot.subgraph() as row1:
        row1.attr(rank='same')
        row1.node('P2_Done')
        row1.node('P1_In')
        row1.edge('P2_Done', 'P1_In', style='invis')

    with dot.subgraph() as row2:
        row2.attr(rank='same')
        row2.node('P3_Feat1')
        row2.node('P4_Solver')
        row2.edge('P4_Solver', 'P3_Feat1', style='invis')

    with dot.subgraph() as row3:
        row3.attr(rank='same')
        row3.node('P5_In')

    # Snake bridges (invisible) to enforce row order
    dot.edge('P1_In', 'P3_Feat1', style='invis')
    dot.edge('P4_Solver', 'P5_In', style='invis')

    # Spacer ranks to create clear gaps between stages
    dot.node('SP12', '', shape='box', width='0.1', height='0.15', style='invis')
    dot.node('SP23', '', shape='box', width='0.1', height='0.15', style='invis')
    with dot.subgraph() as spacer1:
        spacer1.attr(rank='same')
        spacer1.node('SP12')
    with dot.subgraph() as spacer2:
        spacer2.attr(rank='same')
        spacer2.node('SP23')
    dot.edge('P1_In', 'SP12', style='invis')
    dot.edge('SP12', 'P3_Feat1', style='invis')
    dot.edge('P4_Solver', 'SP23', style='invis')
    dot.edge('SP23', 'P5_In', style='invis')

    # Render directly into the paper folder (referenced by T3P.tex as workflow.png).
    output_path = r'D:\MCM-Article\workflow'
    dot.render(output_path, format='png', cleanup=True)
    print(f"Diagram generated: {output_path}.png")

if __name__ == '__main__':
    generate_assets()
    create_scientific_diagram()
