import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
from datetime import datetime
import seaborn as sns
import numpy as np
from tkinter import messagebox
from matplotlib.colors import LinearSegmentedColormap
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import calendar
from PIL import Image, ImageTk

# ========== DATA PREPARATION ==========
# df = pd.read_excel('Unemployment in India.xlsx')
df = pd.read_excel(r'C:\Users\hp\Desktop\CipherByte Technologies Internship Projects\Unemployeement Project\Unemployment in India.xlsx')

# Data cleaning and enhancements
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month_name()
df['Quarter'] = df['Date'].dt.quarter
df['Region'] = df['Region'].fillna('Unknown').astype(str)

# Add calculated metrics
df['Unemployment Change (%)'] = df.groupby(['Region', 'Area'])['Estimated Unemployment Rate (%)'].pct_change() * 100
df['Employment Change'] = df.groupby(['Region', 'Area'])['Estimated Employed'].diff()

# ========== COLOR SCHEME ==========
# Modern, professional color scheme (peaceful tones)
COLOR_SCHEME = {
    'primary': '#3498db',       # Sky Blue
    'secondary': '#2ecc71',     # Emerald Green
    'accent': '#e74c3c',        # Soft Red for highlights
    'background': '#f5f7fa',    # Light gray background
    'text': '#2c3e50',          # Dark text for readability
    'highlight': '#f39c12',     # Orange for highlights
    'success': '#27ae60',       # Darker green for positive metrics
    'warning': '#e67e22',       # Orange for warnings
    'dark': '#34495e',          # Dark blue-gray
    'light': '#ecf0f1',         # Very light gray
    'grid': '#bdc3c7',          # Grid lines color
    'card_bg': '#ffffff',       # Card background
    'card_text': '#34495e'      # Card text color
}

# Custom colormap for heatmaps
cmap = LinearSegmentedColormap.from_list('custom', [COLOR_SCHEME['primary'], COLOR_SCHEME['secondary']])

# ========== STYLING CONFIGURATION ==========
plt.style.use('seaborn-v0_8')
sns.set_style("whitegrid")
sns.set_palette([COLOR_SCHEME['primary'], COLOR_SCHEME['secondary'], COLOR_SCHEME['accent']])

# Configure matplotlib defaults
plt.rcParams['axes.facecolor'] = COLOR_SCHEME['background']
plt.rcParams['figure.facecolor'] = COLOR_SCHEME['background']
plt.rcParams['grid.color'] = COLOR_SCHEME['grid']
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['text.color'] = COLOR_SCHEME['text']
plt.rcParams['axes.labelcolor'] = COLOR_SCHEME['text']
plt.rcParams['xtick.color'] = COLOR_SCHEME['text']
plt.rcParams['ytick.color'] = COLOR_SCHEME['text']

# ========== MAIN APPLICATION ==========
class UnemploymentDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("India Unemployment Analysis Dashboard")
        self.root.geometry("1400x900")
        self.root.configure(bg=COLOR_SCHEME['background'])
        
        # Add modern header
        self.create_header()
        
        # Create main container
        self.main_container = ttk.Frame(root)
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Create control panel
        self.create_control_panel()
        
        # Create visualization area
        self.create_visualization_area()
        
        # Initial plot
        self.update_plots()

    def create_header(self):
        header = tk.Frame(self.root, bg=COLOR_SCHEME['dark'], height=100)
        header.pack(fill=tk.X)
        
        # Left side - Title
        title_frame = tk.Frame(header, bg=COLOR_SCHEME['dark'])
        title_frame.pack(side=tk.LEFT, padx=20, pady=10)
        
        main_title = tk.Label(title_frame, 
                            text="INDIA UNEMPLOYMENT ANALYTICS", 
                            font=("Helvetica", 18, "bold"), 
                            bg=COLOR_SCHEME['dark'], 
                            fg="white")
        main_title.pack(pady=(5,0))
        
        subtitle = tk.Label(title_frame, 
                          text="Comprehensive Analysis of Employment Trends", 
                          font=("Helvetica", 10), 
                          bg=COLOR_SCHEME['dark'], 
                          fg=COLOR_SCHEME['light'])
        subtitle.pack()
        
        # Right side - Date/time and quick stats
        stats_frame = tk.Frame(header, bg=COLOR_SCHEME['dark'])
        stats_frame.pack(side=tk.RIGHT, padx=20, pady=10)
        
        # Current date/time
        self.time_label = tk.Label(stats_frame, 
                                 text="", 
                                 font=("Helvetica", 10), 
                                 bg=COLOR_SCHEME['dark'], 
                                 fg="white")
        self.time_label.pack(anchor='e')
        self.update_time()
        
        # Quick stats
        latest_data = df[df['Date'] == df['Date'].max()]
        avg_unemp = latest_data['Estimated Unemployment Rate (%)'].mean()
        
        stats_text = f"Latest Data: {df['Date'].max().strftime('%b %Y')} | Avg Unemployment: {avg_unemp:.1f}%"
        self.stats_label = tk.Label(stats_frame, 
                                  text=stats_text, 
                                  font=("Helvetica", 9), 
                                  bg=COLOR_SCHEME['dark'], 
                                  fg=COLOR_SCHEME['light'])
        self.stats_label.pack(anchor='e')

    def update_time(self):
        now = datetime.now().strftime("%d %b %Y | %I:%M %p")
        self.time_label.config(text=now)
        self.root.after(60000, self.update_time)  # Update every minute

    def create_control_panel(self):
        control_frame = ttk.LabelFrame(self.main_container, 
                                     text="Dashboard Controls", 
                                     padding=(15, 10),
                                     style='Custom.TLabelframe')
        control_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Row 1: Date range controls
        date_frame = ttk.Frame(control_frame)
        date_frame.grid(row=0, column=0, padx=5, pady=5, sticky='w')
        
        ttk.Label(date_frame, text="Date Range:", font=('Helvetica', 10, 'bold')).pack(side=tk.LEFT)
        
        ttk.Label(date_frame, text="From:").pack(side=tk.LEFT, padx=(10,0))
        self.start_date = ttk.Entry(date_frame, width=12)
        self.start_date.pack(side=tk.LEFT)
        self.start_date.insert(0, df['Date'].min().strftime('%Y-%m-%d'))
        
        ttk.Label(date_frame, text="To:").pack(side=tk.LEFT, padx=(10,0))
        self.end_date = ttk.Entry(date_frame, width=12)
        self.end_date.pack(side=tk.LEFT)
        self.end_date.insert(0, df['Date'].max().strftime('%Y-%m-%d'))
        
        # Row 1: Region selector
        region_frame = ttk.Frame(control_frame)
        region_frame.grid(row=0, column=1, padx=5, pady=5, sticky='w')
        
        ttk.Label(region_frame, text="State/UT:", font=('Helvetica', 10, 'bold')).pack(side=tk.LEFT)
        self.region_var = tk.StringVar()
        self.region_selector = ttk.Combobox(region_frame, 
                                          textvariable=self.region_var, 
                                          values=['All'] + sorted(df['Region'].unique()), 
                                          state='readonly',
                                          width=25)
        self.region_selector.pack(side=tk.LEFT, padx=(10,0))
        self.region_selector.set('All')
        
        # Row 2: Area type and metric
        area_frame = ttk.Frame(control_frame)
        area_frame.grid(row=1, column=0, padx=5, pady=5, sticky='w')
        
        ttk.Label(area_frame, text="Area Type:", font=('Helvetica', 10, 'bold')).pack(side=tk.LEFT)
        self.area_var = tk.StringVar(value='All')
        self.area_selector = ttk.Combobox(area_frame, 
                                        textvariable=self.area_var, 
                                        values=['All', 'Rural', 'Urban'], 
                                        state='readonly',
                                        width=12)
        self.area_selector.pack(side=tk.LEFT, padx=(10,0))
        
        # Metric selector
        metric_frame = ttk.Frame(control_frame)
        metric_frame.grid(row=1, column=1, padx=5, pady=5, sticky='w')
        
        ttk.Label(metric_frame, text="Metric:", font=('Helvetica', 10, 'bold')).pack(side=tk.LEFT)
        self.metric_var = tk.StringVar(value='Estimated Unemployment Rate (%)')
        self.metric_selector = ttk.Combobox(metric_frame, 
                                          textvariable=self.metric_var, 
                                          values=['Estimated Unemployment Rate (%)', 
                                                  'Estimated Employed', 
                                                  'Estimated Labour Participation Rate (%)',
                                                  'Unemployment Change (%)',
                                                  'Employment Change'], 
                                          state='readonly',
                                          width=30)
        self.metric_selector.pack(side=tk.LEFT, padx=(10,0))
        
        # Row 2: Buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=1, column=2, padx=5, pady=5, sticky='e')
        
        # Update button with modern style
        self.update_btn = ttk.Button(button_frame, 
                                   text="UPDATE VIEW", 
                                   command=self.update_plots,
                                   style='Accent.TButton')
        self.update_btn.pack(side=tk.LEFT, padx=5)
        
        # Export button
        self.export_btn = ttk.Button(button_frame, 
                                    text="EXPORT DATA", 
                                    command=self.export_data,
                                    style='Accent.TButton')
        self.export_btn.pack(side=tk.LEFT, padx=5)
        
        # Configure grid weights
        control_frame.grid_columnconfigure(0, weight=1)
        control_frame.grid_columnconfigure(1, weight=1)
        control_frame.grid_columnconfigure(2, weight=1)
    
    def create_visualization_area(self):
        # Create notebook for tabs with modern style
        style = ttk.Style()
        style.configure('TNotebook', background=COLOR_SCHEME['background'])
        style.configure('TNotebook.Tab', 
                       background=COLOR_SCHEME['light'], 
                       foreground=COLOR_SCHEME['text'],
                       padding=[10, 5],
                       font=('Helvetica', 10, 'bold'))
        style.map('TNotebook.Tab', 
                 background=[('selected', COLOR_SCHEME['primary'])],
                 foreground=[('selected', 'white')])
        
        self.tab_control = ttk.Notebook(self.main_container)
        self.tab_control.pack(expand=True, fill=tk.BOTH)
        
        # Tab 1: Trends Analysis
        self.trends_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.trends_tab, text='TRENDS ANALYSIS')
        
        # Tab 2: Regional Comparison
        self.regional_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.regional_tab, text='REGIONAL COMPARISON')
        
        # Tab 3: COVID Impact
        self.covid_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.covid_tab, text='COVID-19 IMPACT')
        
        # Tab 4: Heatmap Analysis (NEW)
        self.heatmap_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.heatmap_tab, text='HEATMAP ANALYSIS')
        
        # Tab 5: State Profile (NEW)
        self.state_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.state_tab, text='STATE PROFILE')
        
        # Create figures for each tab
        self.create_trends_tab()
        self.create_regional_tab()
        self.create_covid_tab()
        self.create_heatmap_tab()
        self.create_state_tab()
    
    def create_trends_tab(self):
        # Main trend plot with enhanced styling
        self.fig1 = plt.Figure(figsize=(12, 5), dpi=100, facecolor=COLOR_SCHEME['background'])
        self.ax1 = self.fig1.add_subplot(111)
        self.ax1.set_facecolor(COLOR_SCHEME['background'])
        
        # Add canvas with padding
        self.canvas1 = FigureCanvasTkAgg(self.fig1, master=self.trends_tab)
        self.canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Summary statistics frame with modern cards
        stats_frame = ttk.Frame(self.trends_tab)
        stats_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        # Card 1: Current Value
        self.current_card = self.create_card(stats_frame, "CURRENT VALUE", "N/A", COLOR_SCHEME['primary'])
        self.current_card.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # Card 2: Change from Previous
        self.change_card = self.create_card(stats_frame, "CHANGE FROM PREVIOUS", "N/A", COLOR_SCHEME['secondary'])
        self.change_card.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # Card 3: 6-Month Avg
        self.avg_card = self.create_card(stats_frame, "6-MONTH AVERAGE", "N/A", COLOR_SCHEME['dark'])
        self.avg_card.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # Card 4: Min/Max
        self.minmax_card = self.create_card(stats_frame, "MIN/MAX", "N/A", COLOR_SCHEME['accent'])
        self.minmax_card.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
    
    def create_regional_tab(self):
        # Regional comparison plot with enhanced styling
        self.fig2 = plt.Figure(figsize=(12, 5), dpi=100, facecolor=COLOR_SCHEME['background'])
        self.ax2 = self.fig2.add_subplot(111)
        self.ax2.set_facecolor(COLOR_SCHEME['background'])
        
        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=self.regional_tab)
        self.canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Top/bottom performers frame with modern cards
        perf_frame = ttk.Frame(self.regional_tab)
        perf_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        # Top performers card
        self.top_card = self.create_card(perf_frame, "TOP PERFORMERS", "Loading...", COLOR_SCHEME['success'])
        self.top_card.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # Bottom performers card
        self.bottom_card = self.create_card(perf_frame, "BOTTOM PERFORMERS", "Loading...", COLOR_SCHEME['warning'])
        self.bottom_card.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # Regional insights card (NEW)
        self.insights_card = self.create_card(perf_frame, "REGIONAL INSIGHTS", "Loading...", COLOR_SCHEME['dark'])
        self.insights_card.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
    
    def create_covid_tab(self):
        # COVID impact analysis with enhanced styling
        self.fig3 = plt.Figure(figsize=(12, 5), dpi=100, facecolor=COLOR_SCHEME['background'])
        self.ax3 = self.fig3.add_subplot(111)
        self.ax3.set_facecolor(COLOR_SCHEME['background'])
        
        self.canvas3 = FigureCanvasTkAgg(self.fig3, master=self.covid_tab)
        self.canvas3.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Impact metrics frame with modern cards
        impact_frame = ttk.Frame(self.covid_tab)
        impact_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        # Max impact card
        self.max_card = self.create_card(impact_frame, "PEAK UNEMPLOYMENT", "Loading...", COLOR_SCHEME['accent'])
        self.max_card.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # Recovery card
        self.recovery_card = self.create_card(impact_frame, "RECOVERY RATE", "Loading...", COLOR_SCHEME['success'])
        self.recovery_card.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # Most affected card
        self.affected_card = self.create_card(impact_frame, "MOST AFFECTED", "Loading...", COLOR_SCHEME['warning'])
        self.affected_card.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # Economic impact card (NEW)
        self.economic_card = self.create_card(impact_frame, "EMPLOYMENT LOSS", "Loading...", COLOR_SCHEME['dark'])
        self.economic_card.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
    
    def create_heatmap_tab(self):
        """NEW: Create heatmap analysis tab"""
        self.fig4 = plt.Figure(figsize=(12, 5), dpi=100, facecolor=COLOR_SCHEME['background'])
        self.ax4 = self.fig4.add_subplot(111)
        self.ax4.set_facecolor(COLOR_SCHEME['background'])
        
        self.canvas4 = FigureCanvasTkAgg(self.fig4, master=self.heatmap_tab)
        self.canvas4.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Heatmap controls frame
        heatmap_controls = ttk.Frame(self.heatmap_tab)
        heatmap_controls.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        ttk.Label(heatmap_controls, text="Heatmap Type:", font=('Helvetica', 10, 'bold')).pack(side=tk.LEFT)
        
        self.heatmap_type = tk.StringVar(value='Monthly')
        ttk.Radiobutton(heatmap_controls, text="Monthly", variable=self.heatmap_type, value='Monthly',
                       command=self.update_heatmap).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(heatmap_controls, text="Quarterly", variable=self.heatmap_type, value='Quarterly',
                       command=self.update_heatmap).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(heatmap_controls, text="Yearly", variable=self.heatmap_type, value='Yearly',
                       command=self.update_heatmap).pack(side=tk.LEFT, padx=5)
        
        # Add colorbar legend
        self.heatmap_legend = ttk.Label(heatmap_controls, text="Color Scale: Blue (Low) → Green (High)")
        self.heatmap_legend.pack(side=tk.RIGHT, padx=5)
    
    
    def create_state_tab(self):
        """NEW: Create state profile tab"""
        self.state_frame = ttk.Frame(self.state_tab)
        self.state_frame.pack(fill=tk.BOTH, expand=True)
        
        # State selector
        state_selector_frame = ttk.Frame(self.state_frame)
        state_selector_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(state_selector_frame, text="Select State:", font=('Helvetica', 10, 'bold')).pack(side=tk.LEFT)
        
        self.state_profile_var = tk.StringVar()
        self.state_selector = ttk.Combobox(state_selector_frame, 
                                         textvariable=self.state_profile_var, 
                                         values=sorted(df['Region'].unique()), 
                                         state='readonly',
                                         width=25)
        self.state_selector.pack(side=tk.LEFT, padx=10)
        self.state_selector.bind('<<ComboboxSelected>>', self.update_state_profile)
        
        # State profile content
        self.state_content = ttk.Frame(self.state_frame)
        self.state_content.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Placeholder for state profile
        ttk.Label(self.state_content, text="Select a state to view detailed profile",
                 font=('Helvetica', 10)).pack(pady=50)
    
    def create_card(self, parent, title, value, color):
        """Helper function to create modern card UI elements"""
        card = tk.Frame(parent, bg=COLOR_SCHEME['card_bg'], bd=0, highlightthickness=0, 
                       relief='ridge', padx=5, pady=5)
        
        # Card header
        header = tk.Frame(card, bg=color, bd=0)
        header.pack(fill=tk.X)
        
        # Title label
        title_label = tk.Label(header, text=title, bg=color, fg='white', 
                              font=('Helvetica', 9, 'bold'), padx=5, pady=2)
        title_label.pack()
        
        # Value label
        value_label = tk.Label(card, text=value, bg=COLOR_SCHEME['card_bg'], 
                              fg=COLOR_SCHEME['card_text'], 
                              font=('Helvetica', 12, 'bold'), pady=5)
        value_label.pack()
        
        # Store reference to update later
        if "CURRENT" in title:
            self.current_label = value_label
        elif "CHANGE" in title:
            self.change_label = value_label
        elif "AVERAGE" in title:
            self.avg_label = value_label
        elif "MIN/MAX" in title:
            self.minmax_label = value_label
        elif "TOP" in title:
            self.top_performers_label = value_label
        elif "BOTTOM" in title:
            self.bottom_performers_label = value_label
        elif "PEAK" in title:
            self.max_unemp_label = value_label
        elif "RECOVERY" in title:
            self.recovery_label = value_label
        elif "MOST" in title:
            self.affected_label = value_label
        elif "EMPLOYMENT" in title:
            self.economic_label = value_label
        elif "INSIGHTS" in title:
            self.insights_label = value_label
        
        return card
    
    def update_plots(self):
        try:
            start = datetime.strptime(self.start_date.get(), '%Y-%m-%d')
            end = datetime.strptime(self.end_date.get(), '%Y-%m-%d')
        except ValueError:
            messagebox.showerror("Error", "Invalid date format. Please use YYYY-MM-DD.")
            return
        
        filtered_df = df[(df['Date'] >= pd.to_datetime(start)) & 
                        (df['Date'] <= pd.to_datetime(end))]
        
        if self.region_var.get() != 'All':
            filtered_df = filtered_df[filtered_df['Region'] == self.region_var.get()]
        
        if self.area_var.get() != 'All':
            filtered_df = filtered_df[filtered_df['Area'] == self.area_var.get()]
        
        metric = self.metric_var.get()
        
        # Clear previous plots
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
            ax.clear()
        
        # ===== TRENDS TAB =====
        if self.region_var.get() == 'All':
            # National trend with enhanced visualization
            trend_data = filtered_df.groupby('Date')[metric].mean().reset_index()
            
            # Main trend line
            self.ax1.plot(trend_data['Date'], trend_data[metric], 
                         color=COLOR_SCHEME['primary'], 
                         linewidth=2.5,
                         marker='o',
                         markersize=6,
                         markerfacecolor='white',
                         markeredgewidth=1.5,
                         markeredgecolor=COLOR_SCHEME['primary'],
                         label='National Average')
            
            # Add confidence interval
            if len(trend_data) > 1:
                rolling_std = trend_data[metric].rolling(window=30).std().fillna(method='bfill')
                self.ax1.fill_between(trend_data['Date'], 
                                     trend_data[metric] - rolling_std, 
                                     trend_data[metric] + rolling_std,
                                     color=COLOR_SCHEME['primary'], 
                                     alpha=0.2,
                                     label='Confidence Interval')
            
            self.ax1.set_title(f'National {metric} Trend', fontsize=12, fontweight='bold', pad=20)
            
            # Update summary cards
            latest_value = trend_data[metric].iloc[-1]
            prev_value = trend_data[metric].iloc[-2] if len(trend_data) > 1 else latest_value
            change = ((latest_value - prev_value) / prev_value * 100) if prev_value != 0 else 0
            
            self.current_label.config(text=f"{latest_value:.1f}")
            self.change_label.config(text=f"{change:+.1f}%", 
                                   fg=COLOR_SCHEME['success'] if change >= 0 else COLOR_SCHEME['accent'])
            
            avg_6mo = trend_data[metric].tail(6).mean()
            self.avg_label.config(text=f"{avg_6mo:.1f}")
            
            min_val = trend_data[metric].min()
            max_val = trend_data[metric].max()
            self.minmax_label.config(text=f"{min_val:.1f} / {max_val:.1f}")
            
        else:
            # State trend with enhanced visualization
            for area_type in filtered_df['Area'].unique():
                area_data = filtered_df[filtered_df['Area'] == area_type]
                self.ax1.plot(area_data['Date'], area_data[metric], 
                             label=area_type,
                             linewidth=2.5,
                             marker='o',
                             markersize=6,
                             markerfacecolor='white',
                             markeredgewidth=1.5)
            
            self.ax1.set_title(f'{self.region_var.get()} - {metric} Trend', 
                             fontsize=12, fontweight='bold', pad=20)
            
            # Update summary cards for state view
            latest = filtered_df[filtered_df['Date'] == filtered_df['Date'].max()]
            if not latest.empty:
                latest_value = latest[metric].mean()
                prev_month = filtered_df[filtered_df['Date'] == (filtered_df['Date'].max() - pd.DateOffset(months=1))]
                prev_value = prev_month[metric].mean() if not prev_month.empty else latest_value
                change = ((latest_value - prev_value) / prev_value * 100) if prev_value != 0 else 0
                
                self.current_label.config(text=f"{latest_value:.1f}")
                self.change_label.config(text=f"{change:+.1f}%", 
                                       fg=COLOR_SCHEME['success'] if change >= 0 else COLOR_SCHEME['accent'])
                
                last_6mo = filtered_df[filtered_df['Date'] >= (filtered_df['Date'].max() - pd.DateOffset(months=6))]
                avg_6mo = last_6mo[metric].mean()
                self.avg_label.config(text=f"{avg_6mo:.1f}")
                
                min_val = filtered_df[metric].min()
                max_val = filtered_df[metric].max()
                self.minmax_label.config(text=f"{min_val:.1f} / {max_val:.1f}")
        
        # Style trend plot
        self.ax1.set_xlabel('Date', fontsize=10, labelpad=10)
        self.ax1.set_ylabel(metric, fontsize=10, labelpad=10)
        self.ax1.legend(frameon=True, facecolor=COLOR_SCHEME['background'])
        self.ax1.grid(True, linestyle='--', alpha=0.5)
        
        # Rotate x-axis labels for better readability
        for label in self.ax1.get_xticklabels():
            label.set_rotation(45)
            label.set_horizontalalignment('right')
        
        # Add padding around the plot
        self.fig1.tight_layout()
        
        # ===== REGIONAL TAB =====
        if self.region_var.get() == 'All':
            # State comparison with enhanced visualization
            state_data = filtered_df.groupby('Region')[metric].mean().sort_values(ascending=False)
            
            # Create bar plot with color gradient
            colors = plt.cm.Blues(np.linspace(0.3, 1, len(state_data)))
            state_data.plot(kind='bar', 
                           ax=self.ax2,
                           color=colors,
                           edgecolor=COLOR_SCHEME['dark'],
                           linewidth=0.5)
            
            self.ax2.set_title(f'State-wise {metric} Comparison', 
                              fontsize=12, fontweight='bold', pad=20)
            self.ax2.set_ylabel(metric, fontsize=10, labelpad=10)
            self.ax2.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for p in self.ax2.patches:
                self.ax2.annotate(f"{p.get_height():.1f}", 
                                 (p.get_x() + p.get_width() / 2., p.get_height()), 
                                 ha='center', va='center', 
                                 xytext=(0, 5), 
                                 textcoords='offset points',
                                 fontsize=8)
            
            # Update top/bottom performers cards
            top5 = state_data.head(5)
            bottom5 = state_data.tail(5)
            
            top_text = "\n".join([f"{state}: {val:.1f}" for state, val in top5.items()])
            bottom_text = "\n".join([f"{state}: {val:.1f}" for state, val in bottom5.items()])
            
            self.top_performers_label.config(text=top_text)
            self.bottom_performers_label.config(text=bottom_text)
            
            # Regional insights (NEW)
            rural_avg = filtered_df[filtered_df['Area'] == 'Rural'][metric].mean()
            urban_avg = filtered_df[filtered_df['Area'] == 'Urban'][metric].mean()
            insight_text = f"Rural: {rural_avg:.1f}\nUrban: {urban_avg:.1f}\nDiff: {abs(rural_avg-urban_avg):.1f}"
            self.insights_label.config(text=insight_text)
            
        else:
            # Monthly breakdown for selected state with enhanced visualization
            month_data = filtered_df.groupby('Month')[metric].mean()
            month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                          'July', 'August', 'September', 'October', 'November', 'December']
            month_data = month_data.reindex(month_order, fill_value=0)
            
            # Create gradient colors based on values
            colors = plt.cm.Blues((month_data - month_data.min()) / (month_data.max() - month_data.min()))
            
            bars = self.ax2.bar(month_data.index, month_data.values, color=colors)
            
            self.ax2.set_title(f'{self.region_var.get()} - Monthly {metric} Breakdown', 
                              fontsize=12, fontweight='bold', pad=20)
            self.ax2.set_ylabel(metric, fontsize=10, labelpad=10)
            self.ax2.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                self.ax2.annotate(f"{height:.1f}",
                                 xy=(bar.get_x() + bar.get_width() / 2, height),
                                                                  xytext=(0, 3),
                                 textcoords='offset points',
                                 ha='center', va='bottom',
                                 fontsize=8)
            
            # Highlight COVID months
            for i, month in enumerate(month_order):
                if month in ['March', 'April', 'May', 'June']:
                    bars[i].set_edgecolor(COLOR_SCHEME['accent'])
                    bars[i].set_linewidth(1.5)
            
            # Update cards for state monthly view
            current_month = filtered_df['Date'].max().strftime('%B')
            current_value = month_data[current_month] if current_month in month_data else 0
            self.current_label.config(text=f"{current_value:.1f}")
            
            # Calculate month-over-month change
            if len(month_data) > 1:
                prev_month_value = month_data.iloc[-2] if current_month != 'January' else month_data.iloc[-1]
                change = ((current_value - prev_month_value) / prev_month_value * 100) if prev_month_value != 0 else 0
                self.change_label.config(text=f"{change:+.1f}%",
                                       fg=COLOR_SCHEME['success'] if change >= 0 else COLOR_SCHEME['accent'])
            
            # Annual average
            annual_avg = month_data.mean()
            self.avg_label.config(text=f"{annual_avg:.1f}")
            
            # Min/max
            min_val = month_data.min()
            max_val = month_data.max()
            self.minmax_label.config(text=f"{min_val:.1f} / {max_val:.1f}")
        
        # Style regional plot
        self.ax2.grid(True, linestyle='--', alpha=0.5)
        self.fig2.tight_layout()
        
        # ===== COVID TAB =====
        # Focus on COVID period (March-June 2020)
        covid_df = df[(df['Date'] >= pd.to_datetime('2020-03-01')) & 
                     (df['Date'] <= pd.to_datetime('2020-06-30'))]
        
        if self.region_var.get() != 'All':
            covid_df = covid_df[covid_df['Region'] == self.region_var.get()]
        
        if self.area_var.get() != 'All':
            covid_df = covid_df[covid_df['Area'] == self.area_var.get()]
        
        # Plot COVID impact with enhanced visualization
        if not covid_df.empty:
            covid_trend = covid_df.groupby('Date')['Estimated Unemployment Rate (%)'].mean()
            
            # Main trend line
            self.ax3.plot(covid_trend.index, covid_trend.values,
                         color=COLOR_SCHEME['primary'],
                         linewidth=2.5,
                         marker='o',
                         markersize=6,
                         markerfacecolor='white',
                         markeredgewidth=1.5,
                         markeredgecolor=COLOR_SCHEME['primary'])
            
            # Add lockdown annotations with modern styling
            self.ax3.axvline(pd.to_datetime('2020-03-25'), color=COLOR_SCHEME['accent'], linestyle='--', alpha=0.7, linewidth=1.5)
            self.ax3.text(pd.to_datetime('2020-03-25'), covid_trend.max()*0.9,
                         'National Lockdown',
                         color=COLOR_SCHEME['accent'],
                         fontsize=9,
                         bbox=dict(facecolor='white', alpha=0.8, edgecolor=COLOR_SCHEME['accent']),
                         rotation=90,
                         va='top')
            
            # Add recovery phase annotation
            recovery_date = pd.to_datetime('2020-06-01')
            if recovery_date in covid_trend.index:
                self.ax3.axvline(recovery_date, color=COLOR_SCHEME['success'], linestyle='--', alpha=0.7, linewidth=1.5)
                self.ax3.text(recovery_date, covid_trend.max()*0.7,
                             'Recovery Phase',
                             color=COLOR_SCHEME['success'],
                             fontsize=9,
                             bbox=dict(facecolor='white', alpha=0.8, edgecolor=COLOR_SCHEME['success']),
                             rotation=90,
                             va='top')
            
            self.ax3.set_title('COVID-19 Impact on Unemployment', fontsize=12, fontweight='bold', pad=20)
            self.ax3.set_xlabel('Date', fontsize=10, labelpad=10)
            self.ax3.set_ylabel('Estimated Unemployment Rate (%)', fontsize=10, labelpad=10)
            self.ax3.grid(True, linestyle='--', alpha=0.5)
            
            # Calculate impact metrics
            max_rate = covid_trend.max()
            min_rate = covid_trend.min()
            recovery = ((max_rate - covid_trend[-1]) / max_rate) * 100
            
            # Update COVID impact cards
            self.max_unemp_label.config(text=f"{max_rate:.1f}%")
            self.recovery_label.config(text=f"{recovery:.1f}%",
                                     fg=COLOR_SCHEME['success'] if recovery >= 0 else COLOR_SCHEME['accent'])
            
            # Calculate employment loss (NEW FEATURE)
            if 'Estimated Employed' in covid_df.columns:
                pre_covid_emp = covid_df[covid_df['Date'] == pd.to_datetime('2020-03-01')]['Estimated Employed'].mean()
                lowest_emp = covid_df['Estimated Employed'].min()
                emp_loss = ((pre_covid_emp - lowest_emp) / pre_covid_emp) * 100
                self.economic_label.config(text=f"{emp_loss:.1f}%")
            
            if self.region_var.get() == 'All':
                worst_state = df[(df['Date'] >= pd.to_datetime('2020-04-01')) & 
                               (df['Date'] <= pd.to_datetime('2020-04-30'))]
                worst_state = worst_state.groupby('Region')['Estimated Unemployment Rate (%)'].mean().idxmax()
                self.affected_label.config(text=worst_state)
        
        # Style COVID plot
        self.fig3.tight_layout()
        
        # ===== HEATMAP TAB (NEW FEATURE) =====
        self.update_heatmap()
        
        
        # ===== STATE PROFILE TAB (NEW FEATURE) =====
        if self.region_var.get() != 'All':
            self.state_profile_var.set(self.region_var.get())
            self.update_state_profile()
        
        # Redraw all canvases
        for canvas in [self.canvas1, self.canvas2, self.canvas3, self.canvas4]:
            canvas.draw()
    
    def update_heatmap(self):
        """NEW FEATURE: Update heatmap visualization"""
        self.ax4.clear()
        
        try:
            start = datetime.strptime(self.start_date.get(), '%Y-%m-%d')
            end = datetime.strptime(self.end_date.get(), '%Y-%m-%d')
        except ValueError:
            return
        
        filtered_df = df[(df['Date'] >= pd.to_datetime(start)) & 
                        (df['Date'] <= pd.to_datetime(end))]
        
        if self.region_var.get() != 'All':
            filtered_df = filtered_df[filtered_df['Region'] == self.region_var.get()]
        
        if self.area_var.get() != 'All':
            filtered_df = filtered_df[filtered_df['Area'] == self.area_var.get()]
        
        metric = self.metric_var.get()
        
        if self.heatmap_type.get() == 'Monthly':
            # Monthly heatmap by year
            filtered_df['Year'] = filtered_df['Date'].dt.year
            filtered_df['Month'] = filtered_df['Date'].dt.month
            
            heatmap_data = filtered_df.pivot_table(index='Year', 
                                                  columns='Month', 
                                                  values=metric, 
                                                  aggfunc='mean')
            
            # Reorder columns by month
            heatmap_data = heatmap_data.reindex(columns=range(1,13))
            heatmap_data.columns = [calendar.month_abbr[i] for i in range(1,13)]
            
            title = f'Monthly {metric} by Year'
            
        elif self.heatmap_type.get() == 'Quarterly':
            # Quarterly heatmap by year
            filtered_df['Year'] = filtered_df['Date'].dt.year
            filtered_df['Quarter'] = filtered_df['Date'].dt.quarter
            
            heatmap_data = filtered_df.pivot_table(index='Year', 
                                                  columns='Quarter', 
                                                  values=metric, 
                                                  aggfunc='mean')
            
            heatmap_data.columns = [f'Q{i}' for i in range(1,5)]
            title = f'Quarterly {metric} by Year'
            
        else:  # Yearly
            # Yearly heatmap by state
            if self.region_var.get() == 'All':
                heatmap_data = filtered_df.pivot_table(index='Region', 
                                                      columns='Year', 
                                                      values=metric, 
                                                      aggfunc='mean')
                title = f'Yearly {metric} by State'
            else:
                # If a specific state is selected, show yearly by area type
                heatmap_data = filtered_df.pivot_table(index='Area', 
                                                      columns='Year', 
                                                      values=metric, 
                                                      aggfunc='mean')
                title = f'Yearly {metric} by Area Type'
        
        # Create heatmap with custom colors
        if not heatmap_data.empty:
            sns.heatmap(heatmap_data, 
                       ax=self.ax4,
                       cmap=cmap,
                       annot=True,
                       fmt=".1f",
                       linewidths=0.5,
                       cbar_kws={'label': metric})
            
            self.ax4.set_title(title, fontsize=12, fontweight='bold', pad=20)
            self.ax4.set_facecolor(COLOR_SCHEME['background'])
            
            # Rotate x-axis labels
            for label in self.ax4.get_xticklabels():
                label.set_rotation(45)
                label.set_horizontalalignment('right')
        
        self.fig4.tight_layout()
        self.canvas4.draw()
    

    def update_state_profile(self, event=None):
        """NEW FEATURE: Update state profile information"""
        state = self.state_profile_var.get()
        if not state:
            return
        
        # Clear previous content
        for widget in self.state_content.winfo_children():
            widget.destroy()
        
        # Get state data
        state_df = df[df['Region'] == state]
        
        # Create tabbed interface for state profile
        state_notebook = ttk.Notebook(self.state_content)
        state_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tab 1: Overview
        overview_tab = ttk.Frame(state_notebook)
        state_notebook.add(overview_tab, text='OVERVIEW')
        
        # Tab 2: Employment Trends
        trends_tab = ttk.Frame(state_notebook)
        state_notebook.add(trends_tab, text='TRENDS')
        
        # Tab 3: Comparison
        comparison_tab = ttk.Frame(state_notebook)
        state_notebook.add(comparison_tab, text='COMPARISON')
        
        # ===== OVERVIEW TAB =====
        # State summary cards
        overview_frame = ttk.Frame(overview_tab)
        overview_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Card 1: Current Unemployment
        latest_data = state_df[state_df['Date'] == state_df['Date'].max()]
        urban_unemp = latest_data[latest_data['Area'] == 'Urban']['Estimated Unemployment Rate (%)'].mean()
        rural_unemp = latest_data[latest_data['Area'] == 'Rural']['Estimated Unemployment Rate (%)'].mean()
        
        unemp_card = self.create_card(overview_frame, 
                                    "CURRENT UNEMPLOYMENT", 
                                    f"Urban: {urban_unemp:.1f}%\nRural: {rural_unemp:.1f}%", 
                                    COLOR_SCHEME['primary'])
        unemp_card.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # Card 2: Employment
        urban_emp = latest_data[latest_data['Area'] == 'Urban']['Estimated Employed'].mean()
        rural_emp = latest_data[latest_data['Area'] == 'Rural']['Estimated Employed'].mean()
        
        emp_card = self.create_card(overview_frame, 
                                  "EMPLOYMENT", 
                                  f"Urban: {urban_emp/1000000:.1f}M\nRural: {rural_emp/1000000:.1f}M", 
                                  COLOR_SCHEME['secondary'])
        emp_card.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # Card 3: Participation Rate
        urban_part = latest_data[latest_data['Area'] == 'Urban']['Estimated Labour Participation Rate (%)'].mean()
        rural_part = latest_data[latest_data['Area'] == 'Rural']['Estimated Labour Participation Rate (%)'].mean()
        
        part_card = self.create_card(overview_frame, 
                                   "PARTICIPATION RATE", 
                                   f"Urban: {urban_part:.1f}%\nRural: {rural_part:.1f}%", 
                                   COLOR_SCHEME['dark'])
        part_card.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # State facts frame
        facts_frame = ttk.Frame(overview_tab)
        facts_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Add state facts (placeholder - could be enhanced with real data)
        facts_text = f"""
        {state} Unemployment Profile:
        - {'Above' if urban_unemp > df['Estimated Unemployment Rate (%)'].mean() else 'Below'} national average
        - {'Urban' if urban_unemp > rural_unemp else 'Rural'} areas have higher unemployment
        - {'Increasing' if state_df['Unemployment Change (%)'].iloc[-1] > 0 else 'Decreasing'} trend
        """
        
        ttk.Label(facts_frame, 
                 text=facts_text,
                 font=('Helvetica', 10),
                 justify=tk.LEFT).pack(side=tk.LEFT)
        
        # ===== TRENDS TAB =====
        # Create figure for trends
        fig = plt.Figure(figsize=(10, 4), dpi=100, facecolor=COLOR_SCHEME['background'])
        ax = fig.add_subplot(111)
        ax.set_facecolor(COLOR_SCHEME['background'])
        
        # Plot urban vs rural trends
        for area in ['Urban', 'Rural']:
            area_data = state_df[state_df['Area'] == area]
            if not area_data.empty:
                trend = area_data.groupby('Date')['Estimated Unemployment Rate (%)'].mean()
                ax.plot(trend.index, trend.values, 
                       label=area,
                       linewidth=2)
        
        ax.set_title(f'{state} - Urban vs Rural Unemployment', fontsize=12, fontweight='bold')
        ax.set_xlabel('Date', fontsize=10)
        ax.set_ylabel('Unemployment Rate (%)', fontsize=10)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # Add canvas to tab
        canvas = FigureCanvasTkAgg(fig, master=trends_tab)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        canvas.draw()
        
        # ===== COMPARISON TAB =====
        # Create figure for comparison
        fig = plt.Figure(figsize=(10, 4), dpi=100, facecolor=COLOR_SCHEME['background'])
        ax = fig.add_subplot(111)
        ax.set_facecolor(COLOR_SCHEME['background'])
        
        # Compare with national average
        national_avg = df.groupby('Date')['Estimated Unemployment Rate (%)'].mean()
        state_avg = state_df.groupby('Date')['Estimated Unemployment Rate (%)'].mean()
        
        ax.plot(national_avg.index, national_avg.values,
               label='National Average',
               color=COLOR_SCHEME['primary'],
               linewidth=2)
        
        ax.plot(state_avg.index, state_avg.values,
               label=state,
               color=COLOR_SCHEME['accent'],
               linewidth=2)
        
        ax.set_title(f'{state} vs National Average', fontsize=12, fontweight='bold')
        ax.set_xlabel('Date', fontsize=10)
        ax.set_ylabel('Unemployment Rate (%)', fontsize=10)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # Add canvas to tab
        canvas = FigureCanvasTkAgg(fig, master=comparison_tab)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        canvas.draw()
    
    def export_data(self):
        """NEW FEATURE: Export filtered data to CSV"""
        try:
            start = datetime.strptime(self.start_date.get(), '%Y-%m-%d')
            end = datetime.strptime(self.end_date.get(), '%Y-%m-%d')
        except ValueError:
            messagebox.showerror("Error", "Invalid date format. Please use YYYY-MM-DD.")
            return
        
        filtered_df = df[(df['Date'] >= pd.to_datetime(start)) & 
                        (df['Date'] <= pd.to_datetime(end))]
        
        if self.region_var.get() != 'All':
            filtered_df = filtered_df[filtered_df['Region'] == self.region_var.get()]
        
        if self.area_var.get() != 'All':
            filtered_df = filtered_df[filtered_df['Area'] == self.area_var.get()]
        
        # Save to CSV
        filename = f"unemployment_data_{start.date()}_to_{end.date()}"
        if self.region_var.get() != 'All':
            filename += f"_{self.region_var.get()}"
        if self.area_var.get() != 'All':
            filename += f"_{self.area_var.get()}"
        filename += ".csv"
        
        filtered_df.to_csv(filename, index=False)
        messagebox.showinfo("Export Successful", f"Data exported to {filename}")

# ========== RUN APPLICATION ==========
if __name__ == "__main__":
    root = tk.Tk()
    
    # Create custom style
    style = ttk.Style()
    style.theme_use('clam')
    
    # Configure styles with our color scheme
    style.configure('.', background=COLOR_SCHEME['background'])
    style.configure('TFrame', background=COLOR_SCHEME['background'])
    style.configure('TLabel', background=COLOR_SCHEME['background'], foreground=COLOR_SCHEME['text'])
    style.configure('TButton', font=('Helvetica', 10, 'bold'), padding=5)
    style.configure('Accent.TButton', foreground='white', background=COLOR_SCHEME['primary'])
    style.map('Accent.TButton',
             background=[('active', COLOR_SCHEME['secondary']), ('pressed', COLOR_SCHEME['accent'])])
    style.configure('TLabelframe', background=COLOR_SCHEME['background'], bordercolor=COLOR_SCHEME['primary'])
    style.configure('TLabelframe.Label', background=COLOR_SCHEME['background'], foreground=COLOR_SCHEME['text'], font=('Helvetica', 10, 'bold'))
    style.configure('Custom.TLabelframe', background=COLOR_SCHEME['background'], bordercolor=COLOR_SCHEME['primary'], relief='solid')
    style.configure('TCombobox', fieldbackground='white')
    style.configure('TEntry', fieldbackground='white')
    
    # Set window icon (placeholder)
    try:
        root.iconbitmap('icon.ico')  # Replace with actual icon file if available
    except:
        pass
    
    app = UnemploymentDashboard(root)
    root.mainloop()