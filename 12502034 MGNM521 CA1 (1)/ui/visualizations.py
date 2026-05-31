import matplotlib
matplotlib.use('qtagg') 
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud # Moved import to the top

class DashboardCharts(FigureCanvas):
    """Embeds Matplotlib charts directly into the PySide6 Desktop UI."""
    def __init__(self, parent=None, width=8, height=6, dpi=100):
        # Changed to a 2x2 grid to fit all 4 plots
        self.fig, self.axes_matrix = plt.subplots(2, 2, figsize=(width, height), dpi=dpi)
        self.axes = self.axes_matrix.flatten() # Flattens to a 1D array (indices 0 to 3)
        self.fig.patch.set_facecolor('#1E1E1E') # Match dark theme
        
        super(DashboardCharts, self).__init__(self.fig)
        if parent:
            self.setParent(parent) # Explicitly set the parent for PySide6 integration

    def plot_data(self, metrics, conf_matrix):
        # Clear all subplots cleanly
        for ax in self.axes:
            ax.clear()

        # 1. Pie Chart (Index 0)
        labels = ['Positive', 'Negative']
        sizes = [metrics.get('Positive_Pct', 50), metrics.get('Negative_Pct', 50)]
        colors = ['#00E676', '#FF1744']
        
        _, texts, autotexts = self.axes[0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        for text in texts + autotexts:
            text.set_color('white')
        self.axes[0].set_title('Sentiment Distribution', color='white')

        # 2. Word Cloud of Negative Texts (Index 1)
        if 'Negative_Text_Corpus' in metrics and metrics['Negative_Text_Corpus']:
            wordcloud = WordCloud(width=400, height=300, background_color='#1E1E1E', colormap='Reds').generate(metrics['Negative_Text_Corpus'])
            self.axes[1].imshow(wordcloud, interpolation='bilinear')
            self.axes[1].axis('off')
            self.axes[1].set_title("Common Negative Words", color='white')
        else:
            self.axes[1].text(0.5, 0.5, 'No Negative Text Data', horizontalalignment='center', verticalalignment='center', color='white', fontsize=12)
            self.axes[1].axis('off')

        # 3. Word Cloud of Positive Texts (Index 2)
        if 'Positive_Text_Corpus' in metrics and metrics['Positive_Text_Corpus']:
            wordcloud = WordCloud(width=400, height=300, background_color='#1E1E1E', colormap='Greens').generate(metrics['Positive_Text_Corpus'])
            self.axes[2].imshow(wordcloud, interpolation='bilinear')
            self.axes[2].axis('off')
            self.axes[2].set_title("Common Positive Words", color='white')
        else:
            self.axes[2].text(0.5, 0.5, 'No Positive Text Data', horizontalalignment='center', verticalalignment='center', color='white', fontsize=12)
            self.axes[2].axis('off')

        # 4. Confusion Matrix (Index 3)
        if conf_matrix is not None:
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, ax=self.axes[3],
                        xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
            self.axes[3].set_title("ML Confusion Matrix", color='white')
            self.axes[3].tick_params(colors='white')
        else:
            self.axes[3].text(0.5, 0.5, 'No Confusion Matrix', horizontalalignment='center', verticalalignment='center', color='white', fontsize=12)
            self.axes[3].axis('off')

        # Adjust Layout and Refresh
        self.fig.tight_layout()
        self.fig.subplots_adjust(bottom=0.15, wspace=0.3, hspace=0.3)
        self.draw()