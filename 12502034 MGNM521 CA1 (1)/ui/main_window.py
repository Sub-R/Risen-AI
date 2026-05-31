from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QPushButton, QLabel, QStackedWidget, QFrame, 
                            QGridLayout, QTextEdit, QLineEdit, QFormLayout, QMessageBox, QFileDialog)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont
from models.transformers_layer import TransformerLayer
from models.data_loader import DataLoader
from models.ml_engine import ClassicalMLEngine
from chatbot.copilot import AICopilot
from ui.visualizations import DashboardCharts
from config.settings import Config
import pandas as pd
from reportlab.pdfgen import canvas

class StartupWorker(QThread):
    """Loads saved models and FAISS index silently in the background on boot."""
    finished = Signal(object, object)
    
    def run(self):
        engine = ClassicalMLEngine()
        metrics, conf_matrix = engine.load_saved_model()
        self.finished.emit(metrics, conf_matrix)
class MLWorker(QThread):
    finished = Signal(object, object)
    
    def run(self):
        df = DataLoader.get_unified_dataset()
        
        # 1. Classical ML
        engine = ClassicalMLEngine()
        metrics, conf_matrix = engine.train_and_evaluate(df)
        
        # 2. HuggingFace Transformer Comparison
        try:
            hf_layer = TransformerLayer()
            # Sampling 10 records so the deep learning model doesn't freeze standard laptops
            sample_texts = df['Text'].head(10).tolist()
            hf_results = hf_layer.run_sentiment_comparison(sample_texts)
            metrics['HF_Comparison'] = hf_results
        except Exception as e:
            metrics['HF_Comparison'] = f"Error: {str(e)}"
            
        self.finished.emit(metrics, conf_matrix)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Business Intelligence Agent - Exec Dashboard")
        self.setGeometry(100, 100, 1400, 850)
        self.setStyleSheet("background-color: #121212; color: #FFFFFF;")
        
        self.metrics = {}
        self.copilot = AICopilot()
        self.init_ui()
        
        # Automatically load saved models to persist state
        self.btn_train.setText("Loading Models...")
        self.btn_train.setEnabled(False)
        self.startup_worker = StartupWorker()
        self.startup_worker.finished.connect(self.on_startup_complete)
        self.startup_worker.start()

    def on_startup_complete(self, metrics, conf_matrix):
        if metrics:
            self.update_dashboard(metrics, conf_matrix)
            self.chat_history.append("<i>System: Restored previous AI model state. Ready.</i>")
        else:
            self.btn_train.setText("▶ Run AI Pipeline")
            self.btn_train.setEnabled(True)
            self.chat_history.append("<i>System: No saved model found. Please run the AI Pipeline.</i>")

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        # ---------------- LEFT SIDEBAR ----------------
        sidebar = QFrame()
        sidebar.setFixedWidth(250)
        sidebar.setStyleSheet("background-color: #1E1E1E; border-right: 1px solid #333;")
        sidebar_layout = QVBoxLayout(sidebar)
        
        title = QLabel("BI COPILOT")
        title.setFont(QFont("Segoe UI", 18, QFont.Bold))
        sidebar_layout.addWidget(title)

        self.btn_dashboard = self.create_nav_button("📊 Dashboard")
        self.btn_chatbot = self.create_nav_button("🤖 AI Chatbot")
        self.btn_settings = self.create_nav_button("⚙️ Settings")
        
        self.btn_dashboard.clicked.connect(lambda: self.stack.setCurrentIndex(0))
        self.btn_chatbot.clicked.connect(lambda: self.stack.setCurrentIndex(1))
        self.btn_settings.clicked.connect(lambda: self.stack.setCurrentIndex(2))

        sidebar_layout.addWidget(self.btn_dashboard)
        sidebar_layout.addWidget(self.btn_chatbot)
        sidebar_layout.addWidget(self.btn_settings)
        sidebar_layout.addStretch()

        # Action Buttons
        self.btn_export_pdf = QPushButton("📄 Export PDF Report")
        self.btn_export_pdf.setStyleSheet("background-color: #03DAC6; color: black; padding: 10px; margin-bottom: 5px;")
        self.btn_export_pdf.clicked.connect(self.export_pdf)
        sidebar_layout.addWidget(self.btn_export_pdf)

        self.btn_train = QPushButton("▶ Retrain ML Pipeline")
        self.btn_train.setStyleSheet("background-color: #BB86FC; color: black; font-weight: bold; padding: 15px;")
        self.btn_train.clicked.connect(self.run_pipeline)
        sidebar_layout.addWidget(self.btn_train)

        layout.addWidget(sidebar)

        # ---------------- MAIN CONTENT AREA ----------------
        self.stack = QStackedWidget()
        layout.addWidget(self.stack)

        self.dashboard_view = QWidget()
        self.build_dashboard(self.dashboard_view)
        self.stack.addWidget(self.dashboard_view)

        self.chatbot_view = QWidget()
        self.build_chatbot(self.chatbot_view)
        self.stack.addWidget(self.chatbot_view)

        self.settings_view = QWidget()
        self.build_settings(self.settings_view)
        self.stack.addWidget(self.settings_view)

    def create_nav_button(self, text):
        btn = QPushButton(text)
        btn.setStyleSheet("text-align: left; padding: 12px; font-size: 14px; border: none;")
        return btn

    def build_dashboard(self, view):
        layout = QVBoxLayout(view)
        kpi_grid = QGridLayout()
        self.kpi_labels = {
            'Accuracy': QLabel("Acc: --%"), 'F1_Score': QLabel("F1: --%"),
            'Precision': QLabel("Prec: --%"), 'Recall': QLabel("Rec: --%"),
            'Customer_Satisfaction': QLabel("CSAT: --%"), 'Total_Analyzed': QLabel("Records: --")
        }
        
        col = 0
        for key, lbl in self.kpi_labels.items():
            lbl.setFont(QFont("Segoe UI", 12))
            lbl.setStyleSheet("background-color: #2D2D2D; padding: 20px; border-radius: 8px;")
            lbl.setAlignment(Qt.AlignCenter)
            kpi_grid.addWidget(lbl, 0, col)
            col += 1
            
        layout.addLayout(kpi_grid)
        self.charts = DashboardCharts(self)
        layout.addWidget(self.charts)

    def build_chatbot(self, view):
        layout = QVBoxLayout(view)
        self.chat_history = QTextEdit()
        self.chat_history.setReadOnly(True)
        self.chat_history.setStyleSheet("background-color: #1E1E1E; border: 1px solid #333; font-size: 14px; padding: 10px;")
        layout.addWidget(self.chat_history)

        input_layout = QHBoxLayout()
        self.chat_input = QLineEdit()
        self.chat_input.returnPressed.connect(self.send_chat)
        btn_send = QPushButton("Send")
        btn_send.clicked.connect(self.send_chat)
        
        input_layout.addWidget(self.chat_input)
        input_layout.addWidget(btn_send)
        layout.addLayout(input_layout)

    def build_settings(self, view):
        layout = QFormLayout(view)
        
        # OpenRouter Key
        self.input_openrouter = QLineEdit(Config.get_openrouter_key())
        self.input_openrouter.setEchoMode(QLineEdit.Password)
        layout.addRow("OpenRouter API Key:", self.input_openrouter)
        
        # HuggingFace Key (Restored)
        self.input_hf = QLineEdit(Config.get_huggingface_key())
        self.input_hf.setEchoMode(QLineEdit.Password)
        layout.addRow("HuggingFace API Key:", self.input_hf)

        btn_save = QPushButton("Save Keys")
        btn_save.clicked.connect(self.save_settings)
        layout.addRow(btn_save)

    def save_settings(self):
        Config.update_key("OPENROUTER_API_KEY", self.input_openrouter.text())
        Config.update_key("HUGGINGFACE_API_KEY", self.input_hf.text())
        QMessageBox.information(self, "Success", "Settings saved securely.")

    def run_pipeline(self):
        self.btn_train.setText("⏳ Retraining Models...")
        self.btn_train.setEnabled(False)
        self.worker = MLWorker()
        self.worker.finished.connect(self.update_dashboard)
        self.worker.start()

    def update_dashboard(self, metrics, conf_matrix):
        self.metrics = metrics
        self.kpi_labels['Accuracy'].setText(f"Acc: {metrics['Accuracy']}%")
        self.kpi_labels['F1_Score'].setText(f"F1: {metrics['F1_Score']}%")
        self.kpi_labels['Precision'].setText(f"Prec: {metrics['Precision']}%")
        self.kpi_labels['Recall'].setText(f"Rec: {metrics['Recall']}%")
        self.kpi_labels['Customer_Satisfaction'].setText(f"CSAT: {metrics['Customer_Satisfaction']}%")
        self.kpi_labels['Total_Analyzed'].setText(f"Records: {metrics['Total_Analyzed']}")

        self.charts.plot_data(metrics, conf_matrix)
        
        # Confirm HuggingFace ran successfully
        if 'HF_Comparison' in metrics and isinstance(metrics['HF_Comparison'], list):
            self.chat_history.append("<i>System: HuggingFace Transformer comparison executed successfully.</i>")
            
        self.btn_train.setText("▶ Retrain ML Pipeline")
        self.btn_train.setEnabled(True)

    def send_chat(self):
        user_text = self.chat_input.text()
        if not user_text.strip(): return
        self.chat_history.append(f"<b style='color:#BB86FC;'>You:</b> {user_text}<br>")
        self.chat_input.clear()
        
        response = self.copilot.get_response(user_text, self.metrics)
        self.chat_history.append(f"<b style='color:#03DAC6;'>AI Copilot:</b> {response.replace(chr(10), '<br>')}<br><br>")
        self.chat_history.verticalScrollBar().setValue(self.chat_history.verticalScrollBar().maximum())

    def export_pdf(self):
        if not self.metrics:
            QMessageBox.warning(self, "Error", "No data to export. Run the pipeline first.")
            return
            
        path, _ = QFileDialog.getSaveFileName(self, "Save PDF Report", "output/Executive_Report.pdf", "PDF Files (*.pdf)")
        if path:
            c = canvas.Canvas(path)
            c.drawString(100, 800, "AI Business Intelligence - Executive Report")
            c.drawString(100, 780, f"Total Analyzed: {self.metrics['Total_Analyzed']}")
            c.drawString(100, 760, f"Customer Satisfaction: {self.metrics['Customer_Satisfaction']}%")
            c.drawString(100, 740, f"Top Complaints: {', '.join(self.metrics['Top_Negative_Words'])}")
            c.drawString(100, 720, f"Top Strengths: {', '.join(self.metrics['Top_Positive_Words'])}")
            c.save()
            QMessageBox.information(self, "Success", "PDF Report Saved Successfully!")