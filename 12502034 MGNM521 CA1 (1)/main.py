import sys
import os
from PySide6.QtWidgets import QApplication
from ui.main_window import MainWindow

if __name__ == "__main__":
    # Ensure directories exist
    os.makedirs('data', exist_ok=True)
    os.makedirs('output', exist_ok=True)
    os.makedirs('charts', exist_ok=True)

    app = QApplication(sys.argv)
    
    # Optional: Set global application font
    font = app.font()
    font.setFamily("Segoe UI")
    app.setFont(font)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())