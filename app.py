"""
app.py
------
Entry point. Run with:  streamlit run app.py
"""

from ui import DashboardUI

if __name__ == "__main__" or True:      # Streamlit always executes top-level
    dashboard = DashboardUI()
    dashboard.render()
