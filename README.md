# EMS SmartGrid — EV + PV + BESS (Streamlit)

App Streamlit pour simuler une journée EMS :
- PV / charge bâtiment / recharge EV
- Batterie (BESS) bornée entre 20% et 80% (ne monte pas à 100%)
- KPI + courbes + audit contraintes

## Lancer en local
```bash
pip install -r requirements.txt
streamlit run app.py