import streamlit as st
import requests

# URL de ta Function en local
AZURE_FUNCTION_URL = "http://localhost:7071/api/recommend"

st.set_page_config(page_title="Recommandations Articles", layout="centered")
st.title("Newspaper Article Recommender (POC local)")

# Utilisateurs et historiques fixes (comme ton fichier)
demo_users = [10, 23, 42, 57, 89, 105, 123, 256, 512, 1024]
history_options = {
    "Historique A": [101, 204, 305, 408, 509],
    "Historique B": [110, 220, 330, 440, 550],
    "Historique C": [115, 215, 315, 415, 515],
    "Historique D": [120, 240, 360, 480, 600],
    "Historique E": [125, 250, 375, 500, 625],
    "Historique F": [130, 260, 390, 520, 650],
    "Historique G": [135, 270, 405, 540, 675],
    "Historique H": [140, 280, 420, 560, 700],
    "Historique I": [145, 290, 435, 580, 725],
    "Historique J": [150, 300, 450, 600, 750],
}

col1, col2 = st.columns(2)
with col1:
    user_id = st.selectbox("Choisir un utilisateur", demo_users, index=0)
with col2:
    history_label = st.selectbox("Choisir un historique", list(history_options.keys()), index=0)

user_history_list = history_options[history_label]
st.caption(f"Historique sélectionné pour {history_label}: {user_history_list}")

if st.button("Obtenir des recommandations"):
    payload = {
        # la Function convertit déjà en str, mais gardons des str pour éviter toute ambiguïté
        "user_id": user_id,
        "history": [int(x) for x in user_history_list],
        # k et alpha sont codés en dur côté Function, on ne les envoie pas
    }
    try:
        resp = requests.post(AZURE_FUNCTION_URL, json=payload, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            items = data.get("items", [])
            st.subheader(f"Top 5 recommandations")
            if not items:
                st.info("Aucune reco renvoyée.")
            else:
                # Affichage lisible (titre/metadata si présentes)
                for i, it in enumerate(items, start=1):
                    article_id = it.get("article_id")
                    score_h = it.get("score_hybrid")
                    st.markdown(f"**{i}. Article {article_id}** — score_hybrid={score_h:.3f}")
                    # extra fields (si fournis par le CSV)
                    extra = {k: v for k, v in it.items() if k not in {"article_id","score_cb","score_cf","score_hybrid"}}
                    if extra:
                        with st.expander("Détails"):
                            st.json(extra)
        else:
            st.error(f"Erreur Function: {resp.status_code}\n{resp.text}")
    except Exception as e:
        st.error(f"Échec de l’appel: {e}")
