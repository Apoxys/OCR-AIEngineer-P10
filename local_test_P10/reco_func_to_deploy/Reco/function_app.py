# function_app.py
# Azure Functions v4 (Python 3.11, decorators)
# Logique identique au notebook : CB via 1 - cosine distance, hybride = alpha*CF + (1-alpha)*CB
# k=5 et alpha=0.6 codés en dur.

import io
import os
import re
import json
import pickle
import binascii
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import azure.functions as func
from sklearn.neighbors import NearestNeighbors


# -----------------------------------------------------------------------------
# Function App (decorators v4)
# -----------------------------------------------------------------------------
app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

# -----------------------------------------------------------------------------
# Cache de process (évite de recharger à chaque requête)
# -----------------------------------------------------------------------------
CACHE: Dict[str, Any] = {
    "embeddings": None,   # np.ndarray (N, D) float32
    "nn_index":   None,   # sklearn NearestNeighbors avec .kneighbors
    "meta_df":    None,   # DataFrame contenant article_id (pas de tri forcé)
    "cf_model":   None,   # Surprise algo avec .predict(uid, iid).est
    "id2idx":     None,   # dict[str -> int] (article_id -> index dans embeddings)
    "emb_ids":    None,   # list[str] mapping index -> article_id (si long. match CSV)
    "n_articles": None
}

# -----------------------------------------------------------------------------
# Constantes métier (codées en dur)
# -----------------------------------------------------------------------------
K_DEFAULT = 5
ALPHA_DEFAULT = 0.6

# -----------------------------------------------------------------------------
# Utils simples (pas de normalisation / pas de tri)
# -----------------------------------------------------------------------------
def _load_pickle(raw: bytes) -> Any:
    # Les blobs binaires sont fournis bruts (grâce à data_type=BINARY)
    # On supporte aussi la signature .npy au cas où :
    if raw.startswith(b"\x93NUMPY"):
        return np.load(io.BytesIO(raw), allow_pickle=True)
    return pickle.loads(raw)

def _load_metadata_csv(raw: bytes) -> pd.DataFrame:
    # On ne modifie pas l'ordre (pas de tri caché)
    df = pd.read_csv(io.BytesIO(raw))
    if "article_id" not in df.columns:
        for c in ("id", "articleId", "item_id"):
            if c in df.columns:
                df = df.rename(columns={c: "article_id"})
                break
        else:
            raise ValueError("La colonne 'article_id' est absente du CSV.")
    df["article_id"] = df["article_id"].astype(str)
    return df


def build_user_profile(user_clicks, embeddings, data_articles):
    """
    user_clicks : liste d'article_id déjà vus (ex. [157541, 280367, 71301])
    Retourne un vecteur moyen des embeddings correspondants.
    Si aucun clic, renvoie un vecteur nul de même dimension.
    """
    if len(user_clicks) == 0:
        # Cold-start user : vecteur nul
        return np.zeros(embeddings.shape[1])

    # Trouver les indices dans data_articles pour chaque article_id
    idxs = data_articles.index[data_articles["article_id"].isin(user_clicks)].tolist()
    if len(idxs) == 0:
        # Aucun match (utilisateur a cliqué sur des articles hors data_articles)
        return np.zeros(embeddings.shape[1])

    user_embs = embeddings[idxs]
    return user_embs.mean(axis=0)

def normalize_minmax(array):
    """
    Ramène array dans [0,1] par un simple min-max scaling.
    Si array.min() == array.max(), on renvoie un vecteur constant à 0.5.
    """
    mn = array.min()
    mx = array.max()
    if mx > mn:
        return (array - mn) / (mx - mn)
    else:
        return np.full_like(array, 0.5, dtype=float)

def score_cf_for_candidates(user_id, candidate_ids, cf_model):
    """
    user_id : int
    candidate_ids : liste d'int (article_id)
    Retourne un numpy array de score CF brute : cf_algo.predict(user_id, iid).est
    """
    cf_scores = []
    for iid in candidate_ids:
        # on met r_ui=None car on ne connaît pas la vraie note
        pred = cf_model.predict(uid=user_id, iid=iid, r_ui=None, verbose=False)
        cf_scores.append(pred.est)
    return np.array(cf_scores)


def recommend_hybrid(user_id, 
                     user_clicks,
                     embeddings, 
                     data_articles,
                     cf_model,
                     nn_index, 
                     k=5, 
                     alpha=0.6):
    """
    Renvoie un DataFrame pandas des k articles recommandés pour user_id,
    en blendant le score CB (similarité cos) et le score CF (prediction SVD).
    
    user_id        : int
    user_clicks    : liste d'article_id déjà cliqués
    k              : nombre d’articles à retourner
    alpha          : poids du CF (0 <= alpha <= 1). ex. 0.5 pour 50% CF / 50% CB
    total_candidates : taille du pool initial de candidats CB
    
    Sortie : DataFrame contenant [
        article_id, category, publisher, words_count, created_at,
        score_cb, score_cf, score_hybrid
    ]
    """
    # ----- 1. Calculer le profil CB de l'utilisateur -----
    user_vec = build_user_profile(user_clicks,embeddings, data_articles)  # vecteur 250-d
    
    # ----- 2. Récupérer le pool initial via NN sur embeddings -----

    total_candidates = len(embeddings) # number of candidates for recommendation based on the number of articles embedded
    distances, indices = nn_index.kneighbors([user_vec], n_neighbors=total_candidates)
    cand_idxs = indices[0]                # indices dans data_articles/embeddings
    sims_cb = 1.0 - distances[0]          # cosinus similarity = 1 - distance
    
    # IDs des articles candidats
    candidate_ids = data_articles.iloc[cand_idxs]["article_id"].tolist()
    
    # ----- 3. Construire le DataFrame brut des candidats -----
    df_cand = pd.DataFrame({
        "article_id": candidate_ids,
        "score_cb": sims_cb
    })
    
    # ----- 4. Calculer le score CF brut (cf_algo.predict) -----
    
    #ensure all types are respected
    user_id = int(user_id)
    candidate_ids = [int(cand) for cand in candidate_ids]
    #print('user_id type :', type(user_id))
    #print('candidate_ids type :',type(candidate_ids[2]))

    raw_cf = score_cf_for_candidates(user_id, candidate_ids, cf_model)
    df_cand["score_cf_raw"] = raw_cf
    print(raw_cf)
    # ----- 5. Normaliser le score CF en [0,1] -----
    df_cand["score_cf"] = normalize_minmax(df_cand["score_cf_raw"].values)
    
    # ----- 6. Calculer le score hybride -----
    df_cand["score_hybrid"] = alpha * df_cand["score_cf"] + (1 - alpha) * df_cand["score_cb"]
    
    # ----- 7. Trier par score_hybrid et prendre les top-k -----
    topk = (
        df_cand
        .sort_values("score_hybrid", ascending=False)
        .head(k)
        .merge(
            data_articles,
            on="article_id",
            how="left"
        )
    )
    
    # ----- 8. Sélection / ordre des colonnes à renvoyer -----
    return topk[[
        "article_id",
        "category_id",
        "publisher_id",
        "words_count",
        "created_at_ts",
        "score_cb",
        "score_cf",
        "score_hybrid"
    ]].reset_index(drop=True)

# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------

# GET de debug pour vérifier qu'on lit les bons blobs (compte/tailles/headers)
@app.function_name(name="DebugBlobs")
@app.route(route="debug-blobs", methods=["GET"])
@app.blob_input(arg_name="embeddings_blob",
                path="https://storageproject10.blob.core.windows.net/reco-assets/articles_embeddings.pickle",
                connection="AzureWebJobsStorage",
                data_type=func.DataType.BINARY)
@app.blob_input(arg_name="index_blob",
                path="https://storageproject10.blob.core.windows.net/reco-assets/nn_index.pkl",
                connection="AzureWebJobsStorage",
                data_type=func.DataType.BINARY)
@app.blob_input(arg_name="metadata_blob",
                path="https://storageproject10.blob.core.windows.net/reco-assets/articles_metadata.csv",
                connection="AzureWebJobsStorage")
@app.blob_input(arg_name="cf_blob",
                path="https://storageproject10.blob.core.windows.net/reco-assets/cf_model.pkl",
                connection="AzureWebJobsStorage",
                data_type=func.DataType.BINARY)
def debug_blobs(req: func.HttpRequest,
                embeddings_blob: func.InputStream,
                index_blob: func.InputStream,
                metadata_blob: func.InputStream,
                cf_blob: func.InputStream) -> func.HttpResponse:
    def info(stream, kind):
        if stream is None:
            return {"present": False}
        try:
            head = stream.read(16)
            length = getattr(stream, "length", None)
            name = getattr(stream, "name", None)
            if kind == "csv":
                return {"present": True, "name": name, "length": length}
            return {"present": True, "name": name, "length": length,
                    "head_hex": binascii.hexlify(head).decode("ascii")}
        except Exception as e:
            return {"present": True, "error": str(e)}

    # On expose aussi le nom de compte issu de la connection string
    cs = os.getenv("AzureWebJobsStorage", "")
    m = re.search(r"AccountName=([^;]+)", cs, re.I)
    account = m.group(1) if m else "unknown"

    payload = {
        "storage_account": account,
        "articles_embeddings.pickle": info(embeddings_blob, "pkl"),
        "nn_index.pkl":               info(index_blob, "pkl"),
        "articles_metadata.csv":      info(metadata_blob, "csv"),
        "cf_model.pkl":               info(cf_blob, "pkl")
    }
    return func.HttpResponse(json.dumps(payload, ensure_ascii=False, indent=2),
                             mimetype="application/json")


# POST /recommend (k=5, alpha=0.6)
@app.function_name(name="Recommend")
@app.route(route="recommend", methods=["POST"])
@app.blob_input(arg_name="embeddings_blob",
                path="https://storageproject10.blob.core.windows.net/reco-assets/articles_embeddings.pickle",
                connection="AzureWebJobsStorage",
                data_type=func.DataType.BINARY)
@app.blob_input(arg_name="index_blob",
                path="https://storageproject10.blob.core.windows.net/reco-assets/nn_index.pkl",
                connection="AzureWebJobsStorage",
                data_type=func.DataType.BINARY)
@app.blob_input(arg_name="metadata_blob",
                path="https://storageproject10.blob.core.windows.net/reco-assets/articles_metadata.csv",
                connection="AzureWebJobsStorage")
@app.blob_input(arg_name="cf_blob",
                path="https://storageproject10.blob.core.windows.net/reco-assets/cf_model.pkl",
                connection="AzureWebJobsStorage",
                data_type=func.DataType.BINARY)
def recommend(req: func.HttpRequest,
              embeddings_blob: func.InputStream,
              index_blob: func.InputStream,
              metadata_blob: func.InputStream,
              cf_blob: func.InputStream) -> func.HttpResponse:

    # 0) Lire payload
    try:
        body = req.get_json()
        user_id = str(body.get("user_id"))
        history = body.get("history", [])
        if not isinstance(history, list):
            raise ValueError
        history = [str(h) for h in history]
    except Exception:
        return func.HttpResponse(
            json.dumps({"error": "Payload invalide. Attendu: {user_id, history:[ids]}."}),
            status_code=400, mimetype="application/json"
        )
    if not user_id:
        return func.HttpResponse(json.dumps({"error": "user_id requis"}), status_code=400,
                                 mimetype="application/json")

    # 1) Charger artefacts (une seule fois)
    if CACHE["embeddings"] is None:
        arr = _load_pickle(embeddings_blob.read())
        if not isinstance(arr, np.ndarray):
            return func.HttpResponse(json.dumps({"error": "embeddings doit être un numpy.ndarray"}), status_code=500,
                                     mimetype="application/json")
        CACHE["embeddings"] = arr.astype(np.float32, copy=False)
        CACHE["n_articles"] = len(arr)

    if CACHE["meta_df"] is None:
        meta_df = _load_metadata_csv(metadata_blob.read())
        CACHE["meta_df"] = meta_df
        # mapping index -> article_id si tailles identiques
        if len(meta_df) == CACHE["n_articles"]:
            ids = meta_df["article_id"].astype(str).tolist()
        else:
            ids = [str(i) for i in range(CACHE["n_articles"])]
        CACHE["emb_ids"] = ids
        CACHE["id2idx"] = {aid: i for i, aid in enumerate(ids)}

    if CACHE["nn_index"] is None:
        idx_obj = _load_pickle(index_blob.read())
        if not hasattr(idx_obj, "kneighbors"):
            return func.HttpResponse(json.dumps({"error": "nn_index.pkl doit exposer .kneighbors (NearestNeighbors attendu)."}),
                                     status_code=500, mimetype="application/json")
        CACHE["nn_index"] = idx_obj

    if CACHE["cf_model"] is None:
        CACHE["cf_model"] = _load_pickle(cf_blob.read())

    # 2) Appliquer la logique métier **comme dans le notebook**
    items = recommend_hybrid(
        user_id=user_id,
        user_clicks=history,
        embeddings=CACHE["embeddings"], 
        data_articles=CACHE["meta_df"],
        cf_model=CACHE["cf_model"],
        nn_index = CACHE["nn_index"],
        k=K_DEFAULT,
        alpha=ALPHA_DEFAULT
    )
    print(items)
    count = len(items) #should be 5
    itemsDict = items.to_dict(orient="records") #to serialize correctly

    resp = {
        "user_id": user_id,
        "count": count,
        "items": itemsDict,
        "meta": {
            "k": K_DEFAULT,
            "alpha": ALPHA_DEFAULT,
            "n_articles": CACHE["n_articles"],
            "index_type": "kneighbors"
        }
    }
    return func.HttpResponse(json.dumps(resp), status_code=200, mimetype="application/json")
