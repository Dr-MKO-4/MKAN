# -*- coding: utf-8 -*-
import sys
sys.stdout.reconfigure(encoding='utf-8')
"""
Patch train.ipynb pour ajouter la reprise après interruption :
  - Recherche heuristique : sauvegarde search_state.json + HTML des graphiques
  - Boucle d'entraînement : sauvegarde training_state.json à chaque epoch
  - Nouvelle cellule bypass-training (chargement checkpoint + état)
"""
import json, copy, sys

NOTEBOOK = r"m:\Ecole\Mémoire\Modelisation\MKAN\train.ipynb"

with open(NOTEBOOK, encoding="utf-8") as f:
    nb = json.load(f)

def src(code: str):
    """Convertit un bloc de code en liste de lignes pour le format notebook."""
    lines = code.split("\n")
    return [l + "\n" for l in lines[:-1]] + [lines[-1]]

def find_cell(cells, cell_id):
    for i, c in enumerate(cells):
        if c.get("id") == cell_id:
            return i, c
    return None, None

cells = nb["cells"]

# ─────────────────────────────────────────────────────────────────────────────
# 1. Cell d9e72ba6  recherche heuristique : ajouter sauvegarde search_state
# ─────────────────────────────────────────────────────────────────────────────
idx, cell = find_cell(cells, "d9e72ba6")
assert cell is not None, "Cellule d9e72ba6 introuvable"

SAVE_SEARCH = """
# ── Sauvegarde de l état de la recherche (replay graphiques post-interruption)
import json as _json
_search_state_path = os.path.join(CHECKPOINT_DIR, 'search_state.json')
try:
    _resume_records = search.resume().to_dict(orient='records')
    with open(_search_state_path, 'w', encoding='utf-8') as _f:
        _json.dump({
            'best_hp': {'score': best_hp['score'], 'params': best_hp['params']},
            'resume':  _resume_records,
        }, _f, ensure_ascii=False, indent=2, default=str)
    print(f'Etat de recherche sauvegarde : {_search_state_path}')
except Exception as _e:
    print(f'Avertissement : impossible de sauvegarder search_state : {_e}')"""

orig_src = "".join(cell["source"])
if "search_state_path" not in orig_src:
    cell["source"] = src(orig_src + SAVE_SEARCH)
    print("✓ Cellule d9e72ba6 : sauvegarde search_state ajoutée")
else:
    print("~ Cellule d9e72ba6 : déjà patchée")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Cell 3861a116  visualisations recherche : write_html + bypass si search absent
# ─────────────────────────────────────────────────────────────────────────────
idx, cell = find_cell(cells, "3861a116")
assert cell is not None, "Cellule 3861a116 introuvable"

NEW_3861 = """\
# Résumé tabulaire de la progression + graphiques (bypass si search non défini)
import json as _json, os, pandas as _pd
import plotly.io as _pio

_search_state_path = os.path.join(CHECKPOINT_DIR, 'search_state.json')

if 'search' in dir() and hasattr(search, 'resume'):
    # ── Cas normal : search est en mémoire ─────────────────────────────────
    cols_resume  = ['generation', 'score', 'diversite', 'mutation_rate',
                    'hidden_size', 'M', 'K', 'lam', 'lr']
    cols_present = [c for c in cols_resume if c in search.resume().columns]
    print("Progression par génération :")
    print(search.resume()[cols_present].to_string(index=False))

    fig_conv  = search.plot_convergence()
    fig_param = search.plot_parameter_space()
    fig_div   = search.plot_diversity()

    fig_conv.write_html(os.path.join(CHECKPOINT_DIR,  'search_convergence.html'))
    fig_param.write_html(os.path.join(CHECKPOINT_DIR, 'search_param_space.html'))
    fig_div.write_html(os.path.join(CHECKPOINT_DIR,   'search_diversity.html'))
    print("Graphiques sauvegardés dans checkpoints/")

    fig_conv.show()
    fig_param.show()
    fig_div.show()

elif os.path.exists(_search_state_path):
    # ── Bypass : rechargement depuis search_state.json ─────────────────────
    print(f"Chargement depuis {_search_state_path} …")
    with open(_search_state_path, encoding='utf-8') as _f:
        _state = _json.load(_f)
    _df = _pd.DataFrame(_state['resume'])
    cols_resume  = ['generation', 'score', 'diversite', 'mutation_rate',
                    'hidden_size', 'M', 'K', 'lam', 'lr']
    cols_present = [c for c in cols_resume if c in _df.columns]
    print("Progression par génération :")
    print(_df[cols_present].to_string(index=False))

    # Rechargement HTML si déjà sauvegardés
    for _name, _fname in [('Convergence',    'search_convergence.html'),
                           ('Espace param',  'search_param_space.html'),
                           ('Diversité',     'search_diversity.html')]:
        _p = os.path.join(CHECKPOINT_DIR, _fname)
        if os.path.exists(_p):
            _fig = _pio.read_json(_p.replace('.html', '.json')) if os.path.exists(_p.replace('.html','.json')) else None
            print(f"  {_name} → ouvrir {_p} dans un navigateur")
        else:
            print(f"  {_name} → fichier HTML absent (graphique non disponible)")
else:
    print("search non défini et search_state.json introuvable.")
    print("Relancer les cellules de recherche heuristique pour générer les graphiques.")
"""

cell["source"] = src(NEW_3861)
cell["outputs"] = []
print("✓ Cellule 3861a116 : réécriture avec bypass et write_html")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Nouvelle cellule bypass-training  AVANT 2678c2ce
# ─────────────────────────────────────────────────────────────────────────────
idx_train, cell_train = find_cell(cells, "2678c2ce")
assert cell_train is not None, "Cellule 2678c2ce introuvable"

BYPASS_TRAINING_CODE = """\
# ══════════════════════════════════════════════════════════════════════════════
#  BYPASS / REPRISE ENTRAÎNEMENT
#  Exécuter cette cellule si l'entraînement a été interrompu (coupure courant,
#  kernel mort, etc.) PUIS exécuter la boucle d'entraînement ci-dessous.
#  La boucle reprendra automatiquement à la dernière epoch sauvegardée.
# ══════════════════════════════════════════════════════════════════════════════
import json as _json, os

_state_path = os.path.join(CHECKPOINT_DIR, 'training_state.json')
_ckpt_path  = os.path.join(CHECKPOINT_DIR, 'best_mkan.pt')

# ── Chargement du checkpoint ─────────────────────────────────────────────────
if os.path.exists(_ckpt_path):
    model.load_state_dict(
        torch.load(_ckpt_path, map_location=DEVICE, weights_only=True))
    _sz = os.path.getsize(_ckpt_path) // 1024
    print(f'Modele charge depuis {_ckpt_path}  ({_sz} KB)')
else:
    print('ERREUR : best_mkan.pt introuvable. Verifier CHECKPOINT_DIR.')

# ── Restauration de l état ───────────────────────────────────────────────────
if os.path.exists(_state_path):
    with open(_state_path, encoding='utf-8') as _f:
        _saved = _json.load(_f)
    history      = _saved['history']
    best_val_mcc = float(_saved['best_val_mcc'])
    best_epoch   = int(_saved['best_epoch'])
    start_epoch  = int(_saved['last_epoch']) + 1
    epochs_left  = N_EPOCHS - int(_saved['last_epoch'])
    print(f'Etat restaure : {int(_saved["last_epoch"])} epochs faites | '
          f'best val_MCC={best_val_mcc:.4f} (epoch {best_epoch})')
    if epochs_left > 0:
        print(f'  → {epochs_left} epoch(s) restante(s)  lancer la boucle ci-dessous')
    else:
        print(f'  → Entraînement COMPLET  passer directement a la section 7')
else:
    # Aucun état sauvegardé : warm-start depuis le checkpoint, tout rereunter
    print('training_state.json absent  historique non récupérable.')
    history = {"epoch": [], "loss": [], "pred_loss": [], "reg": [],
               "l1": [], "entropy": [],
               "val_mcc": [], "val_auc": [], "val_prauc": [], "val_brier": []}
    best_val_mcc = -1.0
    best_epoch   = 0
    start_epoch  = 1
    print(f'Warm-start depuis best_mkan.pt  relance de 1 a {N_EPOCHS} epochs.')
    print(f'Astuce : si tu etais a ~95% (38/40), change N_EPOCHS=5 dans la')
    print(f'  cellule de configuration pour ne faire que les epochs manquantes.')

optimizer = DMLAdam(model.parameters(), lr=LR)
print(f'\\nOptimiseur reconstruit (lr={LR}).  start_epoch = {start_epoch}')
"""

# Vérifier si la cellule bypass-training existe déjà
idx_bypass, _ = find_cell(cells, "bypass-training")
if idx_bypass is None:
    new_cell = {
        "id": "bypass-training",
        "cell_type": "code",
        "metadata": {},
        "source": src(BYPASS_TRAINING_CODE),
        "outputs": [],
        "execution_count": None,
    }
    cells.insert(idx_train, new_cell)
    print("✓ Cellule bypass-training insérée avant 2678c2ce")
else:
    cells[idx_bypass]["source"] = src(BYPASS_TRAINING_CODE)
    print("~ Cellule bypass-training : mise à jour")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Cell 2678c2ce  boucle d'entraînement : start_epoch + sauvegarde état
# ─────────────────────────────────────────────────────────────────────────────
idx_train2, cell_train2 = find_cell(cells, "2678c2ce")
assert cell_train2 is not None

NEW_TRAINING_LOOP = """\
from tqdm.auto import tqdm as _tqdm
import json as _json

# history, best_val_mcc, start_epoch sont fournis par la cellule bypass-training
# si celle-ci a été exécutée ; sinon on part de zéro.
if 'history' not in dir() or not isinstance(history, dict):
    history = {"epoch": [], "loss": [], "pred_loss": [], "reg": [],
               "l1": [], "entropy": [],
               "val_mcc": [], "val_auc": [], "val_prauc": [], "val_brier": []}
if 'best_val_mcc' not in dir():
    best_val_mcc = -1.0
if 'best_epoch' not in dir():
    best_epoch = 0
if 'start_epoch' not in dir():
    start_epoch = 1

_state_path = os.path.join(CHECKPOINT_DIR, 'training_state.json')
_params = list(model.parameters())

pbar = _tqdm(range(start_epoch, N_EPOCHS + 1), desc="Entraînement", unit="epoch")
for epoch in pbar:
    model.train()
    epoch_loss = epoch_pred = epoch_l1 = epoch_ent = 0.0
    n_batches  = 0

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)

        loss, pred_loss, reg_l1, reg_entropy = mkan_total_loss(
            model, X_batch, y_batch, lam=LAM, mu1=MU1, mu2=MU2)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(_params, max_norm=1.0)
        optimizer.step()

        epoch_loss += loss.item()
        epoch_pred += pred_loss.item()
        epoch_l1   += reg_l1.item()
        epoch_ent  += reg_entropy.item()
        n_batches  += 1

    avg_loss = epoch_loss / n_batches
    avg_pred = epoch_pred / n_batches
    avg_l1   = epoch_l1  / n_batches
    avg_ent  = epoch_ent / n_batches

    history["epoch"].append(epoch)
    history["loss"].append(avg_loss)
    history["pred_loss"].append(avg_pred)
    history["reg"].append(avg_loss - avg_pred)
    history["l1"].append(avg_l1)
    history["entropy"].append(avg_ent)

    val_metrics = evaluate(model, val_loader)
    history["val_mcc"].append(val_metrics["mcc"])
    history["val_auc"].append(val_metrics["auc"])
    history["val_prauc"].append(val_metrics["pr_auc"])
    history["val_brier"].append(val_metrics["brier"])

    if val_metrics["mcc"] > best_val_mcc:
        best_val_mcc = val_metrics["mcc"]
        best_epoch   = epoch
        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best_mkan.pt"))

    # Sauvegarde de l état complet après chaque epoch (reprise après interruption)
    with open(_state_path, 'w', encoding='utf-8') as _f:
        _json.dump({
            'history':      history,
            'best_val_mcc': best_val_mcc,
            'best_epoch':   best_epoch,
            'last_epoch':   epoch,
        }, _f)

    pbar.set_postfix({
        "loss":    f"{avg_loss:.4f}",
        "val_MCC": f"{val_metrics['mcc']:.4f}",
        "val_AUC": f"{val_metrics['auc']:.4f}",
        "★" if val_metrics["mcc"] == best_val_mcc else " ": "",
    })

print(f"\\nMeilleur val MCC = {best_val_mcc:.4f} (epoch {best_epoch})")
"""

cell_train2["source"] = src(NEW_TRAINING_LOOP)
cell_train2["outputs"] = [cell_train2["outputs"][0]] if cell_train2["outputs"] else []
print("✓ Cellule 2678c2ce : boucle d'entraînement mise à jour (start_epoch + state saving)")

# ─────────────────────────────────────────────────────────────────────────────
# Sauvegarde
# ─────────────────────────────────────────────────────────────────────────────
with open(NOTEBOOK, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("\n✅ Notebook patché et sauvegardé.")
