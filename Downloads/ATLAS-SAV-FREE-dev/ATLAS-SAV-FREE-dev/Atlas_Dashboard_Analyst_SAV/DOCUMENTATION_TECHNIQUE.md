# Documentation Technique - Prototype Analyse SAV

## 1. Présentation du prototype

### Objectifs
Ce prototype a pour but de démontrer la faisabilité et l'efficacité d'un traitement automatisé des tweets clients via des techniques modernes de LLM (Large Language Models). L'objectif est de transformer des données textuelles brutes et non structurées en insights exploitables pour le service client (SAV).

### Étapes couvertes
Le pipeline de données couvre l'intégralité du flux de traitement :
1.  **Lecture** : Importation des données brutes depuis un fichier CSV (`free_tweet_export.csv`).
2.  **Nettoyage** : Prétraitement du texte (suppression des URLs, emojis, normalisation, traduction si nécessaire).
3.  **Classification LLM** : Envoi des tweets par batchs à l'API Mistral AI pour analyse sémantique.
4.  **Extraction** : Parsing robuste des réponses JSON pour extraire le motif, le sentiment, l'urgence et le risque de churn.
5.  **Restitution** : Sauvegarde des données enrichies (Parquet) et visualisation via un dashboard interactif (React + FastAPI).

---

## 2. Environnement technique

### Langages et Bibliothèques
*   **Langage** : Python 3.9+
*   **Data Science** : `pandas`, `numpy`
*   **IA / LLM** : `mistralai` (Client API), `tenacity` (Gestion des retries)
*   **Backend** : `fastapi`, `uvicorn`
*   **Frontend** : `React`, `TypeScript`, `Tailwind CSS`, `Recharts`

### Gestion de la Sécurité
La clé API Mistral (`MISTRAL_API_KEY`) n'est jamais stockée en dur dans le code. Elle est gérée via des variables d'environnement chargées depuis un fichier `.env` (exclu du contrôle de version via `.gitignore`).

### Architecture Logique
```text
[Données Brutes (CSV)] 
       ⬇
[Pipeline Python (Nettoyage + LLM Mistral)] 
       ⬇
[Données Enrichies (Parquet)] 
       ⬇
[API Backend (FastAPI)] ⬅ (Requêtes HTTP) ➡ [Frontend Dashboard (React)]
```

---

## 3. Extraits de code commentés

### A. Nettoyage des tweets
Cette fonction prépare le texte pour optimiser la compréhension du LLM et réduire le bruit.

```python
# src/cleaning.py

def pipeline_cleaning(text: str) -> Dict[str, str]:
    """
    Pipeline complet de nettoyage
    """
    raw = safe_str(text)
    # Extraction des emojis pour analyse potentielle du sentiment
    emojis = extract_emojis(raw)
    
    # Nettoyage et traduction (si nécessaire)
    # On supprime les URLs, mentions, et on normalise la casse
    lang, t_fr_raw, t_clean = cleaning_with_translation(raw)
    
    return {
        "lang": lang,
        "text_translated_fr": t_fr_raw, # Texte traduit pour le LLM
        "text_clean": t_clean,          # Texte nettoyé pour affichage
        "emojis": emojis
    }
```
*Choix technique* : Séparation du texte traduit et du texte nettoyé pour garder la trace de la langue d'origine tout en fournissant un texte propre au modèle.

### B. Appel au LLM (Classification)
Utilisation de Mistral AI avec une gestion robuste des erreurs (retries) pour garantir la fiabilité du batch processing.

```python
# src/llm_classification.py

@retry(
    reraise=True,
    stop=stop_after_attempt(3), # 3 tentatives max
    wait=wait_exponential(multiplier=1, min=2, max=30), # Backoff exponentiel
    retry=retry_if_exception_type((LLMClassificationError, Exception)),
)
def classify_single_tweet(client: Mistral, tweet_text: str) -> str:
    """
    Classifie un seul tweet avec retry
    """
    try:
        prompt = create_prompt(tweet_text)
        
        # Appel API avec température faible pour des résultats déterministes
        response = client.chat.complete(
            model="mistral-tiny", # Modèle rapide et efficace
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=200
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        logger.error(f"Erreur classification tweet: {e}")
        raise LLMClassificationError(str(e))
```
*Choix technique* : Utilisation de `tenacity` pour gérer les échecs transitoires de l'API et garantir que le pipeline ne plante pas sur un seul tweet.

### C. Parsing Robuste
Transformation de la réponse textuelle (parfois imparfaite) du LLM en données structurées fiables.

```python
# src/parse_llm_outputs.py

def parse_llm_response(raw_response: Optional[str]) -> Dict[str, str]:
    """
    Parse une réponse LLM de manière sécurisée avec fallback
    """
    try:
        # Tentative d'extraction du JSON (même si entouré de texte)
        json_str = extract_json_from_text(raw_response)
        
        if not json_str:
            return DEFAULT_VALUES.copy()
        
        parsed = json.loads(json_str)
        
        # Normalisation des valeurs (ex: "Positif" -> "positif")
        # Utilise des listes de valeurs valides pour éviter les hallucinations
        motif = normalize_value(parsed.get("motif"), VALID_MOTIFS, "Autre")
        sentiment = normalize_value(parsed.get("sentiment"), VALID_SENTIMENTS, "neutre")
        # ... (idem pour urgence et churn)
        
        return {
            "motif": motif,
            "sentiment": sentiment,
            # ...
        }
    except Exception:
        # Fallback en cas d'erreur critique
        return DEFAULT_VALUES.copy()
```
*Choix technique* : Le LLM peut parfois "bavarder" ou faire des erreurs de formatage. Ce parser utilise des expressions régulières et une normalisation stricte pour garantir la qualité des données en sortie.

### D. Enregistrement des résultats
Sauvegarde efficace des données enrichies pour une utilisation ultérieure par le dashboard.

```python
# src/pipeline_enrichment.py

# Conversion des résultats en DataFrame
df_results = pd.DataFrame(parsed_results)

# Fusion avec le DataFrame original
df_final = pd.concat([df, df_results], axis=1)

# Sauvegarde en format Parquet (plus performant que CSV pour la lecture/écriture)
df_final.to_parquet(output_path, index=False)
logger.info(f"Données sauvegardées : {output_path}")
```
*Choix technique* : Le format Parquet est choisi pour sa compression et sa rapidité de lecture, essentielles pour un dashboard réactif.

### E. Affichage Frontend (React)
Composant pour afficher les indicateurs clés de performance (KPIs).

```tsx
// frontend/src/components/KPIGrid.tsx

export function KPIGrid({ data }: KPIGridProps) {
    if (!data) return <div>Chargement...</div>;

    return (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <KPICard 
                title="Total Tweets" 
                value={data.total_tweets.toLocaleString()} 
            />
            <KPICard
                title="Tweets Négatifs"
                value={data.negatifs.toLocaleString()}
                sub={`${data.negatifs_pct}% du total`}
                color="text-red-500"
            />
            {/* Autres cartes KPI... */}
        </div>
    );
}
```
*Choix technique* : Composants React modulaires et typés (TypeScript) pour une maintenance aisée et une interface utilisateur fluide.

---

## 4. Screenshots ou visuels des résultats

### Exemple 1 — Tweet analysé
[INSÉRER CAPTURE OU EXEMPLE DE SORTIE ICI]

Tweet original :
"Mon internet ne marche plus depuis ce matin, c'est inadmissible !"

Sortie LLM :
motif = "Technique"
sentiment = "négatif"
urgence = "élevée"
churn = "faible"

### Exemple 2 — Aperçu Dashboard
[INSÉRER CAPTURE DU DASHBOARD ICI]

*Vue d'ensemble montrant les KPIs, le nuage de mots et l'évolution du volume.*

### Exemple 3 — Données Enrichies (CSV/Parquet)
[INSÉRER CAPTURE D'UN TABLEUR ICI]

| date | full_text | motif | sentiment | urgence |
|------|-----------|-------|-----------|---------|
| 2023-10-01 | Plus de réseau 4G... | Réseau | négatif | moyenne |
| 2023-10-01 | Merci pour l'aide ! | Service client | positif | faible |

---

## 5. Limites du prototype et perspectives

### Limites actuelles
1.  **Coût LLM** : L'utilisation d'une API commerciale (Mistral) a un coût par token. Pour des millions de tweets, cela peut devenir significatif.
2.  **Lenteur du Batching** : Le traitement séquentiel (même batché) prend du temps. Le traitement de 10 000 tweets peut prendre plusieurs dizaines de minutes.
3.  **Nettoyage non exhaustif** : Le nettoyage actuel est généraliste. Il pourrait être affiné pour le jargon spécifique des télécoms.
4.  **Erreurs JSON** : Malgré le parsing robuste, le LLM échoue parfois totalement à produire un JSON valide (~1-2% des cas).

### Perspectives réalistes
1.  **Traitement en masse (Big Data)** : Migrer vers une architecture distribuée (Spark) ou utiliser des modèles locaux quantifiés (ex: Mistral-7B-Quantized) pour réduire les coûts et augmenter la vitesse.
2.  **Création d'une API Temps Réel** : Transformer le script batch en une API qui classifie les tweets au fil de l'eau dès leur réception.
3.  **Multi-tâches** : Entraîner un modèle spécifique (Fine-tuning) pour améliorer la précision sur les motifs "métier" spécifiques à l'opérateur.
4.  **Intégration CRM** : Connecter directement les tweets classifiés "Urgents" ou "Risque Churn" à l'outil de gestion de tickets du SAV (Zendesk, Salesforce).
