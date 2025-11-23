# Documentation Technique - Détails d'Implémentation

Ce document détaille les briques logiques essentielles du projet **ATLAS-analytics**. Il isole les parties de code critiques pour expliquer le fonctionnement interne et justifier les choix d'architecture.

---

## 1. Pipeline de Nettoyage (`src/cleaning.py`)

Le nettoyage est la première étape critique. Des données bruyantes (URLs, emojis, fautes) dégradent la performance du LLM.

```python
def pipeline_cleaning(text: str) -> Dict[str, str]:
    """
    Pipeline complet de nettoyage
    """
    raw = safe_str(text)
    
    # 1. Extraction des emojis (pour analyse de sentiment potentielle)
    emojis = extract_emojis(raw)
    
    # 2. Nettoyage et Traduction
    # - Suppression des URLs et mentions (@user)
    # - Détection de la langue
    # - Traduction en Français si nécessaire (via deep-translator)
    lang, t_fr_raw, t_clean = cleaning_with_translation(raw)
    
    return {
        "lang": lang,
        "text_translated_fr": t_fr_raw, # Texte brut traduit (pour le LLM)
        "text_clean": t_clean,          # Texte nettoyé (pour affichage propre)
        "emojis": emojis
    }
```

### 🔍 Explication
Cette fonction orchestre plusieurs sous-tâches :
1.  **Sécurisation** : `safe_str` gère les valeurs nulles ou non-string.
2.  **Extraction** : On garde les emojis de côté car ils sont de forts marqueurs d'émotion, même si on les retire du texte principal pour simplifier la lecture.
3.  **Traduction** : Si un tweet est en anglais ou espagnol, il est traduit pour que le modèle (et l'analyste) travaille sur une base unifiée en français.

### 💡 Pourquoi ce choix ?
*   **Séparation des concerns** : On sépare le texte "pour la machine" (traduit, brut) du texte "pour l'affichage" (nettoyé, sans URLs).
*   **Robustesse** : Gérer la langue en amont évite d'envoyer du bruit au LLM ou de lui demander de traduire, ce qui consommerait plus de tokens et diluerait son attention sur la classification.

---

## 2. Classification via LLM (`src/llm_classification.py`)

C'est le cœur intelligent du système. Nous utilisons l'API Mistral pour classer les tweets.

```python
@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type((LLMClassificationError, Exception)),
)
def classify_single_tweet(client: Mistral, tweet_text: str) -> str:
    """
    Classifie un seul tweet avec mécanisme de retry
    """
    try:
        # Construction du prompt avec le tweet
        prompt = create_prompt(tweet_text)
        
        # Appel API Mistral
        response = client.chat.complete(
            model="mistral-tiny", # Modèle optimisé pour la vitesse/coût
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1, # Très faible pour des résultats constants
            max_tokens=200
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        logger.error(f"Erreur classification tweet: {e}")
        raise LLMClassificationError(str(e))
```

### 🔍 Explication
*   **Décorateur `@retry`** : Utilise la librairie `tenacity`. Si l'API échoue (timeout, erreur 500), le code attend un peu (backoff exponentiel) et réessaie jusqu'à 3 fois.
*   **Température 0.1** : On veut de la classification, pas de la poésie. Une température basse force le modèle à être déterministe et factuel.
*   **Modèle `mistral-tiny`** : Suffisant pour de la classification simple, beaucoup moins cher et plus rapide que les gros modèles.

### 💡 Pourquoi ce choix ?
*   **Fiabilité** : Les appels réseaux sont instables par nature. Sans mécanisme de retry, un batch de 1000 tweets planterait au moindre pépin réseau.
*   **Coût/Perf** : Le choix du modèle et des paramètres (max_tokens) est optimisé pour traiter de gros volumes sans exploser le budget.

---

## 3. Parsing Robuste (`src/parse_llm_outputs.py`)

Les LLM ne renvoient pas toujours du JSON parfait. Ce module nettoie la sortie.

```python
def parse_llm_response(raw_response: Optional[str]) -> Dict[str, str]:
    """
    Parse une réponse LLM de manière sécurisée avec fallback
    """
    try:
        # 1. Extraction chirurgicale du JSON
        # Cherche le pattern {...} même s'il y a du texte autour
        json_str = extract_json_from_text(raw_response)
        
        if not json_str:
            return DEFAULT_VALUES.copy()
        
        parsed = json.loads(json_str)
        
        # 2. Normalisation stricte
        # Vérifie que "motif" est bien dans notre liste prédéfinie
        motif = normalize_value(parsed.get("motif"), VALID_MOTIFS, "Autre")
        
        # ... idem pour sentiment, urgence, churn ...
        
        return {
            "motif": motif,
            "sentiment": sentiment,
            # ...
        }
    except Exception:
        # 3. Fallback (Filet de sécurité)
        return DEFAULT_VALUES.copy()
```

### 🔍 Explication
*   **Extraction Regex** : Le LLM peut dire "Voici le JSON : { ... }". `json.loads` échouerait sur la phrase complète. La regex va chercher uniquement la partie `{...}`.
*   **Normalisation** : Si le LLM hallucine un motif "Problème Wifi" alors que notre catégorie est "Technique", ou écrit "Positif" avec une majuscule, `normalize_value` corrige cela pour garder des données propres.

### 💡 Pourquoi ce choix ?
*   **Qualité des données** : Pour faire des graphiques (camemberts, barres), il faut des catégories exactes. On ne peut pas avoir "Technique", "technique" et "Pb technique" comme 3 catégories différentes. Ce module garantit l'intégrité des données.

---

## 4. Orchestration (`src/pipeline_enrichment.py`)

Ce script lie tout ensemble et gère le traitement par lots (batching).

```python
def enrich_with_llm(df, ...):
    # ...
    
    # Traitement par lots (Batching)
    # On envoie les tweets par paquets (ex: 10 par 10)
    batch_results = classify_dataframe_batch(client, texts, batch_size=LLM_BATCH_SIZE)
    
    # Parsing des résultats
    parsed_results = parse_batch_responses(batch_results)
    
    # Fusion avec le DataFrame original
    for idx, parsed in zip(df.index, parsed_results):
        for col in llm_columns:
            df.loc[idx, col] = parsed[col]
            
    # Sauvegarde intermédiaire (Checkpoint)
    if checkpoint_path:
        save_dataframe(df, checkpoint_path)
        
    return df
```

### 🔍 Explication
*   **Batching** : Au lieu d'appeler l'API tweet par tweet (trop lent) ou tout d'un coup (trop gros), on traite par petits groupes.
*   **Checkpointing** : Si le script plante après 5000 tweets sur 10000, on sauvegarde l'état. Au prochain lancement, on ne reprend que les 5000 restants.

### 💡 Pourquoi ce choix ?
*   **Performance** : Le batching permet de paralléliser (côté API) et de réduire l'overhead réseau.
*   **Résilience** : Le checkpointing est indispensable pour les longs traitements (plusieurs heures). On ne veut pas tout recommencer à zéro en cas de coupure.

---

## 5. Interface Frontend (`frontend/src/components/KPIGrid.tsx`)

Visualisation des données enrichies pour l'utilisateur final.

```tsx
export function KPIGrid({ data }: KPIGridProps) {
    // État de chargement
    if (!data) return <div className="animate-pulse">Chargement...</div>;

    return (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            {/* Carte KPI simple */}
            <KPICard 
                title="Total Tweets" 
                value={data.total_tweets.toLocaleString()} 
            />
            
            {/* Carte KPI avec indicateur couleur */}
            <KPICard
                title="Tweets Négatifs"
                value={data.negatifs.toLocaleString()}
                sub={`${data.negatifs_pct}% du total`}
                color="text-red-500" // Rouge pour alerter
            />
            
            {/* ... */}
        </div>
    );
}
```

### 🔍 Explication
*   **Composants Réutilisables** : `KPICard` est défini une fois et réutilisé pour chaque métrique, assurant une cohérence visuelle.
*   **Typage TypeScript** : `KPIGridProps` définit exactement quelles données sont attendues, évitant les bugs d'affichage si l'API change.

### 💡 Pourquoi ce choix ?
*   **Expérience Utilisateur** : L'interface doit être immédiate. Les couleurs (rouge/vert) guident l'œil vers l'information importante (négatif/positif) sans effort cognitif.
