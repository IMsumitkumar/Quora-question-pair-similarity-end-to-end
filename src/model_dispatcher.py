from sklearn import feature_extraction
from sklearn import linear_model
from sklearn import tree
import xgboost as xgb

vectorizers = {
    "raw_tfidf": feature_extraction.text.TfidfVectorizer(
        ngram_range = (1, 2),
        min_df = 10
    ),
    "count": feature_extraction.text.CountVectorizer(
        min_df = 10
    )
}

models = {
    'xgb': xgb.XGBClassifier(
        max_depth=8,
        learning_rate=0.2,
        objective='binary:logistic',
        min_child_weight=2,
        n_estimators=100,
        eval_metric='logloss',
        n_jobs=-1)
} 