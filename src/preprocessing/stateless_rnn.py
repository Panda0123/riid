import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import numpy as np

class PreprocessV3RNN(BaseEstimator, TransformerMixin):
    """" Preprocessing for StatelessRNN includes
    isReading features.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.features = ["timestamp", "prior_question_elapsed_time", "prior_question_had_explanation",
                         "isReading", "question_id", "bundle_id", "correct_answer", "part", "tags"]

    def fit(self, df, questionDf):
        self.questionDf = questionDf
        self.scalerTS = StandardScaler()
        self.scalerTS.fit(df["timestamp"].values.reshape(-1, 1))
        self.scalerPQET = StandardScaler()
        self.scalerPQET.fit(df["prior_question_elapsed_time"].fillna(-1).values.reshape(-1, 1))
        self.encoderTags = LabelEncoder()
        # change the type of tags to string and fill na with "-1"
        self.questionDf.loc[:, "tags"] = self.questionDf["tags"].astype(str).fillna("-1")
        self.encoderTags.fit(self.questionDf["tags"])
        self.questionDf.loc[:, "tags"] =  self.encoderTags.transform(self.questionDf["tags"])
        # Add isReading feature for part type
        self.questionDf["isReading"] = (self.questionDf.part.isin([5, 6, 7])).astype(np.int64) # part 5, 6, 7 are for reading part
        self.questionDf  = self.questionDf.astype(np.int64)

    def transform(self, df):
        df = df[["timestamp", "prior_question_elapsed_time", "prior_question_had_explanation", "content_id", "user_id"]]

        # TODO: add feature that tells if prior_question_had_explanation is False
        #       so we can see that the previous test is diagnostic test

        # HANDLING NAs
        df["prior_question_elapsed_time"].fillna(-1, inplace=True)
        df.loc[:, "prior_question_had_explanation"] = \
                (df["prior_question_had_explanation"].fillna(False)).astype(np.int8)

        # Scaling
        df.loc[:, "timestamp"] = \
                self.scalerTS.transform(df["timestamp"].values.reshape(-1, 1))
        df.loc[:, "prior_question_elapsed_time"] = \
                self.scalerPQET.transform(df["prior_question_elapsed_time"].values.reshape(-1, 1))
        df = df.merge(self.questionDf, how="left", left_on="content_id", right_on="question_id")

        return df[self.features + ["user_id"]].astype({"question_id": np.int16, "bundle_id":np.int16,
                        "correct_answer": np.int8, "part":np.int8,
                        "tags":np.int16, "isReading": np.int8, "user_id":np.int64})
