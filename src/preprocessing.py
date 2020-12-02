import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np

class PreprocessV1(BaseEstimator, TransformerMixin):
    """ Aggregate metrics of correctly answered and number of questions
    for each user, content, bundle, and parts. Missing values are filled
    with False for boolean type and -1 for numerical type.
    """
    def __init__(self, features=None):
        if features:
            self.features = features
        else:
            self.features = ['timestamp','mean_user_accuracy', 'num_questions_answered','mean_content_accuracy', 
                               'num_questions_asked',
                               'prior_question_elapsed_time', 'prior_question_had_explanation', 'bundle_size', 
                               'mean_bundle_accuracy','mean_part_accuracy', 'num_content_correctly_answered',
                               'answered_correctly']

    def fit(self, df, df_questions):
        # USERS
        df_questions_only = df
        grouped_user = df_questions_only.groupby("user_id")
        df_user_answered = \
            grouped_user.agg({"answered_correctly" : ["mean", "count"]}).copy()
        df_user_answered.columns = ["mean_user_accuracy", "num_questions_answered"]
        del grouped_user

        # CONTENTS / QUESTIONS
        grouped_content = df_questions_only.groupby("content_id")
        df_questions_answered = \
            grouped_content.agg({"answered_correctly": ["mean", "count"]}).copy()
        df_questions_answered.columns = ["mean_content_accuracy", "num_questions_asked"]
        df_questions_answered["num_content_correctly_answered"] = \
            df_questions_answered.mean_content_accuracy * df_questions_answered.num_questions_asked
        del grouped_content
        df_questions = \
            df_questions.merge(df_questions_answered, left_on="question_id", right_on="content_id", how="left")

        # BUNDLES
        bundle_dict = df_questions["bundle_id"].value_counts().to_dict()
        df_questions.loc[:, "bundle_size"] = df_questions["bundle_id"].apply(lambda x: bundle_dict[x])
        grouped_bundle = df_questions.groupby("bundle_id")
        df_bundle_answered = \
            grouped_bundle.agg({"num_content_correctly_answered": "sum", "num_questions_asked" :"sum"}).copy()
        df_bundle_answered.columns = ["num_bundle_correctly_answered", "num_bundle_asked"]
        df_bundle_answered.loc[:, "mean_bundle_accuracy"] = \
            df_bundle_answered.num_bundle_correctly_answered / df_bundle_answered.num_bundle_asked
        del grouped_bundle

        # PARTS
        grouped_parts = df_questions.groupby("part")
        df_part_answered = \
            grouped_parts.agg({"num_content_correctly_answered": "sum", "num_questions_asked": "sum"}).copy()
        df_part_answered.columns = ["num_part_correctly_answered", "num_part_asked"]
        df_part_answered.loc[:, "mean_part_accuracy"]  = \
            df_part_answered.num_part_correctly_answered / df_part_answered.num_part_asked
        del grouped_parts
        

        self.df_user_answered = df_user_answered
        self.df_questions = df_questions
        self.df_bundle_answered = df_bundle_answered
        self.df_part_answered = df_part_answered

    def transform(self, df):
        # Merge
        df = df.merge(self.df_user_answered, how="left", on="user_id", copy=False)
        df = df.merge(self.df_questions, how="left", left_on="content_id", right_on="question_id", copy=False)
        df = df.merge(self.df_bundle_answered, how="left", on="bundle_id", copy=False)
        df = df.merge(self.df_part_answered, how="left", on="part", copy=False)

        # handle na's
        df["prior_question_had_explanation"].fillna(False, inplace=True)
        df.fillna(-1, inplace=True)

        return df[self.features]


    
class PreprocessV1ANN(BaseEstimator, TransformerMixin):
    """ Aggregate metrics of correctly answered and number of questions
    for each user, content, bundle, and parts. Missing values are filled
    with False for boolean type and 0.5 for numerical type.

    FOR ANN
    """
    def __init__(self, features=None):

        if features:
            self.features = features
        else:
            self.features = ['timestamp','mean_user_accuracy', 'num_questions_answered','mean_content_accuracy', 
                               'num_questions_asked',
                               'prior_question_elapsed_time', 'prior_question_had_explanation', 'bundle_size', 
                               'mean_bundle_accuracy','mean_part_accuracy', 'num_content_correctly_answered',
                               'answered_correctly']

    def fit(self, df, df_questions):
        # USERS
        df_questions_only = df
        grouped_user = df_questions_only.groupby("user_id")
        df_user_answered = \
            grouped_user.agg({"answered_correctly" : ["mean", "count"]}).copy()
        df_user_answered.columns = ["mean_user_accuracy", "num_questions_answered"]
        del grouped_user

        # CONTENTS / QUESTIONS
        grouped_content = df_questions_only.groupby("content_id")
        df_questions_answered = \
            grouped_content.agg({"answered_correctly": ["mean", "count"]}).copy()
        df_questions_answered.columns = ["mean_content_accuracy", "num_questions_asked"]
        df_questions_answered.loc[:, "num_content_correctly_answered"] = \
            df_questions_answered.mean_content_accuracy * df_questions_answered.num_questions_asked
        del grouped_content
        df_questions = \
            df_questions.merge(df_questions_answered, left_on="question_id", right_on="content_id", how="left")

        # BUNDLES
        bundle_dict = df_questions["bundle_id"].value_counts().to_dict()
        df_questions.loc[:, "bundle_size"] = df_questions["bundle_id"].apply(lambda x: bundle_dict[x])
        grouped_bundle = df_questions.groupby("bundle_id")
        df_bundle_answered = \
            grouped_bundle.agg({"num_content_correctly_answered": "sum", "num_questions_asked" :"sum"}).copy()
        df_bundle_answered.columns = ["num_bundle_correctly_answered", "num_bundle_asked"]
        df_bundle_answered.loc[:, "mean_bundle_accuracy"] = \
            df_bundle_answered.num_bundle_correctly_answered / df_bundle_answered.num_bundle_asked
        del grouped_bundle

        # PARTS
        grouped_parts = df_questions.groupby("part")
        df_part_answered = \
            grouped_parts.agg({"num_content_correctly_answered": "sum", "num_questions_asked": "sum"}).copy()
        df_part_answered.columns = ["num_part_correctly_answered", "num_part_asked"]
        df_part_answered.loc[:, "mean_part_accuracy"]  = \
            df_part_answered.num_part_correctly_answered / df_part_answered.num_part_asked
        del grouped_parts

        # ENCODING
        self.feat_encoder = {}
        for feat in self.features[:-1]:
            if feat != "prior_question_had_explanation":
                for temp_df in [df_questions_only, df_user_answered, df_questions, df_bundle_answered, df_part_answered]:
                    if feat in temp_df.columns.values:
                        encoder = StandardScaler()
                        encoder.fit(temp_df[feat].values.reshape(-1, 1))
                        self.feat_encoder[feat] = encoder
                        break
            else:
                encoder = LabelEncoder()
                encoder.fit(df_questions_only[feat].fillna(False).astype(bool).values.reshape(-1, 1))
                self.feat_encoder[feat] = encoder

        self.df_user_answered = df_user_answered
        self.df_questions = df_questions
        self.df_bundle_answered = df_bundle_answered
        self.df_part_answered = df_part_answered

    def transform(self, df):
        df_questions_only = df # the value passed are questions only

        # CUSTOM FEATURE
        df_questions_only.loc[:, "is_first_time"] = df_questions_only["prior_question_had_explanation"].isna().astype(np.int8)

        # MERGE
        df_questions_only = df_questions_only.merge(self.df_user_answered, how="left", on="user_id")
        df_questions_only = df_questions_only.merge(self.df_questions, how="left", left_on="content_id", right_on="question_id")
        df_questions_only = df_questions_only.merge(self.df_bundle_answered, how="left", on="bundle_id")
        df_questions_only = df_questions_only.merge(self.df_part_answered, how="left", on="part")

        # HANDLE NAN
        df_questions_only["prior_question_elapsed_time"].fillna(-1, inplace=True)
        df_questions_only.loc[:, "prior_question_had_explanation"] = df_questions_only["prior_question_had_explanation"].fillna(False).astype(bool)
        df_questions_only["mean_user_accuracy"].fillna(0.5, inplace = True)
        df_questions_only["num_questions_answered"].fillna(0, inplace = True)
        # temporary handling
        df_questions_only["mean_content_accuracy"].fillna(0.5, inplace=True)
        df_questions_only["mean_bundle_accuracy"].fillna(0.5, inplace=True)
        df_questions_only["num_questions_asked"].fillna(0, inplace=True)
        df_questions_only["num_content_correctly_answered"].fillna(0, inplace=True)
        df_questions_only = df_questions_only[self.features[:-1] + ["is_first_time", "user_id"] + [self.features[-1]]] 

        # ENCODING
        for feat in self.features[:-1]:
            df_questions_only[feat] = self.feat_encoder[feat].transform(df_questions_only[feat].values.reshape(-1, 1))

        return df_questions_only


class PreprocessV2RNN(BaseEstimator, TransformerMixin):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.features = ["timestamp", "prior_question_elapsed_time", "prior_question_had_explanation", 
                         "question_id", "bundle_id", "correct_answer", "part", "tags"]

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
        self.questionDf  = self.questionDf.astype(np.int64)
    
    def transform(self, df):
        df = df[["timestamp", "prior_question_elapsed_time", "prior_question_had_explanation", "content_id", "user_id"]]

        # HANDLING NAs
        df["prior_question_elapsed_time"].fillna(-1, inplace=True)
        df.loc[:, "prior_question_had_explanation"] = (df["prior_question_had_explanation"].fillna(False)).astype(np.int8)

        # Scaling
        df.loc[:, "timestamp"] = self.scalerTS.transform(df["timestamp"].values.reshape(-1, 1))
        df.loc[:, "prior_question_elapsed_time"] = self.scalerPQET.transform(df["prior_question_elapsed_time"].values.reshape(-1, 1))
        df = df.merge(self.questionDf, how="left", left_on="content_id", right_on="question_id")

        # Encoding
        # df["tags"] = self.encoderTags.transform(df["tags"].fillna("-1").values.reshape(-1, 1).astype(str)).astype(np.int16)
        

        return df[self.features + ["user_id"]].astype({"question_id": np.int16, "bundle_id":np.int16, 
                        "correct_answer": np.int8, "part":np.int8,
                        "tags":np.int16, "user_id":np.int64})


class PreprocessV3RNN(BaseEstimator, TransformerMixin):

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
