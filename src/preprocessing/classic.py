import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import numpy as np


class PreprocessV1ANN(BaseEstimator, TransformerMixin):
    """ Aggregate metrics of correctly answered and number of questions
    for each user, content, bundle, and parts. Missing values are filled
    with False for boolean type and 0.5 for numerical type.

        Basically it implements target encoding.
    FOR ANN
    """
    def __init__(self, features=None):

        if features:
            self.features = features
        else:
            self.features = ['timestamp', 'mean_user_accuracy',
                             'num_questions_answered', 'mean_content_accuracy',
                             'num_questions_asked',
                             'prior_question_elapsed_time',
                             'prior_question_had_explanation', 'bundle_size',
                             'mean_bundle_accuracy', 'mean_part_accuracy',
                             'num_content_correctly_answered',
                             'answered_correctly']

    def fit(self, df, df_questions):
        # USERS
        df_questions_only = df
        grouped_user = df_questions_only.groupby("user_id")
        df_user_answered = \
            grouped_user.agg({"answered_correctly": ["mean", "count"]}).copy()
        df_user_answered.columns = \
            ["mean_user_accuracy", "num_questions_answered"]
        del grouped_user

        # CONTENTS / QUESTIONS
        grouped_content = df_questions_only.groupby("content_id")
        df_questions_answered = \
            grouped_content\
            .agg({"answered_correctly": ["mean", "count"]}).copy()
        df_questions_answered.columns = \
            ["mean_content_accuracy", "num_questions_asked"]
        df_questions_answered.loc[:, "num_content_correctly_answered"] = \
            df_questions_answered.mean_content_accuracy * \
            df_questions_answered.num_questions_asked
        del grouped_content
        df_questions = df_questions.merge(
                                        df_questions_answered,
                                        left_on="question_id",
                                        right_on="content_id",
                                        how="left")

        # BUNDLES
        bundle_dict = df_questions["bundle_id"].value_counts().to_dict()
        df_questions.loc[:, "bundle_size"] = df_questions["bundle_id"].apply(
                                                lambda x: bundle_dict[x])
        grouped_bundle = df_questions.groupby("bundle_id")
        df_bundle_answered = grouped_bundle.agg(
                                {
                                    "num_content_correctly_answered": "sum",
                                    "num_questions_asked": "sum"
                                }).copy()
        df_bundle_answered.columns = [
                                      "num_bundle_correctly_answered",
                                      "num_bundle_asked"
                                     ]
        df_bundle_answered.loc[:, "mean_bundle_accuracy"] = \
            df_bundle_answered.num_bundle_correctly_answered / \
            df_bundle_answered.num_bundle_asked
        del grouped_bundle

        # PARTS
        grouped_parts = df_questions.groupby("part")
        df_part_answered = \
            grouped_parts.agg(
                                {
                                    "num_content_correctly_answered": "sum",
                                    "num_questions_asked": "sum"
                                }).copy()
        df_part_answered.columns = [
                                    "num_part_correctly_answered",
                                    "num_part_asked"
                                    ]
        df_part_answered.loc[:, "mean_part_accuracy"] = \
            df_part_answered.num_part_correctly_answered / \
            df_part_answered.num_part_asked
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
                encoder.fit(df_questions_only[feat]
                            .fillna(False)
                            .astype(bool)
                            .values.reshape(-1, 1))
                self.feat_encoder[feat] = encoder

        self.df_user_answered = df_user_answered
        self.df_questions = df_questions
        self.df_bundle_answered = df_bundle_answered
        self.df_part_answered = df_part_answered

    def transform(self, df):
        df_questions_only = df  # the value passed are questions only

        # CUSTOM FEATURE
        df_questions_only.loc[:, "is_first_time"] = \
            df_questions_only["prior_question_had_explanation"] \
            .isna().astype(np.int8)

        # MERGE
        df_questions_only = df_questions_only.merge(
                                        self.df_user_answered,
                                        how="left",
                                        on="user_id")
        df_questions_only = df_questions_only.merge(
                                        self.df_questions,
                                        how="left",
                                        left_on="content_id",
                                        right_on="question_id")
        df_questions_only = df_questions_only.merge(
                                        self.df_bundle_answered,
                                        how="left",
                                        on="bundle_id")
        df_questions_only = df_questions_only.merge(
                                        self.df_part_answered,
                                        how="left",
                                        on="part")

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


class PreprocessingLGBM(BaseEstimator, TransformerMixin):
    "Preprocessing for LightGBM"

    def __init__(self, features=None):
        """Preprocessing for LigtGBM

        args:
            features - features to include
        """
        if features:
            self.features = features
        else:
            self.features = ['timestamp', 'mean_user_accuracy',
                             'num_questions_answered', 'mean_content_accuracy',
                             'num_questions_asked',
                             'prior_question_elapsed_time',
                             'prior_question_had_explanation', 'bundle_size',
                             'mean_bundle_accuracy', 'mean_part_accuracy',
                             'num_content_correctly_answered', 'is_first_time',
                             'self_elapsed_time', 'start_time',
                             'lag_time', 'answered_correctly']

    def fit(self, df, df_questions):
        # USERS
        df_questions_only = df
        grouped_user = df_questions_only.groupby("user_id")
        df_user_answered = \
            grouped_user.agg({"answered_correctly" : ["mean", "count"]}).copy()
        df_user_answered.columns = [
                                    "mean_user_accuracy",
                                    "num_questions_answered"
                                   ]
        del grouped_user


        # CONTENTS / QUESTIONS
        grouped_content = df_questions_only.groupby("content_id")
        df_questions_answered = grouped_content.agg(
                            {"answered_correctly": ["mean", "count"]}).copy()
        df_questions_answered.columns = [
                                            "mean_content_accuracy",
                                            "num_questions_asked"
                                        ]
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
        self.encoderPQHE = LabelEncoder()
        self.encoderPQHE.fit(df_questions_only["prior_question_had_explanation"]\
                             .fillna(False).astype(bool).values.reshape(-1, 1))

        self.df_user_answered = df_user_answered
        self.df_questions = df_questions
        self.df_bundle_answered = df_bundle_answered
        self.df_part_answered = df_part_answered

    def transform(self, df):

        # CUSTOM FEATURE
        df.loc[:, "is_first_time"] = df["prior_question_had_explanation"].isna().astype(np.int8)

        # MERGE
        df = df.merge(self.df_user_answered, how="left", on="user_id")
        df = df.merge(self.df_questions, how="left", left_on="content_id", right_on="question_id")
        df = df.merge(self.df_bundle_answered, how="left", on="bundle_id")
        df = df.merge(self.df_part_answered, how="left", on="part")

        # HANDLE NAN
        df["prior_question_elapsed_time"].fillna(-1, inplace=True)
        df.loc[:, "prior_question_had_explanation"] = df["prior_question_had_explanation"].fillna(False).astype(bool)
        df["mean_user_accuracy"].fillna(0.5, inplace = True)
        df["num_questions_answered"].fillna(0, inplace = True)
        # temporary handling
        df["mean_content_accuracy"].fillna(0.5, inplace=True)
        df["mean_bundle_accuracy"].fillna(0.5, inplace=True)
        df["num_questions_asked"].fillna(0, inplace=True)
        df["num_content_correctly_answered"].fillna(0, inplace=True)

        # ENCODING
        df["prior_question_had_explanation"] = self.encoderPQHE\
            .transform(df["prior_question_had_explanation"]\
                       .values.reshape(-1, 1))


        # self elapsed time
        df["self_elapsed_time"] = \
            df.groupby("user_id")["prior_question_elapsed_time"]\
                .shift(-1, fill_value=0)

        # start timestamp
        df["start_time"] = df["timestamp"] - df["self_elapsed_time"]
        mask = df["start_time"] >= 0
        df["start_time"].where(mask, 0, inplace=True)

        # lag time
        df["lag_time"] = \
            (df.groupby("user_id")\
             .apply(
                 lambda tempDf: tempDf["start_time"] - tempDf["timestamp"]\
                 .shift(1, fill_value=0))).values
        mask = df["lag_time"] >= 0
        df["lag_time"].where(mask, 0, inplace=True)

        df = df[self.features[:-1] + ["user_id"] + [self.features[-1]]]

        return df


class PreprocessingLGBMModified(BaseEstimator, TransformerMixin):
    "Preprocessing for LightGBM"

    def __init__(self, features=None):
        """Preprocessing for LigtGBM

        args:
            features - features to include
        """
        if features:
            self.features = features
        else:
            self.conFeatures = ['timestamp', 'prior_question_elapsed_time']

            self.catFeatures = [ 'bundle_id',
                                'part', 'question_id']

            # cat feature that needs encoding
            self.catFeaturesEncoding = [ 'correct_answer',
                                        'prior_question_had_explanation']

            self.customizedCatFeatures = ['is_first_time', 'isReading']

            self.customizedConFeatures = ['self_elapsed_time', 'start_time',
                                          'lag_time', 'n_comb_tags_occur',
                                          'n_tags_occur_added', 'tags_sum',
                                          'n_tags', 'n_tags_occur']
        self.encodings = {}

    def fit(self, df, df_questions):
        self.questionDf = df_questions.copy()
        self._createCustomQuestion()

        # excluding the question_id
        for feature in self.catFeaturesEncoding:
            if feature not in ['prior_question_had_explanation']:
                encoder = LabelEncoder()
                encoder.fit(self.questionDf[feature].values.reshape(-1, 1))
                self.questionDf[feature] = encoder.transform(
                                            self.questionDf[feature].values
                                            .reshape(-1, 1))
            else:
                encoder = LabelEncoder()
                mask = df[feature].isna()
                encoder.fit(
                    df[feature].fillna(False).astype(np.int8).where(~mask, 3))
                self.encodings[feature] = encoder

        dictDtype = dict(
            question_id = np.int16,
            bundle_id = np.int16,
            correct_answer = np.int8,
            part = np.int8,
            n_comb_tags_occur= np.int16,
            n_tags_occur_added = np.int16,
            tags_sum	= np.int16,
            n_tags = np.int8,
            n_tags_occur = np.int16)

        self.questionDf.drop(columns=["tags"], inplace=True)
        self.questionDf = self.questionDf.astype(dictDtype)

    def _createCustomQuestion(self):
        # handling NAN values in questionDf
        self.questionDf["tags"].fillna("-1", inplace=True)

        # CUSTOM FEATURES FOR questionDf
        self.questionDf["isReading"] = \
            (self.questionDf.part.isin([5, 6, 7])).astype(np.int8) # part 5, 6, 7 are for reading part

        count_tag = self.questionDf.tags.value_counts()
        self.questionDf["n_comb_tags_occur"] = self.questionDf.tags.apply(
                                                 lambda x: count_tag.loc[x])
        counts_tag_indiv = \
            self.questionDf.tags.str.split(" ").explode("tags").value_counts()

        self.questionDf["n_tags_occur_added"] = \
            self.questionDf.tags.str.split(" ").apply(
                lambda tags: sum([counts_tag_indiv[tag] for tag in tags]))

        self.questionDf["tags_sum"] = self.questionDf.tags.str.split(" ").apply(
            lambda tags: sum([int(tag) for tag in tags]))

        self.questionDf["n_tags"] = \
            self.questionDf.tags.str.split(" ").apply(lambda x: len(x))

        n_tags_occur = self.questionDf.n_tags.value_counts()
        self.questionDf["n_tags_occur"] = self.questionDf.n_tags.apply(
            lambda x: n_tags_occur.loc[x])


    def transform(self, df):
        "df must contain questions only"

        df["prior_question_elapsed_time"].fillna(0, inplace=True)

        # custom features
        df.loc[:, "is_first_time"] = df["prior_question_had_explanation"]\
                                        .isna().astype(np.int8)

        # exlude correct_answer since already processed in questionDf
        for feature in self.catFeaturesEncoding[1:]:
            if feature not in ['prior_question_had_explanation']:
                df[feature] = self.encodings[feature].transform(
                                    df[feature].values.reshape(-1, 1))
            else:
                mask = df[feature].isna()
                df[feature] = \
                    self.encodings[feature].transform(
                        df[feature].fillna(False)
                        .astype(np.int8).where(~mask, 3)).astype(np.int8)

        # self elapsed time
        df["self_elapsed_time"] = \
            df.groupby("user_id")["prior_question_elapsed_time"]\
            .shift(-1, fill_value=0)

        # start timestamp
        df["start_time"] = df["timestamp"] - df["self_elapsed_time"]
        mask = df["start_time"] >= 0
        df["start_time"].where(mask, 0, inplace=True)

        # lag time
        df["lag_time"] = \
            (df.groupby("user_id")
             .apply(
                 lambda tempDf: tempDf["start_time"] - tempDf["timestamp"]
                 .shift(1, fill_value=0))).values
        mask = df["lag_time"] >= 0
        df["lag_time"].where(mask, 0, inplace=True)

        # MERGE df and questionDf
        df = df.merge(self.questionDf,
                      how="left",
                      left_on="content_id",
                      right_on="question_id")

        return df[self.catFeatures + self.customizedCatFeatures +
                  self.catFeaturesEncoding + self.conFeatures +
                  self.customizedConFeatures]
