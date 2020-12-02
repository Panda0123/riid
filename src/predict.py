
def predict(model, preproc):
    import riiideducation
    env = riiideducation.make_env()
    iter_test = env.iter_test()

    for (test_df, sample_prediction_df) in iter_test:
        test_df = preproc.transform(test_df)
        test_df["answered_correctly"] = model.predict(test_df)
        env.predict(test_df[test_df.content_type_id == 0, ["row_id", "answered_correctly"]])
