
def three_models_combined(self, intrusion_features, avoidance_features, hypertension_features, regression_features,
                          depression_features):

    self.df = self.df[~self.df['intrusion_cutoff'].isna()]
    self.df = self.df[~self.df['avoidance_cutoff'].isna()]
    self.df = self.df[~self.df['hypertention_cutoff'].isna()]
    self.df = self.df[~self.df['PCL3'].isna()]

    print("self.df.shape", self.df.shape)
    X = self.df
    Y = self.df[self.target  ]# strict
    all_Y = [self.target, "intrusion_cutoff", "avoidance_cutoff", "hypertention_cutoff", "PCL3" ,"depression_cutoff", "only_avoidance_cutoff"]

    X_train, X_test, y_train, y_test = train_test_split(X, self.df[all_Y], test_size=0.25, random_state=8526566, stratify=Y)

    # intrusion
    X_intrusion = X_train[intrusion_features].values
    y_intrusion = y_train["intrusion_cutoff"].apply(lambda x: int(x))
    pipe_intrusion = Pipeline(steps=[
        ('rfe', RFE(RandomForestClassifier(n_estimators=100), 10)),
        ('classifier', XGBClassifier(n_estimators=1000, reg_alpha=1, scale_pos_weight=3))])
    scores = cross_val_score(pipe_intrusion, X_intrusion, y_intrusion, scoring='precision', cv=StratifiedKFold(5))
    print(f"intrusion {sum(scores)/5}")
    pipe_intrusion.fit(X_intrusion, y_intrusion)

    # avoidance
    X_avoidance = X_train[avoidance_features].values
    y_avoidance = y_train["avoidance_cutoff"].apply(lambda x: int(x))
    pipe_avoidance = Pipeline(steps=[
        ('rfe', RFE(RandomForestClassifier(n_estimators=100), 10)),
        ('classifier', XGBClassifier(n_estimators=1000, reg_alpha=1, scale_pos_weight=6))])
    scores = cross_val_score(pipe_avoidance, X_avoidance, y_avoidance, scoring='precision', cv=StratifiedKFold(5))
    print(f"avoidance {sum(scores)/5}")
    pipe_avoidance.fit(X_avoidance, y_avoidance)

    # hypertension
    X_hypertension = X_train[hypertension_features].values
    y_hypertention = y_train["hypertention_cutoff"].apply(lambda x: int(x))
    pipe_hypertension = Pipeline(steps=[
        ('rfe', RFE(RandomForestClassifier(n_estimators=100), 10)),
        ('classifier', XGBClassifier(n_estimators=1000, reg_alpha=1, scale_pos_weight=4))])
    scores = cross_val_score(pipe_hypertension, X_hypertension, y_hypertention, scoring='precision', cv=StratifiedKFold(5))
    print(f"hypertension {sum(scores)/5}")
    pipe_hypertension.fit(X_hypertension, y_hypertention)

    # depression
    X_depression = X_train[depression_features].values
    y_depression = y_train["depression_cutoff"].apply(lambda x: int(x))
    pipe_depression = Pipeline(steps=[
        ('classifier', XGBClassifier(n_estimators=500, reg_alpha=1, scale_pos_weight=3))])
    scores = cross_val_score(pipe_depression, X_depression, y_depression, scoring='precision', cv=StratifiedKFold(5))
    print(f"depression {sum(scores)/5}")
    pipe_depression.fit(X_depression, y_depression)

    # only_avoidance
    X_only_avoidance = X_train[avoidance_features].values
    y_only_avoidance = y_train["only_avoidance_cutoff"].apply(lambda x: int(x))
    pipe_only_avoidance = Pipeline(steps=[
        ('classifier', XGBClassifier(n_estimators=500, reg_alpha=1, scale_pos_weight=3))])
    scores = cross_val_score(pipe_only_avoidance, X_only_avoidance, y_only_avoidance, scoring='precision', cv=StratifiedKFold(5))
    print(f"only_avoidance {sum(scores)/5}")
    pipe_only_avoidance.fit(X_only_avoidance, y_only_avoidance)

    # regression
    X_regression = X_train[regression_features].values
    y_regression = y_train["PCL3"]
    pipe_regression = Pipeline(steps=[
        ('classifier', RandomForestRegressor(n_estimators=500))])
    scores = cross_val_score(pipe_regression, X_regression, y_regression)
    print(f"regression {sum(scores)/5}")
    pipe_regression.fit(X_regression, y_regression)


    ## combine three classifiers
    X_test_hypertension = X_test[hypertension_features].values
    X_test_avoidance = X_test[avoidance_features].values
    X_test_intrusion = X_test[intrusion_features].values
    X_test_regression = X_test[regression_features].values
    X_test_depression = X_test[depression_features].values
    X_test_only_avoidance = X_test[avoidance_features].values

    y_pred_hypertension = pipe_hypertension.predict(X_test_hypertension)
    y_pred_avoidance = pipe_avoidance.predict(X_test_avoidance)
    y_pred_intrusion = pipe_intrusion.predict(X_test_intrusion)
    y_pred_regression = pipe_regression.predict(X_test_regression) >= 35
    y_pred_depression = pipe_depression.predict(X_test_depression)
    y_pred_only_avoidance = pipe_only_avoidance.predict(X_test_only_avoidance)

    y_pred = (y_pred_hypertension & y_pred_avoidance & y_pred_intrusion & y_pred_regression)

    y_target = y_test["PCL_Strict3"].apply(lambda x: int(x))

    acc = accuracy_score(y_target, y_pred)
    f1 = f1_score(y_target, y_pred)
    recall = recall_score(y_target, y_pred)
    precision = precision_score(y_target, y_pred)
    print("test scores")
    print(f"acc-{acc}, f1- {f1}, recall-{recall}, precision - {precision}")


class TargetEnsembler(object):
    pass


    def __init__(self, intrusion_features, avoidance_features, hypertension_features, regression_features,
        depression_features):

        self.df = self.df[~self.df['intrusion_cutoff'].isna()]
        self.df = self.df[~self.df['avoidance_cutoff'].isna()]
        self.df = self.df[~self.df['hypertention_cutoff'].isna()]
        self.df = self.df[~self.df['PCL3'].isna()]

        print("self.df.shape", self.df.shape)
        X = self.df
        Y = self.df[self.target]  # strict
        all_Y = [self.target, "intrusion_cutoff", "avoidance_cutoff", "hypertention_cutoff", "PCL3",
                 "depression_cutoff", "only_avoidance_cutoff"]

        def fit(self):

            # intrusion
            X_intrusion = X_train[intrusion_features].values
            y_intrusion = y_train["intrusion_cutoff"].apply(lambda x: int(x))
            pipe_intrusion = Pipeline(steps=[
                ('rfe', RFE(RandomForestClassifier(n_estimators=100), 10)),
                ('classifier', XGBClassifier(n_estimators=1000, reg_alpha=1, scale_pos_weight=3))])
            scores = cross_val_score(pipe_intrusion, X_intrusion, y_intrusion, scoring='precision', cv=StratifiedKFold(5))
            print(f"intrusion {sum(scores)/5}")
            pipe_intrusion.fit(X_intrusion, y_intrusion)

            # avoidance
            X_avoidance = X_train[avoidance_features].values
            y_avoidance = y_train["avoidance_cutoff"].apply(lambda x: int(x))
            pipe_avoidance = Pipeline(steps=[
                ('rfe', RFE(RandomForestClassifier(n_estimators=100), 10)),
                ('classifier', XGBClassifier(n_estimators=1000, reg_alpha=1, scale_pos_weight=6))])
            scores = cross_val_score(pipe_avoidance, X_avoidance, y_avoidance, scoring='precision', cv=StratifiedKFold(5))
            print(f"avoidance {sum(scores)/5}")
            pipe_avoidance.fit(X_avoidance, y_avoidance)

            # hypertension
            X_hypertension = X_train[hypertension_features].values
            y_hypertention = y_train["hypertention_cutoff"].apply(lambda x: int(x))
            pipe_hypertension = Pipeline(steps=[
                ('rfe', RFE(RandomForestClassifier(n_estimators=100), 10)),
                ('classifier', XGBClassifier(n_estimators=1000, reg_alpha=1, scale_pos_weight=4))])
            scores = cross_val_score(pipe_hypertension, X_hypertension, y_hypertention, scoring='precision',
                                     cv=StratifiedKFold(5))
            print(f"hypertension {sum(scores)/5}")
            pipe_hypertension.fit(X_hypertension, y_hypertention)

            # depression
            X_depression = X_train[depression_features].values
            y_depression = y_train["depression_cutoff"].apply(lambda x: int(x))
            pipe_depression = Pipeline(steps=[
                ('classifier', XGBClassifier(n_estimators=500, reg_alpha=1, scale_pos_weight=3))])
            scores = cross_val_score(pipe_depression, X_depression, y_depression, scoring='precision',
                                     cv=StratifiedKFold(5))
            print(f"depression {sum(scores)/5}")
            pipe_depression.fit(X_depression, y_depression)

            # only_avoidance
            X_only_avoidance = X_train[avoidance_features].values
            y_only_avoidance = y_train["only_avoidance_cutoff"].apply(lambda x: int(x))
            pipe_only_avoidance = Pipeline(steps=[
                ('classifier', XGBClassifier(n_estimators=500, reg_alpha=1, scale_pos_weight=3))])
            scores = cross_val_score(pipe_only_avoidance, X_only_avoidance, y_only_avoidance, scoring='precision',
                                     cv=StratifiedKFold(5))
            print(f"only_avoidance {sum(scores)/5}")
            pipe_only_avoidance.fit(X_only_avoidance, y_only_avoidance)

            # regression
            X_regression = X_train[regression_features].values
            y_regression = y_train["PCL3"]
            pipe_regression = Pipeline(steps=[
                ('classifier', RandomForestRegressor(n_estimators=500))])
            scores = cross_val_score(pipe_regression, X_regression, y_regression)
            print(f"regression {sum(scores)/5}")
            pipe_regression.fit(X_regression, y_regression)

    def predict(self):
        ## combine three classifiers
        X_test_hypertension = X_test[hypertension_features].values
        X_test_avoidance = X_test[avoidance_features].values
        X_test_intrusion = X_test[intrusion_features].values
        X_test_regression = X_test[regression_features].values
        X_test_depression = X_test[depression_features].values
        X_test_only_avoidance = X_test[avoidance_features].values

        y_pred_hypertension = pipe_hypertension.predict(X_test_hypertension)
        y_pred_avoidance = pipe_avoidance.predict(X_test_avoidance)
        y_pred_intrusion = pipe_intrusion.predict(X_test_intrusion)
        y_pred_regression = pipe_regression.predict(X_test_regression) >= 35
        y_pred_depression = pipe_depression.predict(X_test_depression)
        y_pred_only_avoidance = pipe_only_avoidance.predict(X_test_only_avoidance)

        y_pred = (y_pred_hypertension & y_pred_avoidance & y_pred_intrusion & y_pred_regression)

        y_target = y_test["PCL_Strict3"].apply(lambda x: int(x))

        acc = accuracy_score(y_target, y_pred)
        f1 = f1_score(y_target, y_pred)
        recall = recall_score(y_target, y_pred)
        precision = precision_score(y_target, y_pred)
        print("test scores")
        print(f"acc-{acc}, f1- {f1}, recall-{recall}, precision - {precision}")


def cv (X_train_2, y_train_2):
    kfold = StratifiedKFold(n_splits=3, shuffle=True)
    cvscores = []
    y_train_2 = np.array(y_train_2)
    X_train_2 = np.array(X_train_2)


    for train, test in kfold.split(X_train_2, y_train_2):

        y_pred = np.zeros_like(y_train_2[test]).reshape(-1, 1)

        model = TargetEnsembler()

        model.fit(X_train_res, y_train_res)
        y_pred +=  model.predict(X_train_2[test])


        y_pred = (y_pre d /t) > lim
        s_f = f1_score(y_pred ,y_train_2[test])
        s_p = precision_score(y_pred ,y_train_2[test])
        s_r = recall_score(y_pred ,y_train_2[test])
        print("\tscores f1", (s_f))
        print("\tscores p", (s_p))
        print("\tscores r", (s_r))
        scores_f.append(s_f)
        scores_p.append(s_p)
        scores_r.append(s_r)

        print("mean scores f1", np.mean(scores_f))
        print("mean scores p", np.mean(scores_p))
        print("mean scores r", np.mean(scores_r))