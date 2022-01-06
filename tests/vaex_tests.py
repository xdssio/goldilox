import vaex

from goldilox.vaex import VaexPipeline
from tests.test_utils import validate_persistence


def test_advance_vaex():
    import vaex

    from vaex.ml.lightgbm import LightGBMModel
    from sklearn.metrics import accuracy_score
    from vaex.ml import LabelEncoder

    df = vaex.open('data/titanic.csv')
    train, test = df.split_random([0.8, 0.2])

    numeric_features = ['PassengerId', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
    string_features = ['Embarked', 'Sex', 'FamilyBin']
    features = numeric_features

    train = train[train['Cabin'].str.contains(' ') != True]
    train['FamilySize'] = train['Parch'] + train['SibSp'] + 1
    train['Name'] = train['Name'].fillna('Mr.')

    train['Initial'] = train['Name'].str.extract_regex(r'(?P<initial>[A-Za-z]+)\.').apply(
        lambda x: x.get('initial', 'Other'))

    initials_map = {k: v for k, v in (zip(['Other', 'Miss', 'Mr', 'Mrs', 'Master', 'Mlle', 'Mme', 'Ms', 'Dr',
                                           'Major', 'Lady', 'Countess',
                                           'Jonkheer', 'Col', 'Rev',
                                           'Capt', 'Sir', 'Don'],
                                          ['Other', 'Miss', 'Mr', 'Mrs', 'Mrs', 'Miss', 'Miss', 'Miss',
                                           'Mr', 'Mr', 'Mrs', 'Mrs',
                                           'Other', 'Other', 'Other',
                                           'Mr', 'Mr', 'Mr']))}
    train['Initial'] = train['Initial'].map(initials_map)

    gb = train.groupby(['Initial']).agg({'value': vaex.agg.mean('Age')})
    means = {k: v for k, v in zip(gb['Initial'].tolist(), gb['value'].tolist())}

    for initial, value in means.items():
        train['Age'] = train.func.where((train.Age.isna() & train.Initial.str.match(initial)), value, train.Age)

    train['AgeGroup'] = train.func.where(((train.Sex.str.match('male')) & (train.Age <= 15)), 'boy', '')
    train['AgeGroup'] = train.func.where(((train.Sex.str.match('female')) & (train.Age <= 15)), 'girl', train.AgeGroup)
    train['AgeGroup'] = train.func.where(((train.Sex.str.match('male')) & (train.Age > 15)), 'adult male',
                                         train.AgeGroup)
    train['AgeGroup'] = train.func.where(((train.Sex.str.match('female')) & (train.Age > 15)), 'adult female',
                                         train.AgeGroup)
    train['FamilyBin'] = train['FamilySize'].digitize(bins=[0, 1, 2, 5, 7, 100, 1000])

    string_features = ['Embarked', 'Sex', 'FamilyBin', 'AgeGroup']
    encoder = LabelEncoder(features=string_features, prefix='le_', allow_unseen=True)
    train = encoder.fit_transform(train)

    features = ['PassengerId', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare'] + [f"{encoder.prefix}{column}" for column in
                                                                             string_features]
    target = 'Survived'
    model = LightGBMModel(features=features,
                          target=target,
                          prediction_name='lgm_predictions',
                          num_boost_round=500, params={'verbose': -1,
                                                       'application': 'binary'})
    model.fit(train)
    train = model.transform(train)
    train['prediction'] = train.func.where(train['lgm_predictions'] > 0.5, 1, 0)
    train['target_label'] = train.func.where(train['lgm_predictions'] > 0.5, 'survived', 'died')
    pipeline = VaexPipeline.from_dataframe(train)
    pipeline = validate_persistence(pipeline)
    assert pipeline.inference(test).shape == (len(test), 23)
    predictions = pipeline.inference(test, fillna=False)['prediction']
    assert 0.5 < accuracy_score(test[target].values, predictions.values)


def test_advance_vaex_fit():
    def fit(df):
        import vaex
        from vaex.ml.lightgbm import LightGBMModel
        from sklearn.metrics import accuracy_score
        from vaex.ml import LabelEncoder

        train, test = df.split_random([0.8, 0.2])

        numeric_features = ['PassengerId', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
        string_features = ['Embarked', 'Sex', 'FamilyBin']
        features = numeric_features

        train = train[train['Cabin'].str.contains(' ') != True]
        train['FamilySize'] = train['Parch'] + train['SibSp'] + 1
        train['Name'] = train['Name'].fillna('Mr.')

        train['Initial'] = train['Name'].str.extract_regex(r'(?P<initial>[A-Za-z]+)\.').apply(
            lambda x: x.get('initial', 'Other'))

        initials_map = {k: v for k, v in (zip(['Other', 'Miss', 'Mr', 'Mrs', 'Master', 'Mlle', 'Mme', 'Ms', 'Dr',
                                               'Major', 'Lady', 'Countess',
                                               'Jonkheer', 'Col', 'Rev',
                                               'Capt', 'Sir', 'Don'],
                                              ['Other', 'Miss', 'Mr', 'Mrs', 'Mrs', 'Miss', 'Miss', 'Miss',
                                               'Mr', 'Mr', 'Mrs', 'Mrs',
                                               'Other', 'Other', 'Other',
                                               'Mr', 'Mr', 'Mr']))}
        train['Initial'] = train['Initial'].map(initials_map)

        gb = train.groupby(['Initial']).agg({'value': vaex.agg.mean('Age')})
        means = {k: v for k, v in zip(gb['Initial'].tolist(), gb['value'].tolist())}

        for initial, value in means.items():
            train['Age'] = train.func.where((train.Age.isna() & train.Initial.str.match(initial)), value, train.Age)

        train['AgeGroup'] = train.func.where(((train.Sex.str.match('male')) & (train.Age <= 15)), 'boy', '')
        train['AgeGroup'] = train.func.where(((train.Sex.str.match('female')) & (train.Age <= 15)), 'girl',
                                             train.AgeGroup)
        train['AgeGroup'] = train.func.where(((train.Sex.str.match('male')) & (train.Age > 15)), 'adult male',
                                             train.AgeGroup)
        train['AgeGroup'] = train.func.where(((train.Sex.str.match('female')) & (train.Age > 15)), 'adult female',
                                             train.AgeGroup)
        train['FamilyBin'] = train['FamilySize'].digitize(bins=[0, 1, 2, 5, 7, 100, 1000])

        string_features = ['Embarked', 'Sex', 'FamilyBin', 'AgeGroup']
        encoder = LabelEncoder(features=string_features, prefix='le_', allow_unseen=True)
        train = encoder.fit_transform(train)

        features = ['PassengerId', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare'] + [f"{encoder.prefix}{column}" for column
                                                                                 in
                                                                                 string_features]
        target = 'Survived'
        model = LightGBMModel(features=features,
                              target=target,
                              prediction_name='lgm_predictions',
                              num_boost_round=500, params={'verbose': -1,
                                                           'application': 'binary'})
        model.fit(train)
        train = model.transform(train)
        train['prediction'] = train.func.where(train['lgm_predictions'] > 0.5, 1, 0)
        train['target_label'] = train.func.where(train['lgm_predictions'] > 0.5, 'survived', 'died')

        pipeline = VaexPipeline.from_dataframe(train)
        predictions = pipeline.inference(test, fillna=False)['prediction']
        accuracy = accuracy_score(test[target].values, predictions.values)
        all_data = pipeline.inference(df, columns=features + [target])
        model.fit(all_data)
        all_data = model.transform(all_data)
        all_data['prediction'] = all_data.func.where(all_data['lgm_predictions'] > 0.5, 1, 0)
        all_data['target_label'] = all_data.func.where(all_data['lgm_predictions'] > 0.5, 'survived', 'died')
        all_data.variables['accuracy'] = accuracy
        return all_data

    df = vaex.open('data/titanic.csv')
    pipeline = VaexPipeline.from_dataframe(df, fit)
    pipeline.fit(df)
    pipeline = validate_persistence(pipeline)
    assert pipeline.inference(df).shape == (len(df), 23)
