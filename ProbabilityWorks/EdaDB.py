import pandas as pd
import sqlite3
from scipy.stats import linregress, f_oneway, chi2_contingency

class EDADB:

    def __init__(self, df, dependent, project_name, department, filename):
        self.df = df
        self.dependent = dependent
        self.features = []
        self.filename = filename
        self.continuous_types = ['int64', 'float64']
        self.dependent_type = 'Continuous' if self.df[dependent].dtype in self.continuous_types else 'Categorical'
        self.project_name = project_name
        self.department = department

    # Method - create_database() initiates and instance of the database and creates a .db file if one does not exist.
    def create_database(self):

        conn = sqlite3.connect(self.filename)

        command = """ CREATE TABLE IF NOT EXISTS projects (
                      project_name TEXT PRIMARY KEY,
                      department TEXT NOT NULL)
                  """
        conn.execute(command)


        command = """ CREATE TABLE IF NOT EXISTS features (
                      feature_name TEXT NOT NULL PRIMARY KEY,
                      project TEXT NOT NULL,
                      data_type TEXT NOT NULL,
                      test TEXT NOT NULL,
                      pvalue TEXT NOT NULL,
                      FOREIGN KEY (project) REFERENCES features(project_name))
                  """
        conn.execute(command)

        command = """ CREATE TABLE IF NOT EXISTS datapoints (
                      feature TEXT NOT NULL,
                      dependent TEXT,
                      independent TEXT,
                      FOREIGN KEY (feature) REFERENCES features(feature_name))
                  """

        conn.execute(command)

        conn.close()

    # pvalue_regression() calculates the pvalue for a regression hypotheses test including a feature and the dependent variable.
    # Returns pvalue.
    # add_features() -> hpy_test()
    def pvalue_regression(self, feature):

        pvalue = linregress(self.df[feature], self.df[self.dependent])

        return pvalue.pvalue

    # pvalue_anova() calculates the pvalue for a ANOVA hypotheses test including a feature and the dependent variable.
    # Returns pvalue.
    # add_features() -> hpy_test()
    def pvalue_anova(self, feature):

        func_arg = []

        feature_elements = self.df[feature].unique()

        for item in feature_elements:

            func_arg.append(self.df[self.df[feature] == item][self.dependent])

        pvalue = f_oneway(*func_arg)

        return pvalue.pvalue

    # pvalue_chisq() calculates the pvalue for a Chi2 hypotheses test including a feature and the dependent variable.
    # Returns pvalue.
    # add_features() -> hpy_test()
    def pvalue_chisq(self, feature):

        cross = pd.crosstab(self.df[feature], self.df[self.dependent])

        pvalue = chi2_contingency(cross)

        return pvalue.p

    # hyp_test() determines type of hypothesis test to perform and routes it to correct function.  Returns test type
    # and pvalue.
    # add_features()
    def hyp_test(self, feature, feature_type):

        if (feature_type == 'Continuous') & (self.dependent_type == 'Continuous'):

            pvalue = self.pvalue_regression(feature)

            return 'Regression', pvalue

        elif (feature_type == 'Categorical') & (self.dependent_type == 'Continuous'):

            pvalue = self.pvalue_anova(feature)

            return 'ANOVA', pvalue

        elif (feature_type == 'Continuous') & (self.dependent_type == 'Categorical'):

            pvalue = self.pvalue_anova(feature)

            return 'ANOVA', pvalue

        else:

            pvalue = self.pvalue_anova(feature)

            return 'Chi Squared', pvalue

    # add_project() adds the project to the projects table.
    # add_features()
    def add_project(self, conn):

        command = "INSERT INTO projects(project_name, department) VALUES('{}', '{}')".format(self.project_name, self.department)

        conn.execute(command)

    # add_datapoints() iterates through data and adds it to datapoints table.
    # add_features()
    def add_datapoints(self, item, conn):

        for i in range(len(self.df.index)):

            command = "INSERT INTO datapoints(feature, dependent, independent) VALUES('{}', '{}', '{}')".format(item, self.df.iloc[i][self.dependent], self.df.iloc[i][item])

            conn.execute(command)

    # Method - add_features() iterates through the features and adds all data to database.
    def add_features(self, features):

        print(self.dependent_type)

        self.features = features

        conn = sqlite3.connect(self.filename)

        self.add_project(conn)

        for feature in self.features:

            print(feature)

            feature_type = 'Continuous' if self.df[feature].dtype in self.continuous_types else 'Categorical'

            feature_test, pvalue = self.hyp_test(feature, feature_type)

            command = "INSERT INTO features(feature_name, project, data_type, test, pvalue) VALUES('{}', '{}', '{}', '{}', '{}')".format(feature, self.project_name,feature_type, feature_test, pvalue)

            conn.execute(command)

            self.add_datapoints(feature, conn)

        conn.commit()

        conn.close()