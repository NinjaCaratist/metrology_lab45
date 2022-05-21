import pandas as pd
import seaborn as sns
import matplotlib.pyplot as ppt
import numpy as np
import copy

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics


class Constants:
    MAX_NUMBER_OF_SUBPLOTS = 5
    GENDER = {'Female': 1, 'Male': 2}


def unknownCount(data: pd.DataFrame) -> dict:
    unknown = dict()
    for column in data:
        unknown[column] = data[data[column] == '?'].shape[0]
    return unknown


def clearData(data: pd.DataFrame):
    data['income'] = data['income'].apply(lambda value: int(value == '>50K'))
    data.replace('?', np.NaN)

def analyzeIncomeByEducationAndHoursPerWeek(data: pd.DataFrame):
    income_data = data.groupby(['education', 'hours-per-week'])['income'].mean().to_frame()
    income_data['income'] = income_data['income'] * 100  # convert to percents
    income_data = income_data.reset_index(level=[0, 1])
    educations = income_data['education'].unique()
    educations_number = educations.shape[0]
    offset = 0
    if educations_number > Constants.MAX_NUMBER_OF_SUBPLOTS:
        while educations_number != 0:
            subplots_number = min(educations_number, Constants.MAX_NUMBER_OF_SUBPLOTS)
            createEducationData(income_data, educations, subplots_number, offset)
            offset += subplots_number
            educations_number -= subplots_number
    else:
        createEducationData(income_data, educations, educations_number, 0)


def createEducationData(income_data, educations, subplots_number, subplots_offset):
    _, axes = ppt.subplots(subplots_number, constrained_layout=True)
    education_offset = 0
    index = 0
    for value in educations:
        if index == subplots_number:
            break

        if education_offset < subplots_offset:
            education_offset += 1
            continue

        education_data = income_data[income_data['education'] == value]
        if subplots_number == 1:
            createSingleGraph(axes,
                              education_data['hours-per-week'],
                              education_data['income'],
                              'hours-per-week',
                              'normalized income',
                              value)
            break

        createSingleGraph(axes[index],
                          education_data['hours-per-week'],
                          education_data['income'],
                          'hours-per-week',
                          'normalized income',
                          value)
        index += 1
    ppt.show()


def analyzeIncomeByGender(data: pd.DataFrame):
    income_data = data.groupby('gender')['income'].mean().to_frame()
    income_data['income'] = income_data['income'] * 100
    bins = ['Female', 'Male']
    values = [income_data.loc['Female', 'income'], income_data.loc['Male', 'income']]
    ppt.bar(bins, values)
    ppt.title("Income by gender")
    ppt.ylabel("Income (percents)")
    ppt.show()


def createSingleGraph(ax, x, y, xlabel, ylabel, title):
    ax.plot(x, y)
    ax.plot(x, y)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc='right')


def indexing(filename) -> dict:
    file = open(filename, 'r')
    result = dict()
    index = 0
    for line in file:
        result[line.rstrip('\n')] = index
        index += 1
    return result




def replaceTextualData(data, filename=None) -> pd.DataFrame:
    clone = copy.deepcopy(data)

    columns = clone.columns
    if "education" in columns:
        education = indexing("./data/education.txt")
        clone["education"] = [education[item] for item in clone["education"]]

    if "marital-status" in columns:
        marital_status = indexing("./data/marital-status.txt")
        clone["marital-status"] = [marital_status[item] for item in clone["marital-status"]]

    if "native-country" in columns:
        native_country = indexing("./data/native-country.txt")
        clone["native-country"] = [native_country[item] for item in clone["native-country"]]

    if "occupation" in columns:
        occupation = indexing("./data/occupation.txt")
        clone["occupation"] = [occupation[item] for item in clone["occupation"]]

    if "race" in columns:
        race = indexing("./data/race.txt")
        clone["race"] = [race[item] for item in clone["race"]]

    if "relationship" in columns:
        relationship = indexing("./data/relationship.txt")
        clone["relationship"] = [relationship[item] for item in clone["relationship"]]

    if "workclass" in columns:
        workclass = indexing("./data/workclass.txt")
        clone["workclass"] = [workclass[item] for item in clone["workclass"]]

    return clone


def createHistogram(data):
    data.hist()
    ppt.show()


def excludeColumnsAndCreateCorrTable(data):
    data = data.drop(columns=['marital-status', 'educational-num', 'fnlwgt', 'capital-gain', 'capital-loss'])
    return data


def arrayToString(array):
    result = ""
    size = len(array)
    for index in range(size - 1):
        result += str(array[index]) + ", "
    result += str(array[size - 1])
    return result


# модель классификации. В качестве алгоритма классификации выбран алгоритм KNN 
def kNN(data, max_number_of_neighbours=None):
    # splitting data
    textual_data = data.select_dtypes(include='object')
    numeric_data = data.select_dtypes(exclude='object')

    textual_data = pd.get_dummies(textual_data)


    # remove income column from numeric data and scaling data
    income = numeric_data['income']
    factors = numeric_data.drop(columns='income')
    scaler = preprocessing.StandardScaler()
    std_factors = pd.DataFrame(scaler.fit_transform(factors))
    std_factors.columns = factors.columns

    # adding classification properties and splitting data to train and test
    train_data = pd.concat([std_factors, textual_data], axis=1)
    train, test, train_result, test_result = train_test_split(train_data, income, test_size=0.3,
                                                              random_state=1)

    # learn model, predict values and record prediction accuracy
    predicted = learnModel(train, train_result, test)

    if max_number_of_neighbours is not None:
        mean_acc = np.zeros((max_number_of_neighbours - 1))
        std_acc = np.zeros((max_number_of_neighbours - 1))

        for neighbours_count in range(1, max_number_of_neighbours):
            predicted = learnModel(train, train_result, test, neighbours_count)
            mean_acc[neighbours_count - 1] = metrics.accuracy_score(test_result, predicted)
            std_acc[neighbours_count - 1] = np.std(predicted == test_result) / np.sqrt(predicted.shape[0])

        # create plot for statistics
        ppt.plot(range(1, max_number_of_neighbours), mean_acc, marker='o')
        ppt.legend(('Accuracy '))
        ppt.ylabel('Accuracy ')
        ppt.xlabel('Number of Neighbors (K)')
        ppt.tight_layout()
        ppt.show()


def learnModel(train, train_result, test, number_of_neighbours=5):
    neigh = KNeighborsClassifier(n_neighbors=number_of_neighbours).fit(train, train_result)
    return neigh.predict(test)


data = pd.read_csv('adult.csv')
print(data.head())
print('------------------------------------ АНАЛИЗ ------------------------------------')
print('Количество ячеек, содержащих неизвестные значения (знак ?)')
print(unknownCount(data))
createHistogram(data)
clearData(data)
analyzeIncomeByEducationAndHoursPerWeek(data)
analyzeIncomeByGender(data)
clone = replaceTextualData(data)
data = excludeColumnsAndCreateCorrTable(data)
kNN(data, 25)
