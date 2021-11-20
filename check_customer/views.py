from django.shortcuts import render
from django.shortcuts import redirect, render, get_object_or_404
from django.contrib import messages
from django.views.generic import CreateView
from .models import customer_details
import pandas as pd
from sklearn.utils.validation import check_array
import numpy as np
from joblib import dump, load
import category_encoders as ce
import pickle

def home(request):
    return render(request, 'check_customer/home.html')

def credit_score(request):
    if(request.method=='POST'):
        print(request.POST)
        #input
        customer = customer_details()
        customer.title = request.POST.get('title')
        customer.name = request.POST.get('name')
        customer.age = request.POST.get('age')
        customer.gender = request.POST.get('gender')
        customer.housing = request.POST.get('housing')
        customer.residence_since = request.POST.get('residence_since')
        customer.marital_status = request.POST.get('marital-status')
        customer.telephone = request.POST.get('telephone')
        customer.foreign = request.POST.get('foreign')
        customer.title = request.POST.get('title')
        customer.employment_status = request.POST.get('employment_status')
        customer.employment_type = request.POST.get('employment_type')
        customer.checking_account = request.POST.get('checking_account')
        customer.savings_account = request.POST.get('savings_account')
        customer.property_type = request.POST.get('property_type')
        customer.installment_plans = request.POST.get('installment_plans')
        customer.existing_credits = request.POST.get('existing_credits')
        customer.credit_history = request.POST.get('credit_history')
        customer.credit_amount = request.POST.get('credit_amount')
        customer.duration = request.POST.get('duration')
        customer.installment_rate = request.POST.get('installment_rate')
        customer.purpose = request.POST.get('purpose')
        customer.debtor = request.POST.get('debtor')
        customer.maintainence = request.POST.get('maintainence')
        Personal_Status_and_Sex = ''
        if(customer.gender=='Female'):
            Personal_Status_and_Sex = 'A92'
        else:
            if(customer.marital_status=='divorced'):
                Personal_Status_and_Sex = 'A91'
            else:
                if(customer.marital_status=='single'):
                    Personal_Status_and_Sex = 'A93'
                else:
                    Personal_Status_and_Sex = 'A94'
        
        checking_dm = int(customer.checking_account)/42.84
        savings_dm = int(customer.savings_account)/42.84
        if(checking_dm==0):
            checking_account_category = 'A11'
        elif(checking_dm<=200):
            checking_account_category = 'A12'
        else:
            checking_account_category = 'A13'
        
        if(savings_dm==0):
            savings_account_category = 'A65'
        elif(savings_dm<100):
            savings_account_category = 'A61'
        elif(savings_dm<500):
            savings_account_category = 'A62'
        elif(savings_dm<1000):
            savings_account_category = 'A63'
        else:
            savings_account_category = 'A64'
        
        
        
        data = {'Checking Account':[checking_account_category],
                'Duration':[int(customer.duration)],
                'Credit History':[customer.credit_history],
                'Purpose':[customer.purpose],
                'Credit Amount':[int(customer.credit_amount)],
                'Savings Account':[savings_account_category],
                'Employment Length':[customer.employment_status],
                'Installment Rate':[int(customer.installment_rate)],
                'Personal Status and Sex':[Personal_Status_and_Sex],
                'Other Debtors':[customer.debtor],
                'Residence Since':[int(customer.residence_since)],
                'Property':[customer.property_type],
                'Age':[int(customer.age)],
                'Other Installments':[customer.installment_plans],
                'Housing':[customer.housing],
                'Number of Credits at this bank':[int(customer.existing_credits)],
                'Job':[customer.employment_type],
                'Number of People Liable':[int(customer.maintainence)],
                'Telephone':[customer.telephone],
                'Foreign Worker':[customer.foreign],
                'Good/Bad':[1]}
        applicant = pd.DataFrame(data)
        
        applicant['Duration']= (applicant['Duration']/10.0).astype(int)
        applicant['Age']= (applicant['Age']/15.0).astype(int)

        applicant = applicant.drop(['Purpose', 'Employment Length', 'Installment Rate', 'Residence Since', 'Other Installments', 'Number of Credits at this bank','Job', 'Number of People Liable', 'Telephone',  ], axis=1)

        encoder= ce.OrdinalEncoder(cols=['Checking Account'],return_df=True, mapping=[{'col':'Checking Account','mapping':{'A11':0,'A12':1,'A13':2,'A14':3}}])
        applicant_transformed = encoder.fit_transform(applicant)

        encoder= ce.OrdinalEncoder(cols=['Credit History'],return_df=True, mapping=[{'col':'Credit History','mapping':{'A30':0,'A31':1,'A32':2,'A33':3, 'A34':4}}])
        applicant_transformed = encoder.fit_transform(applicant_transformed)

        encoder= ce.OrdinalEncoder(cols=['Savings Account'],return_df=True, mapping=[{'col':'Savings Account','mapping':{'A61':0,'A62':1,'A63':2,'A64':3, 'A65':4}}])
        applicant_transformed = encoder.fit_transform(applicant_transformed)

        #Using saved one hot encoder
        #------------------------------------------------------------------------------------------------------------------------------

        encoder = load('check_customer/Files/personal_status_and_sex.joblib')
        applicant_transformed = encoder.transform(applicant_transformed)

        encoder = load('check_customer/Files/other_debtors.joblib')
        applicant_transformed = encoder.transform(applicant_transformed)

        #-------------------------------------------------------------------------------------------------------------------------------              
        encoder= ce.OrdinalEncoder(cols=['Property'],return_df=True, mapping=[{'col':'Property','mapping':{'A121':0,'A122':1,'A123':2,'A124':3}}])
        applicant_transformed = encoder.fit_transform(applicant_transformed)

        encoder= ce.OrdinalEncoder(cols=['Housing'],return_df=True, mapping=[{'col':'Housing','mapping':{'A152':0,'A151':1,'A153':2}}])
        applicant_transformed = encoder.fit_transform(applicant_transformed)

        encoder= ce.OrdinalEncoder(cols=['Foreign Worker'],return_df=True, mapping=[{'col':'Foreign Worker','mapping':{'A201':0,'A202':1}}])
        applicant_transformed = encoder.fit_transform(applicant_transformed)


        #### scale before prediction
        sc = load('check_customer/Files/standard_scaler.joblib')
        applicant_input = sc.transform(applicant_transformed.iloc[:, :-1].values)
        
        #### prediction
        estimators_ = load_estimators()
        X_l_ = load('check_customer/Files/X_l_.joblib')
        y_l_ = load('check_customer/Files/y_l_.joblib')
        machine_predictions_ = load('check_customer/Files/machine_predictions_.joblib')
        predicted_result, predicted_prob = predict(applicant_input, estimators_, X_l_, y_l_, machine_predictions_)
        customer.bad = predicted_result
        customer.res_prob = predicted_prob
        
    return render(request, 'check_customer/credit_score.html', context={'customer':customer})

def pred(estimators_, X_l_, y_l_, machine_predictions_, X, M, info=False):
        """
        Performs the CLassififerCobra aggregation scheme, used in predict method.
        Parameters
        ----------
        X: array-like, [n_features]
        M: int, optional
            M refers to the number of machines the prediction must be close to to be considered during aggregation.
        info: boolean, optional
            If info is true the list of points selected in the aggregation is returned.
        Returns
        -------
        result: prediction
        """

        # dictionary mapping machine to points selected
        select = {}
        for machine in estimators_:
            # machine prediction
            model = estimators_[machine]
            label = model.predict(X)
            select[machine] = set()
            # iterating from l to n
            # replace with numpy iteration
            for count in range(0, len(X_l_)):
                if machine_predictions_[machine][count] == label:
                    select[machine].add(count)

        points = []
        # count is the indice number.
        for count in range(0, len(X_l_)):
            # row check is number of machines which picked up a particular point
            row_check = 0
            for machine in select:
                if count in select[machine]:
                    row_check += 1
            if row_check == M:
                points.append(count)

        # if no points are selected, return 0
        if len(points) == 0:
            if info:
                logger.info("No points were selected, prediction is 0")
                return (0, 0)
            logger.info("No points were selected, prediction is 0")
            return 0

        # aggregate
        classes = {}
        for label in np.unique(y_l_):
            classes[label] = 0

        for point in points:
            classes[y_l_[point]] += 1

        result = int(max(classes, key=classes.get))
        prob = classes[1]/(classes[0]+classes[1])
        if info:
            return result, points, prob
        return result, round(prob, 3)


    
##### estimators_

def load_estimators():
    estimators_ = {}
    machine_list = ['naive_bayes', 'tree', 'knn', 'svm', 'logreg']
    for machine in machine_list:
            try:
                if machine == 'svm':
                    estimators_['svm'] = pickle.load(open('check_customer/Files/Model_svm.sav','rb'))
                if machine == 'knn':
                    estimators_['knn'] = pickle.load(open('check_customer/Files/Model_knn.sav','rb'))
                if machine == 'tree':
                    estimators_['tree'] = pickle.load(open('check_customer/Files/Model_tree.sav','rb'))
                if machine == 'logreg':
                    estimators_['logreg'] = pickle.load(open('check_customer/Files/Model_logreg.sav','rb'))
                if machine == 'naive_bayes':
                    estimators_['naive_bayes'] = pickle.load(open('check_customer/Files/Model_naive_bayes.sav','rb'))
            except ValueError:
                continue
                
    return estimators_
    
def predict( X, estimators_, X_l_, y_l_, machine_predictions_, M=None, info=False):
        """
        Performs the ClassifierCobra aggregation scheme, calls pred.
        ClassifierCobra performs a majority vote among all points which are retained by the COBRA procedure.
        Parameters
        ----------
        X: array-like, [n_features]
        M: int, optional
            M refers to the number of machines the prediction must be close to to be considered during aggregation.
        info: boolean, optional
            If info is true the list of points selected in the aggregation is returned.
        Returns
        -------
        result: prediction
        """
        X = check_array(X)

        if M is None:
            M = len(estimators_)
        if X.ndim == 1:
            return pred(estimators_, X_l_, y_l_, machine_predictions_, X.reshape(1, -1), M=M)

        result = np.zeros(len(X))
        avg_points = 0
        index = 0
        for vector in X:
            if info:
                result[index], points, prob = pred(estimators_, X_l_, y_l_, machine_predictions_, vector.reshape(1, -1), M=M, info=info)
                avg_points += len(points)
            else:
                result[index], prob = pred(estimators_, X_l_, y_l_, machine_predictions_, vector.reshape(1, -1), M=M)
            index += 1
        
        if info:
            avg_points = avg_points / len(X_array)
            return result, avg_points, prob
        
        return result, prob
