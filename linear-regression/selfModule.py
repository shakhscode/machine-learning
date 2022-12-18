class LinearReg():
    def __init__(self) -> None:
        import numpy as np
        self.np = np

        pass

    
    def add_bias(self,features_mat):
        np=self.np
        l = features_mat.shape[0]
        bias = np.ones([l,1])
        features_mat_with_bias = np.hstack((bias,features_mat))
        #print('Shape after added bias:',features_mat_with_bias.shape)
        return features_mat_with_bias

    def predicted_y(self,features,coeffs):
        # features: n X (m+1) matrix
        # coeffs:  (m+1) X 1 matrix

        np = self.np
        h_i = np.round(np.matmul(features,coeffs))

        #print('Shape of h_i:',h_i.shape)

        # h_i: nX1 column
        return h_i
    
    # return the resiaduals as a column
    def residuals(self,hypothetical,actual):
        #hypothetical: nX1 column
        #actual : nX1 column
        np = self.np

        error_column = hypothetical - actual
        
        #print(error_column.shape)

        # error_column: nX1 column
        return error_column

        
    def gradient(self,errors,jth_feature):
        
        # error_column: nX1 column
        # jth_feature: nX1 column

        np = self.np
        
        gradient_w_j = np.multiply(errors,jth_feature)
        #gradient: nX1 column (from elementwise multiplication)

        #return is a single value for a particular feature_j and w_j
        return (1/len(errors))*np.sum(gradient_w_j,axis=1)
    
    
    #define gradient descent algorithm
    def updateCoeffs(self,features, output_y,lr,iterations):
        #features: nX(m+1) matrix
        #output_y: nX1 column
        np = self.np

        num_features = features.shape[1]

        #generate random coefficients
        #coeffs: (m+1)X1 column
        coeffs = np.random.random(size=(num_features,1))


        for iteration in range(iterations):

            #we have features and coeffs, predict y
            ycap = self.predicted_y(features=features,coeffs=coeffs)

            #calculate error in prediction
            errors_ = self.residuals(ycap,output_y)

            temp_coeffs = []

            for j in range(num_features):

                #the particular feature
                x_j = features[:,j]
                
                #the particular coefficient
                w_j = coeffs[j,0]
        
                #gradients 
                grad_w_j = self.gradient(errors=errors_,jth_feature=x_j)

                
                #update the coefficient
                temp_w_j = w_j - lr*grad_w_j

                coeffs[j,0] = temp_w_j


                #temp_coeffs.append(temp_w_j)
                #print(temp_coeffs)
            
            #coeffs = np.reshape(np.array(temp_coeffs),[num_features,1])
        
        #after all return the optimal coefficients
        #coeffs: (m+1)X1 column
        return coeffs    


    # define a function to train the model
    def train_the_model(self,X_train, y_train, learning_rate = 0.01, iterations = 100, show = False):
        np = self.np
        #first convert them into numpy ndarrays (because they are probably pandas DataFrames)
        X_train = X_train.to_numpy(dtype=np.float128)
        y_train = y_train.to_numpy(dtype=np.float128)

        #add bias to the X_train
        X_train = self.add_bias(X_train)
        
        #Get the optimal coefficients using the updateCoeffs() method
        optimal_coeffs = self.updateCoeffs(X_train,y_train,learning_rate,iterations)
        is_inf = np.isinf(optimal_coeffs)
        optimal_coeffs[is_inf] = np.finfo(np.float).max

        # create an instance variable to pass it to the test function
        self.optimal_coeffs = optimal_coeffs

        if show:
            print(f'Intercept: \n{optimal_coeffs[0,0]}')
            print(f'Coefficients: \n{optimal_coeffs[1:,0]}')
    
    # Define a function to test the model
    def test_the_model(self,X_test):
        np = self.np
        X_test = X_test.to_numpy(dtype=np.float128)
        
        #add bias
        X_test = self.add_bias(X_test)

        predicted_output = self.predicted_y(X_test,self.optimal_coeffs)

        return predicted_output
