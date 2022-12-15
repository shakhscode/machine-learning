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

    
    # return the resiaduals as a column
    def residuals(self,features,coefficients,output):
        np = self.np
        #features = self.add_bias(features)
         
        #print('Features shape:',features.shape)

        #print('Coefficients shape:',coefficients.shape)
        
        h_i_mat = np.round(np.multiply(features,coefficients),4)

        #print('h_i_mat_shape', h_i_mat.shape)


        h_i = np.reshape(np.sum(h_i_mat,axis=1),[features.shape[0],])

        #print('Output Shape', output.shape)
        #print('H_i shape', h_i.shape)
        
        
        error_column = np.round(np.subtract(output,h_i),4)
        #print(error_column.shape)

        return error_column

        
    def sum_error_x_j(self,features,coefficients,output,j):
        np = self.np
        features = self.add_bias(features_mat=features)

        errors = self.residuals(features,coefficients,output)
        
        error_x_j = np.round(np.multiply(errors,features[:,j]),4)
        return np.round(np.sum(error_x_j))
    
    
    #define gradient descent algorithm
    def updateCoeffs(self,features, output,lr,iters):
        np = self.np
        m = features.shape[1]
        n =  features.shape[0]
        
        #coeffs = np.zeros((1,m+1),dtype=np.float128)
        coeffs = np.random.random(size=(1, m+1))
        #print(coeffs)
        #print(type(coeffs))

        for iteration in range(iters):

            temp_coeffs = []

            for j in range(m+1):
                #print('j = ',j)

                coeffs_j = coeffs[0][j]
                #print('jth_coeff:', coeffs_j)

                temp_j = coeffs_j - np.round(lr*(-2/n)*self.sum_error_x_j(
                                    features,coeffs,output,j=j),4)

                temp_coeffs.append(temp_j)
            
            coeffs = np.reshape(np.array(temp_coeffs),[1,m+1])
            
            #print('Type of updated coeffs', type(coeffs))
            #print('updated coeffs', coeffs)

        return coeffs    


    # define a function to train the model

    def train_the_model(self,X_train, y_train, learning_rate = 0.01, iterations = 100, show = False):
        np = self.np
        #first convert them into numpy ndarrays
        X_train = X_train.to_numpy(dtype=np.float128)
        y_train = y_train.to_numpy(dtype=np.float128)

        optimal_coeffs = self.updateCoeffs(X_train,y_train,learning_rate,iterations)
        is_inf = np.isinf(optimal_coeffs)
        optimal_coeffs[is_inf] = np.finfo(np.float).max

        # create an instance variable to pass it to the test function
        self.optimal_coeffs = optimal_coeffs

        if show:
            print(f'Intercept: \n{optimal_coeffs[0][0]}')
            print(f'Coefficients: \n{optimal_coeffs[0][1:]}')
    
    # Define a function to test the model
    def test_the_model(self,X_test):
        np = self.np
        X_test = X_test.to_numpy(dtype=np.float128)

        #predict using the obtained coefficients
        X_test = self.add_bias(X_test)

        prediction_mat = np.multiply(X_test,self.optimal_coeffs)
        predictions = np.sum(prediction_mat,axis = 1)
        return predictions
