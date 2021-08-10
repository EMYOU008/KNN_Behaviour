from sklearn import neighbors

from NN_plotit import getExamples,accuracy,NN

from time import time


import matplotlib.pyplot as plt

##############################################Part 2a#######################################
def sample(size):
    mars=NN()
    kitkat=neighbors.KNeighborsClassifier(5)
    Run_time=[]
    Run_time_m=[]

    for n in range(5,size):

        (x_train,y_train)=getExamples(n, d=2)  #taking random take for training from getexample function from NN_Plotit file
        start=time()   #time starting when training start
        kitkat.fit(x_train,y_train)  #fit the model foe the given input


        (x_test,y_test)=getExamples(n,d=2) # taking data for the test purpose
        y_predict=kitkat.predict(x_test) #predict the labels against the features

        end=time() # end time after traing and testing
        t_time=end-start

        # print(t_time)
        # z=accuracy(y_test,y_predict)
        # print(z)
        Run_time.append(t_time)

        # doing the same things as done above but for our own model not sklearn
        (x_train, y_train) = getExamples(n, d=2)
        start_m=time()
        mars.fit(x_train, y_train,5)
        (x_test, y_test) = getExamples(n, d=2)
        mars.predict(x_test)
        end_m=time()
        m_time=end_m-start_m
        Run_time_m.append(m_time)

    plt.plot(range(5,size),Run_time)

    plt.plot(range(5, size), Run_time_m)
    plt.xlabel("sample size")
    plt.ylabel("Time")
    plt.legend(("MY CLassifier","sklearn Classifier"))

    plt.show()
    return Run_time,Run_time_m

###########################################Part 2b#########################################

def sample_1(dimension):
    mars=NN()
    kitkat=neighbors.KNeighborsClassifier(5)
    Run_time=[]
    Run_time_m=[]
    for d in range(1,dimension):

        (x_train,y_train)=getExamples(200,dimension)  #taking random take for training from getexample function from NN_Plotit file
        start=time()   #time starting when training start
        kitkat.fit(x_train,y_train)  #fit the model foe the given input


        (x_test,y_test)=getExamples(200,dimension) # taking data for the test purpose
        y_predict=kitkat.predict(x_test) #predict the labels against the features

        end=time() # end time after traing and testing
        t_time=end-start

        # print(t_time)
        # z=accuracy(y_test,y_predict)
        # print(z)
        Run_time.append(t_time)

        # doing the same things as done above but for our own model not sklearn
        (x_train, y_train) = getExamples(200,dimension)
        start_m=time()
        mars.fit(x_train, y_train,5)
        (x_test, y_test) = getExamples(200, dimension)
        mars.predict(x_test)
        end_m=time()
        m_time=end_m-start_m
        Run_time_m.append(m_time)

    plt.plot(range(1, dimension), Run_time)

    plt.plot(range(1, dimension), Run_time_m)

    plt.xlabel("Dimension size")
    plt.ylabel("Time")
    plt.legend(("MY CLassifier", "sklearn Classifier"))

    plt.show()


    return Run_time,Run_time_m

##################################################Part 2D##################################

def sample_2(cluster):
    mars=NN()

    Run_time=[]
    Run_time_m=[]
    for k in range(1,cluster,2):
        kitkat = neighbors.KNeighborsClassifier(k)

        (x_train,y_train)=getExamples(n=200, d=2)  #taking random take for training from getexample function from NN_Plotit file
        start=time()   #time starting when training start
        kitkat.fit(x_train,y_train)  #fit the model foe the given input


        (x_test,y_test)=getExamples(n=200,d=2) # taking data for the test purpose
        y_predict=kitkat.predict(x_test) #predict the labels against the features

        end=time() # end time after traing and testing
        t_time=end-start

        # print(t_time)
        # z=accuracy(y_test,y_predict)
        # print(z)
        y = accuracy(y_test, y_predict)
        Run_time.append(y)

        # doing the same things as done above but for our own model not sklearn
        (x_train, y_train) = getExamples(n=200, d=2)
        start_m=time()
        mars.fit(x_train, y_train,k)
        (x_test, y_test) = getExamples(n=200, d=2)
        mars.predict(x_test)
        end_m=time()
        y = accuracy(y_test, y_predict)
        #m_time=end_m-start_m
        Run_time_m.append(y)

    plt.plot(range(1, cluster,2), Run_time)

    plt.plot(range(1, cluster,2), Run_time_m)

    plt.xlabel("Number of K")
    plt.ylabel("Accuracy")

    plt.show()
    return Run_time_m

####################################Part 2C############################

def sample_3(cluster):
    mars=NN()
    Run_time_k = []
    Run_time=[]
    Run_time_m=[]
    for k in range(1,cluster,2):
        kitkat = neighbors.KNeighborsClassifier(k)

        (x_train,y_train)=getExamples(n=200, d=2)  #taking random take for training from getexample function from NN_Plotit file
        start=time()   #time starting when training start
        kitkat.fit(x_train,y_train)  #fit the model foe the given input


        (x_test,y_test)=getExamples(n=200,d=2) # taking data for the test purpose
        y_predict=kitkat.predict(x_train) #predict the labels against the features

        end=time() # end time after traing and testing
        t_time=end-start

        # print(t_time)
        z=accuracy(y_test,y_predict)
        # print(z)
        Run_time.append(z)

        # doing the same things as done above but for our own model not sklearn
        (x_train, y_train) = getExamples(n=200, d=2)
        start_m=time()
        mars.fit(x_train, y_train,k)
        (x_test, y_test) = getExamples(n=200, d=2)
        mars.predict(x_train)
        # end_m=time()
        # m_time=end_m-start_m
        # Run_time_m.append(m_time)
        z=accuracy(y_train,y_predict)

        Run_time_k.append(z)

    plt.plot(range(1, cluster, 2), Run_time)

    plt.plot(range(1, cluster, 2), Run_time_k)

    plt.xlabel("Number of K")
    plt.ylabel("Accuracy")

    plt.show()
    return Run_time_k


##################################main#################################

if __name__ == '__main__':
    ML=sample(200)
    print(ML)
    NL=sample_1(10)
    print(NL)
    LL=sample_2(15)
    print(LL)
    OL=sample_3(15)
    print(OL)
    plt.plot()



