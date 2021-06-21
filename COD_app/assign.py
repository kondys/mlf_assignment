from os import system
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import time
import concurrent
from pebble import ProcessPool

np.warnings.filterwarnings('ignore', 'overflow') #disabled warnings for the sigmoid function

CPU_PROCESSES = 12 #CPU processes, reduce this if you are running on a lower spec machine
LOGISTIC_REGRESSION_LOOPS = 50 #Number of iterations inside the logistic regression. #50 should take about 3 mins
LEARNING_RATE = 0.02 #Learning rate... this seems to be the most efficent based on my testing
MIN_GAMES_FOR_GROUPING = 40 #Min number of games required for a group. I.e If the 'mp_nuketown6' map has only been played 30 times and will not be considered

class LogisticRegression:

    #logistic sigmoid function
    def _sigmoid(self, x):                                 
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y):
        #setup variables
        numberOfInputs, variables = X.shape #split the rows and columns into 2 variables
        self.weights = [0.0] * variables #fill array with zeros 
        self.bias = 0 #reset our bias

        #loop for LOGISTIC_REGRESSION_LOOPS number of iterations
        for _ in range(LOGISTIC_REGRESSION_LOOPS):
            linear_model = np.dot(X, self.weights) + self.bias #Computes the weighted sum of input data
            prediction = self._sigmoid(linear_model)

            #Getting the gradients of loss
            dw = (1 / numberOfInputs) * np.dot(X.T, (prediction - y)) #multply numberOfInputs by the results / transposing of X and of the sigmoid less the y (win:1 / loss:0)
            db = (1 / numberOfInputs) * np.sum(prediction - y) #multiply the numberOfInputs by the number of predicted games less y
            
            #Update the weight and bias using our learning rate
            self.weights = self.weights - (LEARNING_RATE * dw)
            self.bias = self.bias- (LEARNING_RATE * db)
            #...then we loop again with the new weight / bias values
            #each loop gets closer to a accurate set of weights and bias (as long as the learning rate wasn't too big to start off with)

    #return the final model
    def getModel(self):
        return {'weights': self.weights, 'bias':self.bias }

    #Predict a result using the current weights and bias
    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        prediction = self._sigmoid(linear_model)
        result = [1 if i >= 0.5 else 0 for i in prediction] #if the result of the sigmoid is more than or equal to 0.5 it is a win, else game will be a loss
        
        return np.array(result)



class CODHelper:
    iv_groups=[]
    independant_vars = ['duration', 'kills', 'ekiadRatio',
        'rankAtEnd', 'shotsLanded', 'highestMultikill', 'score',
        'headshots', 'assists', 'scorePerMinute', 'deaths', 'damageDealt',
        'shotsMissed', 'multikills', 'highestStreak', 'hits', 'timePlayed',
        'suicides', 'timePlayedAlive', 'objectives', 'shotsFired']


    def __init__(self):
        self.loadFullDataSet()

    #Load CSV and prepare data
    #COD_Games.csv has been generated using historical match results from Activision's API using my personal key
    def loadFullDataSet(self):
        df = pd.read_csv('COD_Games.csv', index_col="matchID") 
        df = df[(df.isPresentAtEnd == 1)]   #Restrict games to the playing the whole game till the end
        df = df[(df.result == 'win') | (df.result == 'loss')] #Only interested in wins or losses
        df.result = df.result.map( {'win':1 , 'loss':0} ) #Convert string win to 1 and string loss to 0
        df['map_mode'] = df['map'].str.cat(df['mode'],sep="-") #Combine map and mode to one single column

        self.df = df

    #Create unique independant variable combinations to feed the learning with different data
    #i.e ['kills','headshots','objectives']
    # group_count sets the number of combinations 1, 2 or 3
    def buildUniqueIndependantVariableCombinations(self, group_count):
        iv_groups=[]
        for i in range(0, len(self.independant_vars)-1):
            if group_count==1:
                iv_groups.append([self.independant_vars[i]])
            else:
                for j in range(i+1, len(self.independant_vars)-1):
                    if group_count ==2 :
                        iv_groups.append([self.independant_vars[i], self.independant_vars[j] ])
                    else:
                        for k in range(j+1, len(self.independant_vars)-1):
                            iv_groups.append([self.independant_vars[i], self.independant_vars[j], self.independant_vars[k]])
        
        return iv_groups

    
    #Cacluate the accuracy of the results by comparing the model result to the actual game result
    #Divide total of correct guesses by the total games to get the accruacy (between 0.0 and 1.0)
    def accuracy(self, y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy       

    #Run a test for a particular type of game and loop through independant variables to see what is the best result
    #If the accuracy is higher than the last independant variable set then it becomes the winner
    def runLRFilter(self, pool, filterVariableName, iv_group_length ):
        bestresults = {}
 
        #generate our iv combinations
        iv_groups = self.buildUniqueIndependantVariableCombinations(iv_group_length)

        #no grouping scenario, we are looking at all records
        if filterVariableName == 'no_grouping':
            filters = ['no_grouping']
        else:
            #We filter on the overall data set to find games grouped by type weare interested in.
            #Groupings are ignored if the count falls below the MIN_GAMES_FOR_GROUPING value
            #This is done so that you don't compare low game combinations that don't have enough data
            filters = self.df.groupby(filterVariableName).filter(lambda x: x.shape[0] > MIN_GAMES_FOR_GROUPING)[filterVariableName].unique()  

        for filterValue in filters:
            bestresults[filterValue] = {'score':0}
            
            my_iterable = []
            #build a list of iv combinations we want to run LRs over
            for iv_group in iv_groups:
                my_iterable.append([filterValue, filterVariableName, iv_group])
                
            #run the LRs in a process pool. This speeds things up quite a bit
            results = [pool.schedule(self.runLogisticRegression, args=[value]) for value in my_iterable]
            completed, pending = concurrent.futures.wait(results)

            # cancel pending futures
            for future in pending:
                future.cancel()


            #Once pool is finished compare results and pick the best performing IV group
            #The best performing is the one with the highest accuracy
            for r in completed:
                result = r.result()
                br = bestresults[filterValue]
                if result['score'] > br['score']:
                    br['score']= result['score']
                    br['iv_group'] = result['iv_group']
                    br['model'] = result['model']
                    br['gamesPlayed'] = result['gamesPlayed']
        return bestresults

    #Save results to file + print to screen
    def printResultsSummary(self, game_type, iv_group_count, results):
        totalScore = 0.0

        print("\n\nGame type: {game_type}, iv_groupings: {iv_group_count} with {LOGISTIC_REGRESSION_LOOPS} LR loops and {LEARNING_RATE} learning rate:".format(game_type=game_type, LEARNING_RATE = LEARNING_RATE, LOGISTIC_REGRESSION_LOOPS = LOGISTIC_REGRESSION_LOOPS, iv_group_count=iv_group_count ))
        for key in results:
            r = results[key]
            totalScore += r['score']
            print(f"    {key}, score:{r['score']}, best ivs:{r['iv_group']}\n            model:{r['model']}, gamesPlayed:{r['gamesPlayed']}")

        averageAccuracy = totalScore/len(results)
        print(f"Average accuracy: {averageAccuracy}")
        
        f = open(f"run_logs/stats_LR_{LOGISTIC_REGRESSION_LOOPS}.txt", "a")
        f.write(f"averageAccuracy: {averageAccuracy}, game_type: {game_type}, iv_group_count:{iv_group_count}\n")
        f.close()        


    #Run the Logistic regression for a filter with a specific independant variable
    #i.e for 'map' of type 'mp_miami' run a LR for ['kills','timePlayedAlive', 'objectives']
    #After LR has been run, check the test games against the model to calculate the accuracy
    def runLogisticRegression(self, args):
        filterValue, filterVariableName, iv_group = args
        lr = LogisticRegression()

        filteredDf = self.df if filterVariableName == 'no_grouping' else self.df[(self.df[filterVariableName] == filterValue)]
        y = filteredDf.result #extract the game result (win/loss)
        X = filteredDf.loc[:, iv_group] #only select the columns we want to run a LR on

        #split our data into 70% training, 30% test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        #fit our training data
        lr.fit(X_train, y_train)

        #run the test data to see how we went
        predictions = lr.predict(X_test)

        #work out how well we did
        score = self.accuracy(y_test, predictions)

        #return the stats
        return {'score':score, 'gamesPlayed':len(filteredDf), 'filterValue':filterValue, 'iv_group':iv_group, 'model':lr.getModel()}

'''
Run logistic regressions
For each game type: map, mode and map+mode combined


map = Call of Duty maps https://www.gamesatlas.com/cod-black-ops-cold-war/maps/
mode = Type of game played https://www.callofduty.com/blog/2020/11/Black-Ops-Cold-War-Multiplayer-Modes
map_mode = combinations of game modes played on specific maps


The results:
    - 'map' will return the best independant variables that would tell us if the game was going to result in a win or loss on a given map
    - 'mode' will return the best independant variables for game mode that would tell us if the game was going to be a win or loss
    - 'map_mode' will return the best independant variables for a map and mode combination to predict a win/loss
    - 'no_grouping' will return the best independant variables that would tell us if the game was going to result in a win or loss on any map / mode
    All combinations need to have at least 40 games (MIN_GAMES_FOR_GROUPING) so not to create a poor model

iv_group_count 1-3 will try check different iv groupings:
    (1) ['kills']
    (2) ['kills','headshots'] 
    (3) ['kills','headshots','objectives'] ...etc

LOGISTIC_REGRESSION_LOOPS gives us interesting results (2017 Macbook Pro):
    5000 iterations takes 4+ hours and best avg. accuracy for map_mode is 0.9153439153439153
    1000 iterations, 60 mins, 0.9116090880796763
    100 iterations, 7 mins, 0.8734827264239029
    50 iterations, 3 mins, 0.8930905695611577

The best results are for map_mode combinations with 3 independant variables.
Game type: map_mode, iv_groupings: 3 with 1000 LR loops and 0.02 learning rate:

    mp_kgb-control_cdl, score:0.9411764705882353, best ivs:['kills', 'ekiadRatio', 'headshots']
            model:{'weights': array([-0.12139226,  1.27216461,  0.25889435]), 'bias': -0.27291896604698096}, gamesPlayed:54
    mp_tank-control_cdl, score:0.8888888888888888, best ivs:['ekiadRatio', 'highestMultikill', 'deaths']
            model:{'weights': array([ 1.02329268, -0.32187085, -0.05403984]), 'bias': 0.20085351710537208}, gamesPlayed:59
    mp_raid_rm-control_cdl, score:0.9047619047619048, best ivs:['kills', 'highestStreak', 'objectives']
            model:{'weights': array([-0.34176689,  1.1054231 ,  0.20881256]), 'bias': -1.1330294091635222}, gamesPlayed:68

Average accuracy: 0.9116090880796763

'''
def run(pool):
    start = time.time()
    helper = CODHelper()
    
    #Loop through filters
    for game_type in ['no_grouping','map','mode','map_mode']:
        #iv grouping variations
        for iv_group_count in [1,2,3]:
            results = helper.runLRFilter(pool, game_type, iv_group_count)
            helper.printResultsSummary(game_type, iv_group_count, results)
            
    #calculate total time taken & print to screen
    end = time.time()
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Total Time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))


if __name__ == '__main__':
    with ProcessPool(CPU_PROCESSES) as pool:
        run(pool)