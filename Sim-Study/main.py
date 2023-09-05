import numpy as np
import splaytree as SplayTree
import math
from enum import Enum
import matplotlib.pyplot as plt


## RANDOM GEN PARAMS
TOLERANCE_MEAN = 3
TOLERANCE_STD = 2

QUALITY_MEAN = 0.5
QUALITY_STD = 0.3
        
INTERARRIVAL_TIME_MEAN = 1
INTERARRIVAL_TIME_STD = 0.5
#note (sarah): for the exponential distribution, standard deviation=mean

VIEWING_TIME_MEAN = 3
VIEWING_TIME_STD = 0.5


MIN_SCORE = 130

## SCORE PARAMS
## Adjustes relativve weight of each score.
## all between 0 and 1
STYLE_CONSTANT = 1
PATIENCE_CONSTANT = 1
QUALITY_CONSTANT = 1


DEBUG=False

# enum of event types
class EventType(Enum):
    ARRIVAL = 0
    DEPARTURE = 1
    MOVE = 2

    def __str__(self):
        if self == EventType.ARRIVAL:
            return "ARRIVAL"
        elif self == EventType.DEPARTURE:
            return "DEPARTURE"
        elif self == EventType.MOVE:
            return "MOVE"
        else:
            return "UNKNOWN"

# enum of painting styles
class Style(Enum):
    BAROQUE = 0
    IMPRESSIONIST = 1
    MODERN = 2
    ABSTRACT = 3

    def __str__(self):
        if self == Style.BAROQUE:
            return "BAROQUE"
        elif self == Style.IMPRESSIONIST:
            return "IMPRESSIONIST"
        elif self == Style.MODERN:
            return "MODERN"
        elif self == Style.ABSTRACT:
            return "ABSTRACT"
        else:
            return "UNKNOWN"

    @staticmethod
    def random(rng):
        return rng.integers(0, 4)

class Painting:
    def __init__(self, id: int, style: Style, rng):
        self.id = id
        self.style = style
        self.num_viewers = 0
        self.quality: float = np.clip(rng.normal(QUALITY_MEAN, QUALITY_STD), 0, 0.9999)

class Customer:

    CurrentID = 0

    def __init__(self, num_paintings: int, rng):
        self.id: int = Customer.CurrentID
        Customer.CurrentID += 1

        self.rng = rng
        self.favorite_style: Style = Style.random(rng)

        # positive number, higher the tolerance the less affectec the customer is by the number of viewers
        self.tolerance: float = np.clip(rng.normal(TOLERANCE_MEAN, TOLERANCE_STD), 0.000001, 100)

        self.viewedPaintings = np.full(num_paintings, False)

        self.current_painting: Painting = None


        self.stats = CustomerStats()

    #get the customer's favorite style
    def favoriteStyle(self):
        return self.favorite_style
       

    def beginViewing(self, painting: Painting, time: float):
        painting.num_viewers += 1
        self.viewing_time: float = np.clip(self.rng.normal(VIEWING_TIME_MEAN, VIEWING_TIME_STD), 0.000001, 100)
        # if(DEBUG): #(sarah)
        #     print(self.id,' started viewing', painting.id, 'now there are %d viewers'%painting.num_viewers)
        return self.viewing_time

    def calcViewerScore(self, num_viewers):
        retval = 1/(max(math.sqrt(num_viewers) * 1/self.tolerance, 1)) * PATIENCE_CONSTANT * 100
        if(DEBUG):
            print("Viewer score: " + str(retval))
        return retval
    
    def calcQualityScore(self, quality):
        # Higher this is the lower the slope in the middle
        SLOPE_CONSTANT = 10
        ## SHouldn't edit this, the midpoint of the inverse sigmoid is 0.5 at MIDPOINT_HEIGHT of 1. Instead adjust the QUALITY_CONSTANT
        ## Like you can edit this but you shouldn't need to unless you're doing fancy stuff
        MIDPOINT_HEIGHT = 1

        ## For quality we want to have low quality paintings be very undesirable and high quality very desirable but 25-75 should have less of an effect
        ## We can do this by using an inverse sigmoid function
        retval =  np.clip((0.5 - math.log((1-quality)/(quality * MIDPOINT_HEIGHT))/SLOPE_CONSTANT) * QUALITY_CONSTANT * 100, 0, 100)
        if(DEBUG):
            print("Quality score: " + str(retval))
        return retval
    
    def calcStyleScore(self, style):

        retval= 1 * STYLE_CONSTANT if style == self.favorite_style else 0
        if(DEBUG):
            print("Style score: " + str(retval))
        return retval

    def scorePainting(self, painting: Painting) -> float:
        if self.viewedPaintings[painting.id]:
            return -1
        

        if(DEBUG):
            print("Scoring painting " + str(painting.id) + " for customer " + str(self.id) + " with style " + str(self.favorite_style) + " and tolerance " + str(self.tolerance) + " and quality " + str(painting.quality) + " and viewers " + str(painting.num_viewers) + " and style " + str(painting.style))
        # uses the tolerance and the number of viewers to calculate a score, 
        # score is increased by a multiple if the painting is the same style as the customer's favorite style
        qualityScore = self.calcQualityScore(painting.quality)
        viewerScore = self.calcViewerScore(painting.num_viewers)
        styleScore = self.calcStyleScore(painting.style)

        return qualityScore + viewerScore + styleScore


    
class Event:
    def __init__(self, type: EventType, time: float, customer: Customer):
        self.time = time
        self.customer = customer
        self.type = type
    
    def get_time(self):
        return self.time

    def __lt__(self, other):
        return self.time < other.get_time()

    def __eq__(self, other):
        return self.time == other.get_time()

    def __ne__(self, other):
        return not self.__eq__(other)

    def __gt__(self, other):
        return not self.__lt__(other) and not self.__eq__(other)

    def __le__(self, other):
        return self.__lt__(other) or self.__eq__(other)

    def __ge__(self, other):
        return self.__gt__(other) or self.__eq__(other)

### EVENT LIST CODE ###
class EventList:

    def __init__(self):
        self.splaytree = SplayTree.SplayTree() 

    def enqueue(self, n):
        self.splaytree.insert(n)

    def getMin(self):
        return self.splaytree.findMin()
    
    def dequeue(self):
        #sarah: splaytree.remove() doesn't return the key it removes, so we have to do it ourselves
        min_key = self.splaytree.findMin()
        self.splaytree.remove(min_key)
        return min_key

class CustomerStats:
    def __init__(self):
        self.arrival_time = 0.0
        self.departure_time = 0.0 
        self.departed = False
        self.arrived = False
        self.num_paintings_viewed = 0
        self.total_viewing_time = 0.0
        self.score_history = []
        self.saw_favorite_style = False


class SimStats:
    def __init__(self, num_customers, num_paintings):
        self.num_customers = num_customers
        self.num_paintings = num_paintings

        self.total_viewing_time = 0.0
        self.num_arrived = 0
        self.num_departed = 0
        self.num_paintings_viewed = 0
        self.num_leave_early = 0
        self.painting_scores = []

        #keep track of number of people with favorite style paintings 
        self.num_baroque = 0
        self.num_impressionist = 0
        self.num_modern = 0
        self.num_abstract = 0
        self.attractiveness_for_favourite = 0 #calculated at ProcessMove

        self.num_customers_leave_early = [0 for i in range(num_paintings)]
        
        self.num_painting_views = [0 for i in range(num_paintings)]

        self.leave_early_scores = []
        self.num_painting_left_leave_early = 0





    def printStats(self, customers, paintings):
        average_viewing_time = self.total_viewing_time / self.num_paintings_viewed

        average_paintings_viewed = (((self.num_paintings_viewed / len(paintings))/len(customers)) * 100)

        # percentage of people who saw their favorite style
        # count number of customers who saw their favorite style
        saw_favorite_style = 0
        for customer in customers:
            if customer.stats.saw_favorite_style:
                saw_favorite_style += 1

        percent_saw_favorite_style = ((saw_favorite_style / self.num_customers) * 100)

        # average attractiveness for favourite is the average attractiveness for all customers who saw their favourite style
        avgerage_attractiveness_for_favourite = ((self.attractiveness_for_favourite / saw_favorite_style))

        # average painting score is the average for all painting scores for all customers
        average_painting_score = np.average(np.array(self.painting_scores))

        # make list of painting qualities in one line. From the 'paintings' array
        painting_qualities = [i.quality for i in paintings]


        ###### Performance Stats ######


        ###### General Stats ######
        print("General Stats:")

        print("Number of Customers Arrived: {}".format(self.num_arrived))
        print("Number of Customers Departed: {}".format(self.num_departed))


        ###### Painting Stats ######
        print()
        print("Statistics for Each Painting:")
        print("Number of Paintings: {}".format(self.num_paintings))
        print("Average Number of Views for Each Painting: {:.2f}".format(np.average(np.array(self.num_painting_views))))
        print("Quality of Each Painting: {}".format(painting_qualities))
        ## When printing the average quality round the number to 2 decimal places

        print("Average quality of paintings: {:.2f}".format(np.average(np.array(painting_qualities))))
        print("Maximum quality of paintings: {:.2f}".format(np.max(np.array(painting_qualities))))
        print("Minimum quality of paintings: {:.2f}".format(np.min(np.array(painting_qualities))))

        ###### Customer Stats ######
        print()
        print("Statistics for Each Customer:")
        print("Number of Customers: {}".format(self.num_customers))
        print("Average Percentage of Paintings Viewed per Customer: {:.2f}%".format(average_paintings_viewed))
        print("Average Viewing Time for each painting: {:.2f}".format(average_viewing_time))
        print("Average Painting Score: {:.2f}".format(np.average(np.array(self.painting_scores))))
        print("Maximum Painting Score: {:.2f}".format(np.max(np.array(self.painting_scores))))
        print("Minimum Painting Score: {:.2f}".format(np.min(np.array(self.painting_scores))))


        ###### Favorite Style Stats ######
        print()
        print("Statistics for Customers who saw their favorite style:")

        print("Number of Customers who saw their favorite style: {}".format(percent_saw_favorite_style))
        print("Average Attractiveness when customer views their favourite style: {:.2f}".format(avgerage_attractiveness_for_favourite))


        ###### LEAVE EARLY STATS ######
        avg_num_paintings_left = self.num_painting_left_leave_early/self.num_leave_early if self.num_leave_early > 0 else 0

        print()
        print("Statistics for Customers who leave for Each Painting: {}".format(self.num_customers_leave_early))
        print("Average Score for painting that made Customers leave Early: {:.2f}".format(np.average(np.array(self.leave_early_scores))))
        print("Average number of paintings left when customer leaves early: {:.2f}%".format(avg_num_paintings_left/len(paintings) * 100))
        print("Percentage of Customers who leave early: {}%".format(self.num_leave_early/self.num_customers * 100))





        # print()
        # print("Statistics for Customers who leave for Each Painting: " + str(self.num_customers_leave_early))
        # print("Average Score for painting that made Customers leave Early: " + str(np.average(np.array(self.leave_early_scores))))
        # print("Average number of paintigs left when customer leaves early: " + str(avg_num_paintings_left))


        #timbo - create list of paintings for graph 
        #list_of_paintings = ["painting" +str(i+1) for i in range(self.num_paintings)]
            
        #plt.bar(list_of_paintings, self.num_painting_views)
        #plt.show()

        #create graph that shows percentage of favorite style seen
        #list_of_style_counts = [(self.num_baroque/self.num_customers)*100, (self.num_impressionist/self.num_customers)*100, (self.num_modern/self.num_customers)*100, (self.num_abstract/self.num_customers)*100]

        #plt.bar(["Baroque", "Impressionist", "Modern", "Abstract"], list_of_style_counts)
        #plt.show()
        






class GallerySim:
    def __init__(self, num_paintings: int, num_customers: int, seed: int, DEBUG=False): 
        self.DEBUG=DEBUG
        self.CustomersLeft = num_customers

        self.num_paintings = num_paintings
        self.num_customers = num_customers
        self.seed = seed

        self.rng = np.random.default_rng(seed=self.seed)

        self.paintings = [Painting(i, Style.random(self.rng), self.rng) for i in range(num_paintings)]

        self.stats = SimStats(self.num_customers, self.num_paintings)

        self.time = 0.0
        self.FutureEventList = EventList()

        self.customer = []


        # Schedule the first arrival
        self.ScheduleArrival()

        #THIS is our main loop (processes all events)
        while(self.stats.num_departed < self.num_customers):
            # get next event
            next_event = self.FutureEventList.dequeue()
            self.time = next_event.time

            if(self.DEBUG):
                print("Event Type: " + next_event.type.__str__() + " Time: " + str(self.time) + " Customer: " + str(next_event.customer.id))

            # process event
            if next_event.type == EventType.ARRIVAL:
                self.ProcessArrival(next_event)
            elif next_event.type == EventType.DEPARTURE:
                self.ProcessDeparture(next_event)
            elif next_event.type == EventType.MOVE: #sarah: changed this from VIEWING to MOVE to match rest of code
                self.ProcessMove(next_event)
            else:
                raise Exception("Invalid Event Type")

        #end of sim, print report:
        #we only want to consider the first customer_number departures, so delete the customers that have not yet departed
        self.customer = [i for i in self.customer if i.stats.departed]
        
        #print([c.id for c in self.customer])
        self.stats.printStats(self.customer, self.paintings) 
        # print('\n customer stats:')
        # print(['arrival time %.4f, depart time: %.4f, num paintings: %.4f, total view time: %.4f '
        #       %(c.stats.arrival_time, c.stats.departure_time, c.stats.num_paintings_viewed, c.stats.total_viewing_time) for c in self.customer])
        # print('customer score histories: ', [ str(c.stats.score_history) for c in self.customer])


    def generateInterArrivalTime(self):
        return np.clip(self.rng.exponential(INTERARRIVAL_TIME_MEAN), 0.00001, 100)
        #, INTERARRIVAL_TIME_STD)
        #sarah: rng.exponential takes parameters scale: float, and size: (int or tuple of ints).
        #   doesn't take a parameter for std dev because it is calculated from the rate (ie. std=mean)

    # def generateViewingTime(self):
    #     return np.clip(self.rng.normal(VIEWING_TIME_MEAN, VIEWING_TIME_STD), 0.00001, 100)

    # def generateTolerance(self):
    #     return self.rng.normal(TOLERANCE_MEAN, TOLERANCE_STD)



    def ProcessArrival(self, evt: Event):
        # So when the customer first arrives we basically just want to record the arrival then "move" them to their first painting
        evt.customer.stats.arrived = True
        self.stats.num_arrived += 1
        evt.customer.stats.arrival_time = evt.time #update stats for this customer 

        #if statements to count number of favourite styles
        if evt.customer.favorite_style == 0: #BAROQUE
            self.stats.num_baroque += 1
        elif evt.customer.favorite_style == 1: #IMPRESSIONIST
            self.stats.num_impressionist += 1
        elif evt.customer.favorite_style == 2: #MODERN  
            self.stats.num_modern += 1
        elif evt.customer.favorite_style == 3: #ABSTRACT
            self.stats.num_abstract += 1 

        #process initial move (this occurs immidiately after customer arrives)
        evt.type = EventType.MOVE #event is now Move
        self.ProcessMove(evt)

        # Schedule the next arrival
        self.ScheduleArrival()



    def ScheduleArrival(self):
        # Scheduling the next arrival        
        if(self.stats.num_arrived >= self.num_customers):
            return
        
        next_arrival_time = self.time + self.generateInterArrivalTime()

        new_cust = Customer(self.num_paintings, self.rng) #sarah: added rng
        new_cust.arrival_time = next_arrival_time

        # create the next arrival event
        arrival_event = Event(EventType.ARRIVAL, next_arrival_time, new_cust)

        self.customer.append(new_cust)

        ## add the arrival to the future event list
        self.FutureEventList.enqueue(arrival_event)


    def ProcessDeparture(self, evt: Event):
        #departures will always be processed directly from a Move (ie. not from main loop),
        # so the event will have already been removed from futureEventsList
        #all we need to do is update stats:
        self.stats.num_departed += 1
        evt.customer.stats.departure_time = evt.time
        evt.customer.stats.departed = True
        if(self.DEBUG):
            print('customer', evt.customer.id, 'leaving')




    def ProcessMove(self, evt: Event):
        # Min score at which point the customer will leave instead of going to the next painting

        customer = evt.customer

        #sarah: if customer is already at a painting, need to decrease that painting's num_viewers since somebody is leaving
        if(customer.stats.num_paintings_viewed > 0):
            prev_painting = customer.current_painting #customer.stats.score_history[-1][0]
            prev_painting.num_viewers -= 1
            # if(DEBUG):
            #     print(customer.id,' left painting', prev_painting.id, 'now there are %d viewers' %prev_painting.num_viewers)
        
        # get the painting with the highest score
        painting_scores = np.array([customer.scorePainting(p) for p in self.paintings])
        if(DEBUG):
            print(painting_scores)

        self.stats.painting_scores.extend([x for x in painting_scores if x > 0])

        bestIndex = np.argmax(painting_scores)
        best_painting = self.paintings[bestIndex]

        if(painting_scores[bestIndex] < MIN_SCORE):
            # If the person is leaving early
            if(customer.stats.num_paintings_viewed < self.num_paintings):
                self.stats.num_leave_early += 1
                self.stats.num_customers_leave_early[customer.stats.num_paintings_viewed] += 1
                self.stats.leave_early_scores.append(painting_scores[bestIndex])
                self.stats.num_painting_left_leave_early += self.stats.num_paintings - customer.stats.num_paintings_viewed
            

            # EVent is now a departure
            evt.type = EventType.DEPARTURE
            self.ProcessDeparture(evt)
            return #otherwise customer continues to view painting

        #add view number to painting
        self.stats.num_painting_views[bestIndex] += 1
        if(DEBUG):
            print(self.stats.num_painting_views)

        #check if customer is seeing their favourite style
        if(best_painting.style == evt.customer.favorite_style):
            customer.stats.saw_favorite_style = True

            #keep track of total number of attraciveness levels when all customers see their favourite style
            self.stats.attractiveness_for_favourite += customer.scorePainting(best_painting)


        # begin viewing the painting
        viewing_time = customer.beginViewing(best_painting, self.time)
        customer.viewedPaintings[bestIndex] = True
        customer.current_painting = best_painting

        ## add viewing time to stats
        self.stats.total_viewing_time += viewing_time
        self.stats.num_paintings_viewed += 1

        # Schedule the next MOVE (sarah: for this customer right?)
        self.FutureEventList.enqueue(Event(EventType.MOVE, self.time + viewing_time, customer))

        customer.stats.total_viewing_time += viewing_time
        customer.stats.num_paintings_viewed += 1
        customer.stats.score_history.append((best_painting, painting_scores[bestIndex])) #adding as tuple so we can keep track of both

        return


def main():
    '''Produce data for multiple scenarios (for each scenario, run simulation w/ 5 different initial random seeds)
        and process collected data.'''
    test = GallerySim(50,1000,20, False)

if __name__ == '__main__':
    main()

