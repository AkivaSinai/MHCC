import numpy as np
import  Utils
import json

with open('models/models_meta_data.json', encoding="utf8") as json_file:
    models_meta_data = json.load(json_file)

WAIT_KNOWN= False
#WAIT_CONSTANT= False
NAIVE_WAIT= False
AVERAGE_WAIT= 7.5


def survival_function(device, age, gender):
    if(gender==-1):
        gender=0
    dist_parameter= float(models_meta_data["waiting time"]["mean"][str(device)][str(gender)][str(age)])
    return np.random.exponential(dist_parameter) #todo change to based on class


"""

For good results- need good variance in feedback prediction (not only 5!)- maybe use suboptimal model for that 
(not random forest) or maybe random  forest models with more features (30 ) 
Also (or insted)- need to have more variance in duration of calls and also short calls 
! now I have bbetween 24 and 82 ant that seems to give good results
Without good vairance, it highly depends on the waiting time 
waiting time distributions :
generally, it seems that when the waiting time is higher (and the model is divded into types)- monte carlo outperforms all 


good 
max(np.random.normal(6,4),0)
max(np.random.normal(5,3),0) -not great

bad 
max(np.random.normal(5,2),0)
max(np.random.exponential(6),0)
return max(np.random.exponential(4),0)


"""
#


class Visit(object):
    """
    Visitor class
    Includes relevant Visitor data for Eran Scheduling

    Attributes
    ----------
    Visit dictionary (dict)- Includes all of the of visitor's features
    Arrival time (int)- the time the visistor requested a chat

    """


    def __init__(self, visit_dictionary, arrival_time):
        self.id = visit_dictionary["id"]
        self.age= visit_dictionary["vi age"]
        self.gender = visit_dictionary["vi gender"]
        #self.device = visit_dictionary["vi device"]
        #self.visitor_stats = visit_dictionary["visitor_stats"]
        self.revisit= visit_dictionary["visitor_stats"] is not None
        self.feature_dictionary= visit_dictionary   # todo the parameters above sholud be erased later
        self.waiting=  True
        self.abandoned= False
        self.completed= False
        self.chatting= False
        self.arrival_time= arrival_time

        # todo- think about waiting time. I think we will need to give a few estimations for the distribution of the waiting time
        #it seems like bigger variance in waiting time gives better results
        self.predicted_waiting_time= arrival_time +round(survival_function(self.feature_dictionary[Utils.FN.Visitor.device], self.feature_dictionary[Utils.FN.Visitor.age],self.feature_dictionary[Utils.FN.Visitor.gender]))#self.calculate_waiting_time()
        if(NAIVE_WAIT):
            self.predicted_waiting_time= AVERAGE_WAIT
        self.real_waiting_time=  arrival_time +round(survival_function(self.feature_dictionary[Utils.FN.Visitor.device], self.feature_dictionary[Utils.FN.Visitor.age],self.feature_dictionary[Utils.FN.Visitor.gender]))#self.calculate_waiting_time()
        if(WAIT_KNOWN):
            self.real_waiting_time =   self.predicted_waiting_time
        # the real waiting time is generated from the same distribution
        #self.real_waiting_time =self.predicted_waiting_time
        # self.real_waiting_time =  self.predicted_waiting_time #round(arrival_time +survival_function(self.feature_dictionary[Utils.FN.Visitor.device], self.feature_dictionary[Utils.FN.Visitor.age]) )# self.predicted_waiting_time#
        #
        # self.predicted_waiting_time + np.random.normal(models_meta_data["duration"]["error distribution"][0]/60,
        #                                                                  models_meta_data["duration"]["error distribution"][1]/100) # todo- predict this value

        #self.predicted_waiting_time= max(self.predicted_waiting_time, 0)
        #self.predicted_waiting_time = min(self.predicted_waiting_time, 60)

    def set_visitor_stats(self, visitor_stats):
        self.visitor_stats= visitor_stats
    def get_visit_dictionarty(self):
        visit_dictionary= {}
        visit_dictionary["age"]=self.age
        visit_dictionary["gender"]=self.gender
        visit_dictionary["device"]=self.device
        visit_dictionary["needs"]=self.needs
        visit_dictionary["wrote_background"]=self.wrote_background
        visit_dictionary["revisit"]=self.revisit
        visit_dictionary["visitor_stats"]=self.visitor_stats
        return visit_dictionary
    def calculate_waiting_time(self):
        # todo- the bigger the variance is- better results
        wt= max(0, np.random.normal(7,2))

        return  wt



class AgentStatus(object):
    """

    """
    def __init__(self, id, number_of_chats=0, talking_time=0, finish_talk_at=None, real_finish_talk_at= None ):
        self.id = id
        self.number_of_chats= number_of_chats
        self.talking_time= talking_time
        self.finish_talk_at= finish_talk_at
        self.real_finish_talk_at= real_finish_talk_at

    def update_new_chat(self, duration, time=0):
        self.number_of_chats+=1
        self.talking_time+= duration
        if(self.finish_talk_at is not None):
            self.finish_talk_at+= duration
        else:
            self.finish_talk_at= time+ duration

    def update_new_chat_real(self, predicted_duration, real_duration, time=0):
        self.number_of_chats+=1
        self.talking_time+= real_duration
        if(self.finish_talk_at  is not None and self.finish_talk_at>=time):
            self.finish_talk_at+= predicted_duration
            self.real_finish_talk_at += real_duration
        else:
            self.finish_talk_at= time+ predicted_duration
            self.real_finish_talk_at =time+   real_duration

    def update_status_at_time(self, time):
        if(self.finish_talk_at is not None):
            if (self.finish_talk_at<= time):
                self.finish_talk_at= None


class Agent(object):
    """
        Contains features of and agent
        The featurse include : 1. Demographic Features 2. Statistics from chats 3. Statistics of the conversation
    """
    def __init__(self, agent_dictionary):
        self.id = agent_dictionary["id"]
        self.age= agent_dictionary["ag age"]
        self.gender = agent_dictionary["ag gender"]
        self.special= agent_dictionary["ag special"]
        self.experience= agent_dictionary["ag experience"]
        self.general_statitics= agent_dictionary["agent_statistics"]
        self.agent_by_visitor_stats = agent_dictionary["agent_visitor_stats"]
        self.number_of_chats= 0
        self.talking_time_from_shift_beggining= 0
        self.free= True
        self.finish_talk_at= None
        self.feature_dictionary = agent_dictionary

    def update_agents_time(self, duration, time=0):
        self.number_of_chats+=1
        self.talking_time_from_shift_beggining+= duration
        if(self.finish_talk_at is not None):
            self.finish_talk_at+= duration
        else:
            self.finish_talk_at= time+ duration

    def update_new_chat(self, duration, time=0):
        self.number_of_chats+=1
        self.talking_time_from_shift_beggining+= duration
        if(self.finish_talk_at is not None and self.finish_talk_at>=time):
            self.finish_talk_at+= duration
        else:
            self.finish_talk_at= time+ duration
