import json

import numpy as np

import Utils
from MHCC_Simulator import models_meta_data, prediction_cache, feedback_classiffier, duration_regressor, \
    MODEL_INACCURACY_MISTAKES, Naive_Feedback, AVERAGE_FEEDBACK, Naive_Duration, AVERAGE_DURATION, visits_bank, \
    agents_bank
from Simulator_Actors import Agent, AgentStatus, Visit


def get_model_feature_values(agent: Agent, agents_talking_status: AgentStatus, visitor: Visit, model_name):

    feature_values=[]
    model_features=models_meta_data[model_name]["selected features"]
    for feature in model_features:
       if(feature[0:2]=="ag"):
            if feature in agent.feature_dictionary:
                value= agent.feature_dictionary[feature]
            elif(feature == Utils.FN.Agent.average_feedback_by_gender):
                value=agent.feature_dictionary["agent_visitor_stats"][feature][str(visitor.gender)]
            elif(feature == Utils.FN.Agent.average_feedback_by_age):
                value=agent.feature_dictionary["agent_visitor_stats"][feature][list(Utils.AGE_RANGE.keys())[visitor.age]]
            elif feature in agent.feature_dictionary["agent_statistics"]:
                value= agent.feature_dictionary["agent_statistics"][feature]
            elif feature in agent.feature_dictionary["agent_visitor_stats"]:
                value= agent.feature_dictionary["agent_visitor_stats"][feature]
            # new features from the current shift
            elif feature==  Utils.FN.Agent.shift_number_of_chats:
                value= agents_talking_status.number_of_chats
            elif feature==  Utils.FN.Agent.shift_total_talking_time or feature==  Utils.FN.Agent.shift_total_talking_time+'.1':
                value= agents_talking_status.talking_time
            elif feature==  Utils.FN.Agent.average_duration:
                value= agent.feature_dictionary["agent_statistics"]["avg_duration"]
            else:
                value= 10
       elif(feature[0:2]=="vi"):
           if feature in visitor.feature_dictionary:
               value = visitor.feature_dictionary[feature]
           #elif feature.startswith('vi needs'):
           elif visitor.feature_dictionary["visitor_stats"] is not None and  feature in visitor.feature_dictionary["visitor_stats"]:
               value = visitor.feature_dictionary["visitor_stats"][feature]
           elif(feature== "vi revisit"):
               value = visitor.revisit
       else:
           raise Exception("feature not recognized")
       if(value== None):
           value=0
       value= float(value)
       feature_values.append(value)

    return feature_values


def get_predcition_error_parameters(duration_prediction_error_dict, time_to_end):
    for value in duration_prediction_error_dict.values():
        range= value["range"]
        if(time_to_end>= float(range[0]) and time_to_end< float(range[1])):
            return value["dist values"]
    return value["dist values"]  # in case the time to end is not in of the ranges

"""
This function updates the estimated duration of that the chats that are in currently ongoing
"""


def update_prediction_errors(state, towards_end_prediction= True):
    time= state.time
    #duration_prediction_error_dict=models_meta_data["duration"]["error distribution divided"]
    duration_prediction_error_dict =models_meta_data["remaining time"]["error distribution divided"]
    for agent in state.agent_talking_status:
        #status=state.agent_talking_status[agent]
        if(state.agent_talking_status[agent].finish_talk_at is not None and state.agent_talking_status[agent].finish_talk_at>time and state.agent_talking_status[agent].real_finish_talk_at>time):
            status= state.agent_talking_status[agent]
            real_time_until_the_end= status.real_finish_talk_at- time
            if(towards_end_prediction and  real_time_until_the_end<= 5): # in case the chat is almost finished - use the binary model
                status.finish_talk_at = status.real_finish_talk_at + np.random.normal(0,
                                                      2)
            else:
                error_params=get_predcition_error_parameters(duration_prediction_error_dict, real_time_until_the_end) # the regression model
                updated_inaccuracy_mistake = np.random.normal(error_params[0],
                                                      error_params[1])
                status.finish_talk_at= status.real_finish_talk_at  + updated_inaccuracy_mistake

"""
Returns the prediction of feedback and duration of a chat for each possible 

The prediction models are pre-trained and based on various features
"""
def predict_feedbacks_and_durations(agents_in_shift, agent_talking_status, available_agents_ids,  waiting_visitors,  prediction_error=True):
    prediction_dictionary= {}
    # predict the values of the feedback and duration using the trained models- add a random number in order to
    # to simulate the inaccuracy of the model
    for agent in agents_in_shift:
        if(agent.id in available_agents_ids):

            prediction_dictionary[agent.id]= {}
            for visitor in waiting_visitors:

                prediction_dictionary[agent.id][visitor.id]= {}

                feedback_feature_values= get_model_feature_values(agent, agent_talking_status[agent.id],visitor, model_name="feedback")
                duration_feature_values = get_model_feature_values(agent,agent_talking_status[agent.id], visitor, model_name="duration")

                if((agent.id, visitor.id) in prediction_cache["feedback"]):

                    prediction_dictionary[agent.id][visitor.id]["feedback"]=prediction_cache["feedback"][(agent.id, visitor.id)]
                    #predicted_feedback= prediction_cache["feedback"][(agent.id, visitor.id)]
                    prediction_dictionary[agent.id][visitor.id]["duration"]= prediction_cache["duration"][(agent.id, visitor.id)]
                else:
                    predicted_feedback = feedback_classiffier.predict([feedback_feature_values])[
                                             0]  #- np.random.randint(0,2)
                    predicted_duration = duration_regressor.predict([duration_feature_values])[
                                             0] # +np.random.randint(-10,10)# the duration predicton is in seconds
                    predicted_duration= max(predicted_duration,3 )/2
                    #predicted_feedback= agent.feature_dictionary["agent_statistics"]

                    inaccuracy_mistake = np.random.normal(models_meta_data["feedback"]["error distribution"][0],
                                                        models_meta_data["feedback"]["error distribution"][1])  #


                    inaccuracy_mistake/=2
                    if(not prediction_error or MODEL_INACCURACY_MISTAKES==False):
                        inaccuracy_mistake=0
                    #inaccuracy_mistake=0
                    feedback = np.round(predicted_feedback + inaccuracy_mistake)
                    predicted_feedback= max(predicted_feedback,1)
                    predicted_feedback= min(5,predicted_feedback)
                    feedback= max(feedback,1)
                    feedback= min(5,feedback)
                    if (Naive_Feedback):
                        predicted_feedback = AVERAGE_FEEDBACK
                    prediction_dictionary[agent.id][visitor.id]["feedback"]= {"predicted" :predicted_feedback,
                                                                               "real": feedback}

                    predicted_duration= round(max(predicted_duration,3))
                    duration_inaccuracy_mistake =  np.random.normal(models_meta_data["duration"]["error distribution"][0],
                                          models_meta_data["duration"]["error distribution"][1])

                    if(not prediction_error or MODEL_INACCURACY_MISTAKES==False):
                        duration_inaccuracy_mistake=0

                    predicted_duration= max(predicted_duration,3)
                    predicted_duration= min(50,predicted_duration)
                    predicted_duration= int(predicted_duration)
                    #duration_inaccuracy_mistake=0
                    duration= predicted_duration+ duration_inaccuracy_mistake
                    duration= max(duration,3)
                    duration= min(80,duration)


                    if(Naive_Duration):
                        predicted_duration=AVERAGE_DURATION
                    prediction_dictionary[agent.id][visitor.id]["duration"] = {"predicted" :np.round(predicted_duration),
                                                                               "real": np.round(duration)}
                    prediction_cache["feedback"][(agent.id, visitor.id)]= {"predicted" :predicted_feedback,
                                                                               "real": feedback}#- np.random.randint(0,1)
                    prediction_cache["duration"][(agent.id, visitor.id)]= {"predicted" :np.round(predicted_duration),
                                                                               "real": np.round(duration)}# - np.random.randint(8,12)

    return prediction_dictionary


def filter_visitor_by_revisit():
    filtered_visits_bank=[]
    for visit in visits_bank:
        if(visit["revisit"]==1):
            filtered_visits_bank.append(visit)
    return filtered_visits_bank


def filter_agent_by_feedback():
    filtered_agent_bank=[]
    for agent in agents_bank:
        if(agent["agent_statistics"]["ag avg fbk"]and (agent["agent_statistics"]["ag avg fbk"]<4.2 or agent["agent_statistics"]["ag avg fbk"]>4.8 )):
            filtered_agent_bank.append(agent)

    with open('agents_bank.json', "w") as write_file:
        json.dump(models_meta_data, write_file)
    return filtered_agent_bank